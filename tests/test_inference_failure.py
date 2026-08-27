"""What happens when a *deliberation* call fails — the disposition of a failed infer/ground.

An inference that raised is not evidence that the goal is unreachable. Nothing was attempted, the
world is untouched, and a fresh attempt is free to differ — so the failure degrades rather than
killing the activity, and the runaway-replan breaker (not a counter of its own) decides when trying
again has stopped being an attempt. Two things that used to be silent are now not: the activity's
end reaches episodic memory, and the user hears about it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fakes import ScriptedTransport
from sora.action import default_action_registry, invoke_step
from sora.activity import Activity, ActivityState
from sora.cycle import DecisionCycle
from sora.environment import EnvironmentRegistry
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
    TickResult,
    _inference_defect,
    _report_to_user,
)
from sora.types import InferenceResult, PendingInference, Plan, Step

_PLAN_DEFECT = "the plan inference did not return a usable result (ValueError)"


def _cycle(tmp_path: Path) -> tuple[DecisionCycle, WorkingMemory, ScriptedTransport]:
    registry = EnvironmentRegistry(adapters={})
    working = WorkingMemory(registry=registry)
    transport = ScriptedTransport()
    cycle = DecisionCycle(
        strategies=Strategies(
            observe=DefaultObserveStrategy(),
            reflect=DefaultReflectStrategy(),
            situate=DefaultSituateStrategy(),
            reason=DefaultReasonStrategy(),
            act=DefaultActStrategy(),
        ),
        communication=transport,
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=SemanticMemory(FileMemoryBackend(tmp_path / "semantic")),
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "procedural")),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return cycle, working, transport


def _inferring(kind: str, *, plan: Plan | None = None, **fields: Any) -> Activity:
    return Activity(
        id="a1",
        goal="book the day",
        context={},
        state=ActivityState.RUNNING,
        plan=plan,
        pending_inference=PendingInference(id="inf-1", kind=kind, requested_at=0.0),
        **fields,
    )


def _step(name: str) -> Step:
    return invoke_step("t1", name)


# --------------------------------------------------------------------------------------------------
# the defect string is what the breaker compares, so it has to survive a reworded error
# --------------------------------------------------------------------------------------------------


def test_the_defect_normalizes_away_the_quoted_output() -> None:
    """Two parse failures quote different model output. If the trail carried that verbatim they
    would never compare equal, the precise "same reason twice" check could never fire, and a
    hopeless call would be paid for max_replan_attempts times instead of twice."""
    first = _inference_defect("plan", "ValueError('bad JSON near {\"steps\": [')")
    second = _inference_defect("plan", "ValueError('bad JSON near {\"pending\": ')")
    assert first == second == _PLAN_DEFECT


def test_the_defect_still_distinguishes_two_different_causes() -> None:
    """Normalization must not flatten everything into one string — a wire failure and a parse
    failure are different attempts, and the coarse count is what should bound those."""
    assert _inference_defect("plan", "TimeoutError()") != _inference_defect("plan", "ValueError()")


def test_an_error_without_a_repr_shape_is_carried_whole() -> None:
    assert "boom" in _inference_defect("plan", "boom")


# --------------------------------------------------------------------------------------------------
# a failed plan/sub-goal/ground inference degrades to a replan
# --------------------------------------------------------------------------------------------------


async def test_a_failed_subgoal_inference_replans_rather_than_killing_the_activity(
    tmp_path: Path,
) -> None:
    """The sharpest case for not terminating: the sub-plan is what failed, and the parent plan is
    sitting right there intact. Killing the activity throws away work that nothing is wrong with."""
    cycle, working, _ = _cycle(tmp_path)
    parent = Plan(id="p1", goal="book the day", steps=[_step("search"), _step("send")])
    activity = _inferring("subgoal", plan=parent, step_index=1)
    working.activities["a1"] = activity
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", error="ValueError('nope')"))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.READY
    # The parent is not lost: it is parked for the re-inference to reuse, tagged with why the
    # sub-goal it contained could not be expanded.
    superseded = activity.superseded
    assert superseded is not None
    assert superseded.plan is parent
    assert superseded.step_index == 1
    assert superseded.defect == "the subgoal inference did not return a usable result (ValueError)"


async def test_a_failed_plan_inference_leaves_nothing_stale_behind(tmp_path: Path) -> None:
    cycle, working, _ = _cycle(tmp_path)
    activity = _inferring("plan", grounded_params={"to": "someone"})
    working.activities["a1"] = activity
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", error="ValueError('nope')"))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.READY
    assert activity.pending_inference is None
    assert activity.plan is None
    # Params grounded for a plan that never arrived must not survive into the next one.
    assert activity.grounded_params is None


# --------------------------------------------------------------------------------------------------
# the breaker, not a counter of its own, is what stops the retrying
# --------------------------------------------------------------------------------------------------


async def test_a_second_identical_failure_asks_the_user_instead_of_inferring_again(
    tmp_path: Path,
) -> None:
    """The whole point of routing through the replan path: repeated failure — including a permanent
    one like "no LLM is configured" — converges on a question rather than on either an infinite
    retry or a silent death. No new policy knob: `_replanning_would_loop` already trips at two
    identical defects, which is why the defect string is normalized."""
    cycle, working, transport = _cycle(tmp_path)
    # One failure already on the trail, no operation run since (so nothing forgives it).
    activity = _inferring("plan", replan_trail=[_PLAN_DEFECT])
    working.activities["a1"] = activity
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", error="ValueError('again')"))

    await DefaultObserveStrategy().observe(cycle)
    assert activity.replan_trail == [_PLAN_DEFECT, _PLAN_DEFECT]

    # Reason gates on the breaker *before* spending another inference — there is no LLM wired to
    # this cycle, so reaching one at all would raise here.
    await DefaultReasonStrategy().reason(activity, working, cycle, TickResult())

    assert activity.state is ActivityState.BLOCKED
    wait = activity.blocked_on
    assert wait is not None
    prompt = getattr(wait, "prompt", "")
    assert "book the day" in prompt
    assert _PLAN_DEFECT in prompt  # the question quotes what actually went wrong, twice over
    # And the question was *asked*, not merely recorded on the activity.
    assert transport.sent == [("user", {"text": prompt})]


async def test_one_failure_is_not_enough_to_halt(tmp_path: Path) -> None:
    """A single bad response is a slip, not a pattern — it must still get a second attempt, which
    is the case the JSON repair and the one re-inference exist to rescue."""
    cycle, working, transport = _cycle(tmp_path)
    activity = _inferring("plan")
    working.activities["a1"] = activity
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", error="ValueError('once')"))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.READY
    assert activity.blocked_on is None
    assert transport.sent == []  # nothing to tell the user yet


# --------------------------------------------------------------------------------------------------
# the residual terminate path is no longer silent
# --------------------------------------------------------------------------------------------------


async def test_an_undegradable_failure_records_an_episode_and_says_so(tmp_path: Path) -> None:
    """Terminating is right when there is no defined way to continue, but it must not be invisible.
    It used to be both: no episode (so Reflect's "TERMINATED was already recorded" was untrue for
    this path, and memory never saw the failure) and no word to the user."""
    cycle, working, transport = _cycle(tmp_path)
    activity = _inferring("divine")  # a kind with no degradation of its own
    working.activities["a1"] = activity
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", error="ValueError('nope')"))

    await DefaultObserveStrategy().observe(cycle)

    assert activity.state is ActivityState.TERMINATED
    episodes = await cycle.episodic.consult(activity)
    assert len(episodes) == 1
    assert episodes[0]["succeeded"] is False
    assert "divine" in episodes[0]["summary"]
    assert len(transport.sent) == 1
    to, content = transport.sent[0]
    assert to == "user"
    assert "book the day" in content["text"]


async def test_a_degraded_failure_records_no_episode(tmp_path: Path) -> None:
    """The mirror: an activity that is going to try again has not ended, so writing an episode for
    it would be a claim about an outcome that has not happened."""
    cycle, working, transport = _cycle(tmp_path)
    activity = _inferring("plan")
    working.activities["a1"] = activity
    cycle.inference_sink.push("inf-1", InferenceResult(id="inf-1", error="ValueError('nope')"))

    await DefaultObserveStrategy().observe(cycle)

    assert await cycle.episodic.consult(activity) == []
    assert transport.sent == []


# --------------------------------------------------------------------------------------------------
# reporting must never become the new failure
# --------------------------------------------------------------------------------------------------


class _DeadTransport(ScriptedTransport):
    async def send(self, to: str, content: dict[str, Any]) -> None:
        raise ConnectionError("the channel is gone")


async def test_a_dead_channel_does_not_mask_what_was_being_reported(tmp_path: Path) -> None:
    """Every caller of `_report_to_user` is already delivering bad news. A transport that raises
    must not replace that failure with a different one thrown from the reporting itself."""
    cycle, _, _ = _cycle(tmp_path)
    cycle.communication = _DeadTransport()
    await _report_to_user(cycle, "I got stuck.")  # must not raise


# --------------------------------------------------------------------------------------------------
# an inference that never comes back at all
# --------------------------------------------------------------------------------------------------


async def test_an_inference_that_never_returns_is_given_up_on(tmp_path: Path) -> None:
    """The failure mode ADR-0021's stale guard cannot see: it is identity-based, so it discards a
    LATE result but never notices an ABSENT one. Nothing is pushed here at all — the activity's
    only way out of RUNNING is the deadline. On the run that motivated this, one plan inference
    held the agent for ~14 minutes and cost it the scenario's whole real-time budget."""
    cycle, working, _ = _cycle(tmp_path)
    activity = _inferring("plan")  # requested_at=0.0 — long expired against the host clock
    working.activities["a1"] = activity

    await DefaultObserveStrategy(inference_deadline=1.0).observe(cycle)

    assert activity.pending_inference is None
    assert activity.state is ActivityState.READY  # degraded to a replan, not stranded, not killed


async def test_a_call_still_within_its_deadline_is_left_alone(tmp_path: Path) -> None:
    """The watchdog must not become a latency budget: a thinking model legitimately spends tens of
    seconds, and expiring a call that was about to succeed buys a replan nobody needed."""
    cycle, working, _ = _cycle(tmp_path)
    activity = _inferring("plan")
    activity.pending_inference = PendingInference(id="inf-1", kind="plan", requested_at=time.time())
    working.activities["a1"] = activity

    await DefaultObserveStrategy(inference_deadline=300.0).observe(cycle)

    assert activity.pending_inference is not None
    assert activity.state is ActivityState.RUNNING


async def test_the_deadline_can_be_disabled(tmp_path: Path) -> None:
    """An interactive session may want to wait indefinitely; `None` is not "use the default"."""
    cycle, working, _ = _cycle(tmp_path)
    activity = _inferring("plan")
    working.activities["a1"] = activity

    await DefaultObserveStrategy(inference_deadline=None).observe(cycle)

    assert activity.pending_inference is not None
    assert activity.state is ActivityState.RUNNING


async def test_a_result_that_arrives_after_the_deadline_is_discarded(tmp_path: Path) -> None:
    """Nothing cancels the underlying call — an LLM call cannot be cut mid-generation — so the
    expired request may still answer. It must not resurrect an activity that has already moved on,
    which is exactly what the existing stale-inference guard is for; the deadline reuses it rather
    than adding a second rule."""
    cycle, working, _ = _cycle(tmp_path)
    activity = _inferring("plan")
    working.activities["a1"] = activity

    await DefaultObserveStrategy(inference_deadline=1.0).observe(cycle)
    replanning = activity.state
    cycle.inference_sink.push(
        "inf-1", InferenceResult(id="inf-1", value=Plan(id="p9", goal="late", steps=[_step("go")]))
    )
    await DefaultObserveStrategy(inference_deadline=1.0).observe(cycle)

    assert replanning is ActivityState.READY
    assert activity.plan is None or activity.plan.id != "p9"
