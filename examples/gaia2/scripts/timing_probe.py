"""Per-event timing instrumentation for one charged-clock Gaia2 run.

Answers a question the run's own artifacts cannot: for each oracle event the judge gates on time,
how much of the post-event tolerance went on getting the event *to* the agent, and how much on the
agent's own charged reaction.  ``llm_calls.jsonl`` already carries the charge each crossing billed
and where it landed on the charged axis; what is missing is the pair of instants that bracket a
reaction -- when the scheduled world event was actually dispatched, and when the agent's matching
action was recorded.

Three record kinds go to one JSONL:

``meta``      the run's own identifying fields, written once by the caller.
``dispatch``  one per environment event executed, with its scheduled time and the charged-clock
              reading at the moment it was dispatched.  The gap between the two is delivery
              lateness: time the agent never had.
``gate``      one per ``AgentEventJudge.check_time`` call -- the judge's own inputs, the branch it
              took, and its verdict.  ``gated`` is False for the calls that return True without
              comparing anything, which is most of them outside the Time split.

Patching rather than a constructor argument is deliberate.  This measures the shipped path, and the
shipped path is what the sweep runs; a probe that needed an argument threaded through the runner
would be measuring a different configuration from the one in question.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_stream: Any = None
_installed = False


def _write(record: dict[str, Any]) -> None:
    if _stream is None:
        return
    with _lock:
        _stream.write(json.dumps(record, default=str) + "\n")
        _stream.flush()


def _action_label(event: Any) -> str | None:
    action = getattr(event, "action", None)
    if action is None:
        return getattr(event, "tool_name", None)
    try:
        return f"{action.app_name}.{action.function_name}"
    except Exception:
        return str(action)


def install(path: str | Path) -> None:
    """Patch ARE's dispatch and timing gate to append to ``path``.  Idempotent."""
    global _stream, _installed
    if _installed:
        return

    from are.simulation.environment import Environment
    from are.simulation.validation.event_judge import AgentEventJudge

    _stream = Path(path).open("a", encoding="utf-8")
    _installed = True

    original_process_event = Environment.process_event
    original_check_time = AgentEventJudge.check_time

    def process_event(self: Any, event: Any) -> Any:
        # Read the clock before the handler runs.  Dispatching an event can itself advance the
        # world -- an app that pauses, a successor that ticks -- so sampling afterwards would
        # attribute the handler's own cost to delivery lateness.
        try:
            charged_now = self.time_manager.time()
        except Exception:
            charged_now = None
        _write(
            {
                "probe": "dispatch",
                # A scored run dispatches every event twice: once in the oracle-mode pass that
                # builds the judge's graph, then again in the agent run.  Only the second is
                # subject to the freeze, so pooling the two halves the apparent lateness and hides
                # the finding.  The environment instance is the discriminator -- the passes are
                # different objects, and usually different classes.
                "env_class": type(self).__name__,
                "env_instance": id(self),
                "event_id": getattr(event, "event_id", None),
                "event_type": str(getattr(event, "event_type", None)),
                "action": _action_label(event),
                "scheduled_time": getattr(event, "event_time", None),
                "relative_time": getattr(event, "event_relative_time", None),
                "delivered_charged_time": charged_now,
                "tick_reference_time": getattr(self, "current_time", None),
                "tick_count": getattr(self, "tick_count", None),
                "dependencies": [
                    getattr(dep, "event_id", None)
                    for dep in (getattr(event, "dependencies", None) or [])
                ],
            }
        )
        return original_process_event(self, event)

    def check_time(
        self: Any,
        agent_event: Any,
        oracle_event: Any,
        max_parent_oracle_event_time: float,
        max_parent_agent_event_time: float,
    ) -> bool:
        verdict = bool(
            original_check_time(
                self,
                agent_event=agent_event,
                oracle_event=oracle_event,
                max_parent_oracle_event_time=max_parent_oracle_event_time,
                max_parent_agent_event_time=max_parent_agent_event_time,
            )
        )
        # Re-derive the branch rather than infer it from the verdict: `check_time` returns True
        # both when the agent was punctual and when nothing was compared at all, and telling those
        # two apart is the whole point of the measurement.
        comparator = (
            oracle_event.event_time_comparator.value
            if getattr(oracle_event, "event_time_comparator", None)
            else None
        )
        absolute = getattr(oracle_event, "absolute_event_time", None)
        agent_time = getattr(agent_event, "event_time", None)
        oracle_time = getattr(oracle_event, "event_time", None)
        if absolute is not None:
            gated, basis = True, "absolute"
            agent_axis: float | None = agent_time
            oracle_axis: float | None = absolute
        else:
            agent_axis = None if agent_time is None else agent_time - max_parent_agent_event_time
            oracle_axis = (
                None if oracle_time is None else oracle_time - max_parent_oracle_event_time
            )
            gated = bool(
                oracle_axis is not None
                and (
                    oracle_axis > self.config.check_time_threshold_seconds or comparator is not None
                )
            )
            basis = "relative"
        lateness = None if (agent_axis is None or oracle_axis is None) else agent_axis - oracle_axis
        _write(
            {
                "probe": "gate",
                "gated": gated,
                "basis": basis,
                "verdict": verdict,
                "comparator": comparator,
                "oracle_event_id": getattr(oracle_event, "event_id", None),
                "agent_event_id": getattr(agent_event, "event_id", None),
                "oracle_tool": getattr(oracle_event, "tool_name", None),
                "agent_tool": getattr(agent_event, "tool_name", None),
                "oracle_event_time": oracle_time,
                "agent_event_time": agent_time,
                "max_parent_oracle_event_time": max_parent_oracle_event_time,
                "max_parent_agent_event_time": max_parent_agent_event_time,
                "oracle_axis_time": oracle_axis,
                "agent_axis_time": agent_axis,
                "lateness_seconds": lateness,
                "post_event_tolerance_seconds": self.config.post_event_tolerance_seconds,
                "pre_event_tolerance_seconds": self.config.pre_event_tolerance_seconds,
                "check_time_threshold_seconds": self.config.check_time_threshold_seconds,
                "remaining_budget_seconds": (
                    None
                    if lateness is None
                    else self.config.post_event_tolerance_seconds - lateness
                ),
            }
        )
        return verdict

    Environment.process_event = process_event
    AgentEventJudge.check_time = check_time


def record_meta(**fields: Any) -> None:
    payload = {
        key: (asdict(value) if is_dataclass(value) and not isinstance(value, type) else value)
        for key, value in fields.items()
    }
    _write({"probe": "meta", **payload})


def close() -> None:
    global _stream
    if _stream is not None:
        with _lock:
            _stream.close()
            _stream = None
