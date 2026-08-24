"""Extensible action space: internal actions (memory) and external actions (the world)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from sora.activity import Activity, ActivityState
from sora.llm import current_inference_id
from sora.manual import ToolRecord, WorkspaceRecord
from sora.types import (
    OPERATION_NAME,
    TOOL_ID,
    ActionAck,
    InferenceResult,
    OperationInvocation,
    PendingInference,
    PendingOperation,
    Step,
    UnresolvableGrounding,
    walk_path,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from sora.cycle import DecisionCycle
    from sora.environment import EnvironmentRegistry, Tool, WorkspaceOrigin
    from sora.manual import Manual

log = logging.getLogger("sora.action")


def _spawn_tracked(tasks: set[asyncio.Task[None]], coro: Coroutine[Any, Any, None]) -> None:
    """Fire a background coroutine and hold a strong ref to its task until it finishes, so it isn't
    GC'd mid-flight. The shared off-cycle-dispatch idiom behind _invoke_/_infer_/_ground_ — each
    action keeps its own task set (distinct lifetimes) but routes the spawn through here."""
    task = asyncio.create_task(coro)
    tasks.add(task)
    task.add_done_callback(tasks.discard)


class InternalAction(Protocol):
    name: str

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> Any:
        """No EnvironmentRegistry access — internal actions only ever touch memory."""
        ...


class ExternalAction(Protocol):
    name: str
    # Whether the cycle's Act phase must do *parameter binding* on this step before dispatch —
    # grounding its abstract Step into a concrete, schema-conformant OperationInvocation (not a
    # *protocol binding*, which is the adapter's Tool concern — see ADR-0015). Only _invoke_ does;
    # every other external action dispatches straight from its Step params.
    requires_binding: bool

    async def execute(
        self,
        registry: EnvironmentRegistry,
        cycle: DecisionCycle,
        *,
        activity_id: str,
        **kwargs: Any,
    ) -> ActionAck:
        """Narrower than passing a whole Agent: registry (from working memory) + cycle
        (memory/transport/sinks), nothing else. `activity_id` is always passed by tick()'s
        dispatch, absorbed harmlessly by actions that don't need it (all but _invoke_)."""
        ...


class ActionRegistry:
    def __init__(self) -> None:
        self._internal: dict[str, InternalAction] = {}
        self._external: dict[str, ExternalAction] = {}
        # Data-ops (ADR-0023) are InternalActions too, but kept in their own bucket: they're the set
        # a *plan* may compose as steps, dispatched by Reason. Keeping them out of `_internal`
        # bounds what a (possibly hallucinated) plan step can drive — never a runtime-only lever
        # like _suspend_/_infer_ — and lets the collection-`filter` coexist with the perception one.
        self._data_ops: dict[str, InternalAction] = {}

    def register_internal(self, action: InternalAction) -> None:
        self._internal[action.name] = action

    def register_external(self, action: ExternalAction) -> None:
        self._external[action.name] = action

    def register_data_op(self, action: InternalAction) -> None:
        """Register a plan-composable data-op (a built-in, or a dev's own richer transform)."""
        self._data_ops[action.name] = action

    def internal(self, name: str) -> InternalAction:
        return self._internal[name]

    def external(self, name: str) -> ExternalAction:
        return self._external[name]

    def data_op(self, name: str) -> InternalAction:
        return self._data_ops[name]

    def is_data_op(self, name: str) -> bool:
        return name in self._data_ops


class InvokeAction:  # predefined external action: _invoke_
    name = "invoke"
    requires_binding = True  # abstract Step -> a concrete, schema-conformant OperationInvocation

    def __init__(self) -> None:
        # Hold strong refs to in-flight background invokes so they aren't GC'd mid-flight.
        self._tasks: set[asyncio.Task[None]] = set()

    async def execute(
        self,
        registry: EnvironmentRegistry,
        cycle: DecisionCycle,
        *,
        activity_id: str,
        **kwargs: Any,
    ) -> ActionAck:
        # tool_id/operation_name ride in via **kwargs, keeping this a structural ExternalAction
        # (the README sketch's explicit-param form isn't Protocol-compatible under mypy --strict —
        # see docs/phase-2-findings.md).
        tool_id = kwargs.pop(TOOL_ID)
        operation_name = kwargs.pop(OPERATION_NAME)
        params = kwargs
        tool = registry.get(tool_id)
        invocation = OperationInvocation(
            tool_id=tool_id, operation_name=operation_name, params=params
        )
        invocation_id = uuid.uuid4().hex
        activity = cycle.working.activities[activity_id]
        activity.pending_operation = PendingOperation(
            id=invocation_id, invocation=invocation, invoked_at=time.time()
        )
        activity.state = ActivityState.RUNNING  # implicit, unconditional — see Activities
        log.info("act: invoke %s.%s%s", tool_id, operation_name, f" {params}" if params else "")
        _spawn_tracked(self._tasks, self._call(cycle, tool, operation_name, params, invocation_id))
        return ActionAck(ok=True)  # immediate — the round-trip runs off-cycle, cycle never blocks

    async def _call(
        self,
        cycle: DecisionCycle,
        tool: Tool,
        operation_name: str,
        params: dict[str, Any],
        invocation_id: str,
    ) -> None:
        ack = await tool.invoke(operation_name, **params)
        cycle.result_sink.push(invocation_id, ack)  # keyed by invocation_id, not tool_id


def invoke_step(tool_id: str, operation_name: str, **op_args: Any) -> Step:
    """Assemble an `invoke` Step. tool_id/operation_name are *routing* (decided at Reason time);
    they ride in Step.params under the TOOL_ID/OPERATION_NAME keys alongside the operation's own
    arguments, which Act binds. `invoke` is the one Step whose params bag mixes routing with
    arguments (DefaultActStrategy.bind splits them apart) — use this factory instead of hand-writing
    that magic-keyed dict at every call site."""
    return Step(
        next_action=InvokeAction.name,
        params={TOOL_ID: tool_id, OPERATION_NAME: operation_name, **op_args},
    )


# Every predefined external action takes the uniform (registry, cycle, **kwargs) signature and reads
# its own params out of **kwargs, rather than declaring them as explicit keyword-only args. An extra
# required keyword-only param would break the structural-subtype relation to ExternalAction under
# mypy --strict, so ActionRegistry.register_external() wouldn't type-check. The README's explicit-
# param form is illustration only (see InvokeAction, docs/phase-2-findings.md); activity_id, passed
# by tick()'s dispatch, lands harmlessly in **kwargs for all but invoke.


class FocusAction:  # predefined external action: _focus_
    name = "focus"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        tool_id = kwargs[TOOL_ID]
        tool = registry.get(tool_id)
        await tool.focus(cycle.signal_sink)
        cycle.working.focused_tools[tool_id] = tool
        return ActionAck(ok=True)


class UnfocusAction:  # predefined external action: _unfocus_
    name = "unfocus"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        tool_id = kwargs[TOOL_ID]
        tool = cycle.working.focused_tools.pop(tool_id, None)
        if tool is not None:
            await tool.unfocus()
        # Unfocusing stops re-observing this tool, so its observable-property snapshot is
        # permanently stale — drop it. Signals from the same source stay (their own store, left
        # untouched): fire-and-forget, they may still matter elsewhere (same rationale as _filter_).
        cycle.working.drop_properties(lambda source: source != tool_id)
        return ActionAck(ok=True)


class JoinAction:  # predefined external action: _join_ — implies discover/connect
    name = "join"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        origin: WorkspaceOrigin = kwargs["origin"]
        workspace = await registry.join(origin)
        now = time.time()
        await cycle.semantic.store_workspace_record(
            WorkspaceRecord(id=workspace.id, origin=origin, discovered_at=now, last_seen_at=now)
        )
        for tool in workspace.tools():
            await cycle.semantic.store_manual(tool.manual)
            await cycle.semantic.store_tool_record(
                ToolRecord(
                    id=tool.id,
                    manual_id=tool.manual.id,
                    workspace_id=workspace.id,
                    address=tool.address,  # None unless this tool overrides the workspace's address
                    discovered_at=now,
                    last_seen_at=now,
                )
            )
            # Auto-focus every joined tool (same effect FocusAction performs), so perception no
            # longer hinges on the model emitting — and holding — a `focus` step: an unfocused
            # tool's state isn't observed, so without this a mid-task change (or the agent's own
            # writes) silently goes unseen. This is a **temporary mechanical fallback**, not the
            # intended design: the goal is reliable, intentional model-driven focus/unfocus that
            # attends to only the tools that matter (bounding per-cycle observation cost).
            # `_focus_`/`_unfocus_` stay available as that eventual path and as a manual override;
            # leaving unfocuses these again (LeaveAction), so the join/leave pair stays symmetric.
            await tool.focus(cycle.signal_sink)
            cycle.working.focused_tools[tool.id] = tool
        # workspace_id addresses it (for a later _leave_); tool_ids are a self-contained snapshot
        # of what was gained, legible after leave / across an agent boundary (see EXAMPLES.md).
        # the snapshot is useful for logging, e.g. saving an episode to memory
        tool_ids = [tool.id for tool in workspace.tools()]
        log.info("joined workspace %s (tools: %s)", workspace.id, ", ".join(tool_ids))
        return ActionAck(ok=True, result={"workspace_id": workspace.id, "tool_ids": tool_ids})


class LeaveAction:  # predefined external action: _leave_ — implies close
    name = "leave"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        workspace_id = kwargs["workspace_id"]
        # Unfocus any of this workspace's tools first: leaving deregisters them, and a tool can only
        # be focused via its workspace (focused_tools ⊆ joined tools per A&A), so a departing
        # workspace must not leave a stale focus (a live signal subscription + a dangling handle)
        # behind. Read the tools before registry.leave() pops the workspace.
        for tool in registry.get_workspace(workspace_id).tools():
            focused = cycle.working.focused_tools.pop(tool.id, None)
            if focused is not None:
                await focused.unfocus()
        await registry.leave(workspace_id)
        return ActionAck(ok=True)


class SendAction:  # predefined external action: _send_
    name = "send"
    requires_binding = False

    async def execute(
        self, registry: EnvironmentRegistry, cycle: DecisionCycle, **kwargs: Any
    ) -> ActionAck:
        # registry unused here — every ExternalAction still gets the same uniform signature.
        await cycle.communication.send(kwargs["to"], kwargs["content"])
        return ActionAck(ok=True)


# Predefined internal actions. Each takes the (cycle, **kwargs) InternalAction signature and only
# ever touches memory (no EnvironmentRegistry) — the mechanism half of the working-memory levers
# Situate drives (create/load/unload/filter); the *policy* (which goal, which manuals) lives in the
# SituateStrategy. Params ride in via **kwargs, same reason as the external actions above.


class CreateActivityAction:  # predefined internal action: _create_activity_
    name = "create_activity"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> Activity:
        # goal is the only required input; the strategy derives it from an unhandled message.
        # context defaults empty and activity_id is generated unless the caller pins one.
        activity = Activity(
            id=kwargs.get("activity_id") or uuid.uuid4().hex,
            goal=kwargs["goal"],
            context=kwargs.get("context") or {},
        )
        cycle.working.activities[activity.id] = activity
        log.info("situate: created activity %s from goal %r", activity.id, activity.goal)
        return activity


class LoadManualAction:  # predefined internal action: _load_
    name = "load"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        manual_id = kwargs["manual_id"]
        manual = await cycle.semantic.retrieve_manual(manual_id)
        # unknown id -> no-op, so a stale reference doesn't blow up the cycle
        if manual is not None:
            cycle.working.loaded_manuals[manual_id] = manual


class UnloadManualAction:  # predefined internal action: _unload_
    name = "unload"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        cycle.working.loaded_manuals.pop(kwargs["manual_id"], None)  # absent id -> no-op


class FilterPerceptionsAction:  # predefined internal action: _filter_
    name = "filter"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        # Prune observable-property percepts to the relevant tools, in place (`tool_ids` is the
        # relevant set — the default passes the joined workspaces' tools).
        tool_ids = kwargs["tool_ids"]
        cycle.working.drop_properties(lambda source: source in tool_ids)


class SuspendAction:  # predefined internal action: _suspend_
    name = "suspend"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        # Move a READY activity to BLOCKED, recording the signal it waits for. The *decision* to
        # suspend (a long-running op declared a completion signal not yet observed) is the caller's
        # — mechanically Observe's; this action is just the state flip. Layered on top of the
        # automatic RUNNING->READY resolve, not fused with it (see Activities in README).
        activity = cycle.working.activities[kwargs["activity_id"]]
        activity.state = ActivityState.BLOCKED
        activity.blocked_on = kwargs["wait"]  # a SignalWait
        log.info("observe: suspended activity %s until %s", activity.id, activity.blocked_on)


class ResumeAction:  # predefined internal action: _resume_
    name = "resume"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        # Move a BLOCKED activity back to READY once its awaited signal was observed (the caller
        # matched it — the signal itself is left in working memory, not evicted, so it can still
        # satisfy another activity blocked on the same wait). Clears blocked_on so it's selectable
        # again next Situate.
        activity = cycle.working.activities[kwargs["activity_id"]]
        activity.state = ActivityState.READY
        activity.blocked_on = None
        log.info("observe: resumed activity %s", activity.id)


# The two LLM calls are internal actions too — dispatched off-cycle exactly like _invoke_: set a
# pending marker, go RUNNING, create_task, return at once so the cycle never blocks (ADR-0021).
# Reason fires them when it needs a plan/param it can't produce mechanically; the result resolves a
# cycle or more later via inference_sink. ProceduralMemory still owns the model handle and the
# prompt/parse — the action just wraps that call in the off-cycle dispatch. pending_inference is
# mutually exclusive with pending_operation (a cycle emits one internal *or* one external action).
# A model call that raises (malformed output, no LLM, a network error) resolves with an *error*
# InferenceResult rather than leaving the task to die silently and strand the activity RUNNING
# forever — Observe terminates the activity on it (the failure surfaces, cycle-synchronized).


class InferAction:  # predefined internal action: _infer_ — the async plan model call
    name = "infer"

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        activity = cycle.working.activities[kwargs["activity_id"]]
        catalog: dict[str, Manual] = kwargs["tools"]  # id -> Manual, the planning catalog
        observed = kwargs.get("observed")
        messages = kwargs.get("messages")  # snapshot of recent user messages at fire time
        # kind defaults to "plan" (top-level planning); a mid-plan sub-goal passes kind="subgoal"
        # (it lands the same way, on Activity.plan — see PendingInference) and a `goal` override so
        # the sub-plan is inferred for the sub-goal's goal, not the activity's top-level one (ADR-
        # 0022). pending_inference still lives on the real activity, so Observe resolves it by id.
        kind = kwargs.get("kind", "plan")
        goal = kwargs.get("goal")
        # A signature of the world this plan is inferred against (ADR-0024), captured by Reason at
        # fire time and carried through to plan install so the context-adaptation gate baselines
        # against the assumptions the plan was formed in. Absent for grounding fires.
        baseline = kwargs.get("baseline")
        inf_id = uuid.uuid4().hex
        activity.pending_inference = PendingInference(
            id=inf_id, kind=kind, requested_at=time.time(), baseline=baseline
        )
        activity.state = ActivityState.RUNNING  # off-cycle, like _invoke_ — immediate, never blocks
        # `superseded` is dropped on the sub-goal copy (ADR-0024): the bundle is context for
        # re-planning the *activity's* goal, and default_plan_prompt introduces it as "a previous
        # plan for this goal" — true for the replacement, false for a sub-goal that was never
        # planned, let alone abandoned. Cleared here rather than at each install site so no future
        # path that leaves a bundle parked can leak it into a sub-plan's prompt.
        # The copy also carries the frame Observe *will* push when it installs the sub-plan
        # (strategies.py's kind=="subgoal" branch), so the copy models the stack position the plan
        # is being written for rather than the parent's. That is what lets a PlanPrompt tell a
        # sub-goal from a top-level goal — `parent_frames` is unambiguous here because the only
        # other infer fires when `plan is None`, and a reset clears the whole stack with the plan.
        # Seeding it (rather than adding a flag) keeps the PlanPrompt Protocol unchanged, so an
        # existing custom prompt keeps working and gains the ancestor chain for free.
        frames = list(activity.parent_frames)
        if goal is not None and activity.plan is not None:
            frames.append((activity.plan, activity.step_index, activity.history_mark))
        target = (
            activity
            if goal is None
            else replace(activity, goal=goal, superseded=None, parent_frames=frames)
        )
        log.info("reason: inferring a plan for %r (%d tools)", target.goal, len(catalog))
        _spawn_tracked(self._tasks, self._call(cycle, target, inf_id, catalog, observed, messages))

    async def _call(
        self,
        cycle: DecisionCycle,
        activity: Activity,
        inf_id: str,
        catalog: dict[str, Manual],
        observed: Any,
        messages: Any,
    ) -> None:
        # Tag this task's metered round-trip with the inference id (task-local, isolated per task)
        # so the meter can attribute its cost to *this* inference — and move it to the wasted bucket
        # if the result is later discarded (interrupt/supersede). See sora.llm.current_inference_id.
        current_inference_id.set(inf_id)
        try:
            plan = await cycle.procedural.infer(activity, catalog, observed, messages)  # LLM call
        except Exception as exc:  # noqa: BLE001 — any model/parse/wire failure resolves as an error
            log.exception("reason: infer failed for activity %s", activity.id)
            cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, error=repr(exc)))
            return
        cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, value=plan))


class GroundAction:  # predefined internal action: _ground_ — the async param-grounding escalation
    name = "ground"

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        activity = cycle.working.activities[kwargs["activity_id"]]
        inf_id = uuid.uuid4().hex
        activity.pending_inference = PendingInference(
            id=inf_id, kind="ground", requested_at=time.time()
        )
        activity.state = ActivityState.RUNNING
        log.info("reason: grounding %s params via the model", kwargs["operation_name"])
        _spawn_tracked(
            self._tasks,
            self._call(
                cycle,
                activity,
                inf_id,
                kwargs["operation_name"],
                kwargs.get("manual"),
                kwargs["partial_params"],
                kwargs.get("observed"),
            ),
        )

    async def _call(
        self,
        cycle: DecisionCycle,
        activity: Activity,
        inf_id: str,
        operation_name: str,
        manual: Manual | None,
        partial_params: dict[str, Any],
        observed: Any,
    ) -> None:
        current_inference_id.set(
            inf_id
        )  # attribute this round-trip's cost to this call (see infer)
        try:
            params = await cycle.procedural.ground(
                activity, operation_name, manual, partial_params, observed
            )
        except UnresolvableGrounding as exc:
            # Not a failure of the call — the model followed the contract and reported a gap in the
            # data instead of inventing a value. Kept off the `error` path so Observe can replan
            # rather than terminate.
            log.warning(
                "reason: grounding %s for activity %s found no data (%s)",
                operation_name,
                activity.id,
                exc,
            )
            cycle.inference_sink.push(
                inf_id, InferenceResult(id=inf_id, unresolvable=str(exc) or "unspecified")
            )
            return
        except Exception as exc:  # noqa: BLE001 — any model/parse/wire failure resolves as an error
            log.exception("reason: ground failed for activity %s", activity.id)
            cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, error=repr(exc)))
            return
        cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, value=params))


class RevalidateAction:  # predefined internal action: _revalidate_ — the plan-validity re-check
    name = "revalidate"

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        # The context-adaptation checkpoint (ADR-0024): fire the validity re-check off-cycle,
        # exactly like _infer_/_ground_ — the activity goes RUNNING with
        # pending_inference(kind="revalidate") and the bool verdict lands a later cycle via
        # inference_sink (Observe parks it on reconsider_verdict). Never blocks the cycle.
        activity = cycle.working.activities[kwargs["activity_id"]]
        observed = kwargs.get("observed")
        messages = kwargs.get("messages")
        # The perception signature captured when this re-check fires (ADR-0024). Carried through so
        # Observe can advance reconsider_baseline to *this* moment on resolve: a change that lands
        # during the re-check's flight then stays outside the new baseline and is reconsidered on
        # its own, rather than being folded in silently by re-baselining to the verdict-time world.
        baseline = kwargs.get("baseline")
        inf_id = uuid.uuid4().hex
        activity.pending_inference = PendingInference(
            id=inf_id, kind="revalidate", requested_at=time.time(), baseline=baseline
        )
        activity.state = ActivityState.RUNNING  # off-cycle, like _infer_ — immediate, never blocks
        log.info("reason: revalidating plan for %r", activity.goal)
        _spawn_tracked(self._tasks, self._call(cycle, activity, inf_id, observed, messages))

    async def _call(
        self,
        cycle: DecisionCycle,
        activity: Activity,
        inf_id: str,
        observed: Any,
        messages: Any,
    ) -> None:
        current_inference_id.set(inf_id)
        try:
            valid = await cycle.procedural.revalidate(activity, observed, messages)  # LLM call
        except Exception as exc:  # noqa: BLE001 — any model/parse/wire failure resolves as an error
            log.exception("reason: revalidate failed for activity %s", activity.id)
            cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, error=repr(exc)))
            return
        cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, value=valid))


class EvaluateConditionsAction:  # predefined internal action: _evaluate_conditions_ (ADR-0022)
    name = "evaluate_conditions"

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        # The pending-condition judgement, fired off-cycle exactly like _revalidate_: the activity
        # goes RUNNING with pending_inference(kind="condition") and the verdict lands a later cycle
        # via inference_sink (Observe parks it on condition_verdict). Never blocks the cycle.
        #
        # ONE call covers every eligible condition. The caller passes the conditions it judged
        # eligible, in order, and the verdict's indices refer to that list — so the correspondence
        # stays with the caller and a stale index can't retire the wrong condition.
        activity = cycle.working.activities[kwargs["activity_id"]]
        conditions = kwargs["conditions"]
        changes = kwargs.get("changes") or []
        observed = kwargs.get("observed")
        inf_id = uuid.uuid4().hex
        activity.pending_inference = PendingInference(
            id=inf_id, kind="condition", requested_at=time.time()
        )
        activity.state = ActivityState.RUNNING
        log.info(
            "reason: evaluating %d pending condition(s) for %r", len(conditions), activity.goal
        )
        _spawn_tracked(
            self._tasks, self._call(cycle, activity, inf_id, conditions, changes, observed)
        )

    async def _call(
        self,
        cycle: DecisionCycle,
        activity: Activity,
        inf_id: str,
        conditions: Any,
        changes: Any,
        observed: Any,
    ) -> None:
        current_inference_id.set(inf_id)
        try:
            verdict = await cycle.procedural.evaluate_conditions(
                activity, conditions, changes, observed
            )
        except Exception as exc:  # noqa: BLE001 — any model/parse/wire failure resolves as an error
            log.exception("reason: condition evaluation failed for activity %s", activity.id)
            cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, error=repr(exc)))
            return
        cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, value=verdict))


# --- data-ops: composable structured-value transforms (ADR-0023) ---------------------------------
# A data-op is an InternalAction the *plan* composes as a step (not a runtime lever like _suspend_):
# it reads a run-time collection Reason already resolved (from Activity.history via $from, a prior
# binding via $bind, or a literal) and writes a named binding into Activity.bindings[out], which a
# later step reads via {"$bind": "<name>"}. Registered in ActionRegistry's dedicated data-op bucket
# (not _internal), so only these — never a runtime lever — are dispatchable from a plan step, and
# so the collection-`filter` here never collides with FilterPerceptionsAction's perception-`filter`.
# The pipeline is imperative (one op per step); a declarative $foreach/$select binding spec stays
# rejected (ADR-0022 option (a)). Reason passes the resolved list as `collection` and the op's other
# params as kwargs; the op is a pure transform except FilterAction's $decide escalation.
_REF_DECIDE = "$decide"  # mirrors sora.strategies / ADR-0017's soft reference token


def pluck(element: Any, path: str | None) -> Any:
    """The value at ``path`` of ``element`` (the shared dotted-path grammar), or ``None`` on a bad
    path — a missing key is a non-match/absent value for a data-op, never a crash. Public because
    it is the one path-projection helper shared across the data-op layer: mechanical ``filter``
    evaluation (``_matches`` here) and cross-collection membership projection (Reason, in
    ``strategies``) must read a field the same way."""
    if not path:
        return element
    try:
        return walk_path(element, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _dedup_key(value: Any) -> str:
    """A stable, hashable signature for a (possibly unhashable, e.g. dict) value, for _distinct_."""
    return json.dumps(value, sort_keys=True, default=str)


def _matches(element: Any, where: Any) -> bool:
    """Evaluate a mechanical ``filter`` predicate against one element: ``{"path", "op", "value"}``
    with op in eq/ne/lt/le/gt/ge/between/in/not_in. ``in``/``not_in`` test membership of the
    element's ``path`` value in ``value`` (a literal list, or — resolved upstream in Reason — the
    projected keys of another collection named by a reference). A ``$decide`` predicate never gets
    here (FilterAction escalates it). No predicate keeps everything. A membership set that isn't a
    list is treated as empty: ``in`` matches nothing, ``not_in`` keeps everything (fails open, so a
    malformed exclusion set never silently drops the whole collection)."""
    if not isinstance(where, dict):
        return True
    actual = pluck(element, where.get("path", ""))
    op = where.get("op", "eq")
    value = where.get("value")
    if op == "eq":
        return bool(actual == value)
    if op == "ne":
        return bool(actual != value)
    if op == "in":
        return isinstance(value, (list, tuple)) and actual in value
    if op == "not_in":
        return not (isinstance(value, (list, tuple)) and actual in value)
    if op == "between":
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            return False
        lo, hi = value
        if actual is None:
            return False
        try:
            return bool(lo <= actual <= hi)
        except TypeError:
            return False  # incomparable types -> non-match, never a crash (like lt/le/gt/ge)
    if actual is None:
        return False  # ordered comparisons need a present value
    try:
        if op == "lt":
            return bool(actual < value)
        if op == "le":
            return bool(actual <= value)
        if op == "gt":
            return bool(actual > value)
        if op == "ge":
            return bool(actual >= value)
    except TypeError:
        return False
    log.warning("filter: unknown predicate op %r -> excluding element", op)
    return False


class FilterAction:  # predefined data-op: _filter_
    name = "filter"

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        activity = cycle.working.activities[kwargs["activity_id"]]
        out = kwargs["out"]
        collection = kwargs["collection"]
        where = kwargs.get("where")
        if isinstance(where, dict) and _REF_DECIDE in where:
            # Soft predicate: escalate to one off-cycle model call over the whole collection, like
            # _ground_ — park RUNNING, resolve into bindings[out] a later cycle (via Observe).
            inf_id = uuid.uuid4().hex
            activity.pending_inference = PendingInference(
                id=inf_id, kind="select", requested_at=time.time(), out=out
            )
            activity.state = ActivityState.RUNNING
            log.info("data-op: filter %r via the model (%d items)", out, len(collection))
            _spawn_tracked(
                self._tasks,
                self._call(
                    cycle,
                    activity,
                    inf_id,
                    collection,
                    where[_REF_DECIDE],
                    kwargs.get("observed"),
                ),
            )
            return
        activity.bindings[out] = [e for e in collection if _matches(e, where)]

    async def _call(
        self,
        cycle: DecisionCycle,
        activity: Activity,
        inf_id: str,
        collection: list[Any],
        predicate: str,
        observed: Any = None,
    ) -> None:
        current_inference_id.set(
            inf_id
        )  # attribute this round-trip's cost to this call (see infer)
        try:
            kept = await cycle.procedural.select(activity, collection, predicate, observed)
        except UnresolvableGrounding as exc:
            # The predicate named something the run never produced, and the model said so instead of
            # keeping nothing. Deliberately NOT the `error` path below: that one degrades to an
            # empty binding, which is precisely the wrong answer here. An empty shortlist reads
            # downstream as a real result ("no appointments that day"), so the fail-soft that keeps
            # a flaky call from churning the plan would instead let a whole clause of the task go
            # silently undone. This is a defect in the PLAN — a step assumed an earlier one would
            # yield something it didn't — so it replans, exactly as an unresolvable grounding does.
            log.warning(
                "data-op: $decide filter for activity %s resolved nothing (%s)", activity.id, exc
            )
            cycle.inference_sink.push(
                inf_id, InferenceResult(id=inf_id, unresolvable=str(exc) or "unspecified")
            )
            return
        except Exception as exc:  # noqa: BLE001 — any model/parse/wire failure resolves as an error
            log.exception("data-op: filter select failed for activity %s", activity.id)
            cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, error=repr(exc)))
            return
        cycle.inference_sink.push(inf_id, InferenceResult(id=inf_id, value=kept))


class DistinctAction:  # predefined data-op: _distinct_
    name = "distinct"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        activity = cycle.working.activities[kwargs["activity_id"]]
        by = kwargs.get("by")
        seen: set[str] = set()
        result: list[Any] = []
        for element in kwargs["collection"]:
            signature = _dedup_key(pluck(element, by) if by else element)
            if signature not in seen:
                seen.add(signature)
                result.append(element)
        activity.bindings[kwargs["out"]] = result


class SortAction:  # predefined data-op: _sort_
    name = "sort"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        activity = cycle.working.activities[kwargs["activity_id"]]
        by = kwargs.get("by")
        collection = list(kwargs["collection"])

        def key(element: Any) -> tuple[bool, Any]:
            value = pluck(element, by) if by else element
            return (value is None, value)  # None sorts to one end without comparing None to a value

        try:
            result = sorted(collection, key=key, reverse=bool(kwargs.get("desc", False)))
        except TypeError:
            log.warning("sort: incomparable keys for by=%r -> leaving order unchanged", by)
            result = collection
        activity.bindings[kwargs["out"]] = result


class TakeAction:  # predefined data-op: _take_
    name = "take"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        activity = cycle.working.activities[kwargs["activity_id"]]
        collection = list(kwargs["collection"])
        n = kwargs.get("n")
        activity.bindings[kwargs["out"]] = collection if n is None else collection[: int(n)]


class CollectAction:  # predefined data-op: _collect_ — the map-invoke's MapReduce gather
    name = "collect"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        # Reason pre-gathers the per-element history results (by the `from` operation name) into
        # `collection`; this just materializes them as one binding for a downstream op to consume.
        activity = cycle.working.activities[kwargs["activity_id"]]
        activity.bindings[kwargs["out"]] = list(kwargs["collection"])


class ReduceAction:  # predefined data-op: _reduce_
    name = "reduce"

    async def execute(self, cycle: DecisionCycle, **kwargs: Any) -> None:
        activity = cycle.working.activities[kwargs["activity_id"]]
        collection = kwargs["collection"]
        op = kwargs.get("op", "count")
        out = kwargs["out"]
        if op == "count":
            activity.bindings[out] = len(collection)
            return
        by = kwargs.get("by")
        values = [(pluck(e, by) if by else e) for e in collection]
        nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if op == "sum":
            activity.bindings[out] = sum(nums) if nums else None  # None (not 0) on empty, like mean
        elif op == "min":
            activity.bindings[out] = min(nums) if nums else None
        elif op == "max":
            activity.bindings[out] = max(nums) if nums else None
        elif op == "mean":
            activity.bindings[out] = (sum(nums) / len(nums)) if nums else None
        else:
            log.warning("reduce: unknown op %r -> null result", op)
            activity.bindings[out] = None


def default_action_registry() -> ActionRegistry:
    """The predefined action space, assembled once: the six external actions, the eight internal
    working-memory levers/model calls, plus the six plan-composable data-ops (ADR-0023). bootstrap
    and test harnesses register everything through this rather than naming each action inline."""
    registry = ActionRegistry()
    for external in (
        InvokeAction(),
        FocusAction(),
        UnfocusAction(),
        JoinAction(),
        LeaveAction(),
        SendAction(),
    ):
        registry.register_external(external)
    for internal in (
        CreateActivityAction(),
        LoadManualAction(),
        UnloadManualAction(),
        FilterPerceptionsAction(),
        SuspendAction(),
        ResumeAction(),
        InferAction(),
        GroundAction(),
        RevalidateAction(),
        EvaluateConditionsAction(),
    ):
        registry.register_internal(internal)
    for data_op in (
        FilterAction(),
        DistinctAction(),
        SortAction(),
        TakeAction(),
        CollectAction(),
        ReduceAction(),
    ):
        registry.register_data_op(data_op)
    return registry
