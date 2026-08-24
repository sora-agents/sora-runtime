"""Shared value types referenced throughout; kept minimal on purpose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def walk_path(value: Any, path: str) -> Any:
    """Walk a dotted path into a nested value — a numeric segment indexes a list, else a dict. The
    one path grammar shared by reference resolution (``$from``/``$bind`` in strategies) and the
    data-ops' per-element key paths (``by``/``where.path`` in action). Raises on a bad path; callers
    decide whether that's an escalation signal (grounding) or a skip (a mechanical predicate)."""
    for segment in filter(None, path.split(".")):
        value = value[int(segment)] if segment.isdigit() else value[segment]
    return value


@dataclass(frozen=True)
class ObservableProperty:
    name: str
    value: Any


@dataclass(frozen=True)
class Change:  # WHERE an observable property moved — see Signal.payload["changes"]
    # The one fact a replace-by-key snapshot structurally cannot hold. `properties` answers "what is
    # true now" and never "what just moved", so a contentless signal forces every waiter to
    # re-derive a delta against a store that keeps no previous value to diff against — impossible
    # without the waiter shadowing the whole property itself. The three tuples carry *identities*
    # only, never the values behind them: the snapshot stays in `properties` and this says where to
    # look inside it, which is what makes a property dereferenceable rather than re-scannable. That
    # keeps the thin-signal rule intact (nothing is duplicated, because a delta is not in the
    # snapshot at all).
    # Adapters DEGRADE rather than fail: a WoT property observation is already event-shaped, an MCP
    # resources/updated carries only a URI, and an adapter that cannot identify individual items
    # emits the coarse form — a `path` with all three tuples empty, meaning "something under here
    # moved". Consumers must accept the coarse form.
    path: str = ""  # dotted path into the property's value; "" = the property as a whole
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()


@dataclass(frozen=True)
class Signal:
    name: str
    payload: dict[
        str, Any
    ]  # may carry "changes": list[Change] — never the changed values themselves


@dataclass(frozen=True)
class SignalWait:  # what a `blocked` activity is waiting for — see Activity.blocked_on
    # The specific signal that must be observed before a blocked activity returns to `ready`. Set
    # by the _suspend_ internal action when a long-running operation declares a completion signal
    # (OperationSpecification.completion_signal); matched mechanically in Observe by name (+ source
    # when scoped to the completing tool). A future variant will wait on an observable property
    # reaching a state instead of a signal — deferred; this is why Activity.blocked_on is named
    # generally rather than blocked_on_signal.
    signal_name: str
    source: str | None = None  # tool id the signal must come from; None matches any source
    path: str | None = None  # scope to part of the property; None matches any (today's behavior)


def path_matches(wait_path: str | None, changes: list[Change]) -> bool:
    """Does a path-scoped wait match a signal's change summary?

    Bidirectional prefix on purpose: a change *inside* the watched subtree is relevant, and so is a
    coarser change reported *above* it by a degrading adapter. The second direction is what keeps a
    lossy adapter correct — it costs redundant evaluations but never missed ones, and a missed wake
    is the failure this mechanism exists to prevent. A wait with no path matches any change, and a
    signal with no `changes` at all matches a path-scoped wait too: an adapter that reports nothing
    is indistinguishable from one reporting a change everywhere, so the safe reading is wide.

    Prefix on *segments*, not on characters: these paths are dotted routes, and two siblings are not
    ancestors of one another however much of a leading substring they share. A raw `startswith`
    read `folders.INBOX_ARCHIVE` as living under `folders.INBOX`, and `contacts.contact_10` as
    living under `contacts.contact_1` — and numerically-suffixed ids are exactly what these paths
    are built from, so that is a systematic false wake, not an edge case.
    """
    if wait_path is None or not changes:
        return True
    return any(
        change.path == ""
        or change.path == wait_path
        or change.path.startswith(f"{wait_path}.")
        or wait_path.startswith(f"{change.path}.")
        for change in changes
    )


def changes_of(signal: Signal) -> list[Change]:
    """The `changes` a signal carries, tolerating adapters that emit none or emit plain dicts.

    Payloads cross a serialization boundary (a persisted percept, a JSON-shaped adapter), so a
    `changes` entry may arrive as a dict rather than a Change; normalize instead of trusting the
    producer. An unparseable entry degrades to the coarse form rather than raising — a malformed
    delta must not be able to break a wait that would otherwise have matched.
    """
    raw = signal.payload.get("changes") if isinstance(signal.payload, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[Change] = []
    for entry in raw:
        if isinstance(entry, Change):
            out.append(entry)
        elif isinstance(entry, dict):
            out.append(
                Change(
                    path=str(entry.get("path", "")),
                    added=tuple(entry.get("added", ()) or ()),
                    removed=tuple(entry.get("removed", ()) or ()),
                    updated=tuple(entry.get("updated", ()) or ()),
                )
            )
        else:
            out.append(Change())
    return out


@dataclass(frozen=True)
class InputWait:  # a `blocked` activity awaiting the user's next instruction (see blocked_on)
    # The second blocked_on variant SignalWait's comment foresaw: same `blocked` state, a different
    # wait. Set by the interrupt handler when a hard interrupt (a user stop) pauses an activity to a
    # resumable point; cleared in Observe when a user Message arrives. Not a signal wait — there is
    # no tool signal to match; the awaited stimulus is inbound user input, so it carries only an
    # optional human-facing note on what is being waited for.
    prompt: str | None = None


@dataclass(frozen=True)
class PendingCondition:  # what would make an exhausted plan relevant again — see Plan.pending
    # AgentSpeak's trigger, pointed forward. Only the gate is typed: `when`/`then`/`until` are prose
    # because their consumer is _infer_, which already takes prose goals — so this adds no condition
    # language, no predicate DSL and no event algebra to the plan. *Where to look* is the only part
    # a protocol can answer, so it is the only part given a structure, and it is also the part that
    # has to be cheap. `watch` is required: a condition with no gate degenerates into evaluating
    # every condition against every signal, which is the unbounded keep-alive this design rejected.
    # Firing does not consume a condition — `until` is what ends it — so "whenever X" needs no flag.
    watch: SignalWait
    when: str
    then: str
    until: str | None = None


@dataclass
class PendingConditionState:  # one PendingCondition's per-run state — on Activity, not Plan
    # The plan holds the reusable skeleton; how far this condition has evaluated is per-run, exactly
    # as step_index is to the body. `evaluated_through` is this condition's OWN high-water mark over
    # WorkingMemory.signals_appended — per-waiter on purpose. A single shared cursor (the shape
    # messages_cursor takes, correct there because a message must create an activity at most once)
    # would let the first condition to advance it blind every other reader of that broadcast log.
    condition: PendingCondition
    evaluated_through: int = 0


@dataclass(frozen=True)
class ConditionVerdict:
    """One batched judgement over an activity's *eligible* pending conditions (ADR-0022).

    Indices are into the eligible list the call was made about, not into
    ``Activity.pending_conditions`` — the caller holds the correspondence, so a model that echoes a
    stale index cannot silently retire the wrong condition.

    Deliberately carries no steps. A condition that fires is planned by the ordinary deliberative
    sub-goal path afterwards, which costs a second call only in the rare case where something
    actually holds, and keeps the common case — a gate opened, the change was not the awaited one —
    at exactly one call. It also means this prompt never has to double as a planning prompt.
    """

    fired: tuple[int, ...] = ()
    retired: tuple[int, ...] = ()  # `until` is now satisfied; the condition stops waiting


@dataclass(frozen=True)
class RelevanceCandidate:
    """One terminated episode an observed change may have made relevant again (ADR-0026).

    At most one is ever produced per judgement: a second-best guess about an intention nobody
    declared is not worth the question it would cost. `question` is what the user is asked before
    anything happens, and `goal` is the amending activity's goal if they say yes — kept separate
    because the question has to be answerable without knowing how the runtime words goals.
    """

    episode_id: str  # the terminated activity's id — the amendment points back at it
    goal: str
    question: str


@dataclass(frozen=True)
class ConditionWait:  # the third blocked_on variant — a plan's own declared pending conditions
    # Set when a plan body is exhausted but unsatisfied conditions remain, so the activity blocks
    # instead of terminating. blocked_on stays a single value: the plurality lives in this one wait,
    # whose watches are the union of the unsatisfied conditions'. Matched by the same mechanical
    # name/source/path equality as a SignalWait, but with a different meaning on the far side —
    # opening a gate only makes a condition ELIGIBLE. Whether it actually *holds* is a Reason
    # judgment, because matching prose against an email body is irreducibly semantic.
    watches: tuple[SignalWait, ...] = ()


@dataclass(frozen=True)
class InterruptRequest:  # a pending hard interrupt, recorded on DecisionCycle by interrupt()
    # The authoritative preemption of current work. `signal` carries the "why" (e.g.
    # Signal("user_stop", {})) the interrupt handler reads to decide each targeted activity's
    # follow-up. `target` names the activity to preempt; None is agent-wide (every schedulable
    # activity). A pushed signal only becomes an InterruptRequest through an InterruptPolicy — an
    # ordinary signal that merely matches a wait resumes cooperatively in Observe, never here.
    signal: Signal
    target: str | None = None


@dataclass(frozen=True)
class OperationInvocation:  # the concrete call, different from Step's abstract decision
    tool_id: str
    operation_name: str
    params: dict[str, Any]  # bound, ready for Tool.invoke() — the tool-hallucination-prone step


@dataclass(frozen=True)
class PendingOperation:  # tracks one in-flight invoke — lives on Activity, not WorkingMemory
    id: str  # correlates to what InvokeAction pushed into result_sink
    invocation: OperationInvocation
    invoked_at: float


@dataclass(frozen=True)
class PendingInference:  # tracks one in-flight infer()/ground() — lives on Activity, not WM
    # Mutually exclusive with PendingOperation (a cycle emits either one external action or one
    # internal action, so an activity is RUNNING on at most one of them). `id` correlates to what
    # _infer_/_ground_ pushed into inference_sink; the resolve in Observe discards a result whose id
    # no longer matches the live pending_inference — the stale-inference guard mirroring
    # pending_operation's late-ack guard. `kind` picks the landing zone: "plan" (infer ->
    # Activity.plan) or "ground" (ground -> Activity.grounded_params). See ADR-0021.
    id: str
    # "plan" | "subgoal" | "ground" | "select" | "revalidate". "subgoal" lands like "plan" (into
    # Activity.plan); "select" is a $decide data-op filter (ADR-0023) whose surviving subset lands
    # into Activity.bindings[out] — carries `out`; "revalidate" is the context-adaptation
    # plan-validity re-check (ADR-0024), landing a bool onto Activity.reconsider_verdict.
    kind: str
    requested_at: float
    out: str | None = None  # target binding name for kind=="select"; None for the others
    # A compact signature of the perception the pending deliberation was fired against, captured at
    # fire time. For kind "plan"/"subgoal" it is the world the plan is being inferred against; for
    # "revalidate" it is the world the validity check ran against. Observe moves it onto
    # Activity.reconsider_baseline on resolve, so the context-adaptation gate (ADR-0024) measures
    # change against that fire-time world — not a later, already-drifted one (a change that lands
    # during the call's flight then earns its own reconsideration). None for "ground"/"select" and
    # for a reused plan (no fresh inference), which falls back to an entry-time baseline.
    baseline: object | None = None


class UnresolvableGrounding(Exception):
    """An escalation asked to RESOLVE a reference reported that the reference names data the run
    never produced — an operation that returned an empty list, an absent field, an empty binding —
    rather than inventing a value for it. A defect in the *plan* (it assumed a step would yield
    something it didn't), never a wire or parse failure, so it resolves as
    `InferenceResult.unresolvable` and drives a replan instead of terminating the activity. Carries
    what was missing, for the log and the replanning prompt.

    Raised by both escalations that resolve references: `ground()` for an operation parameter, and
    `select()` for a `$decide` filter predicate. Named for the first, but the contract is one thing
    — in both, the alternative to reporting the gap is a plausible fabrication (an invented
    recipient; an empty shortlist that reads as "nothing qualified")."""


@dataclass(frozen=True)
class InferenceResult:  # what infer()/ground() resolve to — arrives async via inference_sink
    # Deliberation output, not observed environment state, so never a Percept (ADR-0019/0021): it
    # rides its own inference_sink, not result_sink or the perception path. `id` correlates to the
    # PendingInference it resolves; on success `value` is a Plan (kind="plan") or a grounded params
    # dict (kind="ground") and `error` is None. A model call that raised (malformed output, no LLM,
    # a network error) resolves with `error` set and `value` None instead of stranding the activity
    # RUNNING forever — Observe degrades the activity on it (in place, or into a replan carrying the
    # defect; it does not terminate it). DefaultObserveStrategy applies it on
    # resolve.
    id: str
    # Plan (kind "plan"/"subgoal"), grounded params dict (kind "ground"), the surviving-subset list
    # (kind "select" — a $decide data-op filter, ADR-0023), a bool verdict (kind "revalidate" —
    # the context-adaptation plan-validity re-check, ADR-0024), or a ConditionVerdict (kind
    # "condition" — the batched pending-condition judgement, ADR-0022).
    value: Plan | dict[str, Any] | list[Any] | bool | ConditionVerdict | None = None
    error: str | None = None
    # kind=="ground" or kind=="select", and mutually exclusive with both `value` and `error`: the
    # model followed the contract and reported that the data a reference names is not present,
    # instead of fabricating it. Distinct from `error` because it is a report, not a failure: both
    # replan, but only this one is the model doing what it was asked (DefaultObserveStrategy).
    # For "select" the distinction is load-bearing rather than cosmetic — `error` degrades a
    # $decide filter to an empty binding, which is the one answer an unresolvable predicate must
    # NOT produce, since downstream reads it as "nothing qualified" rather than as a question.
    unresolvable: str | None = None


@dataclass(frozen=True)
class ActionAck:  # returned by ExternalAction.execute() — dispatch, not outcome (see EXAMPLES.md)
    ok: bool
    result: Any = None


@dataclass(frozen=True)
class OperationAck:  # returned by Tool.invoke() — the tool's ack, arrives async via result_sink
    ok: bool
    result: Any = None


@dataclass(frozen=True)
class CompletedOperation:  # one resolved invocation + its ack, retained on Activity.history
    # The per-step execution trace a later step grounds against: e.g., a `reply_to_email`'s
    # email_id is resolved from an earlier `list_emails`/`search_emails` result. Kept because
    # last_operation holds only the *most recent* result (overwritten each step), which loses
    # earlier ones.
    invocation: OperationInvocation
    ack: OperationAck


@dataclass(frozen=True)
class Step:
    # next_action names the external action to dispatch (an ExternalAction.name — "invoke", "send",
    # "focus", ...) or the WAIT sentinel. params is that action's own argument bag: the cycle passes
    # it through opaquely and each action destructures it, so its shape is per-action — send ->
    # {to, content}, focus -> {tool_id}, join -> {origin}, and so on. `invoke` is the one whose bag
    # mixes *routing* (tool_id/operation_name, under the TOOL_ID/OPERATION_NAME keys) with the
    # operation's own arguments; DefaultActStrategy.bind splits the routing back out into an
    # OperationInvocation. Build an invoke Step with action.invoke_step() rather than hand-writing
    # those keys.
    next_action: str
    params: dict[str, Any]


@dataclass(frozen=True)
class Plan:  # multi-step, goal-indexed, reusable — the thing ProceduralMemory stores
    id: str  # stable identity for storage/reuse
    goal: str  # matched against future activities' goals — the retrieval key
    steps: list[Step]
    # What would make this plan relevant AGAIN once its body is exhausted. Part of the reusable
    # skeleton (the per-run state lives on Activity.pending_conditions), and deliberately NOT plan
    # control flow: a condition is not a step, never blocks the body, and executes nothing. It
    # declares what a wait would be *for*, while every state transition it implies — the suspend,
    # the match, the resume — stays the cycle's. Declaring is the plan's job; waiting the cycle's.
    pending: tuple[PendingCondition, ...] = ()


@dataclass(frozen=True)
class SupersededPlan:
    """The plan a ``reset_for_replan()`` threw away, kept so the *replacement* inference can see the
    intent it is replacing instead of starting blank (ADR-0024). Carries the whole intention stack,
    since the reset drops it whole: ``plan``/``step_index`` are the active frame at discard time and
    ``parent_frames`` the suspended parents, in the same (plan, subgoal_index) shape ``Activity``
    holds them. Only the *un-run* tail is ever rendered into a prompt — what already ran is
    ``Activity.history``, which the planning prompt renders separately."""

    plan: Plan
    step_index: int
    parent_frames: list[tuple[Plan, int]]
    # Why the plan was dropped, but only when the cause is a DEFECT in the plan itself — a reference
    # naming data the run never produced, say. None (the default) is the reconsideration case: the
    # plan was sound when written and is merely stale. The two need opposite briefs in the
    # replanning prompt — "reuse whatever still applies" makes a planner re-emit the very step that
    # cannot work — so the framing is chosen from this field, and the text names the specific gap.
    defect: str | None = None


# Named constants for Step.next_action values and invoke routing keys — one source of truth instead
# of bare string literals scattered across the cycle/actions/strategies (typos there are invisible
# to mypy). Registered ExternalActions are addressed by their own `.name` (e.g. InvokeAction.name);
# WAIT is the one pseudo-action the cycle special-cases (it dispatches no ExternalAction).
WAIT = "wait"

# The Step.next_action sentinel for a sub-goal — the plan's sole recursion primitive (ADR-0022).
# Reason handles it internally (mechanical fan-out or a mid-plan _infer_) and never dispatches it as
# an ExternalAction, so it too is a pseudo-action like WAIT, not a registered action name.
SUBGOAL = "subgoal"

# Keys under which an `invoke` Step carries its routing in Step.params (and in InvokeAction's
# kwargs), before Act binds them into an OperationInvocation.
TOOL_ID = "tool_id"
OPERATION_NAME = "operation_name"

# The Signal.name a CLI /stop raises through DecisionCycle.interrupt() — the one interrupt the
# runtime default DefaultInterruptHandler recognizes and routes (pause to await input). Shared here
# so the producer (cli.py) and the consumer's guard (strategies.py) agree on one literal, not two.
USER_STOP = "user_stop"
