# Attention scoped to live intentions — analysis, measurement, and why it is opt-in

Design note behind the focus/attention work. Records what focusing a tool actually gates, the rule
`IntentionScopedFocus` implements, and — the part that changed the conclusion — a measurement of
what the narrowing is worth.

**Bottom line: attention is reconciled in Observe (that part is structural and stays), but the
default policy declines to narrow.** `FocusAllJoined` is the default; `IntentionScopedFocus` is a
one-argument opt-in for dynamic, many-workspace runs.

**No ADR-level decision is claimed here, deliberately.** This started as a proposed ADR. It was
withdrawn on two grounds. First, an ADR is for the decisions that are hardest to change, and the
narrowing rule is one pure function behind a one-method Protocol, reverted by a single constructor
argument — the cheapest class of change in the runtime, not the most expensive. Second, the
measurement below contradicts the cost premise the draft argued from.

## What focus gates

Three separate things, and only the first is obvious:

1. **Property snapshotting** — `DefaultObserveStrategy._snapshot_properties` iterates
   `WorkingMemory.focused_tools`, so an unfocused tool's observable properties never enter the
   store, and therefore never enter any prompt.
2. **Signal *production*** — an adapter subscribes on `Tool.focus(sink)`. In `are_sim`, `observe()`
   pushes a `state_changed` signal only when `focus()` set its sink; an unfocused tool is silent.
   So focus does not merely filter what the agent reads, it decides what the environment reports.
3. **Per-cycle work** — the same adapter deep-copies and diffs the whole app state on every
   observation of a focused tool.

## The history

The original design left focus to the model, as `focus`/`unfocus` plan steps. That failed in the way
this kind of thing fails: silently. A plan that forgot a `focus` step ran to completion against a
world it could not see, and neither the runtime nor the model had any way to notice — a missing
percept is indistinguishable from an unchanged one. The response was a deliberately temporary
mechanical fallback: **joining a workspace focuses every tool in it.** That removed the silent
failure and was correct as far as it went, but it discarded the narrowing focus exists to provide.

`IntentionScopedFocus` is the attempt to get the narrowing back without reintroducing the silent
failure, by deriving the attended set from the plans the agent is already committed to rather than
from the model remembering to ask.

## The rule

An activity contributes:

* **every joined tool**, if it has no plan yet;
* otherwise, over its whole intention stack (the active plan plus every suspended parent frame):
  * each step's `tool_id`, *except* an `unfocus` step's;
  * every tool id named by a `{"$prop": "<tool_id>.<name>"}` reference found by walking the step's
    params **recursively** — a `$prop` lives inside the param bag, under a data-op's `in` or a
    sub-goal's collection, not under `tool_id`;
  * every `watch.source` of a declared `PendingCondition`, the source of whatever the activity is
    `blocked_on`, and the tool of any in-flight `pending_operation`.

An agent with **no live activities at all** attends every joined tool. The union is intersected with
the joined set last, so an id left over from a departed workspace can never be re-attended. A
`TERMINATED` activity contributes nothing.

The policy is a `FocusPolicy` Protocol with one pure, set-valued method — `attend(wm) -> set[str]` —
injected into `DefaultObserveStrategy`'s constructor, exactly as `ActivitySelectionStrategy` is
injected into `DefaultSituateStrategy` ([ADR-0016](../adrs/0016-pluggable-activity-selection.md)): a
code-level sub-strategy seam, not an `agent.yaml` key.

### Why "broad while unplanned" is load-bearing

`Activity.reset_for_replan()` sets `plan = None`, so the entire replan window is automatically
broad. That is why attention needs **no grace period** and **no retention of already-executed
steps**: the scan covers all steps of a live plan, not just the un-run tail, so a tool invoked at
step 3 stays attended through step 9, and the only moment a plan is absent is a moment the agent is
broad anyway. It is also what keeps the planner's `$prop`-over-pagination discovery and the
idle-tick relevance judge working unchanged.

### Why recomputed, and not leased

`attended = ⋃ referenced(a)` **is** a refcount, evaluated eagerly. A lease is an incremental cache of
that same quantity, maintained at every site that mutates a plan — a plan landing,
`reset_for_replan`, a sub-goal push and pop, Reflect terminating an activity, an interrupt dropping
one, `LeaveAction`, a condition retiring. A missed decrement leaks a lease and silently regresses
cost; a double decrement unfocuses a tool still in use and is silently blind — the exact failure this
work exists to remove. Recomputation has one site and cannot drift, at O(live activities × plan
steps) per tick.

## The measurement

Model-free and deterministic: both arms read the *same* world state and differ only in the focus
policy, so there is no trajectory confound — which is the flaw that makes an n=1 before/after dollar
total uninterpretable. Real ARE apps, so payload shapes and sizes are the real ones. A task touching
2 of the joined apps (the ARE dynamic scenario's shape, and Gaia2's).

### What an app's `state` actually costs a prompt

`_render_property_value` renders verbatim JSON only while it fits `_TRUNCATE_LIMIT` (400 chars) and
falls back to a **shape sketch** above it. So a property's contribution to a prompt is already
bounded by the renderer, however large the underlying collection gets:

| app | raw JSON bytes | rendered bytes |
|---|---:|---:|
| `EmailClientApp` (6 full emails) | 2,870 | **264** |
| `ContactsApp` (8 contacts) | 2,832 | **2,594** |
| `CabApp` | 342 | 359 |
| `CalendarApp` | 14 | 36 |

**This is what withdrew the ADR.** The draft argued from "each app publishes a `state` property
holding a whole collection … thirteen apps' worth of that, on every call." That premise is false:
`EmailClientApp`'s six full emails cost 264 bytes in a prompt, not 2,870. The sketch fallback was
already doing most of the job the narrowing claimed credit for. `ContactsApp` is the exception — a
keyed collection whose *sketch* is itself large — and it alone accounts for ~72% of the broad
property section below, which makes the headline saving a property of one app's shape rather than a
general result.

### Prompt bytes per model-call site

| joined apps | scoped set | broad bytes | scoped bytes | saved |
|---:|---:|---:|---:|---:|
| 3 | 2 | 2,915 | 320 | 89.0% |
| 5 | 2 | 3,044 | 320 | 89.5% |
| 7 | 2 | 3,491 | 320 | 90.8% |
| 9 | 2 | 3,624 | 320 | 91.2% |

~3,300 bytes ≈ **825 tokens per call site** at 9 apps. Over a run of ~30 model calls that is ~25k
input tokens — cents, and consistent with the ~10% / $0.14-on-$1.41 observed on a static Gaia2 turn.

The honest surviving argument is a **scaling** one, not a current-cost one: the per-app cost is
bounded, but the broad section grows linearly in app count while the scoped section stays constant.
The percentage saved rises with environment size; the absolute saving is small at the sizes we run.

### Judge calls: the saving is zero on both shipped configs

The draft's strongest claim was that narrowing suppresses *model calls*, not just tokens, because an
unattended tool is silent at the adapter and its changes never provoke a judgement. Measured, that
claim does not hold:

| judge | broad arm | scoped arm |
|---|---:|---:|
| condition judge, source-scoped watch (as the plan prompt instructs) | 1 call | 1 call |
| condition judge, source-less watch | 1 call | 1 call |
| relevance judge, churn on tools the plan touches | 1 call | 1 call |
| relevance judge, churn **only** on tools the plan does not touch | 1 call | **0 calls** |

The condition judge is already gated mechanically by `_match_signal` on name **and source**, and
`referenced_tools` attends every declared `watch.source` — so the watched tool is attended under
*both* policies and the gate opens identically. Extra signals from unattended tools were never
reaching that judge in the first place.

The relevance judge saves a call only in the last row, and it batches every unclaimed change into
**one** call per idle tick rather than one per change — so even there the difference is a single
call, not a volume. And it is **opt-in and not enabled in either `examples/are/sim/email_calendar/
agent.yaml` or `examples/gaia2/agent.yaml`**, so on both configs we actually run, the judge-call
saving is exactly zero.

## Where this leaves the default

**`FocusAllJoined` is the default; `IntentionScopedFocus` is the opt-in.** Settled on the
measurement above: the narrowing is worth cents, and its cost is a failure that does not announce
itself. A benchmark result is worth more than that saving, and a silently absorbed change would be
attributed to the agent rather than to the policy. Turning narrowing on is one argument —
`DefaultObserveStrategy(focus=IntentionScopedFocus())` — and is what a dynamic, many-workspace run
should use, where the broad set grows with the environment while the narrow one stays constant.

Note what does **not** revert with it. Attention is still reconciled at the top of every Observe, so
perception no longer hinges on the model emitting and holding a `focus` step — that is the
silent-failure fix, and it is independent of which policy computes the set. `FocusAllJoined` is not
the old auto-focus-on-join fallback restored; it is the same reconciler with a policy that declines
to narrow. What the numbers establish:

* the benefit on the configs we run today is prompt bytes only, ~825 tokens per model call;
* it grows with joined-app count and is dominated by whichever app has the largest *sketch*;
* it is zero for judge calls;
* against it sits an unmeasured cost — attention now churns across replans, and re-attaching
  re-baselines the adapter, so a change occurring while a tool is unattended is absorbed into the
  new baseline rather than reported.

That last risk is why the small win does not carry the default. The run that would justify
flipping it is a dynamic, many-workspace scenario with exogenous churn on tools the plan does not
name — the one row above where a call is actually saved — with the relevance judge enabled.

The per-activity prompt view (`scoped_snapshot`, at `_ground_` and `_select_`) is **not** reverted
with the policy. It is non-destructive, it never narrows the data being operated on — the mechanical
resolve reads the whole store, and a data-op's collection is passed separately — and it only trims
ambient property context from two prompts. It carries none of the subscription-layer risk, so it
keeps the part of the saving that costs nothing.

## Known rough edges

* **An `unfocus` step is a no-op whenever another step of the same plan names the tool.** The
  exclusion in `referenced_tools` stops an `unfocus` step's own `tool_id` from re-attending the
  tool, but the scan covers *all* steps, so an earlier `invoke` re-attends it on the next Observe:

  ```
  plan: [invoke mail.read, unfocus mail, invoke cal.add]
  attended after the unfocus step: ['cal', 'mail']
  ```

  The tool is released and re-attended a tick later, re-baselining the adapter in between. This is
  the no-history-retention property working as designed, but it means the derived floor silently
  overrides an explicit deliberative act.
* **Out-of-band focus does not survive.** A `FocusAction` dispatched outside any plan — by
  application code or a custom strategy — is released on the next Observe. Attention is derived from
  plans, so a focus with no owning intention has no release condition either; giving it one is the
  unsolved part.
* **A bare `{"$prop": "state"}`** names no tool, contributes nothing to attention, and can become
  unresolvable later in a plan even though it resolved when the plan was written. Mitigated by the
  plan prompt asking for qualified names, and by an unresolvable reference reporting a defect rather
  than failing silently.

## The option not yet taken

A third shape, between "the model must remember" and "the runtime derives it":

> **Derived floor ∪ a durable explicitly-held set** — `attend()` returns the intention-derived union
> *plus* a `held` set that only `FocusAction`/`UnfocusAction` mutate.

It is additive, so forgetting an explicit focus degrades to the safe floor and cannot go silently
blind; and it would fix both the `unfocus` no-op and the out-of-band case above. It is not built
because an explicitly held focus needs a **release condition** and an ownerless focus has none — a
model that focuses liberally and forgets to unfocus walks `held` back up to focus-all, which is the
lease-leak failure in declared form. Worth revisiting if out-of-band focus becomes a real
requirement.

## Cross-references

* Replaces the temporary auto-focus-on-join fallback introduced for
  [ADR-0006](../adrs/0006-workspace-join-leave-lifecycle.md)'s join; joining returns to
  discover / connect / persist, and `LeaveAction` still releases whatever ended up attended.
* Refines — and does not contradict —
  [ADR-0022](../adrs/0022-plan-representation-context-guard-and-subgoals.md), which forbids
  auto-focusing on a `$prop` miss *at grounding time*. That prohibition is about grounding
  dispatching a focus mid-resolution, which would break
  [ADR-0009](../adrs/0009-five-phase-decision-cycle.md). Here the tool id is declared in the plan
  text and attention is established by the Observe reconciler, before any reference is resolved.
* Sits beside [ADR-0019](../adrs/0019-blocked-state-machinery-and-percept-storage.md) in Observe —
  that ADR reconciles what an activity *waits for*, this reconciles what it *attends to*; and its
  rule that `_filter_` never touches `signals` is unchanged.
* Interacts with [ADR-0024](../adrs/0024-plan-reconsideration-context-adaptation.md) — the change
  signature hashes the property *set*, so attention transitions move it; this is why the
  per-activity prompt view (`scoped_snapshot`) is non-destructive and why `_revalidate_` keeps the
  agent-level snapshot.
* [ADR-0026](../adrs/0026-undeclared-relevance-recovery.md)'s judge is kept alive by the
  no-live-activities clause: an idle agent attends everything, which is precisely when it runs.
