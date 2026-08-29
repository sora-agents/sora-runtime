# Mechanical predicates over model filters — composition, `overlaps`, and the fired-change bindings

Design note behind three changes that together take a common class of `filter` predicate off the
`$decide` path: boolean composition (`all`/`any`), a closed-form `overlaps` range join, and
runtime-seeded bindings carrying the ids a fired `pending` condition's change reported.

**Bottom line: the data-ops layer had a mechanical predicate path, and the predicates that actually
occur could not be written in it — so they escalated whole, one model call over every item in the
collection.** Nothing here changes what a data-op *is* (ADR-0023's imperative one-op-per-step
pipeline stands); it widens what one `filter` step can say and gives the planner a reference to a
fact the runtime already held.

**No ADR-level decision is claimed, deliberately.** Two of the three are additive grammar behind an
existing extension point — a new op and a new clause shape, in the same position as the
reference-valued `value` that ADR-0023 already grew. The third, the seeded bindings, is the one
worth arguing about, and it is argued below rather than decided by fiat; it is a few lines in
`_pursue_fired_condition` plus one exception in `reset_for_replan`, and reverting it costs nothing
but the prompt paragraph that names it.

## The measurement that motivated it

From `examples/gaia2/logs/aug28-run1-time-gpt-5.5.log` — the `time` scenario, a maintenance goal
watching a calendar for additions and deleting whatever they clash with. 28 model calls,
727,818 input tokens:

| call site | n | input tokens | share |
|---|---|---|---|
| `$decide` filters | 9 | 559,170 | **76.8%** |
| plan | 6 | 123,733 | 17.0% |
| condition / retirement judges | 13 | 44,915 | 6.2% |

Each `$decide` filter serializes the whole collection — about 410 records, ~62k input tokens per
call. Splitting the nine by what they were actually asking:

- **Four asked only "which items were just added"** (~248k tokens, ~34% of the entire run's input).
- Four were the overlap semi-join against those added items.
- One was a fused version of both, and kept nothing.

## Why each one had to escalate

### The added-set question — the runtime already knew

Every one of those four calls re-derived a set the runtime held verbatim. The signal that opened
the condition's gate carried it:

```
Change(path='events', added=('ec416b6f8dbe464e9296ef3a9ea4c7bd',), removed=(), updated=())
```

One id. Recovered by asking a reasoning model to scan 410 serialized records.

The cause is a gap in the reference grammar, not in the planner. `_is_reference` accepts `$from`
(history), `$bind` (a data-op binding), `$prop` (an observed property) and `$decide` — none of
which addresses a *change*. And `_pursue_fired_condition` hands the `then` sub-goal only
`condition.then`, a prose string; the change reaches the planner solely as rendered signal text it
cannot reference. So `$decide` over the whole property was the only mechanical-looking route to
"the ids that just arrived", and the planner took it four times.

### The overlap — no two-sided comparison, and no way to AND two clauses

The remaining clauses were an anti-join and an existential interval comparison:

> keep Calendar events that are not among `newly_added_events` and whose time range overlaps the
> start/end range of at least one event in `newly_added_events`; boundary-only touching does not
> count as overlap

The anti-join half was *already* expressible — `{"path": "event_id", "op": "not_in", "value": {"$bind": ...}, "value_path": "event_id"}`.
It escalated anyway, because a predicate was one flat clause: with no `all`, an expressible clause
and an inexpressible one had to travel together to the model.

## The two designs for the range join, and why the closed form won

The overlap needs an existential over a second collection with a non-equality comparison. Two ways
to write that:

**A general quantifier with a member alias** — `{"any": {"of": {"$bind": "added"}, "as": "n", "where": [... {"$ref": "n", "path": "end_datetime"} ...]}}`.
Genuinely general: any existential, any comparison.

**A closed-form op** — `{"op": "overlaps", "start_path": ..., "end_path": ..., "against": {"$bind": "added"}, "against_start_path": ..., "against_end_path": ...}`.
One shape, extended only by adding siblings.

The closed form was chosen, on one structural ground and two practical ones.

**It preserves the resolve-once invariant.** `_resolve_predicate_value` exists so that a referenced
operand is resolved *in Reason*, once, and `_matches` stays a pure literal comparison.
`overlaps`'s `against` names a collection, so it resolves exactly as an `in` membership set does —
walked once, projected through two paths into `[start, end]` pairs, handed to the evaluator as
literals. A `{"$ref": "n"}` alias cannot be resolved in Reason at all, because `n` does not exist
until evaluation: it would need a deferred-reference kind that Reason recognises and deliberately
skips, plus a binding environment inside `_matches`. That is not extra lines, it is a second
resolution regime living beside the first, which every future op then has to know about.

**Its failure mode is not silent.** An aliased design adds a new one: `{"$ref": "m"}` with a typo'd
alias resolves to nothing, every clause non-matches, the filter keeps zero, and downstream that
reads as "no overlapping events" — a real answer. That is exactly the class of fail-open trap
`_resolve_predicate_value`'s warnings exist to catch. `overlaps` has no alias to typo; its bad case
is an unreadable `against`, which is one check and one reported defect.

**Adherence matters more than generality here.** The whole benefit is contingent on the planner
*choosing* the mechanical form over `$decide`. A three-concept structure nested three deep in JSON,
from a model that already reaches for `$decide` when the shape gets awkward, is a worse bet than
six flat keys. And a small declarative expression language arriving inside plan JSON is precisely
what ADR-0023 declined when it chose an imperative pipeline over a `$foreach`/`$select` binding
spec.

## Composition: why an empty `all` keeps nothing

`all([])` is vacuously true in logic. Here it is a defect, and so is `any([])` and a non-list clause
list — reported by `_composition_defect`, with the evaluator matching nothing if one reaches it
anyway.

The asymmetry is deliberate and specific to what this layer feeds. Elsewhere in the data-ops the
dangerous direction is failing *closed*: an empty binding reads downstream as a fact about the
world ("no appointments that day") and a whole clause of the task goes silently undone. A
composition that keeps *everything* fails the other way, and the consumer is typically a mechanical
sub-goal that fans out one external action per item — on the motivating scenario, one delete per
calendar event. Between silently doing nothing and silently deleting 410 records, the first is the
recoverable one; reporting beats both.

## The seeded bindings, and where they live

At `_pursue_fired_condition`, the runtime writes three bindings from a precise change that opened
the gate — `fired_added_ids`, `fired_removed_ids`, `fired_updated_ids` (`SEEDED_BINDINGS` in
`sora/activity.py`) — and `default_plan_prompt` renders them under *"Ids reported by the change
that triggered this goal"*. A coarse change supplies none of them, as described below.

Three decisions inside that are easy to get wrong:

**Per condition, not per activity.** The changes are recorded on `PendingConditionState` at fire
time, alongside the marks and for the same reason: the judgement runs off-cycle, so by the cycle a
verdict lands the tick carrying the change is gone. Per *condition* because the fire queue outlives
a batch — a verdict fires plural, only one is pursued while the body is busy, and a shared field
would be overwritten by the next batch before the second is reached, planning it against a change
it never fired on. `_eligible_conditions` yields one percept per condition, so this is also
strictly more precise than the batch-wide union the judge sees.

**A binding, not a fifth reference token.** `$change` was considered. It would have required
threading the activity's changes through `_resolve_ref`, `_resolve_nested`, `_resolve_collection`,
`_resolve_predicate_value` and both defect helpers — a wide signature ripple for one fact. A
binding reuses `$bind` end to end: no new token, no new resolution path, and the value is already
in the shape the ops consume.

**Kept across `reset_for_replan`.** That method clears `bindings` because they "were produced by,
and are only meaningful within, the plan being discarded". These were produced by the runtime, so
that rule exempts them rather than covering them — and clearing them would make a replan
unrecoverable: the next plan's `$bind` would resolve to nothing, be reported as a defect, and
replan again, on a reference the runtime itself told it to use.

The same argument is why they may appear in the *plan* prompt at all. `render_bindings`'
docstring is right that ordinary bindings are a lie there — a replan discards them. These survive
it, so `render_seeded_bindings` shows only the reserved names and nothing else.

**Reserved, therefore collidable.** A plan whose data-op writes `out: "fired_added_ids"` would have
it overwritten at the next firing. The prompt says not to; nothing enforces it. Cheap to add a
check if it ever bites.

**Coarse is unavailable, not empty.** A `Change` whose three id tuples are empty means the adapter
knows only that something moved; it does not mean no items moved. Such a firing supplies none of the
reserved bindings, and clears any left by an earlier firing. One coarse member also makes a mixed
batch unavailable: the flattened lists have no completeness marker, so exposing the precise subset
would present a lower bound as the whole delta. The plan prompt says ids are unavailable and warns
that this is not an empty change set.

**Known-empty is still fail-open under `not_in`.** For an entirely precise firing, all three
bindings are present even when one direction is genuinely empty. An empty `fired_added_ids`
excludes nothing, so an exclusion clause standing alone would keep the whole collection. The prompt
tells the planner to pair it with a positive clause under `all`, which is also the shape the worked
example teaches. This is guidance, not a guarantee — the honest statement of what is and is not
enforced.

## What the pipeline looks like now

The motivating round, with no model call in it at all:

```json
{"action": "filter", "in": {"$prop": "<tool>.state", "path": "events"}, "out": "added",
 "where": {"path": "event_id", "op": "in", "value": {"$bind": "fired_added_ids"}}}

{"action": "filter", "in": {"$prop": "<tool>.state", "path": "events"}, "out": "clashing",
 "where": {"all": [
   {"path": "event_id", "op": "not_in", "value": {"$bind": "fired_added_ids"}},
   {"op": "overlaps", "start_path": "start_datetime", "end_path": "end_datetime",
    "against": {"$bind": "added"},
    "against_start_path": "start_datetime", "against_end_path": "end_datetime"}]}}

{"action": "subgoal", "mode": "mechanical", "in": {"$bind": "clashing"}, ...}
```

Two `$decide` calls per firing become zero.

## What this does *not* fix

**The two zero-step rounds were correct.** Five fan-outs on the motivating run produced 3, 1, 1, 0
and 0 steps. The run's signals show exactly four added ids and four removed ones, so by rounds four
and five everything overlapping had already been deleted, and keeping nothing was the right answer.
A guard that skipped a round whose newly-added set adds nothing would be suppressing a correct
result; what was wrong was that reaching it cost ~187k tokens of model-side set arithmetic. Those
rounds become nearly free here without any guard, which is the reason the guard was dropped rather
than built.

**A prompt edit re-baselines the benchmark.** `PLAN_SYSTEM_PROMPT` changed, so Gaia2 numbers do not
compare across this date. The worked example added to it is written in a deliberately neutral
domain (bookings with `starts_at`/`ends_at`) rather than the scenario's own fields, to keep the
prompt from teaching the benchmark it is measured on.

**The predicate still cannot express everything.** No disjunction across *collections*, no
comparison whose operand varies per member other than `overlaps`. If a second such shape appears,
the choice reopens — a second closed-form sibling (`within`, `covers`) versus the general
quantifier this note declined — with one more data point than it had.
