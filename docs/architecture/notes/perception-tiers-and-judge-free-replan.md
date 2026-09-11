# Perception tiers and judge-free replan

Design note behind two changes that go together: a fourth `context_adaptation` level
(`replan_on_change`, resolving a hot change-gate by discarding the plan instead of spending a
model revalidation), and the removal of timer polling from the containerised Gaia2 CLI adapter,
which now declares no observable property at all.

**Bottom line: the runtime's property/signal split was designed against an environment that
supplies both from one cheap read, and it degrades badly on one that supplies neither — unless the
adapter says so honestly and the reconsideration dial has a setting for it.** Nothing here changes
[ADR-0024](../adrs/0024-plan-reconsideration-context-adaptation.md)'s decision: reconsideration is
still cycle-owned, still a per-agent configuration dial, still fronted by the same free mechanical
gate. The dial was built to take new policies; this is one.

**No ADR-level decision is claimed.** The level is a policy behind an extension point the ADR
already anticipated ("a dotted path to a custom one"), and the poll removal is a configuration and
adapter change. What *is* worth writing down is the precondition, because it is invisible at the
call site and unsound to forget — that is most of this note.

## Three tiers, and where the assumption came from

An adapter extracts what the ecosystem offers ([ADR-0003](../adrs/0003-adapters-not-tool-authoring.md)),
so what an agent can perceive varies by environment:

| Tier | Environment offers | What a plan can use |
|---|---|---|
| 3 | observable properties, signals, operations | `$prop`, watches that bind ids, mechanical predicates over observed collections |
| 2 | signals, operations | signals as triggers; state only by invoking a read and binding its output |
| 1 | operations | only what an operation returned |

These tier numbers are study shorthand, not a restriction on the environment model. Observable
properties and signals are independent channels declared by structured specifications or non-empty
authored affordance sections in each tool's manual: a catalog may expose both, either one, or
neither. In particular, property-only is a valid combination even though it has no numbered tier
in this study. Quiet channels remain available from their declarations; conversely, a retained
percept is evidence of its channel when its originating tool has departed and its manual is no
longer live. The prompt's channel profile is therefore the union of live declarations and the
percept kinds actually included in that prompt.

The in-process ARE adapter is tier 3, and it is where the runtime's instincts were formed. ARE
gives it exactly one thing — `app.get_state()`, a whole-state read — and the adapter manufactures
*both* halves from it: one `state` observable property, plus a `state_changed` signal synthesized
by diffing consecutive reads. The signal is deliberately thin, carrying the diff rather than the
state, because the property already publishes the state and duplicating it would reproduce it in
every prompt that renders `wm.signals`. That is a sound trade **given the premise**: property and
signal are two views of one cheap read, so having both costs nothing.

The containerised CLI harness breaks the premise on both sides. State files are readable only by
the `gaia2` user — that is what the setuid wrapper is for — so the only way to see anything is to
run a command, which is an *operation*, not an observation. And there is a genuinely independent
push channel the in-process path has no analogue of: the daemon announces environment-initiated
actions, in prose, with no identifiers.

## What we got wrong twice

The first revision took the notification and dropped it on the floor, perceiving world change only
by polling. That left the agent perceiving strictly less than the sibling harnesses on exactly the
scenarios built around environment-initiated change.

The second kept both, and that is the subtler mistake. To have an observable property at all, the
adapter nominated a zero-argument read per app and published its output *as observed state* on a
one-second timer. Three things are wrong with it:

- **It invents an affordance the environment does not have.** A read command's output is an
  operation result. Calling it an observation makes the manual promise perception the tool cannot
  deliver, and a plan written against the promise fails silently.
- **It degenerates.** The mechanical rule (every read taking no required argument) lands on
  `calendar`'s *tag list* — technically state, useless as one — which is why the benchmark config
  had to hand-pick per-app reads instead, encoding what these scenarios happen to need rather than
  what the environment is.
- **It buys with a subprocess per tool per second** what the sibling harnesses get for free by
  deciding to look.

## What the sibling harnesses actually do

Worth stating because it was assumed rather than checked. Upstream's own agent prompt tells them:

> This environment has **push notifications**… you will automatically receive a notification as a
> follow-up message in this conversation. **Do NOT poll or actively check**…

Hermes delivers the notification as a user message; OpenClaw posts it to a wake hook, where the
agent sees it as a `System:` line prepended to a heartbeat prompt (its heartbeat is configured at
24h — effectively off, so the hook is the only pulse). Neither has a timer, and neither treats the
notification as *data*: they treat it as a cue, and then run a read command inside the turn.

So the notification is sufficient as a trigger and insufficient as evidence, for them and for us.
The tier-2 position is not a handicap relative to the baselines; it is the same position.

## Prompt fitting is part of the honest tier configuration

Perception channels constrain not only what reaches working memory but also which instructions are
actionable. The built-in prompts therefore fit their modules to the channels declared by the tools
relevant to each semantic call. The adaptive setting removes property-only vocabulary — including
`$prop` references and property-backed conditions — when no property channel exists, and removes
signal-backed waiting when no signal channel exists. It does not infer a tier from an empty value
or a quiet cycle: a supported but currently empty channel retains its module. Other prompt material,
including operation results, bindings, history, data operations, safety constraints, and subgoals,
does not depend on the perception profile.

This means the primary tier comparison intentionally changes two coupled things: the environment's
available evidence and the instructions that teach the agent how to use that evidence. That is the
honest deployable configuration. Keeping tier-3 instructions in a lower-tier environment would add
tokens while encouraging plans that refer to capabilities the environment cannot satisfy.

The `fixed-rich` setting is the sensitivity control for the narrower question. It renders the full
property-and-signal prompt for every channel combination, so a comparison under that setting holds
prompt instructions constant and varies perception alone. Its lower-tier result must be interpreted
as a deliberately handicapped counterfactual, not as the expected performance of a correctly
configured lower-tier agent. The gap between adaptive and fixed-rich at the same tier estimates the
effect of fitting vocabulary to declared capabilities; it must not be attributed to perception
quality itself.

## Why a judge-free replan follows

At tier 2 the revalidation judge is shown an announcement and nothing else: it learns that an app
spoke, not what changed. On a task concerning that app it can rarely answer better than "maybe", so
the call is spent to be told what the change-gate already established. The judge earns its price at
tier 3, where it can read the changed state and say "that is a newsletter, proceed" — saving the
whole replan. **The value of a revalidation scales with the quality of the evidence it is shown**,
which is why this is a dial setting and not a new default.

Removing the judge removes the only filter, though, so the discard is scoped: it fires only on a
percept whose source is a tool the current plan references, derived from the plan the same way
`IntentionScopedFocus` derives its attended set. Without that scope an unrelated app's notification
would discard a plan that never touched it — a replan storm precisely in the dynamic scenarios the
level exists for. The judged path stays agent-level, deliberately and for the opposite reason: a
judge has to *see* the change that woke it, or it revalidates against nothing.

The scope needs a position as well as a signature, because a change-gate signature is opaque by
contract (a pluggable gate returns any comparable object) and so can say that perception moved but
never where. `WorkingMemory.perception_cursor()` supplies it, and it is anchored at **infer** time
rather than plan-install time: a change landing during the planning call would otherwise sit behind
the cursor, and the scoped check would rule out the very change the plan never saw. It fails open
in three cases — no cursor, no scope, or a retention cap that outran the cursor — on the standing
rule that a spurious replan costs one call and a missed one costs the scenario.

## The precondition, which is the part to remember

**`replan_on_change` is unsound on an adapter whose signals include the agent's own writes.** Such
a signal lands on a tool the plan references by construction, so every write would discard the plan
that issued it, and the agent would never commit to anything. A judge filters that out by reading
the change; a mechanical trigger cannot.

This is the same reliability problem ADR-0024 cited when it rejected mechanical maintenance
predicates — it needs per-tool identity scoping or efference tagging to exclude self-writes, "a
general-case reliability problem we do not want to own". The level does not solve it. It sidesteps
it, and only where the environment happens to guarantee the exclusion: in the CLI harness every
notification formatter is registered on an environment function, so the agent's own operations
never announce themselves, and with no polling there is no derived property change either. Take
that guarantee away — an adapter that echoes writes back as signals, or a tier-3 environment where
polling reports the agent's own effect — and the level must not be used.

An efference mechanism would retire the precondition and make this safe generally. That is the
principled version and it is not built.

## What this costs

Tier 2 is more expensive per interruption than tier 3, and the honest accounting is worth keeping.
Handling an environment event costs one plan call plus a ground call, against tier 3's revalidation
that may return "still valid" for one call and no replan at all. Against a ReAct baseline the
comparison is *n*-dependent: the baseline pays one call to decide to look plus one per corrective
action, so it wins on a single-action correction and loses from two upward, where the plan
amortizes. Call count is also the wrong denominator on its own — a ReAct call re-feeds the whole
transcript, while a revalidation carries a bounded percept snapshot — so both axes belong in any
reported comparison.

The two harnesses now differ in perception tier rather than only in plumbing, which makes the pair
a cleaner experiment than either alone: same runtime, same scenarios, and the delta isolates what
observable state is worth to a deliberative agent. That comparison is only meaningful if the clock
treatment matches across both; otherwise it measures the clock.
