# Gaia2 charged-clock experiment protocol — 2026-09-19

Status: **freeze candidate**. This protocol becomes the frozen pre-sweep record only when it is
committed together with the exact scenario manifest before any paid sweep result is collected. The
evaluation set is the complete Gaia2 `mini` selection: 32 scenarios from each of the five
capabilities, 160 scenarios total.

## Fixed timing policy

The primary trajectory policy is `serialized_sum`. The environment clock is frozen while one or
more physical agent-model crossings are in flight. Each crossing, including parser repair,
malformed-output retry, and a failed crossing without usage, receives its own non-negative frozen
token charge. The final outstanding crossing releases the environment after the sum of every
crossing's charge has been banked. Overlapping inference is therefore charged as one inference
lane. Tool operations receive no synthetic duration: ARE's apps expose no authored execution
latency, so local Python tool time is an implementation artifact rather than a modeled resource.

This is deliberately conservative against S-ORA's asynchronous architecture, but it does not erase
the architecture's call-amortization effect: every physical crossing still pays the same endpoint
coefficient model on both arms. Changing the primary policy after results exist changes event
delivery and invalidates timing comparison with those results.

`llm_charged_union_seconds` is a declared sensitivity only and never changes the trajectory. It is
constructed online on a settled parallel frontier. Calls admitted before any sibling settles share
a coordinate. A later admission starts at the furthest charged endpoint already settled in the
current uninterrupted freeze. This prevents causal backdating without inheriting the primary
serialized accumulator.

The frontier is not a reconstructed causal DAG. A causally independent call admitted after an
unrelated sibling settles is conservatively placed at the settled frontier rather than the epoch
origin. Its reported parallel elapsed time is therefore conservative-high relative to an ideal
unbounded causal schedule, and `charged_seconds - llm_charged_union_seconds` is a lower bound on
the savings that ideal parallel inference could provide.

## Event delivery under the freeze

Scheduled scenario events are dispatched **synchronously at each final resume**, and **not at all
during an individual frozen inference epoch**. When the last outstanding crossing releases the
environment, the harness ticks ARE's loop before returning to the agent, so everything the frozen
interval skipped — due events and time-based notifications alike — is delivered before the agent
can begin its next crossing.

This is a blocking-inference execution model. It does **not** measure native mid-inference
responsiveness: an event authored inside an epoch is observed at the end of that epoch, late by up
to the epoch's charge, and the agent cannot interrupt, observe, or spawn an activity from it while
the crossing is in flight. The resume-time dispatch bounds that lateness at one epoch. Without it
the delivery would instead race ARE's one-second loop against the agent's next crossing, so
lateness would accumulate across a chain of calls and depend on host thread scheduling — which
would make the Time capability's results non-reproducible rather than merely conservative.

Because overlapping crossings are charged as one lane, the epoch is the union of everything in
flight, not a single call: an event authored during the first of two overlapping crossings waits
for the second as well.

## What the Time capability measures under this policy

The judge's `post_event_tolerance_seconds` is 25 seconds, measured from the oracle event's
scheduled time. Under the freeze, the agent spends that budget on two things, not one: the
delivery lateness described above, plus the charged cost of its own reaction.

The reaction is not a single crossing. Reacting to a scheduled event costs **two inferences** — a
condition evaluation, then a fresh sub-plan inference for the `then` branch, which is re-inferred
per firing — before the resulting call is dispatched mechanically. Measured on the `time` smoke
logs at `medium` reasoning effort, using the frozen charge coefficients, that pair costs a median
of **≈5.3 charged seconds**, with the worst observed pair ≈11.9 s. The sweep runs at `high`, so
these are a lower bound on the sweep's reaction cost.

A Time result under this policy is therefore a measurement of reaction-plus-delivery against a
25-second budget, not of reaction alone. Both terms are charge, so both are reproducible and
provider-independent; neither is wall-clock latency. When reporting, state the budget and the two
terms together — a gated event missed by a margin smaller than one reaction pair is a statement
about the charge model's calibration as much as about the agent.

`llm_charged_union_seconds` is reported as unavailable (null) for any run whose crossings did not
all share one time axis. A crossing admitted after the online judge has stopped the environment
carries no charged coordinate, and sweeping such a window together with charged ones would yield a
figure in no coordinate system at all. An aggregate containing one unavailable run is itself
unavailable rather than a sum over a shorter count. This affects only the declared sensitivity; the
primary trajectory is unchanged.

The separate `--wall-clock` arm is a robustness mode, not timing-comparable to the primary arm.

## Inputs and provenance

The paid scenario set must be supplied through `batch --scenario-manifest`. The manifest pins the
dataset repository, immutable dataset revision, split, ordered exact scenario IDs, and its own
canonical SHA-256 digest. Batch validates the complete selection before opening paid artifacts,
records the digest on every result row, and refuses to pair with an existing counterpart arm whose
digest or scenario/run matrix differs. `--limit` remains smoke-only and is mutually exclusive with
the manifest.

The exact selection is
[`campaigns/aamas2027/mini-validation.json`](campaigns/aamas2027/mini-validation.json). It contains
all 32 `mini` IDs for each of `execution`, `search`, `adaptability`, `time`, and `ambiguity`, grouped
under their actual capability names. At the pinned revision, every selected scenario is present in
the corresponding full capability configuration in the same relative order and with an identical
serialized payload. Runs therefore load each capability configuration and apply the shared
manifest, preserving capability labels and the five-way aggregate while evaluating exactly Gaia2
mini.

Frozen artifacts at this checkpoint candidate:

- `evaluation/campaigns/aamas2027/mini-validation.json` canonical manifest SHA-256:
  `123b92db986d09fea93b92f7bccaa5c3f08c35c77939f91fc97c5c3820fc416b`
- `evaluation/charge_model.json` raw SHA-256:
  `9065b1dc6bda1aa832b64867b0c0c68885930f7c74bb08b6c48ad140dcbca585`
- `evaluation/campaigns/prompt/baseline.json` raw SHA-256:
  `3b3bf50769e3658864d3ca8f7761ff4e755bb59da76203fefa29580c6746a46b`

The charge-model identity and its semantic digest are also written into every charged or
frozen-profile wall-clock result. The manifest digest is canonical over parsed JSON, so formatting
does not change experiment identity; the two raw hashes guard their complete files byte for byte.

## Declared diagnostics

Every scenario result carries:

- `llm_round_trips`: physical agent-model crossings, including repairs and retries;
- `llm_max_in_flight`: maximum simultaneous physical crossings on the host monotonic axis;
- `llm_overlapped_round_trips`: crossings that shared positive host-wall duration with another;
- wall-time sum and union; and
- token-charge serialized sum and settled-frontier union.

All seven current agent-model entry points resolve through `cycle.procedural` and therefore the one
`ProceduralMemory._llm` instance decorated by the Gaia2 harness. Online judge calls are a separate
scoring resource and are not counted as agent inference.

## Ex-ante overlap expectation

The prior 15 recorded Gaia2 runs all created one activity. In those logs, `plan/v1` accounted for
about 85% of inference wall time; the off-cycle `retirement/v1` judge accounted for roughly 78 of
2,084 seconds (about 3.7%), and the other off-cycle caller, `relevance/v1`, did not fire. Under
those observed conditions, the serialized-minus-union difference should be small and smaller on
the charged axis because retirement prompts are token-light.

This is an expectation, not a bound on the new sweep. Multiple activities, relevance calls, or a
different overlap pattern can make the difference larger. A materially larger diagnostic is a
finding that falsifies the prior single-activity premise; it is not a reason to revise the frozen
policy or coefficients after seeing results.

## Operational gates

Before the sweep, run paired real-call range and drift audits for both endpoints and a Time smoke.
The smoke must confirm delivery of all scheduled events, nonzero concurrency when the known
retirement/plan overlap is exercised, and negligible local tool execution time relative to
simulated elapsed. Repeat the drift audit over completed sweep logs. Record a failure against the
frozen coefficients; never re-fit them inside the sweep.

Timing results produced before the token-charged integration are not comparable with results under
this protocol. The semantic prompts and their frozen snapshot are unchanged.
