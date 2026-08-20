# Activities & Concurrency

An activity is the central unit of work for a S-ORA agent: it is a means to achieve a goal and has a context that represents a filtered view of the environment relevant to the activity. An agent can pursue multiple activities concurrently, but only one activity is executed in each decision cycle. It can also drop an activity if it is no longer desirable or achievable, or it can suspend the activity while waiting for external events and conditions.

An activity can be in one of four states:

- running: the activity has an invoked operation in flight — invoked but not yet resolved; the agent won't reselect it until the operation resolves, though other activities may still be picked and progressed meanwhile;
- blocked: the agent is waiting for external events (e.g., signals from tools) to proceed with the activity;
- ready: the agent can pick and pursue the activity;
- terminated: the activity was completed or dropped.

An activity is eligible for selection only when ready; running and blocked activities are skipped until something transitions them back. Invoking an operation always, implicitly, moves an activity to running until that operation's own result comes back — this is unconditional, independent of anything the tool's manual says, and resolving it back to ready is an unambiguous one-to-one match the runtime does automatically, with no strategy code involved. A manual can additionally require blocking on a specific signal before the next step, layered on top of (not instead of) that implicit wait — the two are orthogonal: the operation resolves to ready first, and a *separate* step then blocks the activity until the signal arrives. That block is likewise mechanical: an operation declares its completion signal in its manual (`OperationSpecification.completion_signal`), so entering `blocked` (the `_suspend_` action) and leaving it once the signal is observed (the `_resume_` action — the matched signal itself is left in `wm.signals`, not evicted, so it can also satisfy another activity waiting on the same signal, or a strategy reading it directly; only a fixed retention cap ever removes it) are both name-equality matches the Observe phase performs deterministically — no model call, no judgment. (A completion driven by an observable property reaching a state, rather than a signal, is a foreseen second form — deferred.)
