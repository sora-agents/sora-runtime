# Complex reasoning strategies and `ProceduralMemory.infer()`

Design note answering a recurring question: *if one implements more complex LLM reasoning
strategies (ReAct, tree-of-thoughts, reflexion, replanning, native tool-calling), how do they fit
the `infer()` design shipped with the model-backed Reason phase?* No decision is made here — this
records the intended extension model and where the current seams genuinely bind, so future work
(and future contributors) don't try to grow `infer()` into something it was not meant to be. See
[ADR-0010](adrs/0010-pluggable-phase-strategies.md) / [ADR-0011](adrs/0011-phase-fusion-via-threaded-result.md)
for the phase-strategy and `TickResult` design this builds on.

## Short answer

`infer()` is **not** the extension point for complex reasoning. It is the *default strategy's
one-shot planning primitive*. The extension point is a custom `ReasonStrategy`. If you try to grow
`infer()` into ReAct/ToT/reflexion, you will feel friction — and that friction is the design
telling you to write a strategy instead.

## The two seams, and their division of labor

- **`LLMClient.complete(system, prompt) -> str`** (`sora/llm.py`) — the raw model round-trip. Dumb,
  stateless, text-in/text-out. No tools, no multi-turn, no structured output.
- **`ProceduralMemory.infer(activity, tools) -> Plan`** (`sora/memory.py`) — *one* `complete()`
  call → a static `Plan`. A pure function of `(goal, tool catalog)`; it gets **no `wm`, no `cycle`,
  no memory access** (a memory module must not reach into the environment). In CoALA terms it is a
  *query* against procedural memory's implicit knowledge, not a planning process.
- **`ReasonStrategy.reason(activity, wm, cycle, result)`** (`sora/strategies.py`) — the reasoning
  **policy**, and the actual pluggable extension point. It has the whole context, and via
  `TickResult` threading it can fill `result.step` *and* `result.invocation` directly — i.e. **it
  does not have to produce a `Plan` or call `infer()` at all.**

Mental model: `infer()` = a planning *primitive*; `reason()` = the *policy* that decides
when/whether to plan, replan, reuse, or react. Complexity almost always belongs in the policy.
`DefaultReasonStrategy` is one policy (advance a live plan; else retrieve; else infer); it is not
the only one.

## How concrete "complex" strategies map

| Reasoning style | Fits `infer()` as-is? | How it's expressed |
|---|---|---|
| **Replanning** (invalidate & re-plan on a signal/observation) | Yes | A `ReasonStrategy` reads `wm.perceptions` / `activity.last_operation`, clears `activity.plan` when stale, and re-calls `infer()`. The Protocol already says "deciding when a plan is invalidated is up to the implementation." This is the mid-scenario follow-up-email path in EXAMPLES.md. |
| **ReAct / reactive** (thought→action→observe, no static plan) | Bypasses it | A strategy that each cycle builds a prompt from `wm` (goal + recent percepts + `last_operation` as the last observation), calls the LLM, and returns a single `result.step`. **No `Plan`, no `infer()`.** The observation arrives next cycle via Observe (the invoke resolves onto `last_operation`), so the ReAct loop is spread across decision cycles — the idiomatic S-ORA shape. Scratchpad lives in `activity.context` (strategy-owned data). |
| **Context-rich planning** (consult episodic memory / beliefs / observed state) | Not in `infer()` | `infer()` only sees `(activity, tools)` by design. The strategy gathers the extra context and either (a) passes it into a *widened* `infer(activity, tools, episodes=…, beliefs=…)` — a backward-compatible signature growth that keeps `infer` a pure function of its inputs — or (b) does the LLM call itself. Do **not** push `wm`/environment access *into* the memory module. |
| **ToT / self-critique / multi-sample** (many LLM calls before committing) | Bypasses it | `infer()` is single-call. A strategy can make *several* `complete()` calls within one cycle (the "at most one **external action** per cycle" limit is about actions, not model calls) then commit to one step. `infer()` does not support the burst; the strategy does. |
| **Native tool-calling during reasoning** (model calls tools mid-plan) | Neither seam supports it | Needs a richer client than `complete()` — a `chat(messages, tools) -> response-with-tool-calls` shape — plus multi-turn state (thinking-block / tool-call preservation). That is the deferred two-axis / normalized-response-with-`provider_data` work (see the Phase 4 backlog in [ROADMAP.md](../ROADMAP.md)). |

## Where the current design genuinely binds

Three real limits, stated honestly:

1. **`LLMClient.complete()` is too thin for tool-calling / multi-turn / structured-output
   reasoning.** It is the deliberately-minimal seam. The moment a strategy needs native
   function-calling or preserved thinking blocks, you introduce a *richer* client interface
   alongside it — exactly the Phase 4 two-axis backlog item. Complex reasoning is precisely the
   trigger that promotes that from "deferred" to "needed."

2. **`infer()`'s narrow `(activity, tools)` signature caps context-aware planning.** Good for
   purity, but a strategy wanting episodic/belief-conditioned planning must pass that context in
   (signature growth) or move the LLM call into itself. The narrowness is a feature — it keeps the
   primitive stable and side-effect-free — but it means `infer()` is not where you add smarts.

3. **There is no `ProceduralMemory` Protocol.** `DecisionCycle.procedural` is the concrete class, so
   swapping in a *different* procedural-reasoning engine (a PDDL solver, a hosted planner service,
   an alternative `infer`) currently means subclassing (as `SpyProcedural` does in
   `tests/test_reason.py`). If "complex reasoning" means "pluggable planning backend," the missing
   seam is that Protocol. Cheap to add when needed.

## Conclusions

- Keep `infer()` exactly as narrow as it is — a stable, one-shot, side-effect-free planning
  primitive that any strategy can lean on or ignore.
- Add complexity as **new `ReasonStrategy` implementations**, using `TickResult` to skip
  `Plan`/`infer()` when the paradigm is reactive rather than plan-based.
- Everything short of native tool-calling is already expressible today: replanning (fits `infer()`
  as-is), ReAct across cycles (bypasses it), multi-sample within a cycle (strategy makes several
  `complete()` calls), and context-rich planning (via passed-in inputs).
- Two things become *needed* (not *deferred*) the moment you build tool-calling or multi-turn
  reasoning: the richer LLM-client seam, and — if you want swappable planning engines — a
  `ProceduralMemory` Protocol. Neither is required for the plan-then-execute / replanning scenarios
  targeted for v0.1.0.
