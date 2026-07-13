# S-ORA Implementation Roadmap

Tracks progress from tooling setup through a working implementation of the design in [README.md](README.md) and [EXAMPLES.md](EXAMPLES.md). See [docs/adrs/](docs/adrs/) for why specific decisions were made.

Ordering principle for Phase 3: **fakes and determinism first, real networks and real models last.** Both LLM calls and real external adapters make tests slow, flaky, and non-reproducible — deferring them keeps the test suite fast and deterministic for as long as possible. Phase 2 is the deliberate exception — see its own note below.

## Phase 0 — Tooling & environment

- [x] `.gitignore`
- [x] `.editorconfig`
- [x] `.python-version`
- [x] `pyproject.toml` (project metadata, ruff/mypy/pytest config, optional extras)
- [x] `src/sora/__init__.py` packaging placeholder
- [x] `tests/test_smoke.py` toolchain smoke test
- [x] `.pre-commit-config.yaml`
- [x] `.github/workflows/ci.yml`
- [x] `LICENSE` (Apache 2.0)
- [x] `CONTRIBUTING.md`
- [x] `SECURITY.md`
- [x] `CHANGELOG.md`
- [x] `CODE_OF_CONDUCT.md`
- [x] `ROADMAP.md` (this file)
- [x] `git init`, initial commit
- [x] Verify clean-machine bootstrap: `uv sync --all-extras --dev && uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest` passes
- [x] Confirm copyright holder name in `LICENSE` and contact emails in `SECURITY.md`/`CODE_OF_CONDUCT.md` (currently placeholders)
- [x] Push to `git@github.com:sora-agents/sora-runtime.git`; enable branch protection on `main` (require CI + review before merge)

## Phase 1 — Skeleton

- [x] Create every module named in the README's API sketch (`sora/types.py`, `environment.py`, `perception.py`, `manual.py`, `activity.py`, `action.py`, `memory.py`, `strategies.py`, `transport.py`, `cycle.py`, `cli.py`, `bootstrap.py`)
- [x] All `@dataclass(frozen=True)` value types and `Protocol` interfaces from the sketch, as-is
- [x] Stub concrete classes (`EnvironmentRegistry`, `DecisionCycle`, `Agent`, `ActionRegistry`, the six predefined actions, `NotificationQueueSink`, `DefaultObserveStrategy`, ...) with `...`/`NotImplementedError` bodies
- [x] **Done when:** package imports cleanly, `mypy --strict` passes with zero errors, smoke test passes

## Phase 2 — Walking skeleton against ARE

Goal: get one full Observe→Act decision cycle running against a real external
adapter as fast as possible, to surface conceptual or design gaps in the
README/EXAMPLES.md sketch before investing in the full TDD build-out. This is
the one deliberate exception to "fakes and determinism first": ARE's MCP
server ([EXAMPLES.md](EXAMPLES.md#example-evaluating-a-s-ora-agent-on-are-meta))
runs a fixed scenario locally as a scripted, deterministic subprocess — not a
flaky live network dependency — so pulling it forward here doesn't cost the
reproducibility the TDD-first ordering is protecting. Code from this phase is
explicitly throwaway/minimal: it gets replaced and properly test-driven in
Phase 3, not polished here. Test coverage for this phase is a single
integration-level test, not unit tests per layer.

- [x] Minimal types/actions/cycle wiring needed for exactly one `tick()` — only what this spike touches, not the full API sketch
- [x] Real MCP `WorkspaceAdapter` wired to ARE's local MCP server — over **stdio** via `python -m are.simulation.apps.mcp.server.are_simulation_mcp_server --apps are.simulation.apps.email_client.EmailClientApp --transport stdio` (the sketch's `--scenario scenario_email_calendar --transport sse` form doesn't exist as written — see [docs/phase-2-findings.md](docs/phase-2-findings.md))
- [x] Hardcoded, deterministic `ReasonStrategy` that does just the *first* step — `invoke EmailClientApp.list_emails` (real tool is `EmailClientApp`, not `EmailApp`) — no LLM call, no multi-step plan yet
- [x] **Done when:** the single step completes end-to-end against the real ARE MCP server, captured as one skip-gated integration test (`tests/test_are_walking_skeleton.py`); design gaps written up in [docs/phase-2-findings.md](docs/phase-2-findings.md) (two small typing corrections applied to `src/`; EXAMPLES.md/README diffs proposed, not yet applied; ADR candidates deferred to Phase 3 step 12)

## Phase 3 — TDD rollout

Reorganized after the Phase 2 spike (see [docs/phase-2-findings.md](docs/phase-2-findings.md)). Several
items the original flat 1–15 list treated as greenfield already have *throwaway* spike
implementations (`types.py`, `NotificationQueueSink`, `EnvironmentRegistry.join/leave`, `InvokeAction`,
`DefaultObserveStrategy`, a working `tick()`), covered only by the explicitly-disposable
`tests/test_cycle_wiring.py` — so those tasks are **re-drive as permanent, properly TDD'd code**, not
build-from-scratch. The list is grouped into tracks; `[P]` marks tasks that can proceed in parallel
once their track's gate is met. TDD is the default; two tasks flagged **⚠ harness-risk** are where the
test harness may get complex enough to warrant a fake-vs-real tradeoff — notify before over-investing.

### Track A — Foundations & reconciliation (land first; A1/A3/A4 parallel, A2 gates C/D signatures)

- [x] A1. [P] Apply the Phase-2 *proposed but unapplied* README/EXAMPLES.md diffs — `EmailApp` → `EmailClientApp`, `--scenario scenario_email_calendar` → `--apps …EmailClientApp …CalendarApp`, note the `<App>__<operation>` namespacing and stdio as a valid transport ([docs/phase-2-findings.md](docs/phase-2-findings.md) gaps 1/3/4). Pure docs — parallel with everything.
- [ ] A2. **Reconcile registry access (decided; README/ADR done, `src/` pending).** The agent reasons over what it has joined, so `EnvironmentRegistry` stays reachable from `WorkingMemory` (spike's call, kept — "currently-joined, live workspaces" is per-cycle contextual state, distinct from SemanticMemory's durable records). Refinement: `WorkingMemory` advertises it through a **read-only `EnvironmentView` Protocol** (`get`/`get_workspace`/`all_tools`/`joined_workspaces`) so strategies can reason over the live joined set but `mypy --strict` forbids mutating connections through `wm`; the concrete, mutation-capable `EnvironmentRegistry` remains the single shared instance (per [ADR-0013](docs/adrs/0013-shared-instances-narrow-dependencies.md)) that `tick()` dispatch and the Join/Leave actions use. **Done:** README `EnvironmentView`/`WorkingMemory`/`tick()`/`Agent`/`bootstrap` reconciled + ADR-0013 refined. **Remaining:** the matching `src/` change (add `EnvironmentView`, type `WorkingMemory.registry`, move the mutable handle onto `DecisionCycle`) lands with C2/D4, which it gates.
- [x] A3. [P] De-stringify the cycle constants (§7c) — `Percept.kind` as a `PerceptKind` StrEnum, reuse `InvokeAction.name` instead of the literal `"invoke"`, `WAIT` sentinel for the cycle's no-op step, and `TOOL_ID`/`OPERATION_NAME` key constants. *(Chose named constants over a distinct invoke-param carrier dataclass — a carrier would near-duplicate `OperationInvocation`; the deeper `next_action == "invoke"` special-casing is left for D4/§7b, marked with an in-code NOTE.)* Behavior-preserving; guarded by the existing suite + `mypy --strict`.
- [x] A4. [P] Decide which `tests/test_cycle_wiring.py` assertions become permanent TDD tests vs. get replaced; drop the "throwaway spike" framing as each layer below is re-driven. **Done:** triage recorded in [docs/phase-3-test-triage.md](docs/phase-3-test-triage.md) — 8 of 9 assertions promoted (registry pair extended under C2), the end-to-end tick re-driven under D4, only spike *scaffolding* (wiring helper, in-file fakes, string-typed reason fixture) replaced; no assertion discarded. The blanket "throwaway spike" docstring is replaced with a pointer to the triage; per-group promotion happens as C1/C2/C4/D1/D4 land (annotated below).

### Track B — Long-term memory & manuals (B1 gates B2–B4; B2/B3/B4/B5 then parallel)

- [ ] B1. File-backed `MemoryBackend` + round-trip tests. Gate for B2–B4.
- [ ] B2. [P] `SemanticMemory` — manual + workspace/tool record store/retrieve/list (needed by Join/Leave and `restore()`).
- [ ] B3. [P] `EpisodicMemory` — learn/consult.
- [ ] B4. [P] `ProceduralMemory` retrieve/store (deterministic); `infer()` left as a stub until E3 (it's the LLM path).
- [ ] B5. [P] `Manual` + Markdown `ManualParser` — reuse EXAMPLES.md's manuals as fixtures. Fully independent of A and B1.

### Track C — Environment & actions (after A2; C3-join and C2-restore also need B2)

- [ ] C1. [P] `Tool`/`Workspace`/`WorkspaceAdapter` fake in-process adapter, promoted from the spike into a reusable test fixture. *(A4: absorbs the spike's `FakeTool`/`FakeWorkspace`/`FakeAdapter`.)*
- [ ] C2. `EnvironmentRegistry` — keep join/leave/get; add [ADR-0014](docs/adrs/0014-tool-identity-globally-unique.md) id-uniqueness enforcement (fail loud on duplicate id at join; `leave` never pops a shared id); implement `restore()` (needs B2). *(A4: promotes + extends the spike's join/get/leave assertions → `tests/test_environment.py`.)*
- [ ] C3. [P] The five still-stubbed predefined external actions — Focus/Unfocus (wire `focused_tools` + `signal_sink`), Join/Leave (wrap registry + persist records via B2), Send (needs transport). Independent of each other. *(A4: promotes the spike's `action_registry_lookup` alongside C4.)*
- [ ] C4. [P] Re-drive `InvokeAction` (already implemented) as a permanent TDD test. *(A4: promotes the spike's `invoke_action_sets_running_then_pushes_result` → `tests/test_action.py`.)*

### Track D — Decision cycle proper (mostly serial; needs A3 + the strategies)

- [ ] D1. `DecisionCycle` Observe-only + `DefaultObserveStrategy`, re-driven as permanent. *(A4: promotes the spike's `observe_resolves_running_activity`, and the `NotificationQueueSink` group → `tests/test_perception.py`, its first real consumer.)*
- [ ] D2. `DefaultReflectStrategy` — deterministic completion/failure judgment + store-on-success to episodic (B3) and procedural (B4).
- [ ] D3. `DefaultSituateStrategy` — **fix §7a** (always run; select only if `result.activity is None`), activity-creation-from-message via the `_create_activity_` internal action, and wm adjustment (focus/load/unload/filter). Also update the stale `SituateStrategy` docstring in `src/sora/strategies.py`.
- [ ] D4. Reason + Act end-to-end with a deterministic `ReasonStrategy` (no LLM) — **includes §7b**: introduce an `_act()` bind-then-dispatch boundary and drop the hardcoded `next_action == "invoke"` branch (let the action declare whether it needs binding). Fixes the lingering §7a `if result.activity is None` gate in `cycle.py`. *(A4: re-drives the spike's `tick_end_to_end_invoke_then_resolve` — keep the outcome assertions, rebuild the fixture/harness — and retires `test_cycle_wiring.py` once its last group has moved.)*
- [ ] D5. The `_create_activity_` internal action that the default Situate depends on. (Blocked-state `_suspend_`/`_resume_` + the signal-satisfies-wait judgment moved to Phase 4 — v0.1.0's scenario drives replanning through Observe→Situate, not the blocked path.)

### Track E — Integration & release tail (serial; real network/model)

- [ ] E1. **⚠ harness-risk.** Harden the MCP adapters — extract a protocol-only `McpWorkspaceAdapter` base from `AreMcpWorkspaceAdapter` (grouping + name-assembly hooks; design the default grouping policy here, not from ARE alone — §5); wire resource → `ObservableProperty`/`Signal` and `resource_updated` → `focus()` signal delivery (gap 2); adapter-side ADR-0014 id derivation; write the candidate ADRs (stdio-as-origin, `<App>__` mapping canonical).
- [ ] E2. **⚠ harness-risk.** `Agent` + `sora/bootstrap.py` + `agent.yaml` loading — reproduce EXAMPLES.md's full `scenario_email_calendar` as running code (four-step plan, procedural-memory reuse across runs, signal-driven replanning on the mid-scenario follow-up email). Pin the seeded ARE scenario to the installed ARE version, not the sketch. **Target: tag `v0.1.0` here.**
- [ ] E3. First real, model-backed `ReasonStrategy` + `ProceduralMemory.infer()`. Keep behind a skip-gate / fake by default so the suite stays deterministic.
- [ ] E4. CLI polish — `TerminalSession`, `--verbose`, and `DecisionCycle.interrupt()` (the 10ms hard-interrupt path, currently `NotImplementedError`).

## Phase 4 — Backlog / exploratory

- [ ] Blocked-state machinery — the `_suspend_`/`_resume_` internal actions and the signal-satisfies-wait judgment (Situate/Reflect) that transitions a `blocked` activity back to `ready`. Deferred from Phase 3 D5: no v0.1.0 scenario needs a manual that waits on a specific signal.
- [ ] WoT adapter and the two-agent lab scenario (EXAMPLES.md's additional example)
- [ ] Multi-field `TickResult` fusion in practice, replanning-policy experiments

## Notes

- Update this file as phases/steps complete or get reordered — it's the single place tracking implementation status, referenced from [CONTRIBUTING.md](CONTRIBUTING.md) and [CLAUDE.md](CLAUDE.md).
- If an implementation step reveals that a design decision needs to change, write a new ADR superseding the old one (see [docs/adrs/README.md](docs/adrs/README.md)) rather than silently diverging from README.md/EXAMPLES.md.
