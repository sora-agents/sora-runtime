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
- [x] A2. **Reconcile registry access (decided).** The agent reasons over what it has joined, so `EnvironmentRegistry` stays reachable from `WorkingMemory` (spike's call, kept — "currently-joined, live workspaces" is per-cycle contextual state, distinct from SemanticMemory's durable records) through a **read-only `EnvironmentView` Protocol** (`get`/`get_workspace`/`all_tools`/`joined_workspaces`); the concrete, mutation-capable `EnvironmentRegistry` remains the single shared instance (per [ADR-0013](docs/adrs/0013-shared-instances-narrow-dependencies.md)) that `DecisionCycle` holds and the Join/Leave actions use. **Docs:** README `EnvironmentView`/`WorkingMemory`/`tick()`/`Agent`/`bootstrap` reconciled + ADR-0013 refined. **Code:** `EnvironmentView` added, `WorkingMemory.registry` retyped, mutable handle moved onto `DecisionCycle` (`tick()` dispatches through `self.registry`), `Agent` takes the shared instance; `mypy --strict` verified the boundary (a mutator call through `wm.registry` is rejected). Unblocks Track C and D4.
- [x] A3. [P] De-stringify the cycle constants (§7c) — `Percept.kind` as a `PerceptKind` StrEnum, reuse `InvokeAction.name` instead of the literal `"invoke"`, `WAIT` sentinel for the cycle's no-op step, and `TOOL_ID`/`OPERATION_NAME` key constants. *(Chose named constants over a distinct invoke-param carrier dataclass — a carrier would near-duplicate `OperationInvocation`; the deeper `next_action == "invoke"` special-casing is left for D4/§7b, marked with an in-code NOTE.)* Behavior-preserving; guarded by the existing suite + `mypy --strict`.
- [x] A4. [P] Decide which `tests/test_cycle_wiring.py` assertions become permanent TDD tests vs. get replaced; drop the "throwaway spike" framing as each layer below is re-driven. **Done:** triage recorded in [docs/phase-3-test-triage.md](docs/phase-3-test-triage.md) — 8 of 9 assertions promoted (registry pair extended under C2), the end-to-end tick re-driven under D4, only spike *scaffolding* (wiring helper, in-file fakes, string-typed reason fixture) replaced; no assertion discarded. The blanket "throwaway spike" docstring is replaced with a pointer to the triage; per-group promotion happens as C1/C2/C4/D1/D4 land (annotated below).

### Track B — Long-term memory & manuals (B1 gates B2–B4; B2/B3/B4/B5 then parallel)

- [x] B1. File-backed `MemoryBackend` + round-trip tests. Gate for B2–B4.
- [x] B2. [P] `SemanticMemory` — manual + workspace/tool record store/retrieve/list (needed by Join/Leave and `restore()`).
- [x] B3. [P] `EpisodicMemory` — learn/consult.
- [x] B4. [P] `ProceduralMemory` retrieve/store (deterministic); `infer()` left as a stub until E3 (it's the LLM path).
- [x] B5. [P] `Manual` + Markdown `ManualParser` — parse our clean Markdown format into a `Manual` **envelope**. Fully independent of A and B1. Scope settled while consolidating the manual model (see [ADR-0015](docs/adrs/0015-manuals-protocol-agnostic-adapter-boundary.md)):
  - The Markdown channel fills `id`, `metadata`, `description`, and verbatim `raw_text`; it does **not** lift the bullet-list sections into the structured `observable_properties`/`signals`/`operations` fields. That field extraction proved brittle and no consumer reads it; section prose is served lazily from `raw_text` via `Manual.section(...)`. The structured fields are the *adapter* channel's to fill from native schemas.
  - `# Tool Metadata` `id:` is required (reject-on-no-id — `invalid/missing-metadata.md`); `water-pump.md` is the clean exemplar; inherited manuals under `tests/fixtures/manuals/inherited/` are conversion *source material*, not parser input.
  - Scope boundary: `ManualParser` produces a `Manual` from Markdown only. Extracting a `Manual` from a native description (MCP tool list / WoT TD) is the `WorkspaceAdapter`'s job — the two are parallel `Manual` producers reconciled by `Manual.id`.
  - Deferred (ADR-0015): a later structured-header + prose-body format, for when a consumer needs machine-readable schemas *from* hand-authored manuals; and reconciling a Markdown manual with a native description for the same `Manual.id` (→ E5).

### Track C — Environment & actions (after A2; C3-join and C2-restore also need B2)

- [x] C1. [P] `Tool`/`Workspace`/`WorkspaceAdapter` fake in-process adapter, promoted from the spike into a reusable test fixture. *(A4: absorbs the spike's `FakeTool`/`FakeWorkspace`/`FakeAdapter`.)* **Done:** the fuller double lives in [`tests/fakes.py`](tests/fakes.py) — canned invoke results + call log, `focus(sink)` signal emission, configured `observe()`, and a `connect()` that rebuilds from records with `tool_record.address`→origin fallback; conformance to the three Protocols is enforced by `mypy --strict` (it scans `tests/`), so its own suite ([`tests/test_fakes.py`](tests/test_fakes.py)) stays lean — 4 characterization tests on the non-trivial, not-yet-consumed logic only. The spike's in-file fakes stay in `test_cycle_wiring.py` until C2/C3/C4/D1/D4 lift their groups.
- [x] C2. `EnvironmentRegistry` — keep join/leave/get; add [ADR-0014](docs/adrs/0014-tool-identity-globally-unique.md) id-uniqueness enforcement (fail loud on duplicate id at join; `leave` never pops a shared id); implement `restore()` (needs B2). *(A4: promotes + extends the spike's join/get/leave assertions → `tests/test_environment.py`.)* Done: `_register` now validates ids before mutating (atomic, fail-loud `ValueError` on a duplicate tool id or workspace id); `restore()` reconnects via `adapter.connect()`, resolving manuals from `SemanticMemory` and grouping tool records per workspace; permanent `tests/test_environment.py` promotes+extends the spike's registry group (still live in `test_cycle_wiring.py` until D4 retires that file).
- [x] C3. [P] The five still-stubbed predefined external actions — Focus/Unfocus (wire `focused_tools` + `signal_sink`), Join/Leave (wrap registry + persist records via B2), Send (needs `MessageTransport`). Independent of each other. *(A4: promotes the spike's `action_registry_lookup` alongside C4.)* **Done:** implemented in [`src/sora/action.py`](src/sora/action.py), driven by [`tests/test_action.py`](tests/test_action.py) (7 tests) over the C1 fakes + a real `FileMemoryBackend`. Each action was rewritten to the uniform `(registry, cycle, **kwargs)` signature (reading its params out of `**kwargs`) — the explicit-param stubs from Phase 1 were **not** structural `ExternalAction`s under `mypy --strict`, so `register_external()` didn't type-check; same fix already applied to `InvokeAction` in Phase 2 ([phase-2-findings §B](docs/phase-2-findings.md)). Also fixed the stale `tools.get/join/leave` → `registry.get/join/leave` in the README action sketches (the parameter was renamed to `registry` in A2 but the bodies weren't updated).
- [x] C4. [P] Re-drive `InvokeAction` (already implemented) as a permanent TDD test. *(A4: promotes the spike's `invoke_action_sets_running_then_pushes_result` → `tests/test_action.py`.)* **Done:** the promoted `invoke_action_sets_running_then_pushes_result` was folded **into** the C3 [`tests/test_action.py`](tests/test_action.py) (one file for the whole predefined action space, not a parallel one) as five focused invoke tests — immediate-ack-and-`RUNNING`, bound `pending_operation`/`invoked_at`, off-cycle `result_sink` keyed by `invocation_id`, params pass-through to both tool and invocation, and distinct-id-per-concurrent-invoke — over the C1 fakes + a real `FileMemoryBackend`. The registry lookup keys on each action's `name` constant (A3), not the `"invoke"` literal, and gains internal-lookup + unknown-`KeyError` coverage. The InvokeAction group is lifted out of `test_cycle_wiring.py`; no `src/` change (the Phase-2 implementation already satisfied the contract).
- [ ] C5. Dynamic environments — handle a workspace whose live tool set has drifted since it was last joined. Today `restore()` is a pure snapshot reconstruction (record-driven, skips `discover()`, no write-back): a newly-added tool is silently invisible, a removed tool becomes a stale handle whose failure is deferred (adapter-dependent), and a changed manual restores stale — with **no reconciliation/refresh action and no drift detection**, so an agent that restores into a changed environment silently runs on a stale world model. Decide and build the fix (leading candidate: an explicit, agent-driven `refresh`/`resync` external action that re-`discover()`s a joined workspace, diffs against the registry + records, applies the register/deregister delta, and re-persists — keeping `restore()` fast and pure). Pin the removed-tool `connect()` semantics (eager-validate vs. lazy-rebuild) in an adapter ADR; couples to the Track E MCP-adapter hardening. Full analysis, drift table, and options in [docs/restore-drift-reconciliation.md](docs/restore-drift-reconciliation.md).
- [ ] C6. Cross-workspace tool sharing — decide whether the same tool may be a member of two workspaces. Today `_register` rejects any duplicate `Tool.id` ([ADR-0014](docs/adrs/0014-tool-identity-globally-unique.md)), so a tool cannot appear in two workspaces. That's correct for the A&A/CArtAgO *containment* view (exclusive membership, one owner) but too strict for the Web/hypermedia *index* view, where a workspace is a logical container and two workspaces — each on its own server — could legitimately reference a tool running on a third (via `Tool.address` override). ADR-0014 already put identity on a global URI, which surfaces a tension: it wants global identity *and* rejects duplicate ids as collisions. Leading candidate: split by connection ownership — connection-owned tools (`address is None`) stay exclusive; self-addressed tools (own global URI) may be referenced from multiple workspaces, with `_register` admitting a duplicate id only on canonicalized-equal address, `_workspace_tools` many-to-many, and refcounted deregistration. Couples to E5 (which manual wins for a shared tool) and the C5 refcount bookkeeping. Not needed for v0.1.0 (ARE has no shared tools); realize with the WoT adapter + two-agent lab, as an ADR refining ADR-0014. Full analysis and options in [docs/cross-workspace-tool-sharing.md](docs/cross-workspace-tool-sharing.md).

### Track D — Decision cycle proper (mostly serial; needs A3 + the strategies)

- [x] D1. `DecisionCycle` Observe-only + `DefaultObserveStrategy`, re-driven as permanent. *(A4: promotes the spike's `observe_resolves_running_activity`, and the `NotificationQueueSink` group → `tests/test_perception.py`, its first real consumer.)* Done: `DefaultObserveStrategy` (already implemented from the spike) and `NotificationQueueSink` needed no `src/` change — this promotion pins their full contract. `tests/test_perception.py` lifts the three sink tests verbatim (incl. `drain_snapshots_current_depth`'s within-cycle no-starvation invariant) and adds the `PerceptKind` StrEnum guarantee; `tests/test_cycle.py` is the Observe home — the four channels, the automatic 1:1 running-resolution (matched/only-matched/unmatched-ack guard), and an Observe-only `tick()` that populates percepts+messages then returns without dispatching. Both promoted groups removed from `test_cycle_wiring.py` (registry + end-to-end tick groups remain until D4 retires the file).
- [ ] D2. `DefaultReflectStrategy` — deterministic completion/failure judgment + store-on-success to episodic (B3) and procedural (B4).
- [ ] D3. `DefaultSituateStrategy` — **fix §7a** (always run; select only if `result.activity is None`), activity-creation-from-message via the `_create_activity_` internal action, and wm adjustment (focus/load/unload/filter). Also update the stale `SituateStrategy` docstring in `src/sora/strategies.py`.
- [ ] D4. Reason + Act end-to-end with a deterministic `ReasonStrategy` (no LLM) — **includes §7b**: introduce an `_act()` bind-then-dispatch boundary and drop the hardcoded `next_action == "invoke"` branch (let the action declare whether it needs binding). Fixes the lingering §7a `if result.activity is None` gate in `cycle.py`. *(A4: re-drives the spike's `tick_end_to_end_invoke_then_resolve` — keep the outcome assertions, rebuild the fixture/harness — and retires `test_cycle_wiring.py` once its last group has moved.)*
- [ ] D5. The `_create_activity_` internal action that the default Situate depends on. (Blocked-state `_suspend_`/`_resume_` + the signal-satisfies-wait judgment moved to Phase 4 — v0.1.0's scenario drives replanning through Observe→Situate, not the blocked path.)

### Track E — Integration & release tail (serial; real network/model)

- [ ] E1. **⚠ harness-risk.** Harden the MCP adapters — extract a protocol-only `McpWorkspaceAdapter` base from `AreMcpWorkspaceAdapter` (grouping + name-assembly hooks; design the default grouping policy here, not from ARE alone — §5); wire resource → `ObservableProperty`/`Signal` and `resource_updated` → `focus()` signal delivery (gap 2); adapter-side ADR-0014 id derivation; write the candidate ADRs (stdio-as-origin, `<App>__` mapping canonical).
- [ ] E2. **⚠ harness-risk.** `Agent` + `sora/bootstrap.py` + `agent.yaml` loading — reproduce EXAMPLES.md's full `scenario_email_calendar` as running code (four-step plan, procedural-memory reuse across runs, signal-driven replanning on the mid-scenario follow-up email). Pin the seeded ARE scenario to the installed ARE version, not the sketch. **Target: tag `v0.1.0` here.**
- [ ] E3. First real, model-backed `ReasonStrategy` + `ProceduralMemory.infer()`. Keep behind a skip-gate / fake by default so the suite stays deterministic.
- [ ] E4. CLI polish — `TerminalSession`, `--verbose`, and `DecisionCycle.interrupt()` (the 10ms hard-interrupt path, currently `NotImplementedError`).
- [ ] E5. Reconcile the two `Manual` provenance channels end to end (ADR-0015). Today a tool has *either* a hand-authored Markdown manual (`MarkdownManualParser` → envelope: `raw_text` + semantics, structured specs empty) *or* an adapter-synthesized one (native schemas → structured specs, no `raw_text`) — never a merged view. Pair the two for the same `Manual.id`: load the authored Markdown alongside a native description (WoT TD affordances, MCP tool schemas) and define the **merge policy** deferred in ADR-0015 (which side wins per field — expected: adapter fills the structured specs, Markdown supplies `raw_text`/semantics). First consumer is the WoT sketch's per-Thing `MarkdownManualParser().parse(load_manual(td.id))` pairing (EXAMPLES.md); builds on the E1 MCP-adapter hardening and the Phase-4 WoT adapter.

## Phase 4 — Backlog / exploratory

- [ ] Blocked-state machinery — the `_suspend_`/`_resume_` internal actions and the signal-satisfies-wait judgment (Situate/Reflect) that transitions a `blocked` activity back to `ready`. Deferred from Phase 3 D5: no v0.1.0 scenario needs a manual that waits on a specific signal.
- [ ] WoT adapter and the two-agent lab scenario (EXAMPLES.md's additional example)
- [ ] Multi-field `TickResult` fusion in practice, replanning-policy experiments

## Notes

- Update this file as phases/steps complete or get reordered — it's the single place tracking implementation status, referenced from [CONTRIBUTING.md](CONTRIBUTING.md) and [CLAUDE.md](CLAUDE.md).
- If an implementation step reveals that a design decision needs to change, write a new ADR superseding the old one (see [docs/adrs/README.md](docs/adrs/README.md)) rather than silently diverging from README.md/EXAMPLES.md.
