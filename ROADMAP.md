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

- [ ] Minimal types/actions/cycle wiring needed for exactly one `tick()` — only what this spike touches, not the full API sketch
- [ ] Real MCP `WorkspaceAdapter` wired to ARE's local MCP server: `python are_simulation_mcp_server.py --scenario scenario_email_calendar --transport sse`
- [ ] Hardcoded, deterministic `ReasonStrategy` that does just the *first* step of the `scenario_email_calendar` scenario — `invoke EmailApp.list_emails` — no LLM call, no multi-step plan yet
- [ ] **Done when:** the single step completes end-to-end against the real ARE MCP server and is captured as one integration test; write up whatever design gaps this surfaces (new/updated ADRs if a decision needs to change) before continuing to Phase 3

## Phase 3 — TDD rollout

- [ ] 1. Core value types + `WorkingMemory`
- [ ] 2. `NotificationQueueSink` / `SignalSink`
- [ ] 3. Memory modules against a file-backed `MemoryBackend`
- [ ] 4. `Manual` + `ManualParser` (Markdown) — reuse EXAMPLES.md's manuals as fixtures
- [ ] 5. `Tool`/`Workspace`/`WorkspaceAdapter` against a fake, in-process adapter
- [ ] 6. `EnvironmentRegistry` (join/leave/restore) against the fake adapter
- [ ] 7. The six predefined actions (Invoke/Focus/Unfocus/Join/Leave/Send)
- [ ] 8. `DecisionCycle` — Observe only, `DefaultObserveStrategy`
- [ ] 9. Reflect and Situate default (deterministic) strategies
- [ ] 10. Reason + Act end-to-end with a deterministic `ReasonStrategy` (no LLM)
- [ ] 11. `Plan`/`Step` + `ProceduralMemory` retrieve/infer/store
- [ ] 12. Harden the Phase 2 spike into the real, properly TDD'd MCP `WorkspaceAdapter`
- [ ] 13. `Agent` + `sora/bootstrap.py` + `agent.yaml` loading — reproduce EXAMPLES.md's full `scenario_email_calendar` scenario as running code (four-step plan, procedural-memory reuse across runs, signal-driven replanning on the mid-scenario follow-up email) — **target: tag `v0.1.0` here**
- [ ] 14. First real, model-backed `ReasonStrategy`
- [ ] 15. CLI polish (`TerminalSession`, `--verbose`, interrupt handling)

## Phase 4 — Backlog / exploratory

- [ ] WoT adapter and the two-agent lab scenario (EXAMPLES.md's additional example)
- [ ] Multi-field `TickResult` fusion in practice, replanning-policy experiments

## Notes

- Update this file as phases/steps complete or get reordered — it's the single place tracking implementation status, referenced from [CONTRIBUTING.md](CONTRIBUTING.md) and [CLAUDE.md](CLAUDE.md).
- If an implementation step reveals that a design decision needs to change, write a new ADR superseding the old one (see [docs/adrs/README.md](docs/adrs/README.md)) rather than silently diverging from README.md/EXAMPLES.md.
