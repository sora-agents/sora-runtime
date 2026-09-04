# Contributing to S-ORA

Thanks for your interest in the S-ORA runtime. This project is currently in active development — see [README.md](README.md) for pointers to the conceptual model, [docs/reference/python-api.md](docs/reference/python-api.md) for the exact type/signature reference, [EXAMPLES.md](EXAMPLES.md) for worked scenarios, and [docs/architecture/adrs/](docs/architecture/adrs/) for why specific decisions were made before implementation started. [ROADMAP.md](ROADMAP.md) tracks the open work remaining before the first release.

## Getting set up

```
uv sync --all-extras --dev
uv run pre-commit install
```

`uv` provisions the pinned Python version (see `.python-version`) and installs locked dependencies (`uv.lock`) — no separate virtualenv setup needed.

## Workflow

- Branch from `main`: `feat/...`, `fix/...`, `docs/...`, `refactor/...`, `test/...`, `chore/...`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Open a PR against `main`; PRs are squash-merged, so keep each PR to one logical change.
- Before pushing: `uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest` (or just let `pre-commit` and CI catch it).

## Design changes

If a change touches something documented in [README.md](README.md) or [EXAMPLES.md](EXAMPLES.md), propose the diff in the PR description before or alongside the code change — these files are the spec, not just documentation of whatever the code happens to do.

If a change reverses or refines an accepted decision in [docs/architecture/adrs/](docs/architecture/adrs/), do not edit that ADR's Decision Outcome in place. Write a new ADR that supersedes it, and update the old one's status line to `superseded by ADR-NNNN`. See the [ADR index](docs/architecture/adrs/README.md) for conventions and numbering.

## Testing

This project follows TDD where practical: write a failing test that captures the expected behavior, implement against it, then refactor. Fakes and determinism come before real network adapters or model-backed strategies, so keep new tests in that spirit (prefer a fake `WorkspaceAdapter`/deterministic `ReasonStrategy` over a real one unless the real one is specifically what's being tested).

## Code style

See [CLAUDE.md](CLAUDE.md) for the architectural habits and code style this project follows (Protocol over inheritance, `@dataclass(frozen=True)` value types, async-first, sparse docstrings). Formatting and linting are enforced by `ruff`, not manually — if `ruff format`/`ruff check` pass, style is fine.

## Reporting bugs / requesting features

Open a GitHub issue. For security-sensitive reports, see [SECURITY.md](SECURITY.md) instead of a public issue.
