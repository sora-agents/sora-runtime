# Phase 2 findings — walking skeleton against ARE

Phase 2 (see [ROADMAP.md](../ROADMAP.md)) ran one full Observe→Act decision cycle against the **real**
ARE MCP server and asked, as its explicit deliverable, to "write up whatever design gaps this
surfaces (new/updated ADRs if a decision needs to change) before continuing to Phase 3." This is
that write-up.

**What ran:** a single skip-gated integration test
([`tests/test_are_walking_skeleton.py`](../tests/test_are_walking_skeleton.py)) spawns ARE's
in-package MCP server over **stdio**, exposing `EmailClientApp`, and drives S-ORA's real
`DecisionCycle` + the real ARE-over-MCP adapter `AreMcpWorkspaceAdapter`
([`src/sora/adapters/are_mcp.py`](../src/sora/adapters/are_mcp.py)) through exactly one external
action — `invoke EmailClientApp.list_emails` — asserting the tool's `OperationAck` comes back through
the cycle. All wiring the tick touches was filled in; everything else stays a Phase-1 stub. The code
is throwaway: Phase 3 replaces it and properly test-drives it (ROADMAP steps 7–13, esp. step 12).

The spike used `--apps ...EmailClientApp` over stdio rather than
`--scenario scenario_email_calendar` over SSE (EXAMPLES.md's sketch) because the latter is not
reachable as written — see gap 3. None of the gaps below block Phase 3; they are corrections to the
README/EXAMPLES sketch and two small typing corrections already applied to `src/`.

---

## Gaps vs the EXAMPLES.md / README sketch

### 1. Tool names are namespaced `<App>__<operation>`, and the app is `EmailClientApp`
The ARE MCP server exposes a **flat** tool list across all loaded apps, each named
`EmailClientApp__list_emails`, `EmailClientApp__send_email`, … (double-underscore separator).
EXAMPLES.md assumes a bare `EmailApp` tool with an `list_emails` operation. Real class:
`are.simulation.apps.email_client.EmailClientApp`; calendar is
`are.simulation.apps.calendar.CalendarApp`.

**Consequence for the adapter (implemented):** one MCP server → one S-ORA `Workspace`; group the flat
list by the `<App>__` prefix into one `Tool` per app; split the name back into
`tool_id` + `operation_name` on `invoke`. This matches EXAMPLES.md's "each app becomes a separate
`Tool` within that workspace" — the mapping just has to be derived from the name prefix, which the
sketch didn't spell out.

**Proposed EXAMPLES.md diff (not yet applied — README-driven discipline):** rename `EmailApp` →
`EmailClientApp`, `CalendarApp` path as above, and note the `<App>__<operation>` naming in the
"Connecting via the ARE MCP server" section.

### 2. Observable properties / signals need MCP **resources**, not tools
ARE exposes app state as MCP resources — `app://info`, `app://EmailClientApp/info`,
`app://EmailClientApp/state` — and signals as `resource_updated` notifications on those URIs (as
EXAMPLES.md's "Signals from ARE write operations" already anticipates). The spike's adapter
synthesizes a `Manual` with **empty** `observable_properties`/`signals` and stubs `focus`/`observe`,
because the single `list_emails` step is a read op that needs neither. Wiring resources →
`ObservableProperty`/`Signal` is Phase 3 (ROADMAP step 12). No sketch change needed — just noting the
mechanism is confirmed and deferred.

### 3. `scenario_email_calendar` does not exist; drive apps with `--apps`
There is no `scenario_email_calendar` scenario id in ARE 1.2.0 (the shipped demo scenario is
`scenario_mz_dinner`). For a deterministic walking skeleton, launching the server with
`--apps are.simulation.apps.email_client.EmailClientApp` is the reliable, seed-free path and yields a
stable empty-inbox result. A seeded, multi-app scenario is what Phase 3's full four-step reproduction
(ROADMAP step 13) will need — at which point the exact scenario id / seeding must be pinned against
the installed ARE version, not the sketch.

**Proposed EXAMPLES.md diff:** replace the `--scenario scenario_email_calendar` invocation with an
`--apps …EmailClientApp …CalendarApp` form for the walking-skeleton framing, and flag scenario
selection as version-dependent.

### 4. stdio is the pragmatic transport for a gated test (SSE remains valid)
EXAMPLES.md specifies `--transport sse` with `address: "http://localhost:8080/sse"`. **stdio** was
used instead: no port to bind, no long-lived HTTP server, and the adapter owns the subprocess
lifecycle via a single `AsyncExitStack` — dramatically more robust for a test that spawns and tears
down the server per run. This is a transport *choice*, not a contradiction: SSE stays a valid origin
form. `WorkspaceOrigin.address` cleanly encodes an HTTP/SSE endpoint; for stdio the adapter instead
holds `command`/`args` and the `address` is a nominal label (`stdio:are-email`). Whether stdio
becomes a first-class, addressable origin form is a Phase-3 adapter-hardening decision (candidate
ADR at ROADMAP step 12), deferred deliberately rather than decided from one spike.

---

## Typing corrections already applied to `src/` (mypy --strict)

These are cases where the README API Sketch, if transcribed literally, does not type-check under
`mypy --strict`. Both are small and were fixed in `src/` with an in-code NOTE; **README diffs are
proposed, not yet applied.**

### A. `MessageTransport.receive` must be a non-async `def` returning an `AsyncIterator`
The sketch declares `async def receive(self) -> AsyncIterator[Message]` but its own
`DefaultObserveStrategy` iterates it as `async for message in cycle.communication.receive()` (no
`await`). Those are mutually inconsistent: an `async def` returning an iterator is a *coroutine* and
can't be `async for`-ed without awaiting first. Fixed by declaring
`def receive(self) -> AsyncIterator[Message]` (implementations are async generators), which is what
the call site already assumes. See [`src/sora/transport.py`](../src/sora/transport.py).

### B. Predefined external actions must keep the uniform `**kwargs` signature to *be* `ExternalAction`
The sketch writes `InvokeAction.execute(..., *, activity_id, tool_id, operation_name, **params)` with
explicit `tool_id`/`operation_name`. Adding required keyword-only params beyond the `ExternalAction`
Protocol's `(..., *, activity_id: str, **kwargs)` makes `InvokeAction` **not** a structural subtype,
so `ActionRegistry.register_external(InvokeAction())` fails type-checking. Fixed by keeping the
uniform signature and reading `tool_id`/`operation_name` out of `**kwargs` inside `execute`. The
explicit-param form in the README is fine as *illustration*, but the real, registrable implementation
must match the Protocol. See [`src/sora/action.py`](../src/sora/action.py).

---

## 5. The adapter is ARE-specific — generic MCP base deferred to Phase 3
The Phase-2 adapter speaks MCP as the wire protocol but bakes in ARE's own data model on top of it,
so it is named and defined as an **ARE adapter** (`AreMcpWorkspaceAdapter`, `name = "are-mcp"`), not a
general MCP adapter. What is ARE-specific vs. generic MCP:

| Concern | Nature |
|---|---|
| stdio spawn, `ClientSession`, `initialize`, `list_tools`, `call_tool`, result parsing, workspace lifecycle | generic MCP |
| `<App>__<operation>` tool grouping + the inverse name assembly on `invoke` | ARE-specific |
| "app" manual framing, `app://{app}/state` resources, `resource_updated` signals | ARE-specific |

**Proposed split (Phase 3, ROADMAP step 12):** extract the generic rows into a protocol-only
`McpWorkspaceAdapter` base in `sora.adapters.mcp`, with `AreMcpWorkspaceAdapter(McpWorkspaceAdapter)`
overriding two hooks — *grouping* (`list[mcp.Tool] -> S-ORA Tools`) and its inverse *name assembly*
(`(tool_id, operation) -> mcp_tool_name`) — plus manual synthesis and (later) resource → property/
signal mapping. **Deferred deliberately:** the one undecided piece is the base's *default* grouping
for a plain MCP server with no naming convention (one Tool per MCP tool? one Tool per server?), and
drawing that abstraction from ARE alone would be guessing from a single example. The seam is marked
in-file (`# generic MCP mechanics` vs `# ARE-specific`) so the extraction is mechanical once a second,
non-ARE MCP consumer exists to validate it. Since the spike code is throwaway, the hook signatures
are best committed to under Phase 3's tests, not now.

## 6. `address` is nearly vestigial in ARE — and tool identity is unspecified (→ ADR-0014)
In the ARE-over-MCP-stdio case, neither `address` field carries routing meaning:
- **`Tool.address` is `None` for every ARE tool** — all apps are multiplexed over the workspace's
  single stdio connection (`_AreMcpTool` sets `self.address = None`), so no app has an endpoint of
  its own. (Contrast the two-agent lab, where `robotic-arm` is a physical device with its own URI.)
- **`WorkspaceOrigin.address` is a nominal label** (`"stdio:are-email"`), not a locator: stdio has no
  URL, so the real reconnect info is the `command`/`args` held on the adapter instance. Even the SSE
  variant's `http://…/sse` is a *workspace*-level URI — it locates the one MCP server, not any app.

This matters because it refines the "identity rides on the address (URI), like the Web" model rather
than discarding it: MCP gives no per-tool identity beyond a bare *name*, unique only within one server,
so identity must be *derived from* a global address, not read off the tool directly. Yet the live layer
addresses tools by a bare `Tool.id` everywhere (`EnvironmentRegistry._tools`, `focused_tools`,
`invoke`, `Percept.source`) while nothing specifies that id is unique — and `ToolRecord` scopes an
instance by `workspace_id` only at the persistence layer. One agent joining two workspaces with the
same id therefore hits a **silent overwrite on join**, and — worse — a **cross-workspace
deregistration on `leave`** (leaving A pops a shared id and removes B's still-live tool). The in-flight
result path is unaffected (keyed by `op_id`, not `tool_id`).

**Decision:** [ADR-0014](adrs/0014-tool-identity-globally-unique.md) (proposed) — `Tool.id` is
**globally unique**, guaranteed by the per-protocol adapter via a deterministic derivation from the
tool's global address/origin (URI-based protocols get it free; name-only ones synthesize from the
origin). Global identity is what lets two agents focus the *same* tool, or message about it, under one
name (per the A&A shared-tool model). A single registry can only *enforce* the slice it sees (fail loud
on duplicate at join, `leave` stops popping shared ids), so global uniqueness rests on adapter
correctness with local enforcement as the backstop. Live-layer enforcement lands with the Phase 3
adapter hardening (ROADMAP step 12); no code changed yet.

## 7. Decision-cycle spike review (three seams)
A review of `DecisionCycle.tick()` surfaced three points. The README contract for the first is
corrected now (README-driven); the code fixes land in Phase 3.

### 7a. Situate must always run — the `result.activity` gate is wrong (contract fixed)
`tick()` only calls Situate `if result.activity is None`, mirroring the README's old `SituateStrategy`
line "Only called if `result.activity` is still None." That is a **bug in the contract**: Situate has
two duties — *select* the activity **and** *adjust* working memory for it (focus tools, load/unload
manuals, filter percepts) — but only *select* is a `TickResult` field. Gating the whole phase on that
field means a pre-selected activity skips the wm-adjustment, which must reflect *this* cycle's fresh
percepts regardless of how (or when) the activity was chosen. Even a legitimately forced selection
(e.g. an uncommon Observe pinning the activity that handles a critical signal) must still be situated.

This is **not** a consequence of ADR-0011 — that ADR only defines the field-threading mechanism
("call a phase's strategy only if its field is still None") and says nothing about a fusing phase
inheriting Situate's responsibilities. It is also in tension with the README's own "decision-chain
fusion starts at Situate": normally nothing upstream sets `activity`, so Situate always ran and the
gate was inert — which is why the spike never tripped it and the contract error went unnoticed in
review.

**Fixed in the contract (this change):** the `SituateStrategy` docstring and the `tick()` sketch in
README.md now make Situate **always run**; it selects only if `result.activity is None` (a pre-set
selection is respected and situated, not overridden), and the head-of-chain phase keeps its ability to
fill `step`/`invocation` — only Situate's *own* activity gate is removed. The forward-fusion gates on
Reason (`step`) and Act (`invocation`) are unchanged. **Deferred to Phase 3 (code):** `cycle.py`'s
`if result.activity is None` guard, `DefaultSituateStrategy` (which currently early-returns when
`activity` is set), and the stale `SituateStrategy` docstring in `src/sora/strategies.py` all still
carry the old gate and must catch up when the cycle is TDD'd. This makes Situate a deliberate
*exception* to the uniform "gate on your own field" rule; [ADR-0011](adrs/0011-phase-fusion-via-threaded-result.md)
has been refined in place (it is still `proposed`) to record that exception.

### 7b. The invocation runs *after* the Act strategy — legibility, not a defect
`ActStrategy.bind()` only *binds* a `Step` into a concrete `OperationInvocation` (the pluggable,
hallucination-prone half); the actual tool call happens in the uniform `action.execute(...)` dispatch
after the phase pipeline. This split is deliberate and correct: execution is a *mechanism* (identical
for every external action, routed through the extensible `ActionRegistry` per ADR-0008), not a
per-agent decision, and keeping it out of the strategy is what lets a fused Reason fill `invocation`
directly and what centralizes the one-action-per-cycle gate + off-cycle async firing. The fair
critique is **legibility**: the dispatch is inline in `tick()` and gated by `step.next_action ==
"invoke"`/`"wait"` string compares, so it reads as outside Act when it *is* Act's execution half.
Phase-3 cleanups: name the boundary (an `_act()` doing bind-then-dispatch) and replace the
`next_action == "invoke"` special-case (whether a step needs binding should be a property of the
action, not a hardcoded branch in the generic cycle).

### 7c. Inline string constants — debt, with a category-aware fix (not one enum)
`"invoke"`/`"wait"` (cycle.py), `"property"`/`"signal"` (Percept kinds), and the
`"tool_id"`/`"operation_name"` param keys are stringly-typed: typos invisible to mypy, no single
source of truth. The Phase-3 fix is **not** a blanket enum — that would fight ADR-0008 for the
*extensible* action space. Instead: `Literal`/`StrEnum` for genuinely closed sets (`Percept.kind`);
reuse `InvokeAction.name` rather than the literal `"invoke"`; make `"wait"` a named sentinel (or a
non-string representation, since the cycle special-cases it); and give the invoke param keys a typed
carrier instead of loose dict lookups.

## Not decided here (deferred to Phase 3)
- Cycle code catching up to the corrected Situate contract (§7a): drop the `activity` gate in
  `cycle.py`/`DefaultSituateStrategy` and update the `src/sora/strategies.py` docstring. (The contract
  itself — README + ADR-0011 — is already corrected; only the code lags.)
- Act-phase legibility + de-stringifying the cycle constants (§7b, §7c).
- Enforcing tool-id uniqueness in the live layer per ADR-0014 (registry guard + adapter derivation).
- Extracting the generic `McpWorkspaceAdapter` base and its default tool-grouping policy (section 5).
- Whether `<App>__<operation>` mapping and stdio-as-origin become canonical adapter behavior (ADR
  candidates at ROADMAP step 12, once the adapter is hardened and TDD'd).
- Resource → `ObservableProperty`/`Signal` and `resource_updated` → `focus()` signal delivery.
- Seeded multi-app scenario for the full four-step `schedule-from-email` reproduction (step 13).
