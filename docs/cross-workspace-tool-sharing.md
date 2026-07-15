# Sharing one tool across two workspaces — analysis and open question

Design note behind Track C's cross-workspace-tool-sharing task (C6). Records why the registry
currently forbids the same tool from appearing in two workspaces, why the Web/hypermedia framing makes
that constraint look too strict, and the options for relaxing it. No decision is made here — this is
the analysis that motivates the task; the decision belongs in an ADR that refines
[ADR-0014](adrs/0014-tool-identity-globally-unique.md), realized when the WoT adapter lands.

## The constraint today

`EnvironmentRegistry._register` keeps a flat `id -> Tool` map and fails loud (`ValueError`) whenever a
`Tool.id` is already registered — including when a *second* workspace presents a tool with an id the
*first* already contributed. So the same tool cannot be a member of two workspaces. ADR-0014 justifies
this as collision detection: *"a collision the registry can see must fail loudly, never silently
corrupt a sibling workspace."*

## The motivating use case

In a Web/hypermedia environment a workspace can be a *logical container*, not a connection owner. Two
workspaces — each hosted on its own server — could both reference a tool that actually runs on a
**third** server (reachable at its own address, via `Tool.address` overriding the workspace address,
which the design already permits). Here "the same tool id in two workspaces" is **correct**, not a bug:
both workspaces legitimately link the same globally-identified resource.

## The real question: is a `Workspace` a *container* or an *index*?

- **A&A / CArtAgO (containment).** An artifact is created in and owned by exactly one workspace;
  membership is exclusive. CArtAgO's reference implementation cannot place the same artifact instance
  in two workspaces (cross-artifact `link` operations exist, but not shared membership). This is the
  conservative view, and it is correct for the role it assumes — you do not share a *connection*.
- **Web / hypermedia (referencing).** Identity is a URI; a resource is *linked* from many collections
  without being *contained* by any. A WoT Thing has a URI `id` and can be listed in multiple Thing
  Directories at once. Membership is non-exclusive because identity is decoupled from location and from
  collection.

ADR-0014 already took the first Web step — tool identity moved onto a global URI (*"two agents focusing
the same addressable tool derive the same id"*). This task is the second step (one agent, same tool,
two workspaces) and it exposes a tension **inside ADR-0014**: it wants global identity *and* wants the
registry to treat any duplicate id as a collision. Those conflict exactly here.

## Arguments for allowing it (with verification)

- **The conservative reject can pressure adapters to break global identity.** Faced with a legitimate
  shared tool and a hard reject, an adapter author's only escape is to mint *different* ids for the
  same underlying tool — precisely the failure ADR-0014 warns against (*"globally unique degrades to
  merely locally unique"*). Allowing verified sharing relieves that pressure and keeps the agent's
  focus/invoke-by-id coherent.
- **Standards precedent.** WoT Thing Directories (a Thing in N directories), REST/HATEOAS (a resource
  linked from many parents), Linked Data (same resource, many graphs). "Identity ≠ container" is
  well-trodden on the Web.
- **It fits the existing design.** `Tool.address` already lets a tool have its own endpoint distinct
  from its workspace's, so the design already admits tools the workspace does not own the connection
  to. Same tool in two such workspaces is the natural next increment.

## Arguments against / the hard problems

1. **Lifecycle & `leave()`.** `leave(ws)` closes the connection and deregisters the workspace's tools.
   Sharing needs **refcounted deregistration** (drop the shared tool only when the last referencing
   workspace leaves) and a rule that closing one workspace's connection must not tear down a tool that
   lives on a third server. (This ambiguity already latently exists for `Tool.address`-override tools;
   sharing forces it to be resolved.)
2. **Verification is protocol-dependent and address-equality is fragile.** "Same instance?" is sound
   only with a global discriminator. For URI/WoT tools, id equality already *implies* same instance
   (deterministic derivation), and an address check is cheap belt-and-suspenders — but needs URI
   **canonicalization** (trailing slash, scheme case, default ports) to avoid false negatives. For
   MCP/stdio tools `address` is `None` and the id is synthesized from the workspace origin, so
   same-instance cannot be verified — and ADR-0014's own logic says these *are not shared tools*.
   The policy must therefore keep rejecting where the discriminator is weak/absent.
3. **Loss of loud-failure bug detection.** Reject-always catches a genuine adapter bug — two workspaces
   minting the same id for *different* tools. Relaxing to "allow if addresses match" risks *masking*
   such a collision as a false "same instance" whenever the discriminator is weak, silently merging
   two distinct tools (worse than crashing).
4. **Which handle/manual wins?** "Same id" ≠ "same `Tool` object": each adapter builds its own handle,
   but `_tools[id]` holds one. The registry must pick a canonical handle and, if the two workspaces
   carry *different* manuals for it (authored vs. adapter-synthesized), reconcile them — which is the
   two-channel `Manual` merge deferred to E5. Sharing drags E5 forward.

## The clean boundary (candidate synthesis)

The `Workspace` conflates two roles — (a) a connection/lifecycle boundary and (b) a logical
grouping/index. Exclusivity is intrinsic to (a) but unnatural for (b). The motivating case works
precisely because the shared tool's connection is *not* the workspace's. So split by connection
ownership:

- **Connection-owned tool** (`address is None`, uses the workspace's connection) → **exclusive**, as
  today; sharing is meaningless and reject stays correct.
- **Self-addressed tool** (own global URI, connection-independent) → membership is pure role (b), so it
  may be **referenced from multiple workspaces**; `leave()` never tears down its connection, and its
  deregistration is refcounted.

Registry model: `_tools: id -> Tool` (dedup on the first canonical handle), `_workspace_tools` becomes
many-to-many, a refcount tracks shared self-addressed ids, and `_register` admits a duplicate id **only**
when both handles carry a canonicalized-equal global address — otherwise it raises exactly as now. This
keeps loud failure as the default, confines relaxation to URI-identified tools where verification is
sound, and only relaxes lifecycle for tools the workspace never owned.

## Options to weigh (not yet decided)

1. **Keep conservative (reject always).** Simplest; matches CArtAgO. Cost: pressures adapters to break
   global identity for legitimate shared tools.
2. **Allow verified sharing, scoped to self-addressed tools** (the synthesis above). Enables the Web
   use case with sound verification and refcounted lifecycle; costs registry bookkeeping + URI
   canonicalization + partial E5 coupling.
3. **Explicit aliasing/linking** — a workspace declares a *reference* to a tool owned elsewhere,
   distinct from owning it (closer to CArtAgO's `linkArtifacts`). Most explicit semantics, most
   machinery.

## Timing

Not needed for v0.1.0 — the ARE email/calendar scenario has no shared tools. The case only bites with
the WoT adapter and the two-agent lab (Phase 4 backlog). Building refcounting + canonicalization now
would be speculative and would prematurely entangle E5. Record the container-vs-index resolution and
the "shareable ⟺ self-addressed + canonical-address match" rule as an ADR refining ADR-0014, to be
realized when the WoT adapter lands. Shares the registry-bookkeeping surface (refcounted membership)
with the C5 dynamic-environments work ([docs/restore-drift-reconciliation.md](restore-drift-reconciliation.md)).
