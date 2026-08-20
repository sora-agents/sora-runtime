# Manual Format Reference

The exact grammar `MarkdownManualParser` accepts, as defined in
[`src/sora/manual.py`](https://github.com/sora-agents/sora-runtime/blob/main/src/sora/manual.py).
Malformed input raises `ManualParseError`. For what a manual *is* and why it's structured this way,
see [Concepts — Manuals](../concepts/manuals.md).

## Document shape

A manual is a flat sequence of `# `-level sections. Text before the first `# ` heading is ignored.
Only six headings are canonical (the `ManualSection` enum — matching by exact text, not case- or
whitespace-normalized):

| Heading | Required | Becomes |
| --- | --- | --- |
| `# Tool Metadata` | **yes** | `Manual.id` + `Manual.metadata` |
| `# Functional Description` | no | `Manual.description` |
| `# Observable Properties` | no | `Manual.observable_properties` (only if it carries an interface block — see below) |
| `# Signals` | no | `Manual.signals` (only if it carries an interface block) |
| `# Operations` | no | `Manual.operations` (only if it carries an interface block) |
| `# Usage Protocols & Safety` | no | read via `Manual.section(...)`; not lifted into any typed field |

A manual missing `# Tool Metadata` entirely, or with no `id:` line inside it, raises
`ManualParseError`. Every section's raw prose survives verbatim in `Manual.raw_text` and stays
readable via `Manual.section(ManualSection.X)` even when nothing below lifts it into a typed field.

## `# Tool Metadata`

`key: value` lines (first `:` splits key from value; both stripped). The `id:` line is required and
becomes `Manual.id` — the reconciliation key used to pair a hand-authored manual with an adapter-
synthesized one (ADR-0015/ADR-0018), *not* a live tool-instance id. Every other key lands in
`Manual.metadata` as a plain string.

```text
# Tool Metadata
id: thermostat-v1
category: Critical Infrastructure / Fluid Dynamics
```

## `# Functional Description`

Free prose. Becomes `Manual.description` verbatim (stripped).

## `# Observable Properties`, `# Signals`, `# Operations`

Free prose (conventionally `-` bullet lists, or the literal `(none)` when empty) — this prose is
never regex-lifted. Each section may **additionally** carry one optional fenced code block (` ``` `,
info string ignored — YAML is a JSON superset, so `yaml`/`json`/bare all parse the same way): a
*names-level* interface declaration the parser lifts into the section's typed spec list, so
`merge_manuals` can cross-validate it against an adapter's native schema. No fenced block → the
typed list stays empty; the prose remains the only content.

The block is a YAML list, one entry per named property/signal/operation:

```yaml
- name: target_temperature
  required: [value]
- name: move_to
  required: [x, y]
  completes_on: target_reached
```

| Key | Required | Applies to | Meaning |
| --- | --- | --- | --- |
| `name` | yes | all three sections | Must match an adapter-provided name exactly (`ManualMergeError` on any name-set mismatch once both sides are compared). |
| `required` | no (default `[]`) | all three sections | The field names the adapter's schema must declare as `required`; validated as a subset check, not a full schema comparison. |
| `completes_on` | no (default `None`) | `# Operations` only | The domain signal name that marks this operation's real completion (`OperationSpecification.completion_signal`) — the one field the *author* owns even against an adapter-synthesized manual, since a native description often can't express it. Lifted from the authored `completes_on:` key regardless of which side supplies the rest of the operation's schema. |

Each lifted entry becomes an `ObservablePropertySpecification` / `SignalSpecification` /
`OperationSpecification` with a minimal JSON-Schema-shaped `schema`/`parameters`:
`{"properties": {k: {} for k in required}, "required": required}` — enough to validate names and
required keys, not full field typing (the adapter side supplies real shapes).

## `# Usage Protocols & Safety`

Free prose only — operating instructions, safety constraints, suspend conditions. No interface
block; read via `manual.section(ManualSection.USAGE_AND_SAFETY)`.

---

!!! info "Hand-authored framing pending"
    This page is the exact grammar. Authoring guidance — how to phrase `Preconditions:`/`Effects:`/
    `Behavior:` sub-bullets so a reasoning strategy can use them, when to write an interface block at
    all — belongs in a manual-authoring guide (see the design's `guides/manual-authoring.md`, not yet
    written) and is not duplicated here.
