# Execution and recovery contracts

The default Reason strategy distinguishes an answer from an unreadable plan before dispatch.
These checks establish structural correctness; they do not prove that a model interpreted a
natural-language task correctly.

## Collection and parameter handling

- A mechanical fan-out expands one template step per element. Every referenced loop path must
  resolve before the expansion is installed. A missing path rejects the expansion unless the
  manual declares its containing operation parameter optional; that entire parameter is omitted
  for the affected element. A missing required or unknown parameter path still rejects all steps,
  so a valid first element cannot commit work before a malformed later element is discovered.
  Diagnostics include the element index and path with a bounded record preview.
- Present null values survive substitution. Reason checks required parameters against the manual:
  missing required arguments replan, explicitly nullable required arguments accept null, and
  omitted optional arguments stay omitted. Without a structured schema, required-ness remains
  unknown. The guard recognizes explicit null types and nullable union branches; it is not a
  complete JSON Schema validator.
- An empty collection is a valid zero-element expansion. It is distinct from an unreadable source.
  An unsupported multi-step template is a recoverable defect, not an execution form.
- A paginated pipeline reads pages, collects their results, flattens their payloads, then operates
  on records. Explicit null and empty-list payloads contribute no records. An omitted payload is
  accepted only on an empty object or a metadata-only page with an explicit zero count/total.
  A missing path on an ambiguous or nonempty page rejects the transformation; `has_more: false`
  alone does not prove the current page empty. Generic no-path flatten preserves scalar nulls.
- Dependent calls use lookup fan-out, collect, then follow-up fan-out. Collected entries include
  the invoking arguments, with returned fields taking precedence on name collisions. This
  supports disjoint input/output keys and scalar results without inventing a multi-step template.
  It does not retain both values under a colliding field name; use distinct keys or an adapter's
  explicit result shape for that case.

Reason preflights ordered filter comparisons. A present string versus number, or another
demonstrably incomparable pair, produces a defect when it leaves the predicate undecidable,
before the output binding is published. A valid OR branch may still prove inclusion and a valid
AND branch may prove exclusion, regardless of clause order. A later complement therefore cannot turn that failed
comparison into a confident absence. Comparisons of compatible numeric or string values remain
mechanical. No date parsing, unit inference or conversion is implicit. Equality and membership
still support scalar, null, list and record values. Missing element fields retain their existing
nonmatch semantics; a nonmatch is not a certificate that source data is complete.

## Recovery and originating instructions

A child inference receives its own goal and the full originating request separately as constraint
context. The prompt directs the child to plan only its active goal, leave other work to the parent,
and reuse completed results. The live activity's goal and execution history remain intact. A replacement plan receives the original
qualifiers, recent user instructions, executed results and the reason the previous plan failed.
Recent messages from the user retain their full text so a final restriction cannot be lost to
character truncation. Other senders' text and structured messages retain bounded previews. The recent-message window is still limited to ten messages; this is not a general
solution to deciding which older conversation instructions remain applicable.

A condition branch is not a fresh user request. Its final top-level follow-up sends one closing
reply by default, unless the originating request forbids it. Explicit communication restrictions
govern every branch.
An explicitly requested child question remains work the child owes. This makes the authority
visible to the planner, without attempting to infer universal permissions from arbitrary prose.

The recursion guard remains active. Repeated decomposition pauses on a truthful question rather
than recursing indefinitely, crashing, or asserting completion. Concrete actions can still run
under a condition branch. No exception disables the guard for a child that repeats its parent.

## Reporting and progress accounting

Both user-message routes, operation invocation and `send`, use grounding when a known empty
fan-out or unresolved failed operation in the current frame requires reviewing the report against
execution evidence. Earlier frames' failures and failures followed by a successful invocation with
the same tool, operation and arguments do not force another call; an unrelated success does not
resolve the failure. This applies
to literal text as well as reference-bearing text. The grounding prompt exposes operation results,
failures, zero-operation fan-outs and outstanding conditions. A corrected report is still a model
output, not a mechanically verified proposition; arbitrary literal claims without a known gap are
not independently checked. These extra grounding calls must be included in cost and latency
measurements.

The retry trail resets after a novel successful invocation by the same activity, measured by tool,
operation and arguments. Failed calls and repeated successful calls do not reset it. An unrelated
but novel success does reset it under the existing contract: novelty is a mechanical proxy for
progress, not proof of relevance to the defect. Tests preserve this distinction rather than adding
an unreviewed semantic progress judge.

## Limits requiring evaluation or a separate contract

Task interpretation, substitute selection, outward-write authorization and prose truthfulness
remain semantic responsibilities. Supplying complete constraints does not prove the model follows
them. Arbitrary natural-language prohibitions are not a deterministic permission policy, and
runtime escalation questions still use the existing await-input mechanism. A stricter policy
would require an explicit, separately designed authorization contract. No hidden model judge or
new permission language is introduced here.

Outstanding monitoring conditions are evidence against an unqualified completion claim, but
correct window ownership, catch-up and termination require their own lifecycle checks. Structured
prompt consolidation and future fixed-candidate evaluation must assess the remaining semantic
failures. Passing deterministic tests alone establishes neither scenario success nor campaign
readiness.
