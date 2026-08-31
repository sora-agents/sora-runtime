"""Default Act strategy."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from sora._strategies.contracts import (
    TickResult,
)
from sora._strategies.parameters import (
    _null_required_params,
)
from sora.references import (
    _reference_paths,
)
from sora.types import (
    OPERATION_NAME,
    TOOL_ID,
    OperationInvocation,
    Step,
)

if TYPE_CHECKING:
    from sora.cycle import DecisionCycle
    from sora.manual import Manual

log = logging.getLogger("sora.strategies")


class DefaultActStrategy:
    """The mechanical, no-LLM default for *parameter binding* (not protocol binding — see
    ActStrategy): bind an ``invoke`` Step straight to an OperationInvocation, splitting the
    tool_id/operation_name routing keys out of the operation's own params. A model-backed
    ActStrategy would instead ground under-specified params against the manual's schema here; the
    default assumes the Step already carries concrete params, so binding is just the key-split.

    Two mechanical guards sit here (both structural checks, no judgment, so Act stays mechanistic
    per ADR-0017). The first is a **leak guard**: grounding has already run by the time a step
    reaches binding, so a ``$from``/``$decide``/``$bind`` dict still present in the params is not a
    reference waiting to be filled — it is one the resolver failed to *see*, and binding it would
    serialize the reference itself to the wire as a literal object. The tool then rejects it with a
    message that names the wrong culprit (a type error on the enclosing list), so the guard skips
    the invoke and logs the offending paths instead. It is a backstop for a resolver bug, not part
    of normal flow; a healthy run never trips it.

    The second: a **required** param that resolves to null is a schema violation, so the invoke is
    *skipped* — no
    invocation is emitted and the cycle dispatches nothing this step (`_act`). Grounding (Reason)
    has already run by now, so a null at bind time is a value the model declined or could not fill,
    not an un-grounded reference; dispatching the operation anyway is the historic blind-`delete`
    mis-action, degraded to a probabilistic one and previously held off only by a prompt fragment.
    The guard needs the operation's schema (`OperationSpecification.parameters`, adapter-
    synthesized) to know which params are required; with no manual/spec/declared ``required`` it
    cannot tell, so it does not fire and binds as before — the same structured-spec dependency the
    thread-reading Manual relocation has. It narrows, not eliminates, a null-invoke: an *optional*
    null still passes through by design (many operations take legitimately-optional params)."""

    async def bind(
        self, step: Step, manual: Manual | None, cycle: DecisionCycle, result: TickResult
    ) -> TickResult:
        params = {k: v for k, v in step.params.items() if k not in (TOOL_ID, OPERATION_NAME)}
        operation_name = step.params[OPERATION_NAME]
        leaked = _reference_paths(params)
        if leaked:
            log.error(
                "act: skipping invoke %s.%s — unresolved reference(s) at %s reached parameter "
                "binding; grounding never saw them (resolver bug, not a plan bug)",
                step.params[TOOL_ID],
                operation_name,
                leaked,
            )
            return result  # no invocation -> _act dispatches nothing this step (skip-and-continue)
        null_required = _null_required_params(manual, operation_name, params)
        if null_required:
            log.warning(
                "act: skipping invoke %s.%s — required param(s) %s resolved to null",
                step.params[TOOL_ID],
                operation_name,
                null_required,
            )
            return result  # no invocation -> _act dispatches nothing this step (skip-and-continue)
        invocation = OperationInvocation(
            tool_id=step.params[TOOL_ID],
            operation_name=operation_name,
            params=params,
        )
        return replace(result, invocation=invocation)
