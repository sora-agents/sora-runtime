"""Per-call model records for both benchmark arms, written as one JSONL file per run.

Why this exists
---------------
The charge model that freezes ARE's clock during generation is
``charge = a0 + input_tokens / R_in + output_tokens / R_out``, computed identically for both the
S-ORA arm and the ReAct baseline. Computing it — and, before that, validating the coefficients
against calls the two arms really made — needs one row per model round-trip with the token counts,
the measured latency and enough identity to say which arm and which call site produced it.

Neither arm records that today. S-ORA emits the numbers as ``sora.llm`` log records, but they
survive only as text in whatever the operator redirected, and the *text* form drops
``cached_input_tokens`` — which the cache-aware fit needs. ARE's ReAct agent discards its token
counts entirely (see ``react_engine``). This module is the one schema both write into, so a run's
calls are comparable across arms without a per-arm parser.

The JSONL is the source of truth for the fit and for billing. It is deliberately *not* the same
channel as the charge itself: ARE's clock sees only the single ``completion_duration`` float the
engine hands back per step, and that float can undercount (see ``react_engine``), while every
round-trip lands here regardless.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

Arm = Literal["sora", "react", "grid"]


def read_usage(response: Any) -> tuple[int, int | None, int, int | None, str | None, bool]:
    """(input, cached_input, output, reasoning, finish_reason, captured) from a usage-carrying
    response.

    Duck-typed on the OpenAI response shape, which is why it lives here rather than beside its
    first caller: LiteLLM's ``ModelResponse``, the OpenAI SDK's completion object and a streamed
    usage chunk all carry the same fields, and the latency grid reads them through a client that
    is neither arm's.

    Every field is read defensively: providers differ in which detail blocks they populate, and a
    provider that omits ``cached_tokens`` must not make the run look like a measured cache miss —
    the caller writes ``usage_captured=False`` when the whole block was unreachable, which is the
    signal that separates the two."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, None, 0, None, None, False
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    finish_reason: str | None = None
    choices = getattr(response, "choices", None) or []
    if choices:
        finish_reason = getattr(choices[0], "finish_reason", None)
    # `is not None`, not truthiness: a provider reporting `cached_tokens: 0` measured a cache miss,
    # which is a different fact from a provider that shipped no `prompt_tokens_details` block at all
    # (the ordinary case for most models). Collapsing the second into 0 would put a fabricated cache
    # miss into the cache-aware fit.
    cached = getattr(prompt_details, "cached_tokens", None)
    reasoning = getattr(completion_details, "reasoning_tokens", None)
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(cached) if cached is not None else None,
        int(getattr(usage, "completion_tokens", 0) or 0),
        int(reasoning) if reasoning is not None else None,
        finish_reason,
        True,
    )


@dataclass(frozen=True)
class LLMCallRecord:
    """One logical model call. ``round_trips`` is how many times the wire was actually crossed for
    it — a parser-repair pass on the S-ORA side, a malformed-output retry on the ReAct side — with
    the token fields summed across them, because both arms are billed for every round-trip but
    only one of them is a decision.

    ``seconds`` is measured wall clock at the client, and is what the fit uses; ``charged_seconds``
    is what the run's charge model billed, and is None until a charge model is wired. Keeping both
    is the point: the second is a *pricing rule* and has to stay auditable against the first.

    A None token field means *not reported*, never a measured zero — a provider that omits
    ``cached_tokens`` is unknown coverage, not a measured cache miss.
    """

    call_id: str
    arm: Arm
    model: str | None
    semantic_label: str | None
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    seconds: float | None
    round_trips: int
    finish_reason: str | None
    charged_seconds: float | None = None
    scenario_id: str | None = None
    run_number: int | None = None
    # Groups the rows that one charged step produced. Set on the ReAct arm, where ARE calls the
    # engine again on a malformed output and each call becomes its own row — there is no
    # logical-call id there to reuse, so without this a retry is indistinguishable from two
    # separate decisions. None on the S-ORA arm, where `call_id` already does the grouping.
    bracket_id: str | None = None
    # False when the tokens are not a complete account of what the call cost: the usage block was
    # unreadable, or only some of the round-trips reported one. Either way the fit has to skip the
    # row rather than read the fields it does carry as the whole call.
    usage_captured: bool = True


class LLMCallWriter:
    """JSONL sink, safe to share across threads and across scenarios.

    Flushed per line so an aborted sweep still leaves the scenarios it already paid for on disk.
    The batch harness points every scenario in a capability at one file, so ``scenario_id`` on the
    record — not the filename — is what separates them.

    ``reset=True`` truncates the file once, on the first open of this writer. A sweep re-run into
    an existing output directory replaces ``output.jsonl`` and the traces, so the call log has to
    go with them: appending would leave two sweeps' rows carrying the same ``scenario_id`` and
    ``run_number`` with nothing to tell them apart, and the fit would silently count the stale ones
    twice. The default appends, which is what a driver aimed at an operator-chosen path wants.
    """

    def __init__(self, path: str | Path, *, reset: bool = False) -> None:
        self.path = Path(path)
        self._reset = reset
        self._truncated = False
        self._lock = threading.Lock()
        self._handle: Any = None
        self.written = 0

    def open(self) -> LLMCallWriter:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Only the first open truncates: a close/reopen partway through a sweep must not
            # discard the rows the sweep has already written.
            mode = "w" if self._reset and not self._truncated else "a"
            self._handle = self.path.open(mode, encoding="utf-8")
            self._truncated = True
        return self

    def write(self, record: LLMCallRecord) -> None:
        with self._lock:
            if self._handle is None:
                self.open()
            self._handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            self._handle.flush()
            self.written += 1

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> LLMCallWriter:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@dataclass
class _Partial:
    """Every round-trip seen so far for one logical call id, summed.

    Held until the recorder closes rather than flushed on the call's ``done``. S-ORA's parser repair
    (``_complete_and_parse``) issues its retry as a *second* ``MeteredLLMClient.complete`` under the
    same ``llm_call_scope``, so a repaired call emits ``usage, done, usage, done`` — flushing on the
    first ``done`` would split one logical call across two rows that both claim
    ``round_trips == 1`` and neither of which carries the call's real cost. Nothing in the stream
    says a logical call is *over* (``llm_call_scope`` exits silently, and adding a record there
    would put a line per call into the CLI's verbose trace), so "over" is only known at close.

    Round-trips are counted from ``done``, not ``usage``: ``MeteredLLMClient`` emits ``done`` from a
    ``finally`` on every crossing, while ``usage`` comes from the concrete client and is absent
    entirely when that client is uninstrumented.
    """

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    seconds: float | None = None
    usage_records: int = 0
    done_records: int = 0
    finish_reason: str | None = None
    model: str | None = None
    semantic_label: str | None = None

    @property
    def usage_complete(self) -> bool:
        """True only when every crossing that happened also reported its tokens.

        A repair whose retry fails before ``log_llm_usage`` runs emits ``usage, done, done``: two
        crossings, one token report. Calling that captured would hand the fit a row that looks
        measured, is short by a whole round trip, and is indistinguishable from a call that really
        did cost what its first trip cost.
        """
        return self.usage_records > 0 and self.usage_records >= self.round_trips

    @property
    def round_trips(self) -> int:
        # `done` is the unconditional one; `usage` only covers for a client whose timing decorator
        # is missing, which is also the only way a partial can exist with no `done` at all.
        return self.done_records or self.usage_records

    def add_usage(self, record: logging.LogRecord) -> None:
        self.usage_records += 1
        # Kept None-if-never-reported rather than defaulting to 0: see LLMCallRecord.
        for field_name, attr in (
            ("input_tokens", "llm_input_tokens"),
            ("output_tokens", "llm_output_tokens"),
            ("cached_input_tokens", "llm_cached_input_tokens"),
            ("reasoning_tokens", "llm_reasoning_tokens"),
        ):
            value = getattr(record, attr, None)
            if value is not None:
                current: int | None = getattr(self, field_name)
                setattr(self, field_name, (current or 0) + int(value))
        self.finish_reason = getattr(record, "llm_finish_reason", None) or self.finish_reason
        self.model = getattr(record, "llm_observed_model", None) or self.model
        self.semantic_label = getattr(record, "llm_semantic_label", None) or self.semantic_label

    def add_done(self, record: logging.LogRecord) -> None:
        self.done_records += 1
        seconds = getattr(record, "llm_seconds", None)
        if isinstance(seconds, int | float):
            self.seconds = (self.seconds or 0.0) + float(seconds)
        self.semantic_label = self.semantic_label or getattr(record, "llm_semantic_label", None)


class SoraCallRecorder(logging.Handler):
    """Turns S-ORA's ``sora.llm`` stream into ``LLMCallRecord`` rows.

    The tokens and the latency arrive on *two different records* — ``log_llm_usage`` emits the
    first from inside the concrete client, ``MeteredLLMClient`` the second from its ``finally`` —
    joined here on ``llm_call_id``, together with any further round-trips the same logical call
    made (see ``_Partial`` for why that join can only be settled at close).

    Attach to the ``sora`` logger (or ``sora.llm``) for the duration of a run, and close it when
    the run ends — ``close`` is what writes the rows. With an uninstrumented client no ``usage``
    ever arrives and the rows carry timing only, marked ``usage_captured=False`` so an absent token
    count reads as unmeasured rather than as zero; the same flag catches the partial case, where
    one round-trip of a repaired call reported its tokens and another died before it could.
    """

    def __init__(
        self,
        writer: LLMCallWriter,
        *,
        model: str | None = None,
        scenario_id: str | None = None,
        run_number: int | None = None,
    ) -> None:
        super().__init__(level=logging.INFO)
        self._writer = writer
        self._model = model
        self._scenario_id = scenario_id
        self._run_number = run_number
        self._partials: dict[str, _Partial] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            event = getattr(record, "llm_event", None)
            call_id = getattr(record, "llm_call_id", None)
            if not isinstance(call_id, str):
                return
            if event == "usage":
                self._partials.setdefault(call_id, _Partial()).add_usage(record)
            elif event == "done":
                self._partials.setdefault(call_id, _Partial()).add_done(record)
        except Exception:  # a diagnostic must never take down the run it is observing
            self.handleError(record)

    def _flush(self, call_id: str, partial: _Partial) -> None:
        self._writer.write(
            LLMCallRecord(
                call_id=call_id,
                arm="sora",
                model=partial.model or self._model,
                semantic_label=partial.semantic_label,
                input_tokens=partial.input_tokens,
                cached_input_tokens=partial.cached_input_tokens,
                output_tokens=partial.output_tokens,
                reasoning_tokens=partial.reasoning_tokens,
                seconds=partial.seconds,
                round_trips=partial.round_trips,
                finish_reason=partial.finish_reason,
                scenario_id=self._scenario_id,
                run_number=self._run_number,
                usage_captured=partial.usage_complete,
            )
        )

    def close(self) -> None:
        # Every row is written here, in the order the calls first appeared. A call that never got
        # its `done` (the process died between the two records) still lands, with seconds=None
        # rather than being dropped: the file has to stay a complete account of what was billed.
        for call_id, partial in list(self._partials.items()):
            self._flush(call_id, partial)
        self._partials.clear()
        super().close()
