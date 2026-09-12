"""``MeteredLiteLLMEngine`` — the ReAct baseline arm's instrumentation.

Why this exists
---------------
ARE's own ReAct agent is charged for its thinking through exactly one channel: ``step()`` pauses
the environment clock, calls the engine, and resumes it by ``completion_duration`` taken from the
engine's metadata dict (``base_agent.py``; ``get_offset_from_time_config_mode`` returns that float
verbatim in the default ``"measured"`` mode). The channel is ARE's own and is fully wired — but
``LiteLLMEngine.chat_completion`` ends with ``return res, None``, throwing away a ``ModelResponse``
that carries the entire ``usage`` block. Every shipped engine does the same, which is why every
published leaderboard row describes an agent that thinks instantaneously.

Feeding that dict is the whole integration: no patch to ARE is needed, a subclass suffices. The
same object later carries the frozen charge (pass ``charge=``), so instrumenting the arm and
charging it through one code path shared with S-ORA are the same class rather than two.

Four things that would otherwise bite
-------------------------------------
1. **Retries live inside the pause bracket.** On a malformed output ``step()`` re-calls the engine
   and keeps only the *last* metadata, so a retried step silently loses the earlier round-trips'
   tokens. ``begin_bracket()`` (wire it with ``wrap_pause_env``) makes the returned metadata
   cumulative over the bracket, which is exact. Without that wiring the metadata covers one
   round-trip — undercounting, never double-charging.
   The JSONL keeps its own account: one row per round-trip, carrying ``bracket_id``. The extra
   keys the engine puts in the metadata dict do *not* survive — ``LLMOutputThoughtActionLog`` has
   a fixed field set and drops everything outside it — so the row is the only place a retry can
   still be told apart from two separate decisions after the run.
2. **The fit and the bill want different latency.** Every round-trip, failures included, reaches
   the JSONL and the charge, because the agent really did emit them; a row whose ``finish_reason``
   starts with ``error:`` is what the fit excludes, since it measures the provider's bad day rather
   than the model's work. "Failed" does not imply "unbilled": ARE validates the response after
   ``litellm.completion`` returned, so a content-filtered answer raises with a full ``usage`` block
   already captured, and the row reports what that crossing really cost. Only a crossing that
   produced no response at all bills the charge model's fixed per-call term alone. Either way it
   never reaches ARE's clock, because the exception leaves ``step()`` before metadata is returned.
3. **The response is intercepted, not re-implemented.** ``chat_completion`` still runs ARE's own
   body — including its ``True``/``False`` lowercasing, which downstream checkers depend on — and
   the ``ModelResponse`` is captured from ``litellm.completion`` on the way past. Copying the body
   here to reach the response would fork it from the installed ARE and drift silently.
4. **ARE sends no settings and never streams.** Its ``chat_completion`` ignores its own ``**kwargs``
   and hands ``litellm.completion`` five fixed arguments, so an uninstrumented baseline runs at the
   provider's defaults — a different operating point from S-ORA's, on the one comparison that
   cannot be corrected after the fact. The profile's request kwargs are therefore merged in at the
   same seam the response is captured at. Streaming is *not* part of that operating point: it
   changes how tokens arrive, not what is sampled. The profile field selecting it is read by the
   latency grid too, which is what keeps the fitted coefficients and the arms they are charged to
   on one transport. The shipped profiles are non-streamed — also ARE's native behaviour — and the
   streamed branch stays for a profile that asks for it, adding ``stream=True`` with
   ``stream_options={"include_usage": True}``, the chunks reassembled by LiteLLM's own
   ``stream_chunk_builder`` into the ``ModelResponse`` ARE's assertions expect. That branch carries
   a trap worth keeping on record: ``stream_chunk_builder`` fills a *missing* usage block by
   re-tokenizing prompt and completion locally, so a provider that ignores ``include_usage`` yields
   plausible, wrong counts no downstream reader can tell from reported ones. Taking the row's
   tokens from the raw usage chunk rather than the rebuilt response (see ``_usage_source``) is
   necessary but not sufficient — on OpenRouter the usage *chunk itself* is built that way, and
   measured as content-only completion tokens with no cache detail at all.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from are.simulation.agents.llm.litellm.litellm_engine import (
    LiteLLMEngine,
    LiteLLMModelConfig,
)
from litellm import stream_chunk_builder

from examples.gaia2.evaluation.core import ModelProfile
from examples.gaia2.llm_calls import (
    LLMCallRecord,
    LLMCallWriter,
    read_finish_reason,
    read_usage,
)

# Seconds to charge for one round-trip, given (input_tokens, cached_input_tokens, output_tokens).
# The frozen charge model plugs in here; None bills measured wall clock instead.
ChargeModel = Callable[[int, int, int], float]

_capture = threading.local()

# What LiteLLM wants for a streamed call to report usage at all: without it the final chunk carries
# no usage block and the rebuilt response falls back to a local estimate.
_STREAM_OPTIONS = {"include_usage": True}


@dataclass(frozen=True)
class _CallSettings:
    """What the wrapper applies to one crossing — set by the engine immediately before ARE's
    ``chat_completion`` runs, and cleared immediately after, so an unmetered engine sharing the
    thread is never touched."""

    request_kwargs: Mapping[str, Any] = field(default_factory=dict)
    stream: bool = False


def _install_response_capture() -> None:
    """Wrap the ``completion`` name ARE's engine module calls: apply the caller's request settings,
    stream when asked, and stash the resulting ``ModelResponse`` on a thread-local.

    This is the only seam where both problems can be fixed at once. ARE's ``chat_completion``
    neither forwards settings nor streams, and re-implementing its body here to change that would
    fork it from the installed ARE — including the ``True``/``False`` lowercasing downstream
    checkers depend on — and drift silently.

    Thread-local rather than an instance attribute because a batch may run scenarios in threads
    while sharing one patched module global, and the wrapper always runs on its caller's thread.
    Idempotence is decided by a marker on the *currently installed* function rather than by a
    module flag, and the check is re-run per call: anything that replaces ``module.completion``
    after us — a test double, another harness — would otherwise silently unhook the capture and
    leave every row reading ``usage_captured=False``."""
    from are.simulation.agents.llm.litellm import litellm_engine as module

    inner = module.completion
    if getattr(inner, "_sora_capture", False):
        return

    def capturing(*args: Any, **kwargs: Any) -> Any:
        settings: _CallSettings = getattr(_capture, "settings", None) or _CallSettings()
        kwargs.update(settings.request_kwargs)
        if not settings.stream:
            _capture.response = inner(*args, **kwargs)
            return _capture.response
        chunks = []
        usage_chunk: Any = None
        for chunk in inner(*args, **kwargs, stream=True, stream_options=dict(_STREAM_OPTIONS)):
            chunks.append(chunk)
            # Usage rides a trailing chunk of its own, carrying no choices — so it is collected
            # separately from the content and kept as the row's only token authority.
            if getattr(chunk, "usage", None) is not None:
                usage_chunk = chunk
        _capture.usage_chunk = usage_chunk
        # `messages=` is what the builder needs to *estimate* a missing prompt count. It is passed
        # because the rebuilt response goes to ARE, which asserts on its shape; the estimate it may
        # contain never reaches a row.
        _capture.response = stream_chunk_builder(chunks, messages=kwargs.get("messages"))
        return _capture.response

    capturing._sora_capture = True  # type: ignore[attr-defined]
    module.completion = capturing


@dataclass(frozen=True)
class _RoundTrip:
    input_tokens: int
    # None where the provider reported no such field — never a measured zero (see LLMCallRecord).
    cached_input_tokens: int | None
    output_tokens: int
    reasoning_tokens: int | None
    seconds: float
    charged: float


def _sum_known(values: Iterable[int | None]) -> int | None:
    """Sum of the reported values, or None when nothing in the bracket reported one."""
    known = [value for value in values if value is not None]
    return sum(known) if known else None


class MeteredLiteLLMEngine(LiteLLMEngine):  # type: ignore[misc]  # ARE is untyped
    """``LiteLLMEngine`` that returns a populated metadata dict instead of ``None``.

    ``writer`` receives one ``LLMCallRecord`` per round-trip (``arm="react"``), tagged with the
    ``bracket_id`` of the step it belongs to; ``charge`` decides what ``completion_duration``
    bills. ``scenario_id``/``run_number`` are set per scenario by the driver so one file can hold
    a whole capability's sweep. ``request_kwargs``/``stream`` carry the model profile's operating
    point onto a call ARE would otherwise leave at the provider's defaults — prefer
    :meth:`from_profile`, which derives both from the profile the other arm runs at."""

    def __init__(
        self,
        model_config: LiteLLMModelConfig,
        *,
        writer: LLMCallWriter | None = None,
        charge: ChargeModel | None = None,
        scenario_id: str | None = None,
        run_number: int | None = None,
        request_kwargs: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> None:
        super().__init__(model_config)
        _install_response_capture()
        self.writer = writer
        self.charge = charge
        self.scenario_id = scenario_id
        self.run_number = run_number
        self.stream = stream
        self.settings = _CallSettings(dict(request_kwargs or {}), stream)
        self._bracket: list[_RoundTrip] = []
        self._bracket_wired = False
        self.round_trips = 0
        self.brackets = 0

    @classmethod
    def from_profile(cls, profile: ModelProfile, **kwargs: Any) -> MeteredLiteLLMEngine:
        """Build the engine from the same profile the S-ORA arm runs at.

        The operating point comes from ``profile.request_kwargs()`` unchanged; ``stall_timeout``
        and ``sdk_max_retries`` are added here rather than there because they are transport, and
        LiteLLM takes transport per call (``timeout``/``num_retries``) where a client library takes
        it at construction. Retries stay at the profile's number for the same reason the grid pins
        them: a silently retried call is billed once and measured as the sum of both attempts.

        Which routing string LiteLLM needs for a given provider is LiteLLM's own business and is
        not second-guessed here — the profile's ``provider``/``model``/``endpoint`` are passed
        through as they stand, and a live pilot is what confirms them.

        The one thing that *is* second-guessed is LiteLLM's table of which parameters a model
        accepts. It validates the request against a per-model list it ships, so a model newer than
        the installed version is rejected before the wire — ``reasoning_effort`` on a reasoning
        model it has never heard of — and the run dies at every step without a request ever being
        sent. ``allowed_openai_params`` names the profile's own settings as forwardable, which is
        the narrow fix: the alternative LiteLLM offers, ``drop_params``, would *silently* delete
        them and run the baseline at the provider's defaults while S-ORA runs at the profile's,
        which is exactly the operating-point drift the sweep refuses to start with. A provider that
        genuinely rejects a setting still says so, on the wire, where the row records it."""
        transport: dict[str, Any] = {"num_retries": profile.sdk_max_retries}
        if profile.stall_timeout is not None:
            transport["timeout"] = profile.stall_timeout
        request = profile.request_kwargs()
        # `extra_body`/`extra_headers` are passthrough envelopes LiteLLM never validates; the rest
        # are the request parameters it does.
        allowed = sorted(set(request) - {"extra_body", "extra_headers"})
        return cls(
            LiteLLMModelConfig(
                model_name=profile.model,
                provider=profile.provider,
                endpoint=profile.endpoint,
                # Absent, LiteLLM falls back to its own environment lookup for the provider.
                api_key=os.environ.get(profile.credential_env),
            ),
            request_kwargs={
                **request,
                **({"allowed_openai_params": allowed} if allowed else {}),
                **transport,
            },
            stream=profile.stream,
            **kwargs,
        )

    # -- bracket -----------------------------------------------------------------------------

    def begin_bracket(self) -> None:
        """Start a new generation bracket, discarding the previous one's tally.

        Called from the wrapped ``pause_env``, which ARE invokes once at the top of every
        ``step()`` — the only signal available to an engine for "the metadata you returned last
        time has been consumed"."""
        self._bracket_wired = True
        self.brackets += 1
        self._bracket.clear()

    @property
    def bracketed(self) -> bool:
        """Whether ARE's pause signal ever reached this engine. False means every round-trip was
        counted as its own step, so a retried step's earlier tokens are missing from the metadata
        ARE kept — an undercount, never a double charge, and worth reporting rather than hiding."""
        return self._bracket_wired

    def wrap_pause_env(self, pause_env: Callable[[], None]) -> Callable[[], None]:
        """Wrap ARE's ``env.pause`` so brackets open in step with it. Pass the result as the agent's
        ``pause_env=`` at construction — ``ARESimulationAgent.initialize`` forwards it to the ReAct
        agent, so nothing is monkey-patched."""

        def paused() -> None:
            self.begin_bracket()
            pause_env()

        return paused

    # -- the call ----------------------------------------------------------------------------

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stop_sequences: Any = [],  # noqa: B006 — ARE's signature; overriding it would break callers
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any] | None]:
        _install_response_capture()  # cheap; re-arms if anything replaced the module global
        _capture.response = None
        _capture.usage_chunk = None
        _capture.settings = self.settings
        if not self._bracket_wired:
            # Unwired, a bracket is one round-trip — all the engine can know without ARE's pause.
            self.brackets += 1
        bracket_id = f"react-{self.scenario_id or 'run'}-bracket-{self.brackets:05d}"
        started = time.perf_counter()
        try:
            text, _ = super().chat_completion(messages, stop_sequences, **kwargs)
        except Exception as exc:
            # A failure here does not mean the provider never answered. ARE validates the response
            # *after* `completion()` returned — four asserts, and `assert res is not None` fires on
            # a content-filtered `message.content=None`, a crossing that was billed in full. The
            # captured response is read on this path for exactly that reason; only a failure that
            # produced no response at all falls through to a fixed-term row.
            trip, _reason, captured = self._trip(
                *self._take_capture(), elapsed=time.perf_counter() - started
            )
            self._record(
                trip,
                finish_reason=f"error:{type(exc).__name__}",
                captured=captured,
                bracket_id=bracket_id,
            )
            raise
        finally:
            # Runs on the raising path too. Left set, it would apply this engine's operating point
            # to whatever else calls `completion` on this thread next.
            _capture.settings = None
        trip, finish_reason, captured = self._trip(
            *self._take_capture(), elapsed=time.perf_counter() - started
        )
        self._record(trip, finish_reason=finish_reason, captured=captured, bracket_id=bracket_id)
        self._bracket.append(trip)
        return text, self._metadata(trip)

    def _take_capture(self) -> tuple[Any, Any]:
        """The captured response and, on a streamed call, the provider's raw usage chunk — both
        cleared as they are read so no later call can see them."""
        captured = getattr(_capture, "response", None), getattr(_capture, "usage_chunk", None)
        _capture.response = None
        _capture.usage_chunk = None
        return captured

    def _usage_source(self, response: Any, usage_chunk: Any) -> Any:
        """What the row's token counts are read from.

        On a streamed call that is the provider's own usage chunk and *never* the rebuilt response:
        ``stream_chunk_builder`` fills a missing usage block by re-tokenizing the prompt and the
        completion locally, so a provider that ignores ``include_usage`` produces counts that are
        plausible, wrong, and indistinguishable from reported ones once they are on a row. Falling
        through to None instead is what writes ``usage_captured=False`` — a call charged at the
        fixed per-call term alone, which undercounts visibly rather than mis-fitting silently.

        Necessary but not sufficient, and the reason the shipped profiles are non-streamed: on
        OpenRouter the usage chunk is *itself* assembled by ``stream_chunk_builder``, so both sides
        of this choice are the same locally counted object — measured there as content-only
        completion tokens and a permanently absent cache count. Verify a provider's streamed block
        against the SDK's before trusting an arm's rows on it."""
        return usage_chunk if self.stream else response

    def _trip(
        self, response: Any, usage_chunk: Any, *, elapsed: float
    ) -> tuple[_RoundTrip, str | None, bool]:
        """One round-trip's record, priced. ``response`` of None is the crossing that produced no
        answer at all: no tokens, and a charge model prices it at its fixed per-call term — zero
        would be the one reading the design rules out, since the agent did emit the call and did
        wait for it to fail."""
        input_tokens, cached, output_tokens, reasoning, _, captured = read_usage(
            self._usage_source(response, usage_chunk)
        )
        # From the response either way: a streamed usage chunk carries no choices to read it off.
        finish_reason = read_finish_reason(response)
        charged = self._charge_for(input_tokens, cached, output_tokens, elapsed)
        trip = _RoundTrip(input_tokens, cached, output_tokens, reasoning, elapsed, charged)
        return trip, finish_reason, captured

    def _charge_for(
        self, input_tokens: int, cached: int | None, output_tokens: int, elapsed: float
    ) -> float:
        # An unreported cache figure bills as uncached: the charge model has to return a number,
        # and pricing the unknown at the cheaper rate would understate the arm.
        if self.charge is None:
            return elapsed
        return self.charge(input_tokens, cached or 0, output_tokens)

    def _metadata(self, trip: _RoundTrip) -> dict[str, Any]:
        """Cumulative over the bracket when brackets are wired, this round-trip alone otherwise —
        because ARE keeps only the last metadata a bracket produced."""
        trips = self._bracket if self._bracket_wired else [trip]
        prompt_tokens = sum(t.input_tokens for t in trips)
        completion_tokens = sum(t.output_tokens for t in trips)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # ARE stores this straight into `LLMOutputThoughtActionLog.reasoning_tokens`, an int
            # its exporter collects into a list, so the unknown flattens to 0 on the way into
            # ARE's own artifact. The honest null lives in the JSONL row, which is the artifact
            # whose schema documents the distinction.
            "reasoning_tokens": _sum_known(t.reasoning_tokens for t in trips) or 0,
            "completion_duration": sum(t.charged for t in trips),
            # None of the keys below are read by ARE, and `LLMOutputThoughtActionLog` drops every
            # field outside its own fixed set — so they reach a live trace and nothing else. What
            # has to survive the run is on the JSONL row.
            "round_trips": len(trips),
            "cached_prompt_tokens": _sum_known(t.cached_input_tokens for t in trips),
            "measured_seconds": sum(t.seconds for t in trips),
        }

    def _record(
        self, trip: _RoundTrip, *, finish_reason: str | None, captured: bool, bracket_id: str
    ) -> None:
        self.round_trips += 1
        if self.writer is None:
            return
        self.writer.write(
            LLMCallRecord(
                # ARE's ReAct loop has no logical-call id of its own: one round-trip is one call,
                # and a retry is a distinct row rather than a second trip on the same id — the
                # opposite of S-ORA's repair pass, which reuses its id. `bracket_id` is what puts
                # a step's rows back together, and it has to be *on the row*: the `round_trips`
                # the engine returns is dropped with the rest of the metadata ARE does not read.
                call_id=f"react-{self.scenario_id or 'run'}-{self.round_trips:05d}",
                bracket_id=bracket_id,
                arm="react",
                model=self.model_config.model_name,
                semantic_label="react_step",
                input_tokens=trip.input_tokens if captured else None,
                cached_input_tokens=trip.cached_input_tokens if captured else None,
                output_tokens=trip.output_tokens if captured else None,
                reasoning_tokens=trip.reasoning_tokens if captured else None,
                seconds=trip.seconds,
                round_trips=1,
                finish_reason=finish_reason,
                charged_seconds=trip.charged,
                scenario_id=self.scenario_id,
                run_number=self.run_number,
                usage_captured=captured,
            )
        )
