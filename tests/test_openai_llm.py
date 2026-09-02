"""OpenAICompatLLMClient — the OpenAI SDK pointed at any OpenAI-compatible endpoint (OpenAI itself,
Gemini's OpenAI-compat surface, or a local runtime via ``base_url``). Gated on the optional
``[openai]`` extra; the SDK import lives under adapters. Mirrors the AnthropicLLMClient tests: pure
extraction helpers over ``SimpleNamespace`` fakes, plus ``complete`` over an injected fake SDK."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("openai")

import httpx  # noqa: E402
from openai import APITimeoutError, Timeout  # noqa: E402

from sora.adapters.openai_llm import (  # noqa: E402
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_STREAM_STALL_TIMEOUT,
    OpenAICompatLLMClient,
    _provider_observation_of,
    _text_of,
    _usage_of,
)
from sora.llm import CompletionProfile, CompletionRequest, LLMUsage  # noqa: E402


def _response(
    content: str | None = "ok",
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    cached_tokens: int | None = None,
    finish_reason: str | None = "stop",
    with_usage: bool = True,
) -> SimpleNamespace:
    """A minimal stand-in for an OpenAI ``ChatCompletion`` response — only the fields the client
    reads (``choices[0].message.content`` and the ``usage`` token block)."""
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
    usage = None
    if with_usage:
        details = (
            SimpleNamespace(reasoning_tokens=reasoning_tokens)
            if reasoning_tokens is not None
            else None
        )
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            prompt_tokens_details=(
                SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else None
            ),
            completion_tokens_details=details,
        )
    return SimpleNamespace(choices=[choice], usage=usage)


def _request() -> CompletionRequest:
    return CompletionRequest("s", "p", "test", "1")


# ── pure extraction helpers ───────────────────────────────────────────────────────────────────


def test_text_of_returns_the_first_choice_message_content() -> None:
    assert _text_of(_response("hello there")) == "hello there"


def test_text_of_tolerates_empty_choices_or_null_content() -> None:
    # A refusal / tool-only turn can carry no text; degrade to "" rather than raise.
    assert _text_of(SimpleNamespace(choices=[])) == ""
    assert _text_of(_response(None)) == ""


def test_text_of_tolerates_a_choice_with_no_message() -> None:
    # Some local OpenAI-compatible servers omit `message` on certain finish reasons; degrade to ""
    # rather than raising AttributeError on `choices[0].message`.
    assert _text_of(SimpleNamespace(choices=[SimpleNamespace()])) == ""


def test_usage_of_reads_cached_input_and_the_exact_reasoning_count() -> None:
    # OpenAI (o-series) and Gemini's compat surface report reasoning tokens directly, so thinking is
    # the provider's exact number — NOT the answer-subtraction estimate the Anthropic path falls to.
    usage = _usage_of(
        _response(
            prompt_tokens=1200,
            completion_tokens=800,
            reasoning_tokens=600,
            cached_tokens=900,
        ),
        answer_chars=40,
    )
    assert usage == LLMUsage(
        1200,
        800,
        answer_chars=40,
        reasoning_tokens=600,
        cached_input_tokens=900,
    )
    assert usage.thinking_tokens == 600  # exact, not the 790 the char-estimate would give
    assert usage.thinking_share == 600 / 800


def test_usage_of_falls_back_to_the_estimate_when_no_reasoning_is_reported() -> None:
    # A non-reasoning model (or a server that omits the details block) leaves reasoning_tokens None,
    # so thinking degrades to the same output-minus-answer estimate the Anthropic client uses.
    usage = _usage_of(_response(prompt_tokens=100, completion_tokens=800), answer_chars=40)
    assert usage is not None
    assert usage.reasoning_tokens is None
    assert usage.thinking_tokens == 790  # 800 out - ~10 answer tok


def test_usage_of_tolerates_a_missing_usage_block() -> None:
    # Missing accounting is unavailable, not an exact zero-token provider round trip.
    assert _usage_of(_response(with_usage=False), answer_chars=7) is None


def test_usage_of_distinguishes_missing_cached_input_from_an_explicit_zero() -> None:
    missing = _usage_of(_response(prompt_tokens=42), answer_chars=0)
    explicit_zero = _usage_of(_response(prompt_tokens=42, cached_tokens=0), answer_chars=0)

    assert missing is not None
    assert explicit_zero is not None
    assert missing.cached_input_tokens is None
    assert explicit_zero.cached_input_tokens == 0


def test_provider_observation_reads_openrouter_metadata_and_sdk_model_extra() -> None:
    direct = {"requested": "moonshotai/kimi-k2.5", "summary": "selected=Moonshot AI"}
    assert _provider_observation_of(SimpleNamespace(openrouter_metadata=direct)) == direct
    assert (
        _provider_observation_of(SimpleNamespace(model_extra={"openrouter_metadata": direct}))
        == direct
    )


# ── complete() over an injected fake SDK ──────────────────────────────────────────────────────


def _chunks(response: SimpleNamespace) -> list[SimpleNamespace]:
    """The same content re-shaped as a chat-completions *stream*: one delta chunk per character, so
    a test that asserts on the joined text also proves the client accumulates rather than reading
    one blob, plus the trailing usage-only chunk (no choices) that ``stream_options`` asks for."""
    text = _text_of(response)
    out = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))], usage=None)
        for ch in text
    ]
    finish_reason = response.choices[0].finish_reason if response.choices else None
    out.append(
        SimpleNamespace(
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason=finish_reason)
            ],
            usage=None,
        )
    )
    out.append(SimpleNamespace(choices=[], usage=response.usage))
    return out


class _FakeStream:
    """Models the SDK's ``AsyncStream`` contract, not just its iterability: it is an async context
    manager that closes the underlying response. The difference is the whole point of the
    ``stall_timeout`` path — that timeout raises *mid-iteration*, so a bare ``async for`` would
    leave the httpx response open and never return the connection to the pool.

    ``fail_after`` makes the stream raise where a stalled read does.
    """

    def __init__(self, chunks: list[SimpleNamespace], *, fail_after: int | None = None) -> None:
        self._chunks = chunks
        self._fail_after = fail_after
        self.closed = False

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        self.closed = True

    async def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        for index, chunk in enumerate(self._chunks):
            if self._fail_after is not None and index >= self._fail_after:
                raise APITimeoutError(request=httpx.Request("POST", "http://test"))
            yield chunk


class _FakeCompletions:
    def __init__(self, response: SimpleNamespace, *, fail_after: int | None = None) -> None:
        self.response = response
        self.kwargs: dict[str, Any] | None = None
        self.stream: _FakeStream | None = None
        self._fail_after = fail_after

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if not kwargs.get("stream"):
            return self.response
        self.stream = _FakeStream(_chunks(self.response), fail_after=self._fail_after)
        return self.stream


class _FakeAsyncOpenAI:
    def __init__(self, response: SimpleNamespace, *, fail_after: int | None = None) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(response, fail_after=fail_after))
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def _llm_logging_enabled() -> Iterator[None]:
    # The usage cue is INFO; without an enabled level the record is dropped before any handler sees
    # it (root defaults to WARNING). Real run surfaces enable it — mirror that here.
    logger = logging.getLogger("sora.llm")
    previous = logger.level
    logger.setLevel(logging.INFO)
    yield
    logger.setLevel(previous)


@pytest.mark.asyncio
async def test_complete_maps_system_and_prompt_to_chat_messages_and_returns_the_text() -> None:
    client = OpenAICompatLLMClient(model="some-model", api_key="test")
    fake = _FakeAsyncOpenAI(_response("the answer"))
    client._client = fake  # type: ignore[assignment]

    out = await client.complete(CompletionRequest("be terse", "hi", "plan", "1"))

    assert out == "the answer"
    kwargs = fake.chat.completions.kwargs
    assert kwargs is not None
    assert kwargs["model"] == "some-model"
    assert kwargs["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.asyncio
async def test_request_profile_overrides_output_cap_without_changing_other_defaults() -> None:
    client = OpenAICompatLLMClient(model="m", api_key="test", max_tokens=8192)
    fake = _FakeAsyncOpenAI(_response("the answer"))
    client._client = fake  # type: ignore[assignment]

    request = CompletionRequest(
        "system",
        "user",
        "condition",
        "1",
        profile=CompletionProfile(max_output_tokens=64, reasoning="low"),
    )
    await client.complete(request)

    kwargs = fake.chat.completions.kwargs
    assert kwargs is not None
    assert kwargs["max_completion_tokens"] == 64
    # Provider mappings for qualitative reasoning classes are a later, measured change.
    assert "reasoning_effort" not in kwargs


@pytest.mark.asyncio
async def test_complete_forwards_only_non_null_evaluation_settings() -> None:
    client = OpenAICompatLLMClient(
        model="m",
        api_key="test",
        reasoning_effort="xhigh",
        temperature=0.5,
        top_p=None,
        seed=None,
        verbosity="low",
        service_tier="default",
        max_retries=0,
        stream=False,
    )
    client._client = _FakeAsyncOpenAI(_response("hi"))  # type: ignore[assignment]

    await client.complete(_request())

    kwargs = client._client.chat.completions.kwargs  # type: ignore[attr-defined]
    assert kwargs["reasoning_effort"] == "xhigh"
    assert kwargs["temperature"] == 0.5
    assert kwargs["verbosity"] == "low"
    assert kwargs["service_tier"] == "default"
    assert "top_p" not in kwargs
    assert "seed" not in kwargs


@pytest.mark.asyncio
async def test_complete_forwards_openrouter_reasoning_routing_and_metadata_opt_in() -> None:
    routing = {
        "only": ["moonshotai"],
        "order": ["moonshotai"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    client = OpenAICompatLLMClient(
        model="moonshotai/kimi-k2.5",
        api_key="test",
        reasoning={"enabled": True},
        provider_routing=routing,
        router_metadata=True,
        stream=False,
    )
    client._client = _FakeAsyncOpenAI(_response("hi"))  # type: ignore[assignment]

    await client.complete(_request())

    kwargs = client._client.chat.completions.kwargs  # type: ignore[attr-defined]
    assert kwargs["extra_body"] == {
        "reasoning": {"enabled": True},
        "provider": routing,
    }
    assert kwargs["extra_headers"] == {"X-OpenRouter-Metadata": "enabled"}


@pytest.mark.asyncio
async def test_the_call_streams_by_default_and_joins_the_deltas() -> None:
    # Streaming is what turns the configured read timeout into a *stall* detector: chunks keep
    # resetting it, so a slow-but-healthy thinking model is no longer indistinguishable from a
    # socket that died. Assert the flag actually goes out, not just that the text comes back.
    client = OpenAICompatLLMClient(model="m", api_key="test")
    fake = _FakeAsyncOpenAI(_response("the answer"))
    client._client = fake  # type: ignore[assignment]

    assert await client.complete(_request()) == "the answer"
    kwargs = fake.chat.completions.kwargs
    assert kwargs is not None
    assert kwargs["stream"] is True


@pytest.mark.asyncio
async def test_streaming_can_be_turned_off_for_an_endpoint_that_cannot_do_it() -> None:
    # The escape hatch for an OpenAI-compatible server whose streaming is broken or absent; the
    # single-shot path must keep working unchanged.
    client = OpenAICompatLLMClient(model="m", api_key="test", stream=False)
    fake = _FakeAsyncOpenAI(_response("the answer"))
    client._client = fake  # type: ignore[assignment]

    assert await client.complete(_request()) == "the answer"
    kwargs = fake.chat.completions.kwargs
    assert kwargs is not None
    assert "stream" not in kwargs


@pytest.mark.asyncio
async def test_usage_is_only_requested_when_it_will_be_used() -> None:
    # A streamed response carries no usage block unless `stream_options` asks for it — but a
    # compat server that does not know the field rejects the whole request, so an uninstrumented
    # run must not pay that compatibility cost for a number it would discard.
    plain = OpenAICompatLLMClient(model="m", api_key="test")
    plain._client = _FakeAsyncOpenAI(_response("hi"))  # type: ignore[assignment]
    await plain.complete(_request())
    assert "stream_options" not in (plain._client.chat.completions.kwargs or {})  # type: ignore[attr-defined]

    metered = OpenAICompatLLMClient(model="m", api_key="test", instrument=True)
    metered._client = _FakeAsyncOpenAI(_response("hi"))  # type: ignore[assignment]
    await metered.complete(_request())
    kwargs = metered._client.chat.completions.kwargs  # type: ignore[attr-defined]
    assert kwargs["stream_options"] == {"include_usage": True}


def test_a_stall_timeout_is_configured_rather_than_left_to_the_sdk_default() -> None:
    # The SDK's own default is a 600s read timeout with two retries — up to ~30 minutes of silence,
    # which once cost a whole benchmark scenario. Connect keeps its own, shorter budget: a slow TLS
    # handshake is a different failure from a request that is never answered.
    client = OpenAICompatLLMClient(model="m", api_key="test")
    timeout = client._client.timeout
    assert isinstance(timeout, Timeout)
    assert timeout.read == DEFAULT_STREAM_STALL_TIMEOUT
    assert timeout.connect == DEFAULT_CONNECT_TIMEOUT


def test_the_stall_timeout_can_be_disabled() -> None:
    # An endpoint that legitimately buffers a whole answer before sending anything needs the SDK
    # default back; `None` restores it rather than forcing a number that would cut the call short.
    client = OpenAICompatLLMClient(model="m", api_key="test", stall_timeout=None)
    timeout = client._client.timeout
    assert not isinstance(timeout, Timeout) or timeout.read != DEFAULT_STREAM_STALL_TIMEOUT


@pytest.mark.asyncio
async def test_base_url_is_forwarded_to_the_sdk_so_gemini_and_local_route_by_config() -> None:
    # The whole point of this client: Gemini / Ollama / vLLM are a base_url change, not a new class.
    base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    client = OpenAICompatLLMClient(model="gemini-2.5-pro", api_key="test", base_url=base)
    assert "generativelanguage.googleapis.com" in str(client._client.base_url)


def test_custom_base_url_does_not_inherit_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # A key for OpenAI proper must never leak to an unrelated compatible endpoint. An authenticated
    # custom endpoint receives a key explicitly; an unauthenticated one gets the SDK placeholder.
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    client = OpenAICompatLLMClient(model="llama3", base_url="http://localhost:11434/v1")
    assert client._client.api_key == "not-needed"


def test_openai_endpoint_keeps_the_sdk_environment_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    client = OpenAICompatLLMClient(model="m")
    assert client._client.api_key == "openai-secret"


def test_openai_endpoint_ignores_an_environment_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Endpoint selection is a trust-boundary decision: an environment override must not redirect
    # the SDK-standard OpenAI credential to an unrelated OpenAI-compatible endpoint.
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.example/v1")
    client = OpenAICompatLLMClient(model="m")
    assert str(client._client.base_url) == "https://api.openai.com/v1/"
    assert client._client.api_key == "openai-secret"


def test_explicit_key_still_wins_over_the_placeholder_for_an_authed_endpoint() -> None:
    # An endpoint that does require auth (a gateway) gets the real key, not the placeholder.
    client = OpenAICompatLLMClient(model="m", api_key="real", base_url="http://gw/v1")
    assert client._client.api_key == "real"


@pytest.mark.asyncio
async def test_instrument_emits_a_usage_record_carrying_the_exact_reasoning_tokens(
    _llm_logging_enabled: None,
) -> None:
    client = OpenAICompatLLMClient(model="m", api_key="test", instrument=True)
    client._client = _FakeAsyncOpenAI(  # type: ignore[assignment]
        _response(
            "hi",
            prompt_tokens=100,
            completion_tokens=800,
            reasoning_tokens=600,
            cached_tokens=80,
        )
    )
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    try:
        await client.complete(_request())
    finally:
        logger.removeHandler(handler)

    usage_records = [r for r in records if r.__dict__.get("llm_event") == "usage"]
    assert len(usage_records) == 1
    (record,) = usage_records
    assert record.__dict__["llm_input_tokens"] == 100
    assert record.__dict__["llm_output_tokens"] == 800
    assert record.__dict__["llm_reasoning_tokens"] == 600
    assert record.__dict__["llm_cached_input_tokens"] == 80
    assert record.__dict__["llm_finish_reason"] == "stop"
    assert record.__dict__["llm_semantic_label"] == "test"
    assert record.__dict__["llm_prompt_version"] == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_instrument_carries_provider_finish_reason_for_both_response_modes(
    stream: bool, _llm_logging_enabled: None
) -> None:
    client = OpenAICompatLLMClient(model="m", api_key="test", instrument=True, stream=stream)
    client._client = _FakeAsyncOpenAI(  # type: ignore[assignment]
        _response("hi", finish_reason="length")
    )
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.addHandler(handler)
    try:
        await client.complete(_request())
    finally:
        logger.removeHandler(handler)

    (usage,) = [r for r in records if r.__dict__.get("llm_event") == "usage"]
    assert usage.__dict__["llm_finish_reason"] == "length"


@pytest.mark.asyncio
async def test_uninstrumented_client_emits_no_usage_record() -> None:
    client = OpenAICompatLLMClient(model="m", api_key="test")  # instrument defaults off
    client._client = _FakeAsyncOpenAI(_response("hi"))  # type: ignore[assignment]
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign,assignment]
    logger = logging.getLogger("sora.llm")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        await client.complete(_request())
    finally:
        logger.removeHandler(handler)

    assert not [r for r in records if r.__dict__.get("llm_event") == "usage"]


@pytest.mark.asyncio
async def test_aclose_releases_the_underlying_http_client() -> None:
    client = OpenAICompatLLMClient(model="m", api_key="test")
    fake = _FakeAsyncOpenAI(_response())
    client._client = fake  # type: ignore[assignment]
    await client.aclose()
    assert fake.closed is True


@pytest.mark.asyncio
async def test_a_stalled_stream_closes_the_response_on_its_way_out() -> None:
    """The stall timeout raises *while* the stream is being iterated — that is the path it exists
    for — so the response has to be released there or every timed-out call leaks its connection.
    A run doing thousands of calls against a flaky endpoint exhausts the pool, and since the pool
    timeout is the same `Timeout` value, healthy calls then start failing too."""
    client = OpenAICompatLLMClient(model="m", api_key="test")
    fake = _FakeAsyncOpenAI(_response("hello"), fail_after=2)
    client._client = fake  # type: ignore[assignment]

    with pytest.raises(APITimeoutError):
        await client.complete(_request())

    stream = fake.chat.completions.stream
    assert stream is not None
    assert stream.closed is True


@pytest.mark.asyncio
async def test_a_completed_stream_is_closed_too() -> None:
    client = OpenAICompatLLMClient(model="m", api_key="test")
    fake = _FakeAsyncOpenAI(_response("hello"))
    client._client = fake  # type: ignore[assignment]

    assert await client.complete(_request()) == "hello"

    stream = fake.chat.completions.stream
    assert stream is not None
    assert stream.closed is True
