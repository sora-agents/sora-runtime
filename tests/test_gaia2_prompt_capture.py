from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest
from examples.gaia2.evaluation.campaigns.prompt.capture import (
    BufferedLLMExchangeCapture,
    CapturingLLMClient,
)

from sora.llm import (
    CompletionRequest,
    MeteredLLMClient,
    capture_llm_exchanges,
    current_inference_id,
    current_llm_call_id,
    llm_call_scope,
)


class _RecordingClient:
    instances: ClassVar[list[_RecordingClient]] = []

    def __init__(self, **settings: Any) -> None:
        self.settings = settings
        self.requests: list[CompletionRequest] = []
        self.error: BaseException | None = None
        self.model = settings.get("model")
        self.__class__.instances.append(self)

    async def complete(self, request: CompletionRequest) -> str:
        self.requests.append(request)
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return f"raw:{request.user}"


@pytest.fixture(autouse=True)
def _inner_client_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingClient.instances.clear()
    module = types.ModuleType("gaia2_capture_test_client")
    module.__dict__["RecordingClient"] = _RecordingClient
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _request(*, user: str = "user prompt") -> CompletionRequest:
    return CompletionRequest(
        system="system prompt",
        user=user,
        semantic_label="plan",
        prompt_version="7",
    )


def _rows(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "exchanges.jsonl").read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.asyncio
async def test_capture_forwards_settings_and_records_raw_exchange(tmp_path: Path) -> None:
    root = tmp_path / "llm"
    client = CapturingLLMClient(
        inner_client="gaia2_capture_test_client.RecordingClient",
        capture_dir=root,
        model="model-1",
        api_key="secret",
        base_url="https://provider.invalid/v1",
        max_retries=2,
    )
    call_token = current_llm_call_id.set("call-1")
    inference_token = current_inference_id.set("inference-1")
    try:
        result = await client.complete(_request())
    finally:
        current_inference_id.reset(inference_token)
        current_llm_call_id.reset(call_token)

    assert result == "raw:user prompt"
    assert client.model == "model-1"
    inner = _RecordingClient.instances[0]
    assert inner.settings == {
        "model": "model-1",
        "api_key": "secret",
        "base_url": "https://provider.invalid/v1",
        "max_retries": 2,
    }
    assert inner.requests == [_request()]
    assert not root.exists()
    client.export()

    system_hash = hashlib.sha256(b"system prompt").hexdigest()
    user_hash = hashlib.sha256(b"user prompt").hexdigest()
    assert (root / "prompts" / f"{system_hash}.txt").read_text() == "system prompt"
    assert (root / "prompts" / f"{user_hash}.txt").read_text() == "user prompt"
    row = _rows(root)[0]
    assert row == {
        "sequence": 0,
        "call_id": "call-1",
        "inference_id": "inference-1",
        "semantic_label": "plan",
        "prompt_version": "7",
        "system_sha256": system_hash,
        "user_sha256": user_hash,
        "response": "raw:user prompt",
        "error": None,
        "response_characters": len("raw:user prompt"),
        "elapsed_seconds": pytest.approx(row["elapsed_seconds"]),
    }
    assert row["elapsed_seconds"] >= 0


@pytest.mark.asyncio
async def test_capture_records_error_and_reraises_same_exception(tmp_path: Path) -> None:
    root = tmp_path / "llm"
    client = CapturingLLMClient(
        inner_client="gaia2_capture_test_client.RecordingClient",
        capture_dir=root,
    )
    provider_error = RuntimeError("provider failed")
    _RecordingClient.instances[0].error = provider_error

    with pytest.raises(RuntimeError) as raised:
        await client.complete(_request())

    assert raised.value is provider_error
    assert not root.exists()
    client.export()
    row = _rows(root)[0]
    assert row["response"] is None
    assert row["response_characters"] is None
    assert row["error"] == {"type": "RuntimeError", "message": "provider failed"}


@pytest.mark.asyncio
async def test_capture_write_failure_never_changes_provider_outcome(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied", encoding="utf-8")
    successful = CapturingLLMClient(
        inner_client="gaia2_capture_test_client.RecordingClient",
        capture_dir=blocked,
    )

    assert await successful.complete(_request()) == "raw:user prompt"

    failing = CapturingLLMClient(
        inner_client="gaia2_capture_test_client.RecordingClient",
        capture_dir=blocked,
    )
    provider_error = LookupError("provider failed too")
    _RecordingClient.instances[-1].error = provider_error
    with pytest.raises(LookupError) as raised:
        await failing.complete(_request())
    assert raised.value is provider_error


@pytest.mark.asyncio
async def test_capture_serializes_concurrent_appends_and_continues_sequence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "llm"
    client = CapturingLLMClient(
        inner_client="gaia2_capture_test_client.RecordingClient",
        capture_dir=root,
    )

    async def complete(index: int) -> str:
        call_token = current_llm_call_id.set(f"call-{index}")
        inference_token = current_inference_id.set(f"inference-{index}")
        try:
            return await client.complete(_request(user=f"user {index}"))
        finally:
            current_inference_id.reset(inference_token)
            current_llm_call_id.reset(call_token)

    assert await asyncio.gather(*(complete(index) for index in range(20))) == [
        f"raw:user {index}" for index in range(20)
    ]
    assert not root.exists()
    client.export()
    first_rows = _rows(root)
    assert sorted(row["sequence"] for row in first_rows) == list(range(20))
    assert {row["call_id"] for row in first_rows} == {f"call-{index}" for index in range(20)}

    assert await client.complete(_request(user="resumed")) == "raw:resumed"
    client.export()
    assert [row["sequence"] for row in _rows(root)] == [*range(20), 20]
    assert len(list((root / "prompts").glob("*.txt"))) == 22


@pytest.mark.asyncio
async def test_metered_client_capture_correlates_repairs_without_live_disk_writes(
    tmp_path: Path,
) -> None:
    class _Repairing:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: CompletionRequest) -> str:
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("provider failed")
            return "malformed" if self.calls == 1 else '{"steps": []}'

    capture = BufferedLLMExchangeCapture()
    client = MeteredLLMClient(_Repairing())
    root = tmp_path / "llm"
    with capture_llm_exchanges(capture), llm_call_scope() as call_id:
        await client.complete(_request())
        await client.complete(_request(user="repair prompt"))
        with pytest.raises(RuntimeError, match="provider failed"):
            await client.complete(_request(user="error prompt"))

    assert not root.exists()
    capture.export(root)
    rows = _rows(root)
    assert [row["call_id"] for row in rows] == [call_id, call_id, call_id]
    assert [row["response"] for row in rows] == ["malformed", '{"steps": []}', None]
    assert rows[-1]["error"] == {"type": "RuntimeError", "message": "provider failed"}


@pytest.mark.asyncio
async def test_metered_client_ignores_a_failing_raw_exchange_sink() -> None:
    class _FailingCapture:
        def record(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("capture failed")

    client = MeteredLLMClient(_RecordingClient())
    with capture_llm_exchanges(_FailingCapture()):
        result = await client.complete(_request())

    assert result == "raw:user prompt"
