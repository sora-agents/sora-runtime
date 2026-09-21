"""In-memory raw LLM exchange capture for prompt-evaluation attempts."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from sora.bootstrap import import_object
from sora.llm import CompletionRequest, LLMClient, current_inference_id, current_llm_call_id


class BufferedLLMExchangeCapture:
    """Collect exact exchanges without disk I/O on the live scenario trajectory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompts: dict[str, str] = {}
        self._rows: list[dict[str, Any]] = []

    def record(
        self,
        request: CompletionRequest,
        *,
        response: str | None,
        error: BaseException | None,
        call_id: str,
        inference_id: str | None,
        elapsed_seconds: float,
    ) -> None:
        system_hash = hashlib.sha256(request.system.encode()).hexdigest()
        user_hash = hashlib.sha256(request.user.encode()).hexdigest()
        with self._lock:
            self._prompts.setdefault(system_hash, request.system)
            self._prompts.setdefault(user_hash, request.user)
            self._rows.append(
                {
                    "sequence": len(self._rows),
                    "call_id": call_id,
                    "inference_id": inference_id,
                    "semantic_label": request.semantic_label,
                    "prompt_version": request.prompt_version,
                    "system_sha256": system_hash,
                    "user_sha256": user_hash,
                    "response": response,
                    "error": (
                        {"type": type(error).__name__, "message": str(error)}
                        if error is not None
                        else None
                    ),
                    "response_characters": len(response) if response is not None else None,
                    "elapsed_seconds": elapsed_seconds,
                }
            )

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._rows)

    def export(self, root: Path) -> tuple[Path, ...]:
        """Write the buffered snapshot after the attempt has stopped."""
        with self._lock:
            prompts = dict(self._prompts)
            rows = tuple(dict(row) for row in self._rows)
        prompts_dir = root / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for digest, prompt in sorted(prompts.items()):
            path = prompts_dir / f"{digest}.txt"
            path.write_text(prompt, encoding="utf-8")
            paths.append(path)
        exchanges = root / "exchanges.jsonl"
        exchanges.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        paths.append(exchanges)
        return tuple(paths)


class CapturingLLMClient:
    """Compatibility decorator whose explicit ``export`` preserves the no-live-write rule."""

    def __init__(
        self,
        *,
        inner_client: str,
        capture_dir: str | Path,
        **settings: Any,
    ) -> None:
        client_type = import_object(inner_client)
        self._inner: LLMClient = client_type(**settings)
        configured_model = settings.get("model")
        inner_model = getattr(self._inner, "model", None)
        self.model = (
            configured_model
            if isinstance(configured_model, str)
            else inner_model
            if isinstance(inner_model, str)
            else None
        )
        self._capture_dir = Path(capture_dir)
        self.capture = BufferedLLMExchangeCapture()

    async def complete(self, request: CompletionRequest) -> str:
        call_id = current_llm_call_id.get() or current_inference_id.get() or "unscoped"
        inference_id = current_inference_id.get()
        started = time.perf_counter()
        try:
            response = await self._inner.complete(request)
        except BaseException as error:
            self._record_best_effort(
                request,
                call_id=call_id,
                inference_id=inference_id,
                response=None,
                error=error,
                elapsed=time.perf_counter() - started,
            )
            raise
        self._record_best_effort(
            request,
            call_id=call_id,
            inference_id=inference_id,
            response=response,
            error=None,
            elapsed=time.perf_counter() - started,
        )
        return response

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()

    def export(self) -> tuple[Path, ...]:
        return self.capture.export(self._capture_dir)

    def _record_best_effort(
        self,
        request: CompletionRequest,
        *,
        call_id: str,
        inference_id: str | None,
        response: str | None,
        error: BaseException | None,
        elapsed: float,
    ) -> None:
        try:
            self.capture.record(
                request,
                response=response,
                error=error,
                call_id=call_id,
                inference_id=inference_id,
                elapsed_seconds=elapsed,
            )
        except BaseException:
            return
