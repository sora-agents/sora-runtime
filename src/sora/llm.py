"""Provider-agnostic LLM access for the reasoning path."""

from __future__ import annotations

import logging
import time
from typing import Protocol

# A dedicated child of the `sora` tree so instrumentation records are addressable on their own —
# the CLI presenter surfaces them as a per-call cue, `LLMMeter` tallies them, and neither has to
# reach into the reasoning path to do it.
_llm_log = logging.getLogger("sora.llm")


class LLMClient(Protocol):
    """A single completion round-trip — the runtime's one seam onto a language model.

    Deliberately narrow and wire-format-neutral: a system instruction plus a prompt in, text out.
    It commits to *no* provider shape — not OpenAI ``chat/completions``, not Anthropic ``messages``
    — so the reasoning path (``ProceduralMemory.infer``) stays independent of any one SDK, and the
    concrete client (an optional extra under ``sora/adapters/``) is the only place a wire format
    appears. The canonical format the runtime converts *to* is its own domain (``Plan``/``Step``),
    not a borrowed message schema; that conversion (the anti-corruption boundary) lives in
    ``infer``, never here.

    Non-ownership contract: an ``LLMClient`` owns *only* the round-trip. Retries, streaming,
    credential refresh, prompt caching, and interrupt handling belong to the cycle/agent, never to
    the client. Keeping that boundary explicit is what lets a second provider slot in without
    touching the decision cycle.
    """

    async def complete(self, *, system: str, prompt: str) -> str: ...


class MeteredLLMClient:
    """A transparent ``LLMClient`` decorator that times each round-trip and logs a ``sora.llm`` cue.

    It does *not* violate the ``LLMClient`` non-ownership contract: the contract forbids the
    *client itself* from growing timing/retry responsibilities, keeping every concrete provider
    thin. This wraps one from the outside — an instrumentation layer bootstrap slips in front of the
    real client — so the client stays a bare round-trip while the run gains observability. Each call
    emits one record carrying the elapsed seconds as a structured ``llm_seconds`` field, so a reader
    (`LLMMeter`, the CLI presenter) never has to parse it back out of the message text.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    async def complete(self, *, system: str, prompt: str) -> str:
        start = time.perf_counter()
        try:
            return await self._inner.complete(system=system, prompt=prompt)
        finally:
            elapsed = time.perf_counter() - start
            _llm_log.info(
                "~ llm (%.2fs)", elapsed, extra={"llm_event": "done", "llm_seconds": elapsed}
            )

    async def aclose(self) -> None:
        # Forward lifecycle to the wrapped client if it has any — keeps the decorator drop-in for a
        # client whose teardown someone calls (the Anthropic one holds an HTTP client).
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()


class LLMMeter(logging.Handler):
    """Tallies the ``sora.llm`` per-call records `MeteredLLMClient` emits — call count and summed
    in-model seconds — so a run surface can report them at the end without holding a reference to
    the client (which bootstrap builds and hands off). Attach it to the ``sora`` logger for the run,
    then call ``summary()``. Mirrors how the CLI's ``_Presenter`` reads the same log stream."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.calls = 0
        self.total_seconds = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "llm_event", None) == "done":
            self.calls += 1
            self.total_seconds += getattr(record, "llm_seconds", 0.0)

    def summary(self, wall_seconds: float | None = None) -> str:
        plural = "" if self.calls == 1 else "s"
        text = f"{self.calls} LLM call{plural}, {self.total_seconds:.1f}s in-model"
        if wall_seconds is not None:
            text += f", {wall_seconds:.1f}s wall"
        return text
