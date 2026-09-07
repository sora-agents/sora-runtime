#!/usr/bin/env python3
"""The S-ORA worker inside a Gaia2 container. Runs as the sandboxed ``agent`` user.

Connects to the adapter over a Unix socket, builds the agent from ``/opt/agent.yaml`` with the
``llm:`` block filled in from the environment, and runs one decision cycle for the whole scenario —
every turn the daemon delivers lands in the same running agent's inbox, so it keeps its activities,
memory and in-flight work across turns rather than being re-entered per turn.

**Model selection comes from the environment here**, which inverts this project's usual "agent.yaml
is the only model selector" stance. It is deliberate and scoped to this harness: the runner sets
``PROVIDER``/``MODEL``/``API_KEY``/``BASE_URL`` per invocation so that its own ``--model`` flag
behaves normally across a sweep, and baking a model into the image would make that flag silently
inert. Everything else about the agent still comes from ``agent.yaml``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

import yaml

WORKER_SOCK = os.environ.get("SORA_WORKER_SOCK", "/tmp/sora-worker.sock")
BASE_CONFIG = os.environ.get("SORA_AGENT_CONFIG", "/opt/agent.yaml")
DERIVED_CONFIG = "/tmp/sora-agent.yaml"

logging.basicConfig(
    level=os.environ.get("SORA_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("sora-worker")

# The provider name the runner passes -> the LLMClient that speaks to it. Everything that is not
# Anthropic goes through the OpenAI-compatible client, which is what Gemini, OpenAI itself, and the
# local runtimes all expose.
_ANTHROPIC_CLIENT = "sora.adapters.anthropic_llm.AnthropicLLMClient"
_OPENAI_COMPAT_CLIENT = "sora.adapters.openai_llm.OpenAICompatLLMClient"


def _llm_block() -> dict[str, Any]:
    provider = (os.environ.get("PROVIDER") or "anthropic").strip().lower()
    model = os.environ.get("MODEL") or ""
    api_key = os.environ.get("API_KEY") or ""
    base_url = os.environ.get("BASE_URL") or ""

    block: dict[str, Any] = {"instrument": True}
    if provider == "anthropic":
        block["client"] = _ANTHROPIC_CLIENT
    else:
        block["client"] = _OPENAI_COMPAT_CLIENT
        if base_url:
            block["base_url"] = base_url
    if model:
        block["model"] = model
    if api_key:
        block["api_key"] = api_key
    return block


def _write_derived_config() -> str:
    with open(BASE_CONFIG) as handle:
        config = yaml.safe_load(handle)
    config["agent"]["llm"] = {**config["agent"].get("llm", {}), **_llm_block()}
    with open(DERIVED_CONFIG, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    redacted = {k: ("***" if k == "api_key" else v) for k, v in config["agent"]["llm"].items()}
    log.info("derived agent config at %s with llm=%s", DERIVED_CONFIG, redacted)
    return DERIVED_CONFIG


# The adapter's ``backend_connect`` returns *before* its listener exists (it has to: ``/health``
# must answer while the worker is still coming up), and nothing on either side waits for the socket.
# So "no such file" here is the ordinary startup order, not a dead adapter — whichever of the
# adapter's imports and this worker's ``build_agent`` finishes first decides it, which is why it
# can work for weeks and then not. One un-retried connect made that race fatal: the 2026-09-07
# time-scenario run died on ``FileNotFoundError`` while a perfectly healthy adapter was still
# binding. Retry to a deadline, then fail naming the adapter and its log rather than the syscall.
_CONNECT_TIMEOUT = float(os.environ.get("SORA_WORKER_CONNECT_TIMEOUT", "60"))


async def _connect_to_adapter() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _CONNECT_TIMEOUT
    delay = 0.1
    announced = False
    while True:
        try:
            return await asyncio.open_unix_connection(WORKER_SOCK)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"the adapter never opened {WORKER_SOCK} within {_CONNECT_TIMEOUT:.0f}s "
                    "— it is not running; see /tmp/gaia2-adapter.log"
                ) from exc
            if not announced:
                log.info("waiting for the adapter to open %s ...", WORKER_SOCK)
                announced = True
            await asyncio.sleep(delay)
            delay = min(delay * 2, 1.0)


async def _main() -> None:
    from sora.adapters.gaia2_cli import Gaia2CliTransport
    from sora.bootstrap import build_agent

    agent = build_agent(_write_derived_config())
    transport = agent.communication
    # `transport: {kind: gaia2-cli}` in agent.yaml is what makes this hold; assert rather than
    # cast, so a config that selected some other transport fails here instead of at the first turn.
    assert isinstance(transport, Gaia2CliTransport), (
        f"agent.yaml must select `transport: {{kind: gaia2-cli}}`, got {type(transport).__name__}"
    )

    reader, writer = await _connect_to_adapter()
    log.info("connected to the adapter on %s", WORKER_SOCK)

    def _emit(message: dict[str, Any]) -> None:
        # Called synchronously from the agent's own send; the socket write is buffered, and the
        # drain happens on the event loop's next pass.
        writer.write((json.dumps(message) + "\n").encode())

    transport.emit = _emit
    writer.write((json.dumps({"type": "ready"}) + "\n").encode())
    await writer.drain()

    async def _read_turns() -> None:
        async for raw in reader:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                log.warning("adapter sent non-JSON: %r", raw)
                continue
            if msg.get("type") == "notification":
                # An environment notification, not a turn: it announces that the world moved, so it
                # goes to the naming tool's signal sink rather than the inbox. Offered to every
                # joined workspace — each drops a label it does not hold, and scenario pruning means
                # a notification for an app nobody joined is expected rather than an error.
                text = msg.get("text", "")
                for workspace in agent.registry.joined_workspaces():
                    note = getattr(workspace, "note_env_notification", None)
                    if note is not None:
                        note(text)
                log.info("env notification: %s", text[:200])
                continue
            if msg.get("type") != "message":
                continue
            text = msg.get("text", "")
            log.info("turn delivered (run_id=%s): %s", msg.get("run_id"), text[:200])
            transport.submit_user_message(text, run_id=msg.get("run_id", ""))

    # Raced, not fire-and-forget. A bare `create_task(agent.run())` whose exception nobody
    # retrieves leaves this process alive and silent on the socket read: no log, no response, and
    # the scenario runs out its clock looking like a slow agent rather than a crashed one. Racing
    # the two means whichever ends first ends the worker, and the loser's failure is reported.
    run = asyncio.create_task(agent.run(), name="agent")
    turns = asyncio.create_task(_read_turns(), name="turns")
    try:
        done, _ = await asyncio.wait({run, turns}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc is not None:
                log.error("the %s task failed", task.get_name(), exc_info=exc)
                # Answer the turn in flight, so the harness records a boundary and the daemon can
                # close the scenario instead of waiting out its clock on a dead agent.
                _emit(
                    {
                        "type": "response",
                        "run_id": transport._run_id or "",
                        "state": "error",
                        "message": f"agent failed: {exc}",
                        "errorMessage": str(exc),
                    }
                )
                await writer.drain()
    finally:
        await agent.stop()
        for task in (run, turns):
            task.cancel()
        await asyncio.gather(run, turns, return_exceptions=True)
        writer.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except Exception:  # a crash here is invisible otherwise — the adapter just never sees "ready"
        log.exception("S-ORA worker failed")
        sys.exit(1)
