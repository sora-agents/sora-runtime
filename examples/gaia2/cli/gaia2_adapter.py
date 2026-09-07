#!/usr/bin/env python3
"""HTTP bridge between the Gaia2 runner/daemon and the S-ORA worker.

Runs as the ``gaia2`` user (it needs ``/var/gaia2/state``, which the agent user must not reach);
the worker runs as ``agent`` and connects back over a Unix socket. Structure and protocol follow
the sibling harnesses' adapters — the two deliberate differences are:

* **``/send_notifications`` is claimed here rather than aliased to ``/notify``.** The base handler
  routes ``/notify``, ``/send_user_message`` and ``/send_notifications`` to one handler with an
  identical body, so the route is the only thing that distinguishes "the user said something" from
  "the environment changed". Forwarding the rendered notification as a *user message* would
  attribute it to someone who did not speak; S-ORA has a third slot for exactly this, so it is
  delivered to the worker as a **Signal** on the tool the notification names. The synthetic
  ``send_message_to_agent`` event the base would have written is still written here, so the
  daemon's own bookkeeping is unchanged.
* **``backend_connect`` returns immediately.** ``/health`` must answer before the worker exists,
  or the runner's health probe times out waiting on a worker that is waiting on the socket.

Unix-socket protocol (JSON lines):
    adapter -> worker: {"type": "message", "text": ..., "run_id": ...}
    worker  -> adapter: {"type": "ready"}
                        {"type": "response", "run_id": ..., "state": "final"|"error",
                         "message": ...}
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from typing import Any

_this_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_this_dir, os.path.join(_this_dir, "..", "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gaia2_adapter_base import (  # noqa: E402
    AdapterState,
    create_client_handler,
    http_response,
    run_adapter,
    write_aui_event,
)
from gaia2_cli.daemon.cli_executor import execute_cli_action  # noqa: E402

STATE_DIR = os.environ.get("GAIA2_STATE_DIR", "/var/gaia2/state")
WORKER_SOCK = os.environ.get("SORA_WORKER_SOCK", "/tmp/sora-worker.sock")

_writer: asyncio.StreamWriter | None = None
_connected = False
_active_run_id: str | None = None
_run_lock: asyncio.Lock | None = None

_state = AdapterState(buffer_size=int(os.environ.get("GAIA2_BUFFER_SIZE", "200")))


# ── worker socket ────────────────────────────────────────────────────────────────────────────────


def _handle_worker_response(msg: dict[str, Any]) -> None:
    global _active_run_id

    run_id = msg.get("run_id", "")
    state = msg.get("state", "")
    if run_id and run_id == _active_run_id:
        _active_run_id = None
    if state == "final":
        # THE turn boundary. Written even for an empty message: without it the daemon never ticks
        # and the scenario ends "error: no turn boundary detected".
        write_aui_event("send_message_to_user", msg.get("message", "") or "")
    entry = _state.buffer_and_broadcast(
        {
            "run_id": run_id,
            "runId": run_id,
            "state": state,
            "message": msg.get("message", ""),
            **({"errorMessage": msg["errorMessage"]} if msg.get("errorMessage") else {}),
        }
    )
    print(f"[gaia2-adapter] buffered {state} seq={entry['seq']} runId={run_id}")


async def _on_worker_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    global _writer, _connected

    if _writer is not None:
        print("[gaia2-adapter] new worker connected; dropping the previous connection")
        try:
            _writer.close()
        except Exception:
            pass
    _writer = writer
    print("[gaia2-adapter] worker connected")
    try:
        async for raw in reader:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"[gaia2-adapter] worker sent non-JSON: {exc}: {raw!r}")
                continue
            kind = msg.get("type")
            if kind == "ready":
                _connected = True
                print("[gaia2-adapter] worker ready")
            elif kind == "response":
                _handle_worker_response(msg)
            else:
                print(f"[gaia2-adapter] unknown worker message type: {kind!r}")
    finally:
        _connected = False
        if _writer is writer:
            _writer = None
        try:
            writer.close()
        except Exception:
            pass
        print("[gaia2-adapter] worker disconnected")


async def _worker_listener() -> None:
    try:
        os.unlink(WORKER_SOCK)
    except FileNotFoundError:
        pass
    server = await asyncio.start_unix_server(_on_worker_conn, path=WORKER_SOCK)
    os.chmod(WORKER_SOCK, 0o666)  # the agent user has to be able to connect
    print(f"[gaia2-adapter] listening for the S-ORA worker on {WORKER_SOCK}")
    async with server:
        await server.serve_forever()


# ── inbound ──────────────────────────────────────────────────────────────────────────────────────


def _write_line(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
    writer.write((json.dumps(msg) + "\n").encode())


async def send_message(text: str) -> dict[str, Any]:
    """Deliver a user turn to the worker. Unlike the sibling harnesses there is nothing to queue
    while a run is in flight: the S-ORA cycle drains its inbox every Observe, so a turn arriving
    mid-run is picked up by the next tick rather than after the current one finishes."""
    global _active_run_id

    if not _connected or _writer is None or _run_lock is None:
        raise ConnectionError("S-ORA worker not connected")
    async with _run_lock:
        run_id = str(uuid.uuid4())
        _active_run_id = run_id
        _write_line(_writer, {"type": "message", "text": text, "run_id": run_id})
        await _writer.drain()
    return {"run_id": run_id}


def is_connected() -> bool:
    return _connected


def get_health_info() -> dict[str, Any]:
    return {"backend": "sora", "worker_connected": _connected, "activeRun": _active_run_id}


# The daemon writes daemon_status.json and exits when the scenario ends — and tells no one else.
# This HTTP server serves forever and the worker's agent loop has no reason to stop, so a standalone
# `docker run` sits there looking busy long after the last turn was judged. Saying so is
# unconditional. *Acting* on it is opt-in, because the runner learns about completion by polling
# `GET /status` from this very process: exiting the moment the daemon does would race the poll that
# was going to read the result.
_TERMINAL_STATUSES = frozenset({"complete", "error", "stopped"})
_EXIT_ON_COMPLETE = (os.environ.get("SORA_EXIT_ON_COMPLETE") or "").lower() in {"1", "true", "yes"}
_EXIT_GRACE_SECONDS = 15.0


def _read_daemon_status(path: str) -> str:
    with open(path) as handle:
        return str(json.load(handle).get("status", ""))


async def _watch_for_scenario_end() -> None:
    status_file = os.path.join(STATE_DIR, "daemon_status.json")
    while True:
        await asyncio.sleep(2.0)
        try:
            status = await asyncio.to_thread(_read_daemon_status, status_file)
        except (OSError, json.JSONDecodeError, ValueError):
            continue  # not written yet, or caught mid-write
        if status not in _TERMINAL_STATUSES:
            continue
        print(f"[gaia2-adapter] the daemon finished (status={status}); the scenario is over")
        if not _EXIT_ON_COMPLETE:
            print(
                "[gaia2-adapter] the container stays up so the runner can still poll GET /status — "
                "stop it with `docker stop`, or pass -e SORA_EXIT_ON_COMPLETE=1 to have a "
                "standalone run shut itself down"
            )
            return
        print(
            f"[gaia2-adapter] SORA_EXIT_ON_COMPLETE is set; exiting in {_EXIT_GRACE_SECONDS:.0f}s"
        )
        await asyncio.sleep(_EXIT_GRACE_SECONDS)  # a window for the runner's last poll to land
        if _writer is not None:
            _writer.close()  # EOF on the worker's socket: it unwinds, and the container with it
        os._exit(0)


async def backend_connect() -> None:
    # Return at once; the worker comes up later via entrypoint.sh.
    asyncio.create_task(_worker_listener())
    asyncio.create_task(_watch_for_scenario_end())


def _on_notify_sent(text: str) -> None:
    write_aui_event("send_message_to_agent", text)


async def _execute_action(
    app: str, action: str, args: dict[str, Any], event_id: str
) -> dict[str, Any]:
    """Environment actions from the daemon, run through the app CLIs as the gaia2 user."""
    result: dict[str, Any] = execute_cli_action(app, action, args, event_id, state_dir=STATE_DIR)
    return result


async def _extra_routes(
    method: str,
    route: str,
    parsed: Any,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    body: bytes,
) -> bool:
    """Claim ``/send_notifications`` so an environment notification reaches the agent as a Signal
    rather than as a user message — see the module docstring. The event is still recorded, so the
    daemon's own view of what it told the agent is unchanged."""
    if not (method == "POST" and route == "/send_notifications"):
        return False
    try:
        text = json.loads(body or b"{}").get("message", "")
    except (json.JSONDecodeError, ValueError):
        text = ""
    write_aui_event("send_message_to_agent", text)
    delivered = bool(text) and _connected and _writer is not None
    if delivered:
        assert _writer is not None
        # Deliberately no `_run_lock`: a notification is not a turn and must not queue behind the
        # run in flight. The worker pushes it into the signal sink of the tool it names, where the
        # cycle drains it on the next Observe — the same path any tool-pushed signal takes.
        _write_line(_writer, {"type": "notification", "text": text})
        await _writer.drain()
    print(f"[gaia2-adapter] env notification (delivered={delivered}): {text[:120]!r}")
    http_response(writer, 200, {"ok": True, "delivered": delivered})
    await writer.drain()
    writer.close()
    return True


async def main() -> None:
    global _run_lock
    _run_lock = asyncio.Lock()
    handler = create_client_handler(
        state=_state,
        send_message=send_message,
        is_connected=is_connected,
        get_health_info=get_health_info,
        extra_routes=_extra_routes,
        on_notify_sent=_on_notify_sent,
        execute_action=_execute_action,
    )
    await run_adapter(_state, handler, backend_connect, backend_name="S-ORA")


if __name__ == "__main__":
    asyncio.run(main())
