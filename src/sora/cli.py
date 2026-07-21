"""The runtime's minimal terminal interface."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from sora.activity import ActivityState
from sora.bootstrap import build_agent
from sora.perception import Message
from sora.transport import InProcessTransport

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.cycle import Agent

log = logging.getLogger("sora.cli")

_PHASES = ("observe", "situate", "reason", "reflect", "act")
_CYCLE_BEGIN = re.compile(r"^\[cycle (\d+)\] begin$")
_EXIT_COMMANDS = ("exit", "quit")

# No trailing "> " prompt: in a plain line-buffered terminal it has no way to survive
# asynchronous output landing mid-line (it's not a real prompt-toolkit-style redraw), so it just
# reads as noise. This banner is printed once at startup instead, explaining how to interact.
_BANNER = (
    "+----------------------------------------------+\n"
    "| S-ORA -- minimal terminal interface          |\n"
    "|                                              |\n"
    "| Type a goal in plain English to delegate it. |\n"
    "| Type 'exit' or 'quit' to quit.               |\n"
    "+----------------------------------------------+"
)


class _Console:
    """Tracks whether the terminal cursor sits at the start of a line, so lines printed through
    here (log trace lines, the agent's replies) are always separated by exactly one newline —
    never bunched onto the same physical line as whatever printed right before them."""

    def __init__(self) -> None:
        self._at_line_start = True

    def line(self, text: str) -> None:
        if not self._at_line_start:
            print()
        print(text)
        self._at_line_start = True


class _Presenter(logging.Handler):
    """Formats the runtime's existing ``sora.*`` log records into the terminal trace — the
    ``[cycle N] Phase - ...`` line under ``--verbose``, a terse ``[invoking tool.op...]`` cue
    otherwise — without adding any new log call sites. Cycle number is tracked purely by parsing
    ``cycle.py``'s existing ``"[cycle %d] begin"`` debug record: ``DecisionCycle`` deliberately
    exposes no current-phase/current-activity state for a presenter to read directly."""

    def __init__(self, *, verbose: bool, console: _Console) -> None:
        super().__init__(level=logging.DEBUG if verbose else logging.INFO)
        self._verbose = verbose
        self._console = console
        self._cycle = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        begin = _CYCLE_BEGIN.match(message)
        if begin:
            self._cycle = int(begin.group(1))
            return
        if self._verbose:
            self._emit_verbose(message)
        else:
            self._emit_terse(message)

    def _emit_verbose(self, message: str) -> None:
        prefix, sep, rest = message.partition(": ")
        if sep and prefix.lower() in _PHASES:
            label = f"{prefix.capitalize():<9}-"
            self._console.line(f"[cycle {self._cycle}] {label} {rest}")
        else:
            self._console.line(message)  # no recognized phase prefix — passed through as-is

    def _emit_terse(self, message: str) -> None:
        if message.startswith("act: invoke "):
            target = message.removeprefix("act: invoke ").split(" ", 1)[0]
            self._console.line(f"[invoking {target}...]")


class TerminalSession:
    """Streams cycle output to stdout; queues stdin as Message(sender="user", ...) — not a
    Percept, since terminal input is user communication, not environment stimuli. No UI beyond
    this."""

    def __init__(
        self,
        agent: Agent,
        verbose: bool = False,
        *,
        poll_interval: float = 0.02,
        initial_task: str | None = None,
    ) -> None:
        communication = agent.communication
        if not isinstance(communication, InProcessTransport):
            raise TypeError(
                "TerminalSession requires an InProcessTransport "
                f"(got {type(communication).__name__}); peer transports are not supported yet"
            )
        self._agent = agent
        self._transport = communication
        self._verbose = verbose
        # Neither of these is README-documented (the sketch's __init__ takes only agent/verbose)
        # — implementation details, same role as Agent's own tick_interval. initial_task lets
        # `sora run --task/--task-file` (and a test) seed the first Observe without needing a
        # real stdin line, mirroring what examples/*/run.py does by hand via transport.submit().
        self._poll_interval = poll_interval
        self._initial_task = initial_task

    async def run(self) -> None:
        console = _Console()
        presenter = _Presenter(verbose=self._verbose, console=console)
        sora_log = logging.getLogger("sora")
        previous_level = sora_log.level
        sora_log.setLevel(logging.DEBUG)
        sora_log.addHandler(presenter)

        if self._initial_task:
            self._transport.submit(
                Message(
                    sender="user", content={"text": self._initial_task}, received_at=time.time()
                )
            )

        runner = asyncio.create_task(self._agent.run())
        stop_reading = asyncio.Event()
        reader = asyncio.create_task(self._read_stdin(stop_reading))
        printed_trajectories: set[str] = set()
        try:
            console.line(_BANNER)
            sent_seen = 0
            while not runner.done() and not stop_reading.is_set():
                sent = self._transport.sent
                while sent_seen < len(sent):
                    _, content = sent[sent_seen]
                    sent_seen += 1
                    console.line(str(content.get("text", content)))
                self._print_new_trajectories(console, printed_trajectories)
                await asyncio.sleep(self._poll_interval)
        finally:
            await self._agent.stop()
            if not runner.done():
                runner.cancel()
                try:
                    await runner
                except (Exception, asyncio.CancelledError, BaseExceptionGroup) as exc:
                    # We triggered this cancellation ourselves (mid-join/mid-tick teardown can
                    # surface as a messy anyio BaseExceptionGroup, not a plain CancelledError —
                    # see the MCP stdio client's own task group) — treat it as shutdown noise,
                    # not a crash, but keep it visible at debug level.
                    log.debug("agent.run() raised while shutting down after cancel: %r", exc)
            else:
                await runner  # finished on its own — a real failure here should propagate
            if not reader.done():
                reader.cancel()
            try:
                await reader  # connect_read_pipe-based, so this responds to cancel immediately
            except asyncio.CancelledError:
                pass
            sora_log.removeHandler(presenter)
            sora_log.setLevel(previous_level)
            console.line("Goodbye.")

    def _print_new_trajectories(self, console: _Console, printed: set[str]) -> None:
        for activity in self._agent.working.activities.values():
            if activity.state is ActivityState.TERMINATED and activity.id not in printed:
                printed.add(activity.id)
                console.line(_trajectory(activity))

    async def _read_stdin(self, stop_reading: asyncio.Event) -> None:
        # A plain `run_in_executor(None, sys.stdin.readline)` blocks a real OS thread with no way
        # to interrupt it once started — cancelling *our* task doesn't stop it, and both
        # asyncio.run()'s own shutdown and Python's ThreadPoolExecutor atexit hook then wait for
        # that thread to actually finish (i.e. for a real Enter keypress) before the process can
        # exit. connect_read_pipe reads the fd directly through the event loop instead, so it's
        # cancellable immediately, with nothing left dangling. (Unix-only: connect_read_pipe isn't
        # implemented on Windows' default ProactorEventLoop — not a target platform today.)
        loop = asyncio.get_running_loop()
        stream = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(stream)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        while True:
            raw = await stream.readline()
            if raw == b"":
                stop_reading.set()
                return
            line = raw.decode(errors="replace").strip()
            if line.lower() in _EXIT_COMMANDS:
                stop_reading.set()
                return
            if line:
                self._transport.submit(
                    Message(sender="user", content={"text": line}, received_at=time.time())
                )


def _trajectory(activity: Activity) -> str:
    lines = [f"activity {activity.id!r}: {activity.goal!r} -> {activity.state.name}"]
    if activity.plan is not None:
        for i, step in enumerate(activity.plan.steps):
            marker = ">" if i == activity.step_index else " "
            lines.append(f"  {marker} {step.next_action} {step.params}")
    return "\n".join(lines)


def main() -> None:
    # Not in the README sketch — added so `[project.scripts] sora = "sora.cli:main"` resolves to a
    # real callable. Only `run` is implemented; `sora init` (project scaffolding, also shown in
    # README's walkthrough) is a separate, unbuilt feature.
    parser = argparse.ArgumentParser(prog="sora")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Start a persistent terminal session")
    run_parser.add_argument("config", nargs="?", default="agent.yaml", help="Path to agent.yaml")
    run_parser.add_argument(
        "--verbose", action="store_true", help="Print the per-phase decision-cycle trace"
    )
    task_group = run_parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task", help="Submit this text as the initial user message at startup"
    )
    task_group.add_argument(
        "--task-file", help="Read the initial user message from this file at startup"
    )
    args = parser.parse_args()

    # A console-script entry point (unlike `python -m`) doesn't put the current directory on
    # sys.path, but agent.yaml's dotted strategy/adapter-factory paths are meant to resolve
    # project-local code from wherever `sora run` is invoked — import_object's own docstring
    # already promises this ("anything on sys.path resolves"). Match `python -m`'s behavior.
    if "" not in sys.path:
        sys.path.insert(0, "")

    initial_task = args.task
    if args.task_file:
        initial_task = Path(args.task_file).read_text(encoding="utf-8").strip()

    agent = build_agent(args.config)
    session = TerminalSession(agent, verbose=args.verbose, initial_task=initial_task)
    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        pass
