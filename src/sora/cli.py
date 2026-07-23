"""The runtime's minimal terminal interface."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from sora import scaffold
from sora.activity import ActivityState
from sora.bootstrap import build_agent
from sora.llm import LLMMeter
from sora.perception import Message
from sora.transport import InProcessTransport

if TYPE_CHECKING:
    from sora.activity import Activity
    from sora.cycle import Agent

log = logging.getLogger("sora.cli")

_PHASES = ("observe", "situate", "reason", "reflect", "act")
_CYCLE_BEGIN = re.compile(r"^\[cycle (\d+)\] begin$")
_EXIT_COMMANDS = ("exit", "quit")

# ANSI styling. Raw escapes, no dependency: the core stays dependency-light and the CLI already
# speaks straight to the terminal. Every styled write goes through `_paint`, a no-op when color is
# off — so with color disabled (tests, pipes, NO_COLOR) output is byte-for-byte the pre-color text.
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_CYAN = "\x1b[36m"
_MAGENTA = "\x1b[35m"


def _paint(text: str, code: str, *, enabled: bool) -> str:
    return f"{code}{text}{_RESET}" if enabled else text


def _color_enabled(setting: bool | None) -> bool:
    """Resolve the --color / --no-color / auto tri-state. Auto (``None``) colors only a real TTY
    with ``NO_COLOR`` unset — the de-facto convention — so redirected/piped output and CI stay
    plain, and captured test output has no escapes to assert around."""
    if setting is not None:
        return setting
    try:
        return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    except (AttributeError, ValueError):  # stdout may be a capture object without a real fileno
        return False


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


def _restore_blocking_stdout() -> None:
    """Force stdout back to blocking mode before a write. A stdio subprocess sharing our terminal
    (the MCP server) and asyncio's own ``connect_read_pipe`` on stdin both call
    ``os.set_blocking(fd, False)``; on a tty, fds 0/1/2 share one open file description, so that
    also makes stdout non-blocking. A write larger than the tty buffer then raises
    ``BlockingIOError`` (EAGAIN) instead of blocking — which struck the final, oversized trajectory
    dump. Restoring blocking mode is safe for the async stdin reader: ``add_reader``-gated reads
    only fire when data is already available, so they never block on a blocking fd. Best-effort:
    stdout may be captured/redirected with no real fd (tests)."""
    try:
        os.set_blocking(sys.stdout.fileno(), True)
    except (AttributeError, OSError, ValueError):
        pass


class _Console:
    """Tracks whether the terminal cursor sits at the start of a line, so lines printed through
    here (log trace lines, the agent's replies) are always separated by exactly one newline —
    never bunched onto the same physical line as whatever printed right before them."""

    def __init__(self) -> None:
        self._at_line_start = True

    def line(self, text: str) -> None:
        _restore_blocking_stdout()
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

    def __init__(self, *, verbose: bool, console: _Console, color: bool = False) -> None:
        super().__init__(level=logging.DEBUG if verbose else logging.INFO)
        self._verbose = verbose
        self._console = console
        self._color = color
        self._cycle = 0

    def emit(self, record: logging.LogRecord) -> None:
        # The presenter formats the *runtime's* trace; ignore the CLI layer's own diagnostics
        # (e.g. the expected shutdown-cancel CancelledError), which are children of `sora` too but
        # aren't part of the decision-cycle trace the user is watching.
        if record.name.startswith(log.name):
            return
        # `MeteredLLMClient`'s per-call cue: a live "the model is thinking" signal, shown only under
        # --verbose (it's noise in the terse view, but the end-of-run summary still counts it).
        if record.name == "sora.llm":
            if self._verbose:
                self._console.line(_paint(record.getMessage(), _MAGENTA, enabled=self._color))
            return
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
            line = f"[cycle {self._cycle}] {label} {rest}"
        else:
            line = message  # no recognized phase prefix — passed through as-is
        self._console.line(_paint(line, _DIM, enabled=self._color))

    def _emit_terse(self, message: str) -> None:
        if message.startswith("act: invoke "):
            target = message.removeprefix("act: invoke ").split(" ", 1)[0]
            self._console.line(_paint(f"[invoking {target}...]", _CYAN, enabled=self._color))


class TerminalSession:
    """Streams cycle output to stdout; queues stdin as Message(sender="user", ...) — not a
    Percept, since terminal input is user communication, not environment stimuli. No UI beyond
    this."""

    def __init__(
        self,
        agent: Agent,
        verbose: bool = False,
        *,
        color: bool | None = None,
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
        self._color = _color_enabled(color)
        # Neither of these is README-documented (the sketch's __init__ takes only agent/verbose)
        # — implementation details, same role as Agent's own tick_interval. initial_task lets
        # `sora run --task/--task-file` (and a test) seed the first Observe without needing a
        # real stdin line, mirroring what examples/*/run.py does by hand via transport.submit().
        self._poll_interval = poll_interval
        self._initial_task = initial_task

    async def run(self) -> None:
        console = _Console()
        presenter = _Presenter(verbose=self._verbose, console=console, color=self._color)
        # Tallies the model round-trips `MeteredLLMClient` logs, for the end-of-run summary. A
        # separate handler from the presenter (which only *displays*), so counting is independent
        # of display: the terse view hides the per-call cue but still totals it in the summary.
        meter = LLMMeter()
        sora_log = logging.getLogger("sora")
        previous_level = sora_log.level
        sora_log.setLevel(logging.DEBUG)
        sora_log.addHandler(presenter)
        sora_log.addHandler(meter)
        wall_start = time.monotonic()

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
            console.line(_paint(_BANNER, _BOLD, enabled=self._color))
            sent_seen = 0
            while not runner.done() and not stop_reading.is_set():
                sent = self._transport.sent
                while sent_seen < len(sent):
                    _, content = sent[sent_seen]
                    sent_seen += 1
                    # A blank line above + bold sets the agent's reply apart from the dim trace
                    # around it — the one line the user is actually waiting for.
                    console.line("")
                    console.line(
                        _paint(str(content.get("text", content)), _BOLD, enabled=self._color)
                    )
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
            sora_log.removeHandler(meter)
            sora_log.setLevel(previous_level)
            wall = time.monotonic() - wall_start
            console.line(_paint(f"-- {meter.summary(wall)} --", _DIM, enabled=self._color))
            console.line(_paint("Goodbye.", _DIM, enabled=self._color))

    def _print_new_trajectories(self, console: _Console, printed: set[str]) -> None:
        for activity in self._agent.working.activities.values():
            if activity.state is ActivityState.TERMINATED and activity.id not in printed:
                printed.add(activity.id)
                console.line(_paint(_trajectory(activity), _DIM, enabled=self._color))

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


def _run(args: argparse.Namespace) -> None:
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
    session = TerminalSession(
        agent, verbose=args.verbose, color=args.color, initial_task=initial_task
    )
    try:
        asyncio.run(session.run())
    except KeyboardInterrupt:
        pass


def _init(args: argparse.Namespace) -> None:
    project_dir = Path(args.dir)
    scaffold.write_project(project_dir)
    print(f"Created {project_dir}/")
    for path in sorted(project_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(project_dir)}")
    print()
    print(f"  cd {project_dir}")
    print("  uv sync")
    print("  export ANTHROPIC_API_KEY=sk-ant-...")
    print("  uv run sora run")


def main() -> None:
    # Not in the README sketch — added so `[project.scripts] sora = "sora.cli:main"` resolves to a
    # real callable.
    parser = argparse.ArgumentParser(prog="sora")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start a persistent terminal session")
    run_parser.add_argument("config", nargs="?", default="agent.yaml", help="Path to agent.yaml")
    run_parser.add_argument(
        "--verbose", action="store_true", help="Print the per-phase decision-cycle trace"
    )
    color_group = run_parser.add_mutually_exclusive_group()
    color_group.add_argument(
        "--color",
        dest="color",
        action="store_true",
        default=None,
        help="Force ANSI color output (default: auto — on only for a TTY, off if NO_COLOR is set)",
    )
    color_group.add_argument(
        "--no-color", dest="color", action="store_false", help="Disable ANSI color output"
    )
    task_group = run_parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task", help="Submit this text as the initial user message at startup"
    )
    task_group.add_argument(
        "--task-file", help="Read the initial user message from this file at startup"
    )

    init_parser = subparsers.add_parser("init", help="Scaffold a minimal example agent")
    init_parser.add_argument("dir", help="Directory to create (must not already exist)")

    args = parser.parse_args()
    if args.command == "run":
        _run(args)
    elif args.command == "init":
        _init(args)
