"""``TerminalSession`` (``sora/cli.py``) — the runtime's minimal terminal interface.

Three things are tested at different levels. ``_Console`` (fresh-line bookkeeping) and
``_Presenter`` (the `[cycle N] Phase - ...` / terse `[invoking ...]` formatter) are tested directly
— pure formatters over the runtime's *existing* sora.* log records, no agent needed.
``TerminalSession.run``'s own plumbing (stdin -> Message, transport.sent -> stdout, the typed
exit command, initial-task seeding, trajectory printing, clean shutdown) is tested against a real
``Agent`` built from ``FakeAdapter``/``FakeWorkspace``/``FakeTool`` (the same subprocess-free
double ``test_agent_run.py`` uses) with a spied/replaced stdin, isolating it from Reason/Act —
matching that file's own isolation rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from fakes import FakeAdapter, FakeTool, FakeWorkspace
from sora.action import default_action_registry
from sora.activity import Activity, ActivityState
from sora.cli import _BANNER, TerminalSession, _Console, _Presenter, main
from sora.cycle import Agent, DecisionCycle
from sora.environment import EnvironmentRegistry, WorkspaceOrigin
from sora.memory import (
    EpisodicMemory,
    FileMemoryBackend,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from sora.perception import Message
from sora.strategies import (
    DefaultActStrategy,
    DefaultObserveStrategy,
    DefaultReasonStrategy,
    DefaultReflectStrategy,
    DefaultSituateStrategy,
    Strategies,
)
from sora.transport import InProcessTransport
from sora.types import Plan, Signal, Step

_ORIGIN = WorkspaceOrigin(adapter="fake", address="fake://ws")


def _build_agent(tmp_path: Path) -> Agent:
    workspace = FakeWorkspace(
        "clock", _ORIGIN, [FakeTool("Clock", invoke_results={"get_time": {}})]
    )
    registry = EnvironmentRegistry(adapters={_ORIGIN: FakeAdapter("fake", workspace)})
    working = WorkingMemory(registry=registry)
    semantic = SemanticMemory(FileMemoryBackend(tmp_path / "semantic"))
    strategies = Strategies(
        observe=DefaultObserveStrategy(),
        reflect=DefaultReflectStrategy(),
        situate=DefaultSituateStrategy(),
        reason=DefaultReasonStrategy(),
        act=DefaultActStrategy(),
    )
    cycle = DecisionCycle(
        strategies=strategies,
        communication=InProcessTransport(),
        actions=default_action_registry(),
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=ProceduralMemory(FileMemoryBackend(tmp_path / "procedural")),
        episodic=EpisodicMemory(FileMemoryBackend(tmp_path / "episodic")),
    )
    return Agent(
        cycle=cycle,
        registry=registry,
        working=working,
        semantic=semantic,
        procedural=cycle.procedural,
        episodic=cycle.episodic,
        communication=cycle.communication,
        tick_interval=0.0,  # run as fast as the event loop allows
    )


async def _run_until(predicate: object, task: asyncio.Task[None]) -> None:
    for _ in range(1000):
        if predicate():  # type: ignore[operator]
            return
        await asyncio.sleep(0)
    task.cancel()
    raise AssertionError("condition not reached before the loop budget ran out")


async def _collect_until(
    capsys: pytest.CaptureFixture[str], needle: str, task: asyncio.Task[None]
) -> str:
    collected = ""
    for _ in range(1000):
        collected += capsys.readouterr().out
        if needle in collected:
            return collected
        await asyncio.sleep(0)
    task.cancel()
    raise AssertionError(f"{needle!r} not seen; got {collected!r}")


def _record(name: str, level: int, msg: str, *args: object) -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 0, msg, args, None)


class _PipeStdin:
    """A real OS pipe standing in for ``sys.stdin``. ``TerminalSession`` reads via
    ``loop.connect_read_pipe``, which reads the underlying fd directly through the event loop
    (bypassing any Python-level buffering on the object passed in) — a plain duck-typed fake
    without a real fd wouldn't exercise that path faithfully. Gives the test explicit control over
    stdin timing without touching a real terminal; exposes ``.fileno()`` so it can stand in for
    ``sys.stdin`` wherever that's read."""

    def __init__(self) -> None:
        read_fd, write_fd = os.pipe()
        self._read_end = os.fdopen(read_fd, "rb", buffering=0)
        self._write_end = os.fdopen(write_fd, "wb", buffering=0)

    def fileno(self) -> int:
        return self._read_end.fileno()

    def push_line(self, line: str) -> None:
        data = line if line.endswith("\n") else line + "\n"
        self._write_end.write(data.encode())

    def close(self) -> None:
        self._write_end.close()  # EOF on the read end once buffered data is drained


async def _stop(session: TerminalSession, task: asyncio.Task[None], stdin: _PipeStdin) -> None:
    stdin.close()
    await asyncio.wait_for(task, timeout=2)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class _NoOutboundLogTransport:
    """No `.sent` — TerminalSession can't stream a reply from this, regardless of transport kind."""

    async def send(self, to: str, content: dict[str, object]) -> None: ...

    def receive(self) -> AsyncIterator[Message]:
        async def _empty() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover

        return _empty()


class _SentOnlyTransport:
    """A presentable-but-not-submittable transport: an outbound `.sent` log, but no `.submit()` —
    exercises TerminalSession's graceful degradation when ad hoc input has nowhere to go. (Both
    shipped transports are now submittable; this stands in for a custom one that isn't.)"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, object]]] = []

    async def send(self, to: str, content: dict[str, object]) -> None:
        self.sent.append((to, content))

    def receive(self) -> AsyncIterator[Message]:
        async def _empty() -> AsyncIterator[Message]:
            return
            yield  # pragma: no cover

        return _empty()


async def test_init_rejects_transport_without_an_outbound_log(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.communication = _NoOutboundLogTransport()
    with pytest.raises(TypeError, match="`.sent`"):
        TerminalSession(agent)


async def test_init_accepts_a_duck_typed_transport_without_submit(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.communication = _SentOnlyTransport()
    TerminalSession(agent)  # must not raise: .sent is all __init__ requires


# ---------------------------------------------------------------------------
# _Console — fresh-line bookkeeping
# ---------------------------------------------------------------------------


def test_console_consecutive_lines_do_not_add_blank_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    console = _Console()
    console.line("first")
    console.line("second")
    out = capsys.readouterr().out
    assert out == "first\nsecond\n"


def test_console_writes_to_a_supplied_stream(capsys: pytest.CaptureFixture[str]) -> None:
    # The --log-file capture path: a stream-backed console writes there, not to stdout, so the
    # same presenter formatting can mirror the trace to a file.
    import io

    sink = io.StringIO()
    console = _Console(stream=sink)
    console.line("first")
    console.line("second")
    assert sink.getvalue() == "first\nsecond\n"
    assert capsys.readouterr().out == ""


def test_file_presenter_captures_the_full_verbose_trace() -> None:
    # A file-backed presenter is always verbose (independent of the terminal's --verbose), so a
    # --log-file always holds the complete per-phase trace even when the terminal view is terse.
    import io

    sink = io.StringIO()
    presenter = _Presenter(verbose=True, console=_Console(stream=sink), color=False)
    presenter.emit(_record("sora.cycle", logging.DEBUG, "[cycle %d] begin", 1))
    presenter.emit(
        _record("sora.action", logging.INFO, "act: invoke %s.%s%s", "Clock", "get_time", "")
    )
    assert sink.getvalue() == "[cycle 1] Act      - invoke Clock.get_time\n"


# ---------------------------------------------------------------------------
# _Presenter — pure formatting, no agent needed
# ---------------------------------------------------------------------------


def test_presenter_verbose_formats_the_cycle_phase_trace(
    capsys: pytest.CaptureFixture[str],
) -> None:
    presenter = _Presenter(verbose=True, console=_Console())
    presenter.emit(_record("sora.cycle", logging.DEBUG, "[cycle %d] begin", 1))
    presenter.emit(
        _record(
            "sora.strategies",
            logging.INFO,
            "observe: message from %s: %r",
            "user",
            "what time is it?",
        )
    )
    presenter.emit(
        _record("sora.action", logging.INFO, "act: invoke %s.%s%s", "Clock", "get_time", "")
    )
    presenter.emit(_record("sora.cycle", logging.DEBUG, "[cycle %d] begin", 2))
    presenter.emit(
        _record(
            "sora.strategies",
            logging.INFO,
            "reflect: activity %s completed; stored to episodic memory",
            "ask-time",
        )
    )

    out = capsys.readouterr().out
    assert out == (
        "[cycle 1] Observe  - message from user: 'what time is it?'\n"
        "[cycle 1] Act      - invoke Clock.get_time\n"
        "[cycle 2] Reflect  - activity ask-time completed; stored to episodic memory\n"
    )


def test_presenter_verbose_hides_debug_detail_but_still_tracks_cycles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The terminal --verbose trace must NOT show sub-INFO detail (model prompts, full plan/result
    # reprs) — those are for the --log-file mirror only — while the DEBUG "[cycle N] begin" marker
    # is still consumed for cycle tracking.
    presenter = _Presenter(verbose=True, console=_Console())  # show_debug defaults False
    presenter.emit(_record("sora.cycle", logging.DEBUG, "[cycle %d] begin", 4))
    presenter.emit(_record("sora.memory", logging.DEBUG, "reason: system prompt\n%s", "SECRET"))
    presenter.emit(
        _record("sora.action", logging.INFO, "act: invoke %s.%s%s", "Clock", "get_time", "")
    )
    out = capsys.readouterr().out
    assert "SECRET" not in out  # the DEBUG prompt did not reach the terminal
    assert out == "[cycle 4] Act      - invoke Clock.get_time\n"  # cycle 4 tracked from the marker


def test_file_presenter_shows_debug_detail() -> None:
    # The --log-file mirror (show_debug=True) is the complete trace: the same DEBUG prompt the
    # terminal suppresses is captured here.
    import io

    sink = io.StringIO()
    presenter = _Presenter(
        verbose=True, console=_Console(stream=sink), color=False, show_debug=True
    )
    presenter.emit(_record("sora.cycle", logging.DEBUG, "[cycle %d] begin", 4))
    presenter.emit(_record("sora.memory", logging.DEBUG, "reason: system prompt\n%s", "SECRET"))
    assert sink.getvalue() == "[cycle 4] Reason   - system prompt\nSECRET\n"


def test_presenter_verbose_passes_through_unrecognized_prefix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    presenter = _Presenter(verbose=True, console=_Console())
    presenter.emit(
        _record(
            "sora.cycle", logging.INFO, "startup: joining workspace %s (%s)", "fake://ws", "fake"
        )
    )
    out = capsys.readouterr().out
    assert out == "startup: joining workspace fake://ws (fake)\n"


def test_presenter_non_verbose_only_surfaces_invoke_cues(
    capsys: pytest.CaptureFixture[str],
) -> None:
    presenter = _Presenter(verbose=False, console=_Console())
    presenter.emit(
        _record("sora.strategies", logging.INFO, "observe: message from %s: %r", "user", "hi")
    )
    presenter.emit(
        _record(
            "sora.action", logging.INFO, "act: invoke %s.%s%s", "Clock", "get_time", " {'x': 1}"
        )
    )
    out = capsys.readouterr().out
    assert out == "[invoking Clock.get_time...]\n"


def test_presenter_verbose_surfaces_the_llm_cue(capsys: pytest.CaptureFixture[str]) -> None:
    presenter = _Presenter(verbose=True, console=_Console())
    presenter.emit(_record("sora.llm", logging.INFO, "~ llm (%.2fs)", 1.24))
    assert capsys.readouterr().out == "~ llm (1.24s)\n"


def test_presenter_terse_hides_the_llm_cue(capsys: pytest.CaptureFixture[str]) -> None:
    presenter = _Presenter(verbose=False, console=_Console())
    presenter.emit(_record("sora.llm", logging.INFO, "~ llm (%.2fs)", 1.24))
    assert capsys.readouterr().out == ""  # counted in the summary, but not shown live in terse mode


# ---------------------------------------------------------------------------
# TerminalSession.run — stdin/output plumbing, against a real (fakes-backed) Agent
# ---------------------------------------------------------------------------


async def test_run_prints_the_banner_and_submits_stdin_lines_as_user_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    transport = agent.communication
    assert isinstance(transport, InProcessTransport)
    submitted: list[Message] = []
    original_submit = transport.submit

    def _spy(message: Message) -> None:
        submitted.append(message)
        original_submit(message)

    monkeypatch.setattr(transport, "submit", _spy)

    session = TerminalSession(agent, poll_interval=0.0)
    task = asyncio.create_task(session.run())
    try:
        stdin.push_line("what time is it?")
        await _run_until(lambda: len(submitted) == 1, task)
        assert submitted[0].sender == "user"
        assert submitted[0].content == {"text": "what time is it?"}
        out = capsys.readouterr().out
        assert out.startswith(_BANNER)  # no "> " prompt — it can't survive async output anyway
    finally:
        await _stop(session, task, stdin)


async def test_run_treats_slash_stop_as_a_hard_interrupt_not_a_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `/stop` is a reserved authoritative control: it must raise a hard interrupt straight into the
    # cycle (halt current work), never be submitted as an ordinary user Message.
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    transport = agent.communication
    assert isinstance(transport, InProcessTransport)
    submitted: list[Message] = []
    original_submit = transport.submit

    def _spy_submit(message: Message) -> None:
        submitted.append(message)
        original_submit(message)

    monkeypatch.setattr(transport, "submit", _spy_submit)

    interrupts: list[Signal] = []
    original_interrupt = agent.cycle.interrupt

    async def _spy_interrupt(signal: Signal, *, target: str | None = None) -> None:
        interrupts.append(signal)
        await original_interrupt(signal, target=target)

    monkeypatch.setattr(agent.cycle, "interrupt", _spy_interrupt)

    session = TerminalSession(agent, poll_interval=0.0)
    task = asyncio.create_task(session.run())
    try:
        stdin.push_line("/stop")
        await _run_until(lambda: len(interrupts) == 1, task)
        assert interrupts[0].name == "user_stop"
        assert submitted == []  # never routed through the cooperative Message path
    finally:
        await _stop(session, task, stdin)


async def test_run_streams_conversational_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    transport = agent.communication
    assert isinstance(transport, InProcessTransport)
    session = TerminalSession(agent, poll_interval=0.0)
    task = asyncio.create_task(session.run())
    try:
        await transport.send("user", {"text": "It's 14:32."})
        await _collect_until(capsys, "It's 14:32.", task)
    finally:
        await _stop(session, task, stdin)


async def test_run_returns_cleanly_on_stdin_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    session = TerminalSession(agent, poll_interval=0.0)
    task = asyncio.create_task(session.run())
    stdin.close()
    await asyncio.wait_for(task, timeout=2)  # must not hang
    assert task.exception() is None


async def test_run_task_cancel_exits_promptly_without_a_stdin_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mirrors what a real Ctrl-C does at the *asyncio* level: cancelling session.run()'s task
    # directly, mid-flight, with nobody ever writing to stdin (no EOF, no typed exit command).
    # Note this alone does NOT prove the real bug is fixed: cancelling a task blocked on
    # run_in_executor(readline) is honored promptly at the asyncio level either way (old or new
    # implementation) — the actual hang happened later, at Python interpreter shutdown, when
    # concurrent.futures' atexit hook joins the still-blocked reader thread. That's a process-level
    # effect this in-process test can't see; see
    # test_process_exits_promptly_after_cancel_even_with_stdin_left_open below for the test that
    # actually catches it. This test still earns its keep as the fast, in-process check that
    # run()/_read_stdin's own cancellation handling doesn't regress.
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    session = TerminalSession(agent, poll_interval=0.0)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0)  # let it start up (create the reader/runner tasks) before cancelling
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=2)  # must not hang waiting on stdin
    except asyncio.CancelledError:
        pass
    stdin.close()


def test_process_exits_promptly_after_cancel_even_with_stdin_left_open(tmp_path: Path) -> None:
    # The real regression: run.run_in_executor(sys.stdin.readline) leaks a genuinely-blocked OS
    # thread on cancel (asyncio-level cancellation doesn't stop it — it just stops *waiting* on
    # it). That thread is invisible to any in-process test, because pytest itself never restarts
    # its process between tests. It only bites at real process exit, when
    # concurrent.futures.thread's atexit hook joins every pool thread, including the leaked one —
    # which never returns, because in a real terminal session nothing else closes stdin's write
    # end for you (unlike a same-process test fixture, which incidentally closes it during its own
    # teardown and masks the bug). So: spawn a real subprocess, give it a pipe stdin whose write
    # end *this test* deliberately keeps open throughout, cancel its main task from inside the
    # subprocess (mirroring what asyncio.run()'s own SIGINT-triggered cleanup does — real signal
    # delivery to a spawned child isn't reliable across sandboxes, so the child triggers it
    # itself), and assert the process actually exits within a bound — never writing a line, never
    # closing the pipe.
    script = f"""
import asyncio, sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})
from sora.cli import TerminalSession
from sora.transport import InProcessTransport

class _FakeWorkingMemory:
    activities: dict = {{}}

class _FakeAgent:
    def __init__(self):
        self.communication = InProcessTransport()
        self.working = _FakeWorkingMemory()
        self._stopped = False

    async def run(self):
        while not self._stopped:
            await asyncio.sleep(0.01)

    async def stop(self):
        self._stopped = True

async def main():
    session = TerminalSession(_FakeAgent(), poll_interval=0.0)
    task = asyncio.create_task(session.run())
    await asyncio.sleep(0.3)  # let _read_stdin actually start blocking on the (empty) pipe
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

asyncio.run(main())
print("PROCESS_EXITING_CLEANLY", flush=True)
"""
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=read_fd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    os.close(read_fd)  # the child has its own duplicate; this process doesn't need it
    try:
        # write_fd deliberately never written to or closed here — simulates a real terminal/`tail
        # -f /dev/null` keeping stdin open for the process's whole lifetime.
        stdout, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise AssertionError(
            "process did not exit within 5s with stdin left open — the Ctrl-C hang regressed"
        ) from None
    finally:
        os.close(write_fd)
    assert process.returncode == 0, stdout
    assert "PROCESS_EXITING_CLEANLY" in stdout, stdout


@pytest.mark.parametrize("command", ["/exit", "/quit", "/EXIT", "/Quit"])
async def test_run_exits_cleanly_on_typed_exit_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    transport = agent.communication
    assert isinstance(transport, InProcessTransport)
    submitted: list[Message] = []
    monkeypatch.setattr(transport, "submit", lambda m: submitted.append(m))

    session = TerminalSession(agent, poll_interval=0.0)
    task = asyncio.create_task(session.run())
    stdin.push_line(command)
    await asyncio.wait_for(task, timeout=2)  # must not hang waiting for EOF
    assert task.exception() is None
    assert submitted == []  # the exit command itself is never submitted as a goal


async def test_run_submits_initial_task_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    transport = agent.communication
    assert isinstance(transport, InProcessTransport)
    submitted: list[Message] = []
    monkeypatch.setattr(transport, "submit", lambda m: submitted.append(m))

    session = TerminalSession(agent, poll_interval=0.0, initial_task="what time is it?")
    task = asyncio.create_task(session.run())
    try:
        await _run_until(lambda: len(submitted) == 1, task)
        assert submitted[0].sender == "user"
        assert submitted[0].content == {"text": "what time is it?"}
    finally:
        await _stop(session, task, stdin)


async def test_run_prints_trajectory_once_when_an_activity_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    session = TerminalSession(agent, poll_interval=0.0)
    task = asyncio.create_task(session.run())
    try:
        plan = Plan(
            id="p1",
            goal="what time is it?",
            steps=[
                Step(
                    next_action="invoke", params={"tool_id": "Clock", "operation_name": "get_time"}
                )
            ],
        )
        agent.working.activities["a1"] = Activity(
            id="a1",
            goal="what time is it?",
            context={},
            state=ActivityState.TERMINATED,
            plan=plan,
            step_index=1,
        )
        collected = await _collect_until(capsys, "activity 'a1'", task)
        assert "what time is it?" in collected
        assert "-> TERMINATED" in collected

        # Printed exactly once — later polls must not re-print an already-seen termination.
        for _ in range(20):
            await asyncio.sleep(0)
        collected += capsys.readouterr().out
        assert collected.count("activity 'a1'") == 1
    finally:
        await _stop(session, task, stdin)


async def test_run_notifies_once_when_transport_does_not_support_ad_hoc_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    agent = _build_agent(tmp_path)
    agent.communication = _SentOnlyTransport()
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    session = TerminalSession(agent, poll_interval=0.0)
    task = asyncio.create_task(session.run())
    try:
        stdin.push_line("hello?")
        stdin.push_line("still there?")
        collected = await _collect_until(capsys, "doesn't accept ad hoc input", task)
        for _ in range(20):  # a second typed line must not print a second notice
            await asyncio.sleep(0)
        collected += capsys.readouterr().out
        assert collected.count("doesn't accept ad hoc input") == 1
    finally:
        await _stop(session, task, stdin)


async def test_run_exits_via_typed_exit_even_without_submit_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _build_agent(tmp_path)
    agent.communication = _SentOnlyTransport()
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)

    session = TerminalSession(agent, poll_interval=0.0)
    task = asyncio.create_task(session.run())
    stdin.push_line("/exit")
    await asyncio.wait_for(task, timeout=2)
    assert task.exception() is None


async def test_exit_when_idle_stops_the_session_without_stdin_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)
    agent.working.activities["a1"] = Activity(
        id="a1", goal="what time is it?", context={}, state=ActivityState.TERMINATED
    )

    session = TerminalSession(agent, poll_interval=0.0, exit_when_idle=0.01)
    task = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(task, timeout=2)  # must exit on its own, no EOF/typed exit
        assert task.exception() is None
    finally:
        stdin.close()


async def test_log_file_captures_the_trace_and_is_written_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A --log-file run writes the complete trace to the file even in the terse (non-verbose)
    # terminal view — the startup "joined workspace" line is a DEBUG/INFO record the terse terminal
    # would still show, but here it proves the file-backed presenter ran and flushed to disk.
    agent = _build_agent(tmp_path)
    agent.working.activities["a1"] = Activity(
        id="a1", goal="what time is it?", context={}, state=ActivityState.TERMINATED
    )
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)
    log_path = tmp_path / "trace.log"

    session = TerminalSession(
        agent, verbose=False, poll_interval=0.0, exit_when_idle=0.01, log_file=log_path
    )
    task = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(task, timeout=2)
        assert task.exception() is None
    finally:
        stdin.close()

    assert log_path.exists()
    captured = log_path.read_text(encoding="utf-8")
    assert "joined workspace clock" in captured  # a real runtime log record reached the file
    assert "\x1b[" not in captured  # no ANSI escapes in the file mirror


async def test_banner_reflects_headless_scenario_driven_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Neither of the default banner's two lines applies to a headless `--scenario ...
    # --exit-when-idle` run: this transport won't accept a typed goal, and nobody is watching
    # stdin to type 'exit' — telling the user either would be actively misleading.
    agent = _build_agent(tmp_path)
    agent.communication = _SentOnlyTransport()
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)
    agent.working.activities["a1"] = Activity(
        id="a1", goal="what time is it?", context={}, state=ActivityState.TERMINATED
    )

    session = TerminalSession(agent, poll_interval=0.0, exit_when_idle=0.01)
    task = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(task, timeout=2)  # exits on its own via the idle timer
        assert task.exception() is None
    finally:
        stdin.close()

    out = capsys.readouterr().out
    assert "Type a goal in plain English" not in out
    assert "Type '/exit' or '/quit' to quit" not in out
    assert "Driven by the running scenario's timeline." in out
    assert "Exits automatically once idle." in out


async def test_exit_when_idle_resets_while_activities_are_still_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _build_agent(tmp_path)
    stdin = _PipeStdin()
    monkeypatch.setattr(sys, "stdin", stdin)
    # RUNNING (not READY): only READY activities are selected for Reason, so this stays "live" for
    # the duration of the test without the fake agent's LLM-less ProceduralMemory ever being asked
    # to infer a plan for it.
    agent.working.activities["a1"] = Activity(
        id="a1", goal="what time is it?", context={}, state=ActivityState.RUNNING
    )

    session = TerminalSession(agent, poll_interval=0.0, exit_when_idle=0.01)
    task = asyncio.create_task(session.run())
    try:
        for _ in range(20):  # a live (non-terminated) activity must not trigger the idle exit
            await asyncio.sleep(0)
        assert not task.done()
    finally:
        # A RUNNING activity never reaches TERMINATED on its own, so stdin EOF (which
        # `--exit-when-idle` deliberately ignores — see `_read_stdin`) won't stop this session;
        # only an explicit typed command does.
        stdin.push_line("/exit")
        await asyncio.wait_for(task, timeout=2)


# ---------------------------------------------------------------------------
# main() — argparse wiring only (build_agent/TerminalSession faked out)
# ---------------------------------------------------------------------------


class _FakeSession:
    last_calls: dict[str, object] = {}

    def __init__(
        self,
        agent: object,
        verbose: bool = False,
        *,
        color: bool | None = None,
        poll_interval: float = 0.02,
        initial_task: str | None = None,
        exit_when_idle: float | None = None,
        log_file: str | None = None,
    ) -> None:
        _FakeSession.last_calls = {
            "agent": agent,
            "verbose": verbose,
            "color": color,
            "initial_task": initial_task,
            "exit_when_idle": exit_when_idle,
            "log_file": log_file,
        }

    async def run(self) -> None:
        return None


def test_main_run_defaults_to_agent_yaml_and_non_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake_build_agent(config_path: str, *, simulation: object | None = None) -> object:
        calls["config"] = config_path
        calls["simulation"] = simulation
        return object()

    monkeypatch.setattr("sora.cli.build_agent", _fake_build_agent)
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", ["sora", "run"])

    main()

    assert calls["config"] == "agent.yaml"
    assert calls["simulation"] is None
    assert _FakeSession.last_calls["verbose"] is False
    assert _FakeSession.last_calls["color"] is None  # auto: TerminalSession resolves TTY/NO_COLOR
    assert _FakeSession.last_calls["initial_task"] is None
    assert _FakeSession.last_calls["exit_when_idle"] is None
    assert _FakeSession.last_calls["log_file"] is None


def test_main_run_passes_log_file_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sora.cli.build_agent", lambda config_path, **_: object())
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", ["sora", "run", "--log-file", "/tmp/trace.log"])
    main()
    assert _FakeSession.last_calls["log_file"] == "/tmp/trace.log"


def test_main_run_color_flags_pass_tri_state_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sora.cli.build_agent", lambda config_path, **_: object())
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)

    monkeypatch.setattr(sys, "argv", ["sora", "run", "--no-color"])
    main()
    assert _FakeSession.last_calls["color"] is False

    monkeypatch.setattr(sys, "argv", ["sora", "run", "--color"])
    main()
    assert _FakeSession.last_calls["color"] is True


def test_main_run_accepts_config_path_and_verbose_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake_build_agent(config_path: str, *, simulation: object | None = None) -> object:
        calls["config"] = config_path
        return object()

    monkeypatch.setattr("sora.cli.build_agent", _fake_build_agent)
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", ["sora", "run", "other.yaml", "--verbose"])

    main()

    assert calls["config"] == "other.yaml"
    assert _FakeSession.last_calls["verbose"] is True


def test_main_run_passes_task_flag_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sora.cli.build_agent", lambda config_path, **_: object())
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", ["sora", "run", "--task", "do the thing"])

    main()

    assert _FakeSession.last_calls["initial_task"] == "do the thing"


def test_main_run_reads_task_file_and_strips_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_file = tmp_path / "task.txt"
    task_file.write_text("  do the thing from a file  \n")

    monkeypatch.setattr("sora.cli.build_agent", lambda config_path, **_: object())
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", ["sora", "run", "--task-file", str(task_file)])

    main()

    assert _FakeSession.last_calls["initial_task"] == "do the thing from a file"


def test_main_run_rejects_task_and_task_file_together(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sora", "run", "--task", "a", "--task-file", "b"])
    with pytest.raises(SystemExit):
        main()


def test_main_run_rejects_scenario_and_task_together(monkeypatch: pytest.MonkeyPatch) -> None:
    # A scenario delivers its own task through the AUI timeline — combining it with --task/
    # --task-file is a usage error, caught immediately by argparse rather than silently ignored.
    monkeypatch.setattr(sys, "argv", ["sora", "run", "--scenario", "pkg.Scenario", "--task", "a"])
    with pytest.raises(SystemExit):
        main()


def test_main_run_scenario_builds_and_injects_a_simulation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, object] = {}

    class _FakeAreSimulation:
        def __init__(self, scenario: object) -> None:
            self.scenario = scenario

    def _fake_load_scenario(ref: str) -> str:
        calls["scenario_ref"] = ref
        return f"loaded:{ref}"

    monkeypatch.setattr("sora.adapters.are_sim.AreSimulation", _FakeAreSimulation)
    monkeypatch.setattr("sora.adapters.are_sim.load_scenario", _fake_load_scenario)

    def _fake_build_agent(config_path: str, *, simulation: object | None = None) -> object:
        calls["simulation"] = simulation
        return object()

    monkeypatch.setattr("sora.cli.build_agent", _fake_build_agent)
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", ["sora", "run", "--scenario", "pkg.mod.MyScenario"])

    main()

    assert calls["scenario_ref"] == "pkg.mod.MyScenario"
    simulation = calls["simulation"]
    assert isinstance(simulation, _FakeAreSimulation)
    assert simulation.scenario == "loaded:pkg.mod.MyScenario"
    # Feedback printed *before* the (potentially slow, first-import-of-ARE) load_scenario call —
    # otherwise the terminal sits silent until TerminalSession's own startup banner.
    assert "loading scenario 'pkg.mod.MyScenario'" in capsys.readouterr().out


def test_main_run_without_scenario_prints_no_loading_notice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sora.cli.build_agent", lambda config_path, **_: object())
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", ["sora", "run"])

    main()

    assert "loading scenario" not in capsys.readouterr().out


def test_main_run_without_scenario_injects_no_simulation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def _fake_build_agent(config_path: str, *, simulation: object | None = None) -> object:
        calls["simulation"] = simulation
        return object()

    monkeypatch.setattr("sora.cli.build_agent", _fake_build_agent)
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", ["sora", "run"])

    main()

    assert calls["simulation"] is None


def test_main_run_passes_exit_when_idle_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sora.cli.build_agent", lambda config_path, **_: object())
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(sys, "argv", ["sora", "run", "--exit-when-idle", "8"])

    main()

    assert _FakeSession.last_calls["exit_when_idle"] == 8.0


def test_main_run_calls_the_report_hook_with_agent_and_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built_agent = object()
    report_calls: list[tuple[object, object]] = []

    monkeypatch.setattr("sora.cli.build_agent", lambda config_path, **_: built_agent)
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr(
        "sora.cli.import_object",
        lambda path: lambda agent, simulation: report_calls.append((agent, simulation)),
    )
    monkeypatch.setattr(sys, "argv", ["sora", "run", "--report", "pkg.mod.report"])

    main()

    assert report_calls == [(built_agent, None)]


def test_main_run_skips_the_report_hook_when_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def _boom(path: str) -> object:
        nonlocal called
        called = True
        raise AssertionError("import_object should not be called without --report")

    monkeypatch.setattr("sora.cli.build_agent", lambda config_path, **_: object())
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)
    monkeypatch.setattr("sora.cli.import_object", _boom)
    monkeypatch.setattr(sys, "argv", ["sora", "run"])

    main()

    assert called is False


def test_main_run_makes_the_cwd_importable_for_agent_yaml_dotted_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A console-script entry point (unlike `python -m`) doesn't put the current directory on
    # sys.path, so a project-local `strategies.reason: local_strategy.NoOpReason` (the whole point
    # of agent.yaml naming project code, not just sora.* built-ins) used to fail with
    # ModuleNotFoundError. Exercises the real build_agent/import_object/load_yaml path — only
    # TerminalSession is faked, so we don't need a real ReasonStrategy or a running cycle.
    (tmp_path / "local_strategy.py").write_text(
        "class NoOpReason:\n"
        "    async def reason(self, activity, wm, cycle, result):\n"
        "        return result\n"
    )
    (tmp_path / "agent.yaml").write_text(
        "agent:\n"
        "  name: local-project\n"
        "  strategies:\n"
        "    reason: local_strategy.NoOpReason\n"
        "  memory:\n"
        "    semantic: file://./semantic\n"
        "    procedural: file://./procedural\n"
        "    episodic: file://./episodic\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sora", "run"])
    monkeypatch.setattr("sora.cli.TerminalSession", _FakeSession)

    main()  # must not raise ModuleNotFoundError: No module named 'local_strategy'

    assert _FakeSession.last_calls["agent"] is not None


# ---------------------------------------------------------------------------
# main() — `sora init` dispatch
# ---------------------------------------------------------------------------


def test_main_init_scaffolds_the_named_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sora", "init", "my-agent"])

    main()

    project = tmp_path / "my-agent"
    assert (project / "pyproject.toml").is_file()
    assert (project / "agent.yaml").is_file()
    assert (project / "manuals" / "clock.md").is_file()
    assert (project / "clock_tool.py").is_file()
    out = capsys.readouterr().out
    assert "my-agent" in out
    assert "uv run sora run" in out


def test_main_init_reports_an_already_existing_target_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "my-agent").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sora", "init", "my-agent"])

    with pytest.raises(FileExistsError, match="my-agent"):
        main()
