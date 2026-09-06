"""Workspace adapter over the containerised Gaia2 CLI apps.

The apps are ten command-line binaries reachable on a locked-down ``PATH``, each fronted by a
setuid wrapper that runs it as the state-owning user. Every binary answers a ``schema`` subcommand
with a JSON description of its subcommands and options, so operations are *imported* from the
native description rather than authored here (ADR-0003/ADR-0015) — the same provenance channel the
MCP and ARE adapters use, over a different wire.

Two things about this environment shape the design:

* **This environment offers operations and signals, and no observable state.** The setuid wrapper
  exists so the agent user cannot read the state files: the only way to see anything is to run a
  command, which is an *operation*, not an observation. So the manuals declare no observable
  property, and nothing is polled. Two earlier revisions each got half of this wrong — one dropped
  the daemon's notification and kept only a timer poll, leaving the agent perceiving less than the
  sibling harnesses on exactly the scenarios built around environment-initiated change; the other
  kept both, which nominated some zero-argument read per app and published its output *as* observed
  state. That manufactures an affordance the environment does not have, degenerates where no read
  represents the state (``calendar`` yields its tag list), and buys with a subprocess per tool per
  second what the sibling harnesses get by deciding to look.
* **So world change is perceived on one channel: the announcement.** ``/send_notifications``
  arrives as an ``env_notification`` signal on the tool it names — immediate, but rendered prose
  with no ids, absent for the two apps upstream registers no formatter for, and never fired by the
  agent's own writes. It says *that* the world moved, never what it moved to, so an agent that
  needs the detail plans a read. That is what OpenClaw and Hermes do on a wake, and it is why
  ``context_adaptation: replan_on_change`` fits here: shown an announcement and nothing else, a
  revalidation judge can rarely do better than "maybe", so the call is spent to learn what the
  change-gate already established.
* **A declared poll is still supported, and is how a *different* CLI ecosystem would bind a real
  observable** (a WoT poll form does exactly this). It is simply not used here, because these apps
  have nothing a poll would honestly represent. Polling, when declared, runs *off-cycle* in a task
  started at ``focus`` and ``observe()`` returns the last snapshot — a subprocess inside the
  synchronous ``observe()`` would stall the loop an in-flight model call is being awaited on.
  Nothing here caches a phase result.
* **Sibling harnesses drive these apps through a shell.** That is a property of those harnesses,
  not of the environment: grading reads ``events.jsonl``, which each app CLI writes from inside
  itself, so an invocation is recorded however it was issued. Operations are typed here, and the
  subprocess is an implementation detail of ``invoke`` exactly as an in-process call is for the
  in-process adapter.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from sora.manual import (
    Manual,
    ObservablePropertySpecification,
    OperationSpecification,
    SignalSpecification,
)
from sora.perception import Message
from sora.types import ObservableProperty, OperationAck, Signal, diff_values

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence

    from sora.environment import DomainClock, Tool, Workspace, WorkspaceOrigin
    from sora.manual import ToolRecord, WorkspaceRecord
    from sora.perception import SignalSink

log = logging.getLogger(__name__)

GAIA2_CLI_ADAPTER = "gaia2-cli"  # matches WorkspaceOrigin.adapter

# The ten binaries the setuid wrapper allow-lists. Named here only to bound `discover`'s probing to
# things that can possibly be Gaia2 apps; which of them a given scenario actually exposes is
# decided by the container (it deletes the symlinks for apps the scenario does not use), and this
# adapter takes that pruning as given.
GAIA2_CLI_BINARIES: tuple[str, ...] = (
    "calendar",
    "contacts",
    "emails",
    "messages",
    "chats",
    "rent-a-flat",
    "city",
    "cabs",
    "shopping",
    "cloud-drive",
)

# Where the harness's event daemon publishes the scenario's simulated time, for libfaketime to
# interpose into the app CLIs. The agent reads the same file rather than being preloaded — see
# _FaketimeClock.
DEFAULT_FAKETIME_PATH = "/tmp/faketime.rc"
DEFAULT_COMMAND_TIMEOUT = 30.0
DEFAULT_POLL_INTERVAL = 1.0

# Read verbs, as the *ecosystem's* naming convention rather than a per-tool table — see
# `classify_side_effecting` for why this may only ever downgrade a command to a read.
_READ_VERBS = frozenset(
    {
        "list", "get", "search", "read", "lookup", "find", "show", "view",
        "cat", "ls", "tree", "stat", "exists", "info", "head", "tail", "count",
    }
)  # fmt: skip

_CLICK_TO_JSON = {
    "text": "string",
    "integer": "integer",
    "float": "number",
    "boolean": "boolean",
}


@dataclass(frozen=True)
class _FaketimeClock:
    """Domain time as the harness's event daemon publishes it, read from its libfaketime timestamp
    file on every call (the daemon rewrites it as the scenario advances).

    This exists instead of preloading the agent process with libfaketime, which is how the
    environment's own CLIs get simulated time. Preloading works for the clock and breaks everything
    else that reads one: OpenSSL validates the model provider's certificate against it, so a
    scenario set in the past fails every outbound call with "certificate is not yet valid" — and
    the agent's HTTP timeouts, poll period and inference watchdog would be measured on a clock that
    is not wall time either. Reading the file keeps the split exact: domain time is simulated,
    infrastructure time is real, which is the distinction DomainClock exists to draw.

    Falls back to host wall clock when the file is absent — that means nothing is faking time here,
    so host time *is* domain time. It never returns None: this workspace can always say what time
    it is, it is only ever a question of whose clock answers.
    """

    path: str = DEFAULT_FAKETIME_PATH

    def now(self) -> datetime:
        try:
            with open(self.path) as handle:
                raw = handle.read().strip()
        except OSError:
            return datetime.now(UTC)
        if not raw:
            return datetime.now(UTC)
        # The harness writes "%Y-%m-%d %H:%M:%S" (UTC — its init derives it with tz=utc).
        # libfaketime
        # also accepts an "@" epoch form and relative offsets; the epoch form is cheap to honour,
        # and an offset is not a wall time at all, so it degrades to the host clock rather than
        # being guessed at.
        try:
            if raw.startswith("@"):
                return datetime.fromtimestamp(float(raw[1:]), tz=UTC)
            return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        except ValueError:
            log.warning("gaia2-cli: unparseable faketime stamp %r; using host time", raw)
            return datetime.now(UTC)


class CommandRunner(Protocol):
    """Runs one argv and returns ``(returncode, stdout, stderr)``. A seam so the parsing and
    polling logic is testable without a container."""

    # ASYNC109: the timeout belongs in the signature here rather than at each call site — it is
    # the wrapped subprocess's own budget, and an implementation may enforce it however it can.
    async def __call__(
        self,
        argv: list[str],
        timeout: float,  # noqa: ASYNC109
    ) -> tuple[int, str, str]: ...


@dataclass(frozen=True)
class PollSpec:
    """One observable property, produced by running one read command with fixed arguments.

    Which read best represents an app's state is a *domain* judgement (an inbox listing, a calendar
    window), so it is declared in configuration rather than decided here; `default_polls` is only
    the mechanical fallback for an app nobody declared."""

    name: str
    command: str
    params: dict[str, Any] = field(default_factory=dict)


# ── schema -> Manual ────────────────────────────────────────────────────────────────────────────


def classify_side_effecting(command: str) -> bool | None:
    """Whether a CLI command mutates state — ``False`` for a read, ``None`` (unknown) otherwise.

    The apps *do* classify themselves, in the ``write=`` keyword each passes to its own
    ``log_action``, but ``schema`` does not surface it; filling that gap from the adapted
    ecosystem's naming convention is the adapter's job. Deliberately asymmetric: a leading read
    verb downgrades a command to a read, and nothing ever upgrades one to a write, so an unknown
    or newly added command stays ``None`` — which the pre-write checkpoint already treats as a
    write. A read left unknown costs a redundant check; a write called a read skips the guard.
    """
    return False if command.split("-", 1)[0] in _READ_VERBS else None


def _parameters_schema(parameters: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in parameters:
        name = str(param["name"])
        prop: dict[str, Any] = {
            "type": _CLICK_TO_JSON.get(str(param.get("type", "text")), "string"),
            "description": param.get("description") or "",
        }
        if param.get("required"):
            required.append(name)
        elif "default" in param and param["default"] is not None:
            # A required option has no default by construction; click >= 8.2 nevertheless hands
            # `build_schema` an UNSET sentinel that reaches the JSON as an anonymous object's repr,
            # so the `required` branch above must win rather than this one.
            prop["default"] = param["default"]
        properties[name] = prop
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def schema_to_manual(
    binary: str,
    schema: Sequence[Mapping[str, Any]],
    *,
    app_class: str,
    option_flags: Mapping[str, Mapping[str, str]] | None = None,
    polls: Sequence[PollSpec] = (),
) -> Manual:
    """One app's ``schema`` output as a Manual.

    Operations are named by ``oracle_function`` — the Python callback name, which is what the app
    writes to ``events.jsonl`` and what the graded oracle is matched on — not by the kebab command
    the shell would type. That also makes the operation names identical to the in-process adapter's,
    so a plan, condition or manual written against one harness reads the same on the other. The
    kebab command survives in ``metadata['commands']`` so a reconnect can rebuild argv without
    re-running ``schema``.

    ``completion_signal`` is never set: every notification formatter upstream is registered on a
    ``hidden=True`` environment function, and ``schema`` omits hidden commands, so no operation the
    agent can invoke has a machine-readable completion signal to declare.

    **What the manual declares is exactly what this environment offers, and no more.** These apps
    expose no observable state: their state files are readable only by the ``gaia2`` user, so the
    only way to see anything is to run a command — an *operation*, not an observation. An earlier
    revision papered over that by nominating some zero-argument read per app and polling it on a
    timer, which manufactures an observable the environment does not have and lands on a degenerate
    one where no read represents the state (``calendar`` yields its tag list that way). So
    ``observable_properties`` is empty unless a poll was *declared* — the adapter never invents one
    — and each signal is declared only where it can actually fire: ``env_notification`` for an app
    the ecosystem registers a notification formatter for, ``state_changed`` only where there is a
    polled snapshot to diff. A manual that promises perception the tool cannot deliver is worse
    than one that admits the limit, because a plan written against the promise fails silently.
    """
    operations: list[OperationSpecification] = []
    commands: dict[str, str] = {}
    options: dict[str, dict[str, str]] = {}
    for entry in schema:
        command = str(entry["command"])
        name = str(entry.get("oracle_function") or entry.get("function") or command)
        commands[name] = command
        renamed = {
            param: flag
            for param, flag in ((option_flags or {}).get(command) or {}).items()
            if flag != f"--{param.replace('_', '-')}"
        }
        if renamed:
            options[name] = renamed
        operations.append(
            OperationSpecification(
                name=name,
                description=str(entry.get("description") or ""),
                parameters=_parameters_schema(entry.get("parameters") or ()),
                # The CLI prints JSON but never describes its shape, so a `$from` path into a
                # result is left to the reference resolver rather than guessed at here.
                returns=None,
                side_effecting=classify_side_effecting(command),
            )
        )
    return Manual(
        id=app_class,
        metadata={
            "source": GAIA2_CLI_ADAPTER,
            "binary": binary,
            "app": app_class,
            "commands": commands,
            # Only the parameters whose flag is not the kebab of their name — see build_argv.
            "options": options,
        },
        description=f"Gaia2 {app_class}, driven through the {binary} CLI",
        observable_properties=[
            ObservablePropertySpecification(name=poll.name, description="", schema={})
            for poll in polls
        ],
        signals=_signal_specs(binary, polls),
        operations=operations,
        raw_text=None,
    )


def _signal_specs(binary: str, polls: Sequence[PollSpec]) -> list[SignalSpecification]:
    """The signals this tool can actually emit — never a signal it merely might.

    ``env_notification`` is declared only for an app the ecosystem registers a notification
    formatter for: two of the ten (``contacts``, ``cloud-drive``) register none, so nothing will
    ever announce a change to them and saying otherwise would invite a `watch` that can never fire.
    ``state_changed`` is derived here by diffing a polled snapshot, so it exists only where a poll
    was declared. When the registry is not importable (outside the container) the formatter set is
    unknown, and the declaration fails OPEN — a watch that never fires costs nothing, while an
    undeclared signal is one the planner cannot author against at all."""
    specs: list[SignalSpecification] = []
    formatters = _upstream_formatters(binary)
    if formatters is None or formatters:
        specs.append(
            SignalSpecification(
                name="env_notification",
                description=(
                    "The environment announced that this app's world moved. Prose only: it carries "
                    "no item identifiers and no structured changes, so it says THAT something "
                    "happened, never what the new state is. Read the app to find out."
                ),
                schema={
                    "type": "object",
                    "properties": {"app": {"type": "string"}, "text": {"type": "string"}},
                },
            )
        )
    if polls:
        specs.append(SignalSpecification(name="state_changed", description="", schema={}))
    return specs


# ── argv ────────────────────────────────────────────────────────────────────────────────────────


def build_argv(
    executable: str,
    command: str,
    params: Mapping[str, Any],
    options: Mapping[str, str] | None = None,
) -> list[str]:
    """``{'folder_name': 'INBOX'}`` -> ``--folder-name INBOX``, mirroring the conventions the
    daemon's own CLI executor uses: a list or dict is JSON-encoded (the apps parse those options
    with ``json.loads``), a true bool is a bare flag, and a false or absent one is simply omitted.

    ``options`` maps a parameter name to its actual flag, for the options whose flag is *not* the
    kebab-case of the name. ``schema`` reports each parameter by the callback's variable name, and
    an app is free to declare a different flag for it: ``calendar get-events`` takes
    ``--start-date`` into ``start_datetime``, and every ``cloud-drive`` path option is declared
    ``--path``. Twelve of 273 options across the apps — but they cover all of ``cloud-drive`` and
    calendar's range query, so the kebab rule alone leaves those unusable. A missing entry falls
    back to the kebab rule, which is right for the other 261.
    """
    argv = [executable, command]
    for name, value in params.items():
        if value is None or value is False:
            continue
        option = (options or {}).get(str(name)) or f"--{str(name).replace('_', '-')}"
        if value is True:
            argv.append(option)
            continue
        argv.append(option)
        argv.append(json.dumps(value) if isinstance(value, (list, dict)) else str(value))
    return argv


async def _run_subprocess(
    argv: list[str],
    timeout: float,  # noqa: ASYNC109
) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timed out after {timeout}s"
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


def _parse_stdout(text: str) -> Any:
    """The apps print JSON, but a command that prints nothing (or prints a bare line) is not a
    failure — it succeeded and said little, so return the text rather than erroring."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


# ── environment notifications ───────────────────────────────────────────────────────────────────

_NOTIFICATION_PREFIX = "[Notification] "
_NOTIFICATIONS_PREFIX = "[Notifications]"


def parse_notification(text: str) -> list[tuple[str, str]]:
    """Split a daemon environment notification into ``(cli name, rendered message)`` pairs.

    The daemon sends either ``[Notification] <cli>: <message>`` or, when several fire in one turn,
    a ``[Notifications]`` bundle of ``- <cli>: <message>`` lines, labelling each with the
    agent-visible CLI name. Only the first ``": "`` separates the label, so a rendered message may
    contain its own colon.

    A line carrying no label yields nothing. The label is the only thing tying a notification to a
    tool, and a signal pushed on a guess would name the wrong source — worse than the silence,
    because a watch would fire against an app that did not move.
    """
    body = text.strip()
    if body.startswith(_NOTIFICATION_PREFIX):
        lines = [body[len(_NOTIFICATION_PREFIX) :]]
    elif body.startswith(_NOTIFICATIONS_PREFIX):
        lines = [
            line[2:]
            for line in body[len(_NOTIFICATIONS_PREFIX) :].strip().splitlines()
            if line.strip().startswith("- ")
        ]
    else:
        # Unprefixed, so the daemon's framing changed. The label is what matters; try to read it
        # rather than dropping a real world-change over a missing prefix.
        lines = [body]
    out: list[tuple[str, str]] = []
    for line in lines:
        label, sep, message = line.strip().partition(": ")
        if sep and label.strip() and message.strip():
            out.append((label.strip(), message.strip()))
    return out


# ── Tool / Workspace / Adapter ──────────────────────────────────────────────────────────────────


class _Gaia2CliTool:
    """One live tool over one app binary."""

    def __init__(
        self,
        *,
        tool_id: str,
        manual: Manual,
        executable: str,
        commands: Mapping[str, str],
        options: Mapping[str, Mapping[str, str]],
        polls: Sequence[PollSpec],
        runner: CommandRunner,
        timeout: float,
        poll_interval: float | None,
    ) -> None:
        self.id = tool_id
        self.manual = manual
        self.address: str | None = None
        self._executable = executable
        self._commands = dict(commands)
        self._options = {op: dict(flags) for op, flags in options.items()}
        self._polls = list(polls)
        self._runner = runner
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._sink: SignalSink | None = None
        self._snapshot: dict[str, Any] = {}
        self._poll_errors: dict[str, str] = {}  # last failure per property, to log on change only
        self._task: asyncio.Task[None] | None = None

    async def invoke(self, operation_name: str, **params: Any) -> OperationAck:
        command = self._commands.get(operation_name)
        if command is None:
            return OperationAck(ok=False, result=f"unknown operation {operation_name!r}")
        code, out, err = await self._runner(
            build_argv(self._executable, command, params, self._options.get(operation_name)),
            self._timeout,
        )
        if code != 0:
            # An app rejecting a call is a failed ack, not a runtime crash: the plan gets to see
            # why and re-plan. stderr is where the apps put their reasons (`cli_error`).
            return OperationAck(ok=False, result=(err.strip() or out.strip() or f"exit {code}"))
        return OperationAck(ok=True, result=_parse_stdout(out))

    async def focus(self, sink: SignalSink) -> None:
        self._sink = sink
        # Establish the baseline synchronously so the very first observe() after focusing is not
        # empty, then hand the cadence to a background task — see the module docstring.
        await self._read_snapshot()
        # No declared poll means no observable property, so there is nothing for a loop to read;
        # a timer that subprocesses nothing every second is pure cost against the scenario clock.
        if self._polls and self._poll_interval is not None and self._task is None:
            self._task = asyncio.create_task(self._poll_loop())

    async def unfocus(self) -> None:
        self._sink = None
        self._snapshot = {}
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def observe(self) -> list[ObservableProperty]:
        return [ObservableProperty(name=k, value=v) for k, v in self._snapshot.items()]

    def note_env_notification(self, message: str) -> None:
        """Push an environment notification as a Signal — the fast path for a world change.

        The daemon announces environment-initiated actions the moment they happen, which is the
        only channel that reports them *when* they occur; the poll below reports the same world a
        snapshot later, and only within the slice a `PollSpec` happens to read. Dropping the
        announcement (as this adapter first did) leaves the agent perceiving strictly less than the
        harness offers, on exactly the scenarios that turn on environment-initiated change.

        Deliberately **not** ``state_changed``. That signal carries a structured ``changes`` list
        diffed from the polled snapshot, so a consumer can read paths and ids off it; this one
        carries the daemon's rendered prose ("New calendar event added by Bob") and no structure at
        all. Publishing it under the same name would claim a precision it does not have. Note the
        consequence, which is intended: `watch_matches` treats a signal with no changes as matching
        every watch on the source, so this opens the gate and lets the condition's own judge decide
        — the announcement, not the evidence.
        """
        if self._sink is None:
            return  # unfocused: nothing is attending this tool, so there is no sink to push into
        self._sink.push(
            self.id,
            Signal(
                "env_notification",
                {"app": self.manual.metadata.get("app"), "text": message},
            ),
        )

    async def poll_once(self) -> None:
        """Re-read every declared property and, when something moved, push the diff.

        Thin like the in-process adapter's: the values themselves are published as observable
        properties, and the signal carries only *where* they moved — the one thing a replace-by-key
        snapshot cannot express."""
        previous = self._snapshot
        current = await self._read_snapshot()
        if self._sink is not None and current != previous:
            self._sink.push(
                self.id,
                Signal(
                    "state_changed",
                    {
                        "app": self.manual.metadata.get("app"),
                        "changes": diff_values(previous, current),
                    },
                ),
            )

    def _flags_for(self, command: str) -> Mapping[str, str] | None:
        operation = next((op for op, c in self._commands.items() if c == command), None)
        return self._options.get(operation or "")

    async def _read_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        for poll in self._polls:
            code, out, err = await self._runner(
                build_argv(
                    self._executable, poll.command, poll.params, self._flags_for(poll.command)
                ),
                self._timeout,
            )
            if code != 0:
                # A poll that fails leaves the property absent rather than poisoning the snapshot
                # with an error string a plan would then try to path into. Logged only when the
                # failure *changes*: several apps exit non-zero for a perfectly ordinary empty
                # state (`cabs get-current-ride-status` -> "No ride ordered."), which at the poll
                # cadence would otherwise bury every real warning in the run.
                message = err.strip() or out.strip() or f"exit {code}"
                if self._poll_errors.get(poll.name) != message:
                    self._poll_errors[poll.name] = message
                    log.warning(
                        "gaia2-cli: poll %s on %s failed: %s", poll.command, self.id, message
                    )
                continue
            self._poll_errors.pop(poll.name, None)
            snapshot[poll.name] = _parse_stdout(out)
        self._snapshot = snapshot
        return snapshot

    async def _poll_loop(self) -> None:
        assert self._poll_interval is not None
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # a transient app/subprocess failure must not kill perception
                log.exception("gaia2-cli: poll loop error on %s", self.id)


class _Gaia2CliWorkspace:
    def __init__(
        self,
        ws_id: str,
        origin: WorkspaceOrigin,
        tools: list[Tool],
        clock: DomainClock | None = None,
    ) -> None:
        self.id = ws_id
        self.origin = origin
        # Domain time is the scenario's simulated time, read from the file the harness's daemon
        # publishes it in — not this process's wall clock, and not a libfaketime preload. See
        # _FaketimeClock for why the agent reads the clock rather than running under it.
        self.clock: DomainClock | None = clock or _FaketimeClock()
        self._tools = tools

    def tools(self) -> list[Tool]:
        return self._tools

    def note_env_notification(self, text: str) -> None:
        """Route a daemon notification to the tool it names, as a Signal.

        Dropped rather than broadcast when the label names an app this workspace does not hold:
        scenario pruning means the joined set is a subset of the ten, so a notification for a
        pruned app is expected, not an error.
        """
        # Narrowed rather than cast: `_tools` is typed to the `Tool` protocol, and only this
        # adapter's own tool knows how to turn prose into a signal on itself.
        by_binary = {
            tool.manual.metadata.get("binary"): tool
            for tool in self._tools
            if isinstance(tool, _Gaia2CliTool)
        }
        for label, message in parse_notification(text):
            tool = by_binary.get(label)
            if tool is None:
                log.debug("gaia2-cli: notification for an app not joined here: %s", label)
                continue
            tool.note_env_notification(message)

    async def close(self) -> None:
        for tool in self._tools:
            await tool.unfocus()


class Gaia2CliWorkspaceAdapter:
    """Imports whichever Gaia2 app binaries this scenario left on the sandbox PATH.

    ``schemas``/``app_classes``/``binaries`` are injectable so the whole adapter is testable without
    a container; left unset they are probed from the filesystem, from ``<binary> schema``, and from
    the upstream app registry.
    """

    name = GAIA2_CLI_ADAPTER

    def __init__(
        self,
        *,
        workspace_id: str,
        origin: WorkspaceOrigin,
        binaries: Sequence[str] | None = None,
        polls: Mapping[str, Sequence[PollSpec]] | None = None,
        schemas: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        app_classes: Mapping[str, str] | None = None,
        option_flags: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None,
        runner: CommandRunner | None = None,
        timeout: float = DEFAULT_COMMAND_TIMEOUT,
        poll_interval: float | None = DEFAULT_POLL_INTERVAL,
        faketime_path: str = DEFAULT_FAKETIME_PATH,
    ) -> None:
        self._workspace_id = workspace_id
        self._origin = origin
        self._binaries = list(binaries) if binaries is not None else None
        self._polls = {k: list(v) for k, v in (polls or {}).items()}
        self._schemas = schemas
        self._app_classes = app_classes
        self._option_flags_by_binary = option_flags
        self._runner: CommandRunner = runner or _run_subprocess
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._clock = _FaketimeClock(faketime_path)

    async def discover(self) -> list[Workspace]:
        tools: list[Tool] = []
        for binary in self._present_binaries():
            schema = await self._schema_for(binary)
            if schema is None:
                continue
            tools.append(self._build_tool(binary, schema))
        return [_Gaia2CliWorkspace(self._workspace_id, self._origin, tools, self._clock)]

    async def connect(
        self,
        workspace_record: WorkspaceRecord,
        tool_records: list[ToolRecord],
        manuals: dict[str, Manual],
    ) -> Workspace:
        # No session to re-establish — a binary is reachable or it is not — and the manual already
        # carries the command table, so nothing needs re-probing.
        tools: list[Tool] = []
        for record in tool_records:
            manual = manuals.get(record.manual_id)
            if manual is None:
                continue
            binary = str(manual.metadata.get("binary", record.manual_id))
            tools.append(self._tool_from_manual(record.id, binary, manual))
        return _Gaia2CliWorkspace(workspace_record.id, workspace_record.origin, tools, self._clock)

    def _present_binaries(self) -> Iterable[str]:
        if self._binaries is not None:
            return self._binaries
        from pathlib import Path

        root = Path(self._origin.address)
        return [b for b in GAIA2_CLI_BINARIES if (root / b).exists()]

    async def _schema_for(self, binary: str) -> Sequence[Mapping[str, Any]] | None:
        if self._schemas is not None:
            return self._schemas.get(binary)
        code, out, err = await self._runner([self._executable(binary), "schema"], self._timeout)
        if code != 0:
            log.warning("gaia2-cli: %s schema failed (%s): %s", binary, code, err.strip())
            return None
        parsed = _parse_stdout(out)
        if not isinstance(parsed, list):
            log.warning("gaia2-cli: %s schema was not a list", binary)
            return None
        return parsed

    def _build_tool(self, binary: str, schema: Sequence[Mapping[str, Any]]) -> Tool:
        manual = schema_to_manual(
            binary,
            schema,
            app_class=self._app_class(binary),
            option_flags=self._option_flags(binary),
            polls=self._polls.get(binary, ()),
        )
        return self._tool_from_manual(self._tool_id(binary), binary, manual, schema=schema)

    def _tool_from_manual(
        self,
        tool_id: str,
        binary: str,
        manual: Manual,
        *,
        schema: Sequence[Mapping[str, Any]] | None = None,
    ) -> Tool:
        # No fallback: an app nobody declared a poll for publishes no observable property. These
        # CLIs expose state only by running a command, so nominating one and calling its output an
        # observation invents an affordance the environment does not have (see schema_to_manual).
        # A *declared* poll still survives a reconnect, rebuilt from the manual that recorded it.
        polls = self._polls.get(binary)
        if polls is None:
            polls = [
                PollSpec(name=spec.name, command=_command_for(manual, spec.name))
                for spec in manual.observable_properties
            ]
        return _Gaia2CliTool(
            tool_id=tool_id,
            manual=manual,
            executable=self._executable(binary),
            commands=manual.metadata.get("commands", {}),
            options=manual.metadata.get("options", {}),
            polls=polls,
            runner=self._runner,
            timeout=self._timeout,
            poll_interval=self._poll_interval,
        )

    def _option_flags(self, binary: str) -> Mapping[str, Mapping[str, str]]:
        if self._option_flags_by_binary is not None:
            return self._option_flags_by_binary.get(binary, {})
        return _upstream_option_flags(binary)

    def _app_class(self, binary: str) -> str:
        if self._app_classes is not None and binary in self._app_classes:
            return self._app_classes[binary]
        return _upstream_app_class(binary) or binary

    def _executable(self, binary: str) -> str:
        return f"{self._origin.address.rstrip('/')}/{binary}"

    def _tool_id(self, binary: str) -> str:
        # ADR-0014: adapter-derived, deterministic, globally unique.
        return self._executable(binary)


def _command_for(manual: Manual, operation: str) -> str:
    commands: Mapping[str, str] = manual.metadata.get("commands", {})
    return commands.get(operation, operation.replace("_", "-"))


def _upstream_option_flags(binary: str) -> dict[str, dict[str, str]]:
    """``{command: {parameter name: actual long flag}}``, read off the app's own click group.

    The ``schema`` output reports a parameter by the callback's variable name and never says which
    flag sets it, so an option declared under a different flag is unreachable from the schema alone.
    The apps are importable in the environment they run in, and this is the same introspection the
    ecosystem's own environment-action executor does to recognise boolean flags. Not importable
    (a host without the package, a unit test) -> empty, and build_argv falls back to the kebab rule.
    """
    module = _upstream_module(binary)
    if module is None:
        return {}
    try:
        import click
    except Exception:
        return {}
    group = getattr(module, "cli", None)
    if group is None:
        return {}
    flags: dict[str, dict[str, str]] = {}
    for name, command in getattr(group, "commands", {}).items():
        per_command: dict[str, str] = {}
        for param in getattr(command, "params", ()):
            if not isinstance(param, click.Option):
                continue
            long_opts = [opt for opt in param.opts if opt.startswith("--")]
            if long_opts and param.name:
                per_command[param.name] = long_opts[0]
        if per_command:
            flags[name] = per_command
    return flags


def _upstream_module(binary: str) -> Any | None:
    try:
        import importlib

        from gaia2_cli.app_registry import APP_REGISTRY
    except Exception:
        return None
    for entry in APP_REGISTRY:
        if entry.get("cli") == binary or entry.get("agent_cli") == binary:
            try:
                return importlib.import_module(str(entry["module"]))
            except Exception:
                return None
    return None


def _upstream_formatters(binary: str) -> set[str] | None:
    """The ENV functions the ecosystem renders a notification for, or ``None`` if its registry is
    not importable (outside the container). An app with an empty set can never be announced."""
    try:
        from gaia2_cli.app_registry import APP_REGISTRY
    except Exception:
        return None
    for entry in APP_REGISTRY:
        if entry.get("cli") == binary or entry.get("agent_cli") == binary:
            return set(entry.get("formatters") or ())
    return set()


def _upstream_app_class(binary: str) -> str | None:
    """The app's canonical Gaia2 class name, read from the ecosystem's own registry when it is
    importable (it is, inside the container). Falling back to the binary name only costs a less
    recognisable ``Manual.id``; nothing about grading depends on it, since the events the judge
    reads are written by the app CLI itself."""
    try:
        from gaia2_cli.app_registry import APP_REGISTRY
    except Exception:
        return None
    for entry in APP_REGISTRY:
        if entry.get("cli") == binary or entry.get("agent_cli") == binary:
            return str(entry["canonical"])
    return None


# ── Transport ───────────────────────────────────────────────────────────────────────────────────


class Gaia2CliTransport:
    """``MessageTransport`` over the worker's socket to the container adapter.

    ``receive`` yields the user turns the daemon delivered; ``send`` emits the final response for
    the run in flight, which the container adapter turns into the synthetic
    ``AgentUserInterface.send_message_to_user`` event that *is* the daemon's turn boundary. An empty
    reply is still emitted: silence ends the scenario as "no turn boundary detected".

    An environment notification is deliberately **not** a message: delivering the rendered text as
    a user message would attribute it to someone who did not speak. It is a *signal* — see
    ``_Gaia2CliTool.note_env_notification``, which is where one lands. It is also the agent's only
    channel for world change here, and it announces rather than describes: no ids, nothing for a
    watch to bind against, nothing at all for the two apps upstream registers no formatter for, and
    never a word about the agent's own writes. Reading the detail is a plan step, not a percept.
    """

    def __init__(self, emit: Callable[[dict[str, Any]], None] | None = None) -> None:
        # Public and settable: bootstrap builds this from config alone, and whoever hosts the agent
        # attaches the socket writer afterwards.
        self.emit = emit
        self._inbox: asyncio.Queue[Message] = asyncio.Queue()
        self._run_id: str | None = None
        self.sent: list[tuple[str, dict[str, Any]]] = []  # outbound log, for tests/inspection
        self.notifications: list[str] = []  # inbound env notifications, for traces

    def submit_user_message(self, text: str, *, run_id: str) -> None:
        import time

        self._run_id = run_id
        self._inbox.put_nowait(
            Message(sender="user", content={"text": text}, received_at=time.time())
        )

    def note_env_notification(self, text: str) -> None:
        self.notifications.append(text)

    async def send(self, to: str, content: dict[str, Any]) -> None:
        self.sent.append((to, content))
        if self.emit is not None:
            self.emit(
                {
                    "type": "response",
                    "run_id": self._run_id or "",
                    "state": "final",
                    "message": str(content.get("text", "")),
                }
            )

    def receive(self) -> AsyncIterator[Message]:
        async def _drain() -> AsyncIterator[Message]:
            for _ in range(self._inbox.qsize()):
                yield self._inbox.get_nowait()

        return _drain()
