"""Unit tests for the gaia2-cli workspace adapter — schema synthesis, argv building, the
read/write classifier, polling, and the worker-socket transport. All against fakes and fixture
JSON; nothing here starts a container."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
from typing import Any

import pytest

from sora.adapters.gaia2_cli import (
    GAIA2_CLI_ADAPTER,
    Gaia2CliTransport,
    Gaia2CliWorkspaceAdapter,
    PollSpec,
    _FaketimeClock,
    build_argv,
    classify_side_effecting,
    parse_notification,
    schema_to_manual,
)
from sora.environment import WorkspaceOrigin
from sora.types import Signal

# A trimmed but faithful slice of `emails schema` / `calendar schema` output (click 8.1 shapes).
EMAILS_SCHEMA: list[dict[str, Any]] = [
    {
        "command": "list-emails",
        "function": "list_emails",
        "oracle_function": "list_emails",
        "description": "List emails in a folder with pagination.",
        "parameters": [
            {
                "name": "folder_name",
                "type": "text",
                "description": "Folder.",
                "required": False,
                "default": "INBOX",
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Max.",
                "required": False,
                "default": 10,
            },
        ],
    },
    {
        "command": "send-email",
        "function": "send_email",
        "oracle_function": "send_email",
        "description": "Send an email.",
        "parameters": [
            {"name": "recipients", "type": "text", "description": "JSON list.", "required": True},
            {
                "name": "subject",
                "type": "text",
                "description": "Subject.",
                "required": False,
                "default": "",
            },
            {"name": "cc", "type": "text", "description": "JSON list.", "required": False},
        ],
    },
    {
        "command": "delete-email",
        "function": "delete_email",
        "oracle_function": "delete_email",
        "description": "Delete an email.",
        "parameters": [
            {"name": "email_id", "type": "text", "description": "Id.", "required": True},
        ],
    },
]

CALENDAR_SCHEMA: list[dict[str, Any]] = [
    {
        "command": "add-event",
        "function": "add_event",
        # The kebab command and the graded oracle function disagree here — the case that matters.
        "oracle_function": "add_calendar_event",
        "description": "Add a calendar event.",
        "parameters": [
            {"name": "title", "type": "text", "description": "Title.", "required": False},
            {"name": "attendees", "type": "text", "description": "JSON list.", "required": False},
        ],
    },
    {
        "command": "today-events",
        "function": "today_events",
        "oracle_function": "read_today_calendar_events",
        "description": "Today's events.",
        "parameters": [],
    },
    {
        "command": "get-all-tags",
        "function": "get_all_tags",
        "oracle_function": "get_all_tags",
        "description": "All tags.",
        "parameters": [],
    },
]


# ── schema -> Manual ────────────────────────────────────────────────────────────────────────────


def test_schema_to_manual_names_operations_by_oracle_function() -> None:
    """The graded name, not the kebab command: `log_action` writes the callback's Python name and
    the judge matches the oracle on it, so that is the name a plan must be able to state."""
    manual = schema_to_manual("calendar", CALENDAR_SCHEMA, app_class="CalendarApp")
    assert [op.name for op in manual.operations] == [
        "add_calendar_event",
        "read_today_calendar_events",
        "get_all_tags",
    ]
    assert manual.id == "CalendarApp"
    assert manual.metadata["source"] == GAIA2_CLI_ADAPTER
    assert manual.metadata["binary"] == "calendar"
    # The kebab command survives, keyed by operation name, so build_argv can recover it after a
    # reconnect that never re-runs `schema`.
    assert manual.metadata["commands"]["add_calendar_event"] == "add-event"


def test_schema_to_manual_builds_json_schema_parameters() -> None:
    manual = schema_to_manual("emails", EMAILS_SCHEMA, app_class="EmailClientV2")
    send = manual.operation("send_email")
    assert send is not None
    assert send.parameters["properties"]["recipients"]["type"] == "string"
    assert send.parameters["required"] == ["recipients"]
    listing = manual.operation("list_emails")
    assert listing is not None
    assert listing.parameters["properties"]["limit"]["type"] == "integer"
    assert listing.parameters["properties"]["folder_name"]["default"] == "INBOX"
    assert "required" not in listing.parameters


def test_schema_to_manual_leaves_completion_signal_unset() -> None:
    """Every notification formatter is registered on a `hidden=True` ENV function, and `schema`
    omits hidden commands — so no agent-invokable operation can declare a completion signal."""
    manual = schema_to_manual("emails", EMAILS_SCHEMA, app_class="EmailClientV2")
    assert all(op.completion_signal is None for op in manual.operations)


def test_schema_to_manual_ignores_a_default_on_a_required_parameter() -> None:
    """click >= 8.2 hands `build_schema` an UNSET sentinel for a required option, which reaches the
    JSON as the repr of an anonymous object. A required parameter has no default by definition."""
    schema = [
        {
            "command": "get-email-by-id",
            "function": "get_email_by_id",
            "oracle_function": "get_email_by_id",
            "description": "",
            "parameters": [
                {
                    "name": "email_id",
                    "type": "text",
                    "description": "",
                    "required": True,
                    "default": "<object object at 0x102dec6f0>",
                },
            ],
        }
    ]
    manual = schema_to_manual("emails", schema, app_class="EmailClientV2")
    op = manual.operation("get_email_by_id")
    assert op is not None
    assert "default" not in op.parameters["properties"]["email_id"]


def test_schema_to_manual_marks_reads_and_leaves_writes_unknown() -> None:
    manual = schema_to_manual("emails", EMAILS_SCHEMA, app_class="EmailClientV2")
    assert manual.operation("list_emails").side_effecting is False  # type: ignore[union-attr]
    assert manual.operation("send_email").side_effecting is None  # type: ignore[union-attr]


# ── the read/write classifier ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "command", ["list-emails", "get-event", "search-contacts", "read-conversation"]
)
def test_classify_side_effecting_reads(command: str) -> None:
    assert classify_side_effecting(command) is False


@pytest.mark.parametrize("command", ["send-email", "add-event", "checkout", "today-events"])
def test_classify_side_effecting_is_unknown_for_everything_else(command: str) -> None:
    """Never `True`: the classifier can only ever *downgrade* a command to a read on an explicit
    verb, so an unrecognised one stays UNKNOWN — which the reconsideration checkpoint already
    treats as a write."""
    assert classify_side_effecting(command) is None


def _upstream_write_flags(apps_dir: pathlib.Path) -> dict[str, bool]:
    """Every *visible* command of every gaia2-cli app, mapped to whether its own
    ``log_action(..., write=...)`` call records a write. Read off the source rather than executed:
    the flag is a literal keyword at each call site, and running the commands would need container
    state. Hidden commands are excluded because ``schema`` omits them, so the agent never sees
    them."""
    import ast

    flags: dict[str, bool] = {}
    for module_path in sorted(apps_dir.glob("*.py")):
        if module_path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(module_path.read_text())):
            if not isinstance(node, ast.FunctionDef):
                continue
            command: str | None = None
            hidden = False
            for dec in node.decorator_list:
                if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                    continue
                if dec.func.attr != "command":
                    continue
                command = (
                    str(dec.args[0].value)
                    if dec.args and isinstance(dec.args[0], ast.Constant)
                    else node.name
                )
                hidden = any(
                    kw.arg == "hidden" and isinstance(kw.value, ast.Constant) and kw.value.value
                    for kw in dec.keywords
                )
            if command is None or hidden:
                continue
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "log_action":
                    write = any(
                        kw.arg == "write" and isinstance(kw.value, ast.Constant) and kw.value.value
                        for kw in call.keywords
                    )
                    flags[command] = flags.get(command, False) or write
    return flags


def test_classify_side_effecting_agrees_with_upstream_write_flags() -> None:
    """Drift guard, run against a local gaia2-cli checkout named by ``SORA_GAIA2_CLI_DIR`` — set it
    in the repo-root ``.env``, which ``tests/conftest.py`` loads (skipped without one — the shipped
    container has only compiled modules, and CI has no checkout).

    Soundness only, and that is the property that matters: a write classified as a read would skip
    the pre-write checkpoint. A read left UNKNOWN merely buys a redundant check."""
    checkout = os.environ.get("SORA_GAIA2_CLI_DIR")
    if not checkout:
        pytest.skip("set SORA_GAIA2_CLI_DIR (see .env.example) to run the drift guard")
    # Expanded, because the natural way to write this in `.env` is `~/...` and the loader does no
    # shell expansion — an unexpanded `~` would fail the `is_dir()` below and skip as if the
    # variable were unset, which reads as "no checkout" rather than "wrong path".
    apps_dir = pathlib.Path(checkout).expanduser() / "cli" / "gaia2_cli" / "apps"
    if not apps_dir.is_dir():
        pytest.skip(f"no gaia2-cli app sources under {apps_dir}")
    flags = _upstream_write_flags(apps_dir)
    assert flags, "found no commands to check — the scan, not the classifier, is broken"
    for command, writes in flags.items():
        if classify_side_effecting(command) is False:
            assert not writes, f"{command!r} classified read but upstream logs write=True"


# ── argv ────────────────────────────────────────────────────────────────────────────────────────


def test_build_argv_kebabs_the_option_names() -> None:
    assert build_argv("/home/agent/bin/emails", "list-emails", {"folder_name": "INBOX"}) == [
        "/home/agent/bin/emails",
        "list-emails",
        "--folder-name",
        "INBOX",
    ]


def test_build_argv_json_encodes_collections() -> None:
    argv = build_argv("emails", "send-email", {"recipients": ["a@b.c", "d@e.f"]})
    assert argv[:3] == ["emails", "send-email", "--recipients"]
    assert json.loads(argv[3]) == ["a@b.c", "d@e.f"]


def test_build_argv_emits_a_bare_flag_for_true_and_drops_false_and_none() -> None:
    argv = build_argv(
        "cloud-drive", "rm", {"path": "/x", "recursive": True, "force": False, "owner": None}
    )
    assert argv == ["cloud-drive", "rm", "--path", "/x", "--recursive"]


def test_build_argv_uses_the_declared_flag_when_it_is_not_the_kebab_of_the_name() -> None:
    """`schema` names a parameter after the callback variable and never says which flag sets it.
    `calendar get-events` declares `--start-date` into `start_datetime`; every `cloud-drive` path
    option is declared `--path`. Without the map those commands are simply unusable."""
    argv = build_argv(
        "calendar",
        "get-events",
        {"start_datetime": "2020-01-01 00:00:00", "limit": 5},
        {"start_datetime": "--start-date"},
    )
    assert argv == ["calendar", "get-events", "--start-date", "2020-01-01 00:00:00", "--limit", "5"]


def test_schema_to_manual_records_only_the_renamed_flags() -> None:
    manual = schema_to_manual(
        "calendar",
        CALENDAR_SCHEMA,
        app_class="CalendarApp",
        option_flags={"add-event": {"title": "--title", "attendees": "--with"}},
    )
    # `title` maps to exactly the kebab rule, so it is not worth carrying; `attendees` is.
    assert manual.metadata["options"] == {"add_calendar_event": {"attendees": "--with"}}


def test_a_renamed_flag_reaches_the_invoked_argv() -> None:
    runner = _FakeRunner({"get-events": (0, "[]", "")})
    adapter = Gaia2CliWorkspaceAdapter(
        workspace_id="gaia2",
        origin=WorkspaceOrigin(adapter=GAIA2_CLI_ADAPTER, address="/home/agent/bin"),
        binaries=["calendar"],
        schemas={"calendar": CALENDAR_SCHEMA},
        app_classes={"calendar": "CalendarApp"},
        option_flags={"calendar": {"add-event": {"title": "--subject"}}},
        runner=runner,
        poll_interval=None,
    )

    async def _run() -> None:
        tool = (await adapter.discover())[0].tools()[0]
        await tool.invoke("add_calendar_event", title="Standup")

    asyncio.run(_run())
    assert runner.calls[-1][1:] == ["add-event", "--subject", "Standup"]


def test_build_argv_stringifies_scalars() -> None:
    assert build_argv("emails", "list-emails", {"limit": 10}) == [
        "emails",
        "list-emails",
        "--limit",
        "10",
    ]


# ── what the manual declares ────────────────────────────────────────────────────────────────────


def test_an_app_with_no_declared_poll_publishes_no_observable_property() -> None:
    """The adapter never invents an observable. These apps expose state only by running a command
    — an operation, not an observation — so nominating some zero-argument read and publishing its
    output as observed state would manufacture an affordance the environment does not have."""
    manual = schema_to_manual("calendar", CALENDAR_SCHEMA, app_class="CalendarApp")
    assert manual.observable_properties == []


def test_a_declared_poll_is_what_the_manual_publishes() -> None:
    manual = schema_to_manual(
        "calendar",
        CALENDAR_SCHEMA,
        app_class="CalendarApp",
        polls=[PollSpec(name="events", command="get-events")],
    )
    assert [p.name for p in manual.observable_properties] == ["events"]


def test_state_changed_is_declared_only_where_there_is_a_poll_to_diff_it_from() -> None:
    """It is derived by diffing a polled snapshot, so with no poll it can never fire — and a signal
    a tool cannot emit is one a `watch` would wait on forever."""
    without = schema_to_manual("calendar", CALENDAR_SCHEMA, app_class="CalendarApp")
    assert "state_changed" not in {sig.name for sig in without.signals}

    with_poll = schema_to_manual(
        "calendar",
        CALENDAR_SCHEMA,
        app_class="CalendarApp",
        polls=[PollSpec(name="events", command="get-events")],
    )
    assert "state_changed" in {sig.name for sig in with_poll.signals}


def test_env_notification_is_declared_so_a_plan_can_watch_for_it() -> None:
    """Without this the signal is pushed but undeclared, so a planner reading the manual has no way
    to know it exists and cannot author a condition against it."""
    manual = schema_to_manual("emails", EMAILS_SCHEMA, app_class="EmailClientV2")
    spec = next(sig for sig in manual.signals if sig.name == "env_notification")
    # The payload schema states the tier: an announcement, with no structured changes to read.
    assert set(spec.schema["properties"]) == {"app", "text"}
    assert "changes" not in spec.schema["properties"]


def test_a_declared_poll_replaces_the_mechanical_default() -> None:
    """Which read best represents an app's state is a domain judgement, so it is configured rather
    than derived — the mechanical fallback would poll `calendar`'s tag list and call it state."""
    runner = _FakeRunner({"get-events": (0, '{"events": []}', "")})
    adapter = Gaia2CliWorkspaceAdapter(
        workspace_id="gaia2",
        origin=WorkspaceOrigin(adapter=GAIA2_CLI_ADAPTER, address="/home/agent/bin"),
        binaries=["calendar"],
        schemas={"calendar": CALENDAR_SCHEMA},
        app_classes={"calendar": "CalendarApp"},
        polls={
            "calendar": [
                PollSpec(
                    name="events",
                    command="get-events",
                    params={"start_datetime": "2020-01-01 00:00:00"},
                )
            ]
        },
        runner=runner,
        poll_interval=None,
    )

    async def _run() -> list[Any]:
        tool = (await adapter.discover())[0].tools()[0]
        await tool.focus(_RecordingSink())
        return list(tool.observe())

    props = asyncio.run(_run())
    assert [p.name for p in props] == ["events"]
    assert runner.calls[-1][1:3] == ["get-events", "--start-datetime"]


# ── the live tool ───────────────────────────────────────────────────────────────────────────────


class _FakeRunner:
    """Stands in for the subprocess: records argv, replays canned (rc, stdout, stderr)."""

    def __init__(self, replies: dict[str, tuple[int, str, str]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.replies = replies or {}

    async def __call__(
        self,
        argv: list[str],
        timeout: float,  # noqa: ASYNC109
    ) -> tuple[int, str, str]:
        self.calls.append(list(argv))
        return self.replies.get(argv[1], (0, "null", ""))


def _adapter(runner: _FakeRunner, **kw: Any) -> Gaia2CliWorkspaceAdapter:
    return Gaia2CliWorkspaceAdapter(
        workspace_id="gaia2",
        origin=WorkspaceOrigin(adapter=GAIA2_CLI_ADAPTER, address="/home/agent/bin"),
        binaries=["emails"],
        schemas={"emails": EMAILS_SCHEMA},
        app_classes={"emails": "EmailClientV2"},
        runner=runner,
        **kw,
    )


def _tool(runner: _FakeRunner, **kw: Any) -> Any:
    ws = asyncio.run(_adapter(runner, **kw).discover())[0]
    return ws.tools()[0]


async def _atool(runner: _FakeRunner, **kw: Any) -> Any:
    ws = (await _adapter(runner, **kw).discover())[0]
    return ws.tools()[0]


def test_invoke_returns_the_parsed_stdout_as_an_ok_ack() -> None:
    runner = _FakeRunner({"list-emails": (0, '{"emails": [{"email_id": "e1"}]}', "")})
    tool = _tool(runner)
    ack = asyncio.run(tool.invoke("list_emails", folder_name="INBOX"))
    assert ack.ok is True
    assert ack.result == {"emails": [{"email_id": "e1"}]}
    assert runner.calls[-1][1:] == ["list-emails", "--folder-name", "INBOX"]


def test_invoke_reports_a_nonzero_exit_as_a_failed_ack_carrying_stderr() -> None:
    runner = _FakeRunner({"delete-email": (1, "", "no such email 'e9'")})
    ack = asyncio.run(_tool(runner).invoke("delete_email", email_id="e9"))
    assert ack.ok is False
    assert "no such email" in str(ack.result)


def test_invoke_rejects_an_operation_the_manual_does_not_describe() -> None:
    runner = _FakeRunner()
    ack = asyncio.run(_tool(runner).invoke("drop_database"))
    assert ack.ok is False
    assert runner.calls == []


def test_focus_polls_once_so_the_first_observe_is_not_empty() -> None:
    runner = _FakeRunner({"list-emails": (0, '{"emails": []}', "")})

    async def _run() -> list[Any]:
        tool = await _atool(
            runner,
            poll_interval=None,  # no background loop
            polls={"emails": [PollSpec(name="list_emails", command="list-emails")]},
        )
        await tool.focus(_RecordingSink())
        return list(tool.observe())

    props = asyncio.run(_run())
    assert [p.name for p in props] == ["list_emails"]
    assert props[0].value == {"emails": []}


class _RecordingSink:
    def __init__(self) -> None:
        self.pushed: list[tuple[str, Signal]] = []

    def push(self, source: str, signal: Signal) -> None:
        self.pushed.append((source, signal))


def test_a_changed_poll_pushes_state_changed_naming_where_it_moved() -> None:
    """Same contract the in-process ARE adapter publishes, so a condition or a `blocked` wait
    written against one harness matches on the other."""
    runner = _FakeRunner({"list-emails": (0, '{"emails": []}', "")})
    sink = _RecordingSink()

    async def _run() -> None:
        tool = await _atool(
            runner,
            poll_interval=None,
            polls={"emails": [PollSpec(name="list_emails", command="list-emails")]},
        )
        await tool.focus(sink)
        runner.replies["list-emails"] = (0, '{"emails": [{"email_id": "e1"}]}', "")
        await tool.poll_once()

    asyncio.run(_run())
    assert [s.name for _, s in sink.pushed] == ["state_changed"]
    assert sink.pushed[0][1].payload["app"] == "EmailClientV2"
    assert sink.pushed[0][1].payload["changes"]


def test_an_unchanged_poll_pushes_nothing() -> None:
    runner = _FakeRunner({"list-emails": (0, '{"emails": []}', "")})
    sink = _RecordingSink()

    async def _run() -> None:
        tool = await _atool(
            runner,
            poll_interval=None,
            polls={"emails": [PollSpec(name="list_emails", command="list-emails")]},
        )
        await tool.focus(sink)
        await tool.poll_once()

    asyncio.run(_run())
    assert sink.pushed == []


def test_an_unfocused_tool_stops_polling() -> None:
    runner = _FakeRunner({"list-emails": (0, "null", "")})

    async def _run() -> list[Any]:
        tool = await _atool(
            runner,
            poll_interval=None,
            polls={"emails": [PollSpec(name="list_emails", command="list-emails")]},
        )
        await tool.focus(_RecordingSink())
        await tool.unfocus()
        return list(tool.observe())

    assert asyncio.run(_run()) == []


# ── discovery ───────────────────────────────────────────────────────────────────────────────────


def test_discover_surfaces_only_the_binaries_present_so_scenario_pruning_is_respected() -> None:
    """The init entrypoint deletes `/home/agent/bin` symlinks for apps this scenario does not use;
    an adapter that discovered the full ten would hand the planner tools that cannot run."""
    runner = _FakeRunner()
    adapter = Gaia2CliWorkspaceAdapter(
        workspace_id="gaia2",
        origin=WorkspaceOrigin(adapter=GAIA2_CLI_ADAPTER, address="/home/agent/bin"),
        binaries=["emails", "calendar"],
        schemas={"emails": EMAILS_SCHEMA},  # calendar's symlink was pruned -> no schema
        app_classes={"emails": "EmailClientV2"},
        runner=runner,
    )
    ws = asyncio.run(adapter.discover())[0]
    assert [t.id for t in ws.tools()] == ["/home/agent/bin/emails"]


def test_discovered_workspace_reports_domain_time() -> None:
    """The container is under libfaketime, so its own wall clock *is* scenario time — the one
    environment where reading host time is the right answer rather than the 1970-vs-2024 bug."""
    ws = asyncio.run(_adapter(_FakeRunner()).discover())[0]
    assert ws.clock is not None
    assert ws.clock.now() is not None


# ── transport ───────────────────────────────────────────────────────────────────────────────────


def test_transport_receive_drains_submitted_messages() -> None:
    transport = Gaia2CliTransport()
    transport.submit_user_message("Book me a flat", run_id="r1")

    async def _drain() -> list[Any]:
        return [m async for m in transport.receive()]

    messages = asyncio.run(_drain())
    assert [m.content["text"] for m in messages] == ["Book me a flat"]
    assert messages[0].sender == "user"


def test_transport_send_emits_one_final_response_per_reply() -> None:
    """The synthetic `send_message_to_user` event this becomes is the daemon's turn boundary: no
    reply means the run ends `error: no turn boundary detected`."""
    emitted: list[dict[str, Any]] = []
    transport = Gaia2CliTransport(emit=emitted.append)
    transport.submit_user_message("hi", run_id="r1")
    asyncio.run(transport.send("user", {"text": "done"}))
    assert emitted == [{"type": "response", "run_id": "r1", "state": "final", "message": "done"}]


def test_transport_answers_an_empty_reply_rather_than_staying_silent() -> None:
    emitted: list[dict[str, Any]] = []
    transport = Gaia2CliTransport(emit=emitted.append)
    transport.submit_user_message("hi", run_id="r1")
    asyncio.run(transport.send("user", {}))
    assert emitted[0]["state"] == "final"
    assert emitted[0]["message"] == ""


def test_transport_does_not_deliver_an_env_notification_as_a_user_message() -> None:
    """A daemon notification is not the user speaking. Perception of an ENV change is the polled
    property and its `state_changed` diff — the same channel the in-process arm uses — so routing
    the rendered text in as a Message would both duplicate it and mis-attribute it."""
    transport = Gaia2CliTransport()
    transport.note_env_notification("[Notification] emails: New email received from x@y.z")

    async def _drain() -> list[Any]:
        return [m async for m in transport.receive()]

    assert asyncio.run(_drain()) == []
    assert transport.notifications == ["[Notification] emails: New email received from x@y.z"]


# ── domain time ─────────────────────────────────────────────────────────────────────────────────


def test_the_workspace_clock_reads_the_harnesss_simulated_time(tmp_path: Any) -> None:
    """The agent reads the daemon's timestamp file rather than running under libfaketime. Under the
    preload, OpenSSL validates the model provider's certificate against the scenario clock and every
    outbound call fails "certificate is not yet valid" on any scenario set in the past."""
    stamp = tmp_path / "faketime.rc"
    stamp.write_text("2024-10-15 07:04:14\n")
    adapter = Gaia2CliWorkspaceAdapter(
        workspace_id="gaia2",
        origin=WorkspaceOrigin(adapter=GAIA2_CLI_ADAPTER, address="/home/agent/bin"),
        binaries=[],
        schemas={},
        runner=_FakeRunner(),
        faketime_path=str(stamp),
    )
    ws = asyncio.run(adapter.discover())[0]
    assert ws.clock is not None
    assert ws.clock.now().isoformat() == "2024-10-15T07:04:14+00:00"


def test_the_workspace_clock_follows_the_file_as_the_scenario_advances(tmp_path: Any) -> None:
    stamp = tmp_path / "faketime.rc"
    stamp.write_text("2024-10-15 07:00:00")
    clock = _FaketimeClock(str(stamp))
    first = clock.now()
    stamp.write_text("2024-10-15 08:30:00")
    assert (clock.now() - first).total_seconds() == 5400


def test_an_absent_stamp_file_means_nothing_is_faking_time() -> None:
    """Host time *is* domain time then. Never None: this workspace can always say what time it is;
    the only question is whose clock answers."""
    from datetime import UTC, datetime

    now = _FaketimeClock("/nonexistent/faketime.rc").now()
    assert abs((now - datetime.now(UTC)).total_seconds()) < 5


def test_an_unparseable_stamp_falls_back_rather_than_guessing(tmp_path: Any) -> None:
    from datetime import UTC, datetime

    stamp = tmp_path / "faketime.rc"
    stamp.write_text("+3600")  # libfaketime's relative-offset form: not a wall time at all
    now = _FaketimeClock(str(stamp)).now()
    assert abs((now - datetime.now(UTC)).total_seconds()) < 5


# ── environment notifications ───────────────────────────────────────────────────────────────────


def test_parse_notification_reads_the_single_form() -> None:
    assert parse_notification("[Notification] emails: New email from alice@example.com") == [
        ("emails", "New email from alice@example.com")
    ]


def test_parse_notification_reads_the_bundle_form() -> None:
    text = "[Notifications]\n- emails: New email from bob\n- calendar: New event added by carol"
    assert parse_notification(text) == [
        ("emails", "New email from bob"),
        ("calendar", "New event added by carol"),
    ]


def test_parse_notification_keeps_a_hyphenated_app_label() -> None:
    """The daemon labels with the agent-visible CLI name, and three of the ten contain a hyphen."""
    assert parse_notification("[Notification] rent-a-flat: New apartment added") == [
        ("rent-a-flat", "New apartment added")
    ]


def test_parse_notification_keeps_a_colon_inside_the_message() -> None:
    """Only the first `: ` is the label separator — a rendered message may contain its own."""
    assert parse_notification("[Notification] emails: New email: dinner?") == [
        ("emails", "New email: dinner?")
    ]


def test_parse_notification_drops_a_line_with_no_app_label() -> None:
    """The label is the only thing tying a notification to a tool, so an unlabelled line yields
    nothing rather than a signal pushed on a guess at the source."""
    assert parse_notification("[Notification] something happened") == []


def test_an_env_notification_pushes_a_signal_distinct_from_state_changed() -> None:
    runner = _FakeRunner({"list-emails": (0, '{"emails": []}', "")})
    sink = _RecordingSink()

    async def _run() -> None:
        tool = await _atool(
            runner,
            poll_interval=None,
            polls={"emails": [PollSpec(name="list_emails", command="list-emails")]},
        )
        await tool.focus(sink)
        tool.note_env_notification("New email from alice@example.com")

    asyncio.run(_run())
    assert [s.name for _, s in sink.pushed] == ["env_notification"]
    source, signal = sink.pushed[0]
    assert source == "/home/agent/bin/emails"
    assert signal.payload["app"] == "EmailClientV2"
    assert signal.payload["text"] == "New email from alice@example.com"
    # No `changes` key: the daemon renders prose, and claiming a structured diff it never carried
    # would let a consumer read ids off it that do not exist.
    assert "changes" not in signal.payload


def test_an_env_notification_on_an_unfocused_tool_is_dropped() -> None:
    """Focus is what wires the sink; a notification for a tool nobody attends has nowhere to go."""
    runner = _FakeRunner({"list-emails": (0, '{"emails": []}', "")})
    tool = _tool(runner, poll_interval=None)
    tool.note_env_notification("New email from alice@example.com")  # must not raise


def test_the_workspace_routes_a_notification_to_the_tool_it_names() -> None:
    runner = _FakeRunner({"list-emails": (0, '{"emails": []}', "")})
    emails_sink, calendar_sink = _RecordingSink(), _RecordingSink()

    async def _run() -> None:
        adapter = Gaia2CliWorkspaceAdapter(
            workspace_id="gaia2",
            origin=WorkspaceOrigin(adapter=GAIA2_CLI_ADAPTER, address="/home/agent/bin"),
            binaries=["emails", "calendar"],
            schemas={"emails": EMAILS_SCHEMA, "calendar": CALENDAR_SCHEMA},
            app_classes={"emails": "EmailClientV2", "calendar": "CalendarApp"},
            runner=runner,
            poll_interval=None,
        )
        ws: Any = (await adapter.discover())[0]
        by_binary = {t.manual.metadata["binary"]: t for t in ws.tools()}
        await by_binary["emails"].focus(emails_sink)
        await by_binary["calendar"].focus(calendar_sink)
        ws.note_env_notification("[Notification] calendar: New event added by carol")

    asyncio.run(_run())
    assert emails_sink.pushed == []
    assert [s.name for _, s in calendar_sink.pushed] == ["env_notification"]


def test_a_notification_for_an_app_this_workspace_does_not_hold_is_dropped() -> None:
    runner = _FakeRunner({"list-emails": (0, '{"emails": []}', "")})
    sink = _RecordingSink()

    async def _run() -> None:
        adapter = _adapter(runner, poll_interval=None)
        ws: Any = (await adapter.discover())[0]
        await ws.tools()[0].focus(sink)
        ws.note_env_notification("[Notification] cabs: Your ride was cancelled")

    asyncio.run(_run())
    assert sink.pushed == []
