"""Store/retrieve/list round-trip tests for ``SemanticMemory``.

``SemanticMemory`` is the durable store for *knowledge about the world* — tool-type
``Manual``s, ``WorkspaceRecord``s, and ``ToolRecord``s. It layers dataclass<->dict
(de)serialization on top of the generic key->JSON ``FileMemoryBackend`` (see
``tests/test_memory_backend.py``): the backend stays type-agnostic, so this module pins the
contract that the domain (de)serialization is the memory module's job.

Two invariants get extra attention because they are the reason the backend can hold three
record kinds at once:

* a ``kind`` discriminator kept in the stored value, so ``list_*`` returns just one kind; and
* namespaced storage keys, so the three independent id-spaces never clobber each other's files.
"""

from __future__ import annotations

from pathlib import Path

from sora.environment import WorkspaceOrigin
from sora.manual import (
    Manual,
    ObservablePropertySpecification,
    OperationSpecification,
    SignalSpecification,
    ToolRecord,
    WorkspaceRecord,
)
from sora.memory import FileMemoryBackend, SemanticMemory

# --------------------------------------------------------------------------------------------------
# Fixtures — one of each record kind, with enough nesting that (de)serialization is real work.
# --------------------------------------------------------------------------------------------------


def make_manual(manual_id: str = "email-client") -> Manual:
    return Manual(
        id=manual_id,
        metadata={"vendor": "ARE", "version": 1},
        description="Read and send email.",
        observable_properties=[
            ObservablePropertySpecification(
                name="unread_count",
                description="Number of unread emails.",
                schema={"type": "integer"},
            )
        ],
        signals=[
            SignalSpecification(
                name="email_received",
                description="Fired when a new email arrives.",
                schema={"type": "object"},
            )
        ],
        operations=[
            OperationSpecification(
                name="list_emails",
                description="List emails in the inbox.",
                parameters={"type": "object", "properties": {"limit": {"type": "integer"}}},
            ),
            OperationSpecification(
                name="send_email",
                description="Send an email.",
                parameters={"type": "object", "properties": {"to": {"type": "string"}}},
            ),
        ],
        raw_text="Call list_emails before send_email.",
    )


def make_workspace_record(workspace_id: str = "ws-email") -> WorkspaceRecord:
    return WorkspaceRecord(
        id=workspace_id,
        origin=WorkspaceOrigin(adapter="mcp", address="mcp://localhost:6000/email"),
        discovered_at=1.0,
        last_seen_at=2.0,
    )


def make_tool_record(tool_id: str = "EmailClientApp__list_emails") -> ToolRecord:
    return ToolRecord(
        id=tool_id,
        manual_id="email-client",
        workspace_id="ws-email",
        address=None,
        discovered_at=1.0,
        last_seen_at=2.0,
    )


def semantic(tmp_path: Path) -> SemanticMemory:
    return SemanticMemory(FileMemoryBackend(tmp_path))


# --------------------------------------------------------------------------------------------------
# Manual round-trip
# --------------------------------------------------------------------------------------------------


async def test_store_then_retrieve_manual_returns_equal_manual(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    manual = make_manual()
    await mem.store_manual(manual)
    assert await mem.retrieve_manual(manual.id) == manual


async def test_retrieved_manual_nested_specs_are_dataclasses_not_dicts(tmp_path: Path) -> None:
    # The whole point of the module owning (de)serialization: you get typed objects back,
    # not the raw JSON dicts the backend stores.
    mem = semantic(tmp_path)
    await mem.store_manual(make_manual())
    got = await mem.retrieve_manual("email-client")
    assert got is not None
    assert isinstance(got.operations[0], OperationSpecification)
    assert isinstance(got.observable_properties[0], ObservablePropertySpecification)
    assert isinstance(got.signals[0], SignalSpecification)
    assert got.operations[0].name == "list_emails"


async def test_retrieve_missing_manual_returns_none(tmp_path: Path) -> None:
    assert await semantic(tmp_path).retrieve_manual("nope") is None


async def test_store_manual_overwrites_same_id(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    await mem.store_manual(make_manual())
    updated = make_manual()
    updated = Manual(
        id=updated.id,
        metadata={"vendor": "ARE", "version": 2},  # bumped
        description="Read and send email (v2).",
        observable_properties=updated.observable_properties,
        signals=updated.signals,
        operations=updated.operations,
        raw_text=updated.raw_text,
    )
    await mem.store_manual(updated)
    got = await mem.retrieve_manual("email-client")
    assert got is not None
    assert got.metadata == {"vendor": "ARE", "version": 2}
    assert got.description == "Read and send email (v2)."


# --------------------------------------------------------------------------------------------------
# WorkspaceRecord round-trip + listing
# --------------------------------------------------------------------------------------------------


async def test_store_then_retrieve_workspace_record_returns_equal(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    record = make_workspace_record()
    await mem.store_workspace_record(record)
    assert await mem.retrieve_workspace_record(record.id) == record


async def test_retrieved_workspace_record_origin_is_workspace_origin(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    await mem.store_workspace_record(make_workspace_record())
    got = await mem.retrieve_workspace_record("ws-email")
    assert got is not None
    assert isinstance(got.origin, WorkspaceOrigin)
    assert got.origin.adapter == "mcp"


async def test_retrieve_missing_workspace_record_returns_none(tmp_path: Path) -> None:
    assert await semantic(tmp_path).retrieve_workspace_record("nope") is None


async def test_list_workspace_records_returns_all(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    a = make_workspace_record("ws-a")
    b = make_workspace_record("ws-b")
    await mem.store_workspace_record(a)
    await mem.store_workspace_record(b)
    listed = await mem.list_workspace_records()
    assert {r.id for r in listed} == {"ws-a", "ws-b"}
    assert set(listed) == {a, b}


async def test_list_workspace_records_empty_returns_empty(tmp_path: Path) -> None:
    assert await semantic(tmp_path).list_workspace_records() == []


# --------------------------------------------------------------------------------------------------
# ToolRecord round-trip + listing
# --------------------------------------------------------------------------------------------------


async def test_store_then_retrieve_tool_record_returns_equal(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    record = make_tool_record()
    await mem.store_tool_record(record)
    assert await mem.retrieve_tool_record(record.id) == record


async def test_tool_record_with_address_round_trips(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    record = ToolRecord(
        id="device-1",
        manual_id="lamp",
        workspace_id="ws-home",
        address="coap://192.168.1.5/lamp",  # tool overrides the workspace address
        discovered_at=3.0,
        last_seen_at=4.0,
    )
    await mem.store_tool_record(record)
    assert await mem.retrieve_tool_record("device-1") == record


async def test_retrieve_missing_tool_record_returns_none(tmp_path: Path) -> None:
    assert await semantic(tmp_path).retrieve_tool_record("nope") is None


async def test_list_tool_records_returns_all(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    a = make_tool_record("EmailClientApp__list_emails")
    b = make_tool_record("EmailClientApp__send_email")
    await mem.store_tool_record(a)
    await mem.store_tool_record(b)
    listed = await mem.list_tool_records()
    assert set(listed) == {a, b}


async def test_list_tool_records_empty_returns_empty(tmp_path: Path) -> None:
    assert await semantic(tmp_path).list_tool_records() == []


# --------------------------------------------------------------------------------------------------
# The two invariants that let three record kinds share one backend
# --------------------------------------------------------------------------------------------------


async def test_same_id_across_kinds_does_not_clobber(tmp_path: Path) -> None:
    # A manual, a workspace record, and a tool record that all share the string id "shared"
    # must be stored under distinct keys — none overwrites another.
    mem = semantic(tmp_path)
    manual = make_manual("shared")
    ws = make_workspace_record("shared")
    tool = make_tool_record("shared")
    await mem.store_manual(manual)
    await mem.store_workspace_record(ws)
    await mem.store_tool_record(tool)
    assert await mem.retrieve_manual("shared") == manual
    assert await mem.retrieve_workspace_record("shared") == ws
    assert await mem.retrieve_tool_record("shared") == tool


async def test_list_returns_only_its_own_kind_even_with_overlapping_ids(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    await mem.store_manual(make_manual("shared"))
    await mem.store_workspace_record(make_workspace_record("shared"))
    await mem.store_tool_record(make_tool_record("shared"))
    workspaces = await mem.list_workspace_records()
    tools = await mem.list_tool_records()
    assert [r.id for r in workspaces] == ["shared"]
    assert all(isinstance(r, WorkspaceRecord) for r in workspaces)
    assert [r.id for r in tools] == ["shared"]
    assert all(isinstance(r, ToolRecord) for r in tools)
    # The same-id manual is neither listed among the records above nor clobbered by them.
    assert await mem.retrieve_manual("shared") == make_manual("shared")


# --------------------------------------------------------------------------------------------------
# Persistence across instances (the "durable" point) and copy isolation
# --------------------------------------------------------------------------------------------------


async def test_records_persist_across_semantic_memory_instances(tmp_path: Path) -> None:
    await SemanticMemory(FileMemoryBackend(tmp_path)).store_manual(make_manual())
    # A brand-new SemanticMemory over the same root (i.e. a process restart) sees the write.
    reloaded = SemanticMemory(FileMemoryBackend(tmp_path))
    assert await reloaded.retrieve_manual("email-client") == make_manual()


async def test_mutating_retrieved_manual_does_not_corrupt_store(tmp_path: Path) -> None:
    mem = semantic(tmp_path)
    await mem.store_manual(make_manual())
    got = await mem.retrieve_manual("email-client")
    assert got is not None
    got.operations.append(
        OperationSpecification(name="injected", description="", parameters={})
    )  # mutate the returned value's list
    again = await mem.retrieve_manual("email-client")
    assert again is not None
    assert [op.name for op in again.operations] == ["list_emails", "send_email"]
