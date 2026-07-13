"""Round-trip tests for the file-backed ``MemoryBackend`` default.

These tests pin the contract that ``SemanticMemory``/``ProceduralMemory``/``EpisodicMemory`` will
build on. The backend is a generic key -> JSON-serializable-value store; domain dataclass
(de)serialization is the memory modules' job, so everything here is exercised with plain
dict/list/scalar values, exactly what those modules will hand it.
"""

from __future__ import annotations

from pathlib import Path

from sora.memory import FileMemoryBackend

# --------------------------------------------------------------------------------------------------
# Basic put / get round-trip
# --------------------------------------------------------------------------------------------------


async def test_put_then_get_returns_equal_value(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    value = {"id": "m1", "description": "a manual", "operations": ["a", "b"]}
    await backend.put("m1", value)
    assert await backend.get("m1") == value


async def test_get_missing_key_returns_none(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    assert await backend.get("nope") is None


async def test_get_on_missing_root_returns_none(tmp_path: Path) -> None:
    # Constructing against a path that doesn't exist yet must not raise on read.
    backend = FileMemoryBackend(tmp_path / "not_created_yet")
    assert await backend.get("nope") is None


async def test_put_overwrites_existing_key(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    await backend.put("k", {"v": 1})
    await backend.put("k", {"v": 2})
    assert await backend.get("k") == {"v": 2}


async def test_distinct_keys_do_not_clobber(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    await backend.put("a", {"v": "a"})
    await backend.put("b", {"v": "b"})
    assert await backend.get("a") == {"v": "a"}
    assert await backend.get("b") == {"v": "b"}


# --------------------------------------------------------------------------------------------------
# Value shapes the memory modules will actually store
# --------------------------------------------------------------------------------------------------


async def test_nested_dict_and_list_round_trip_unchanged(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    value = {
        "id": "plan-1",
        "goal": "schedule from email",
        "steps": [
            {"next_action": "invoke", "params": {"tool_id": "EmailClientApp", "n": 1}},
            {"next_action": "wait", "params": {}},
        ],
        "discovered_at": 1.5,
    }
    await backend.put("plan-1", value)
    assert await backend.get("plan-1") == value


async def test_scalar_and_none_values_round_trip(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    await backend.put("s", "just a string")
    await backend.put("n", None)
    await backend.put("num", 42)
    assert await backend.get("s") == "just a string"
    assert await backend.get("n") is None
    assert await backend.get("num") == 42


# --------------------------------------------------------------------------------------------------
# Keys with characters unsafe for filenames (tool ids are URIs / <App>__<op> namespaced)
# --------------------------------------------------------------------------------------------------


async def test_uri_like_and_namespaced_keys_round_trip(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    keys = ["mcp://localhost:6000/email", "EmailClientApp__list_emails", "a/b/c", "with space"]
    for i, key in enumerate(keys):
        await backend.put(key, {"i": i})
    for i, key in enumerate(keys):
        assert await backend.get(key) == {"i": i}


# --------------------------------------------------------------------------------------------------
# Persistence across backend instances (the whole point of "file-backed")
# --------------------------------------------------------------------------------------------------


async def test_value_persists_across_backend_instances(tmp_path: Path) -> None:
    await FileMemoryBackend(tmp_path).put("k", {"v": "durable"})
    # A brand-new instance over the same root (i.e. a process restart) sees the prior write.
    assert await FileMemoryBackend(tmp_path).get("k") == {"v": "durable"}


# --------------------------------------------------------------------------------------------------
# query(**filters)
# --------------------------------------------------------------------------------------------------


async def test_query_no_filters_returns_all_values(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    await backend.put("a", {"kind": "manual", "id": "a"})
    await backend.put("b", {"kind": "tool_record", "id": "b"})
    results = await backend.query()
    assert len(results) == 2
    assert {"kind": "manual", "id": "a"} in results
    assert {"kind": "tool_record", "id": "b"} in results


async def test_query_on_missing_root_returns_empty(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path / "not_created_yet")
    assert await backend.query() == []
    assert await backend.query(kind="manual") == []


async def test_query_filters_match_top_level_fields(tmp_path: Path) -> None:
    # This is the discriminator pattern SemanticMemory will use to keep three record kinds
    # (manuals / workspace records / tool records) in one backend and list just one kind.
    backend = FileMemoryBackend(tmp_path)
    await backend.put("m1", {"kind": "manual", "id": "m1"})
    await backend.put("t1", {"kind": "tool_record", "id": "t1", "workspace_id": "ws"})
    await backend.put("t2", {"kind": "tool_record", "id": "t2", "workspace_id": "ws"})
    tool_records = await backend.query(kind="tool_record")
    assert len(tool_records) == 2
    assert {r["id"] for r in tool_records} == {"t1", "t2"}


async def test_query_multiple_filters_are_conjunctive(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    await backend.put("t1", {"kind": "tool_record", "workspace_id": "ws-a"})
    await backend.put("t2", {"kind": "tool_record", "workspace_id": "ws-b"})
    matched = await backend.query(kind="tool_record", workspace_id="ws-a")
    assert matched == [{"kind": "tool_record", "workspace_id": "ws-a"}]


async def test_query_with_filters_excludes_non_dict_values(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    await backend.put("scalar", "not a dict")
    await backend.put("d", {"kind": "manual"})
    assert await backend.query(kind="manual") == [{"kind": "manual"}]


async def test_query_results_are_in_stable_key_order(tmp_path: Path) -> None:
    # FileMemoryBackend has no relevance ranking, so it honors query()'s "deterministic tiebreak"
    # clause by returning matches in stable on-disk-key order — the property ProceduralMemory relies
    # on to treat result[0] as canonical.
    backend = FileMemoryBackend(tmp_path)
    for key in ["p3", "p1", "p2"]:
        await backend.put(key, {"kind": "plan", "id": key})
    ids = [v["id"] for v in await backend.query(kind="plan")]
    assert ids == ["p1", "p2", "p3"]
    assert ids == [v["id"] for v in await backend.query(kind="plan")]  # stable across calls


# --------------------------------------------------------------------------------------------------
# Copy isolation: a file backend re-reads from disk, so a caller can't mutate stored state
# through a returned reference.
# --------------------------------------------------------------------------------------------------


async def test_get_result_is_isolated_from_store(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    await backend.put("k", {"nested": {"v": 1}})
    got = await backend.get("k")
    assert got is not None
    got["nested"]["v"] = 999  # mutate the returned value
    assert await backend.get("k") == {"nested": {"v": 1}}  # store is untouched


async def test_query_results_are_isolated_from_store(tmp_path: Path) -> None:
    backend = FileMemoryBackend(tmp_path)
    await backend.put("k", {"kind": "manual", "tags": ["x"]})
    (result,) = await backend.query(kind="manual")
    result["tags"].append("y")  # mutate the returned value
    assert await backend.get("k") == {"kind": "manual", "tags": ["x"]}
