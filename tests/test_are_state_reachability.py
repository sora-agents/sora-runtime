"""The `state` observable publishes only what some operation on the same app can return.

S-ORA snapshots one `state` property per ARE app; ARE's own ReAct agent never sees app state at
all. Most of that gap is a CALL-COUNT asymmetry — the same information, reachable through a tool,
just at the price of a call plus a model round-trip. That gap is the architecture, and it is
measured rather than removed (`WorkingMemory.prop_reads`). A handful of state fields are different:
no operation returns them at any call count, so publishing them would make a paired run a
comparison of information rather than of architectures. `_UNREACHABLE_STATE_KEYS` withholds exactly
those, and this file holds it to that rule from both ends — the filter behaves, and the upstream
key set it was authored against has not moved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sora.adapters.are_sim import _UNREACHABLE_STATE_KEYS, _AreTool, _publishable_state
from sora.manual import Manual

CENSUS = Path(__file__).parent / "fixtures" / "are_app_state_census.json"


# ------------------------------------------------------------------------------------------------
# The filter itself (fakes, no ARE — runs in CI)
# ------------------------------------------------------------------------------------------------


def test_an_unreachable_key_is_withheld_and_its_neighbours_are_not() -> None:
    published = _publishable_state(
        "CityApp", {"crime_data": {"94110": 7}, "api_call_limit": 100, "api_call_count": 3}
    )
    # `api_call_limit` stays: `get_api_call_limit` returns it, so withholding it would narrow the
    # snapshot below what a tool call already gives, which is not what the rule says.
    assert published == {"api_call_limit": 100, "api_call_count": 3}


def test_an_app_with_nothing_withheld_passes_through_unchanged() -> None:
    state = {"events": [{"id": "e1"}]}
    assert _publishable_state("CalendarApp", state) == state


def test_a_non_dict_state_passes_through_rather_than_being_guessed_at() -> None:
    # ARE builds every app state with `asdict()`, so a non-dict is an upstream shape change.
    # Guessing a route into it would silently publish what this exists to withhold; passing it
    # through leaves the census test to fail loudly instead.
    assert _publishable_state("CityApp", ["not", "a", "dict"]) == ["not", "a", "dict"]


class _FakeCity:
    """Claims CityApp's name so the real table applies; the shape is CityApp's."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {"crime_data": {"94110": 7}, "api_call_count": 0}

    def app_name(self) -> str:
        return "CityApp"

    def get_state(self) -> dict[str, Any]:
        return {k: v for k, v in self.state.items()}

    def get_tools(self) -> list[Any]:
        return []


class _RunInline:
    def run(self, fn: Any) -> Any:
        return fn()


class _Sink:
    def __init__(self) -> None:
        self.pushed: list[Any] = []

    def push(self, tool_id: str, signal: Any) -> None:
        self.pushed.append(signal)


def _city_tool(app: _FakeCity) -> _AreTool:
    return _AreTool(
        tool_id="city",
        manual=Manual(
            id="CityApp",
            description="",
            metadata={},
            observable_properties=[],
            signals=[],
            operations=[],
        ),
        app=app,
        ops={},
        simulation=_RunInline(),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_the_withheld_key_is_absent_from_the_published_property() -> None:
    tool = _city_tool(_FakeCity())
    await tool.focus(_Sink())
    (prop,) = tool.observe()
    assert prop.name == "state"
    assert prop.value == {"api_call_count": 0}


@pytest.mark.asyncio
async def test_the_withheld_key_does_not_leak_back_out_through_the_change_signal() -> None:
    # The trap this guards: filtering at observe()'s RETURN would narrow the property and leave
    # `diff_values(previous, state)` computed over the unfiltered snapshots, republishing every
    # withheld key in the `state_changed` payload. Filtering at the single read closes both.
    app = _FakeCity()
    tool = _city_tool(app)
    sink = _Sink()
    await tool.focus(sink)

    app.state["crime_data"] = {"94110": 7, "94103": 9}  # a change ONLY in the withheld key
    tool.observe()
    assert sink.pushed == [], "a change confined to a withheld key must not even be announced"

    app.state["api_call_count"] = 1
    tool.observe()
    assert len(sink.pushed) == 1
    assert "crime_data" not in repr(sink.pushed[0].payload)  # Change objects; repr, not json


# ------------------------------------------------------------------------------------------------
# The upstream key set the table was authored against (needs ARE: `uv sync --group are`)
# ------------------------------------------------------------------------------------------------


def _live_census() -> dict[str, list[str]]:
    """Every ARE app's `get_state()` keys, keyed by `app_name()` — the same identity the filter is
    keyed by, so an alias class (`Calendar` vs `CalendarApp` vs `CalendarV2`) is its own row and
    cannot be classified in one place and forgotten in another."""
    import importlib
    import inspect
    import pkgutil

    import are.simulation.apps as pkg
    from are.simulation.apps.app import App

    classes: dict[str, type] = {}
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        try:
            mod = importlib.import_module(f"are.simulation.apps.{mod_info.name}")
        except Exception:  # an app module that needs optional extras is not part of the surface
            continue
        for obj in vars(mod).values():
            if inspect.isclass(obj) and issubclass(obj, App) and obj is not App:
                classes[obj.__name__] = obj

    census: dict[str, list[str]] = {}
    for cls in classes.values():
        try:
            app = cls()
            name, state = app.app_name(), app.get_state()
        except Exception as exc:  # recorded, not skipped: losing an app from the census is silent
            census[cls.__name__] = [f"<unavailable: {type(exc).__name__}>"]
            continue
        census[name] = sorted(state) if isinstance(state, dict) else ["<non-dict state>"]
    return census


def test_the_upstream_state_surface_still_matches_the_one_the_table_was_authored_against() -> None:
    # Fail LOUD, not open or closed. Fail-open (publish an unclassified key) reopens the hole on an
    # ARE bump with nothing red to show for it; fail-closed (withhold it) silently degrades what the
    # agent perceives. This makes an upstream key a red test that forces a human classification.
    # It needs ARE installed, so it SKIPS in CI — but an ARE version bump only ever happens on a
    # host that has ARE, which is exactly where it fires.
    pytest.importorskip("are.simulation")
    assert _live_census() == json.loads(CENSUS.read_text()), (
        "ARE's app state surface moved. Classify each new key against the rule — an app's `state` "
        "publishes only what some operation on that same app can return — updating "
        "`_UNREACHABLE_STATE_KEYS` if it is unreachable, then refresh the census fixture."
    )


def test_every_withheld_key_is_a_key_the_app_actually_has() -> None:
    # A typo withholds nothing and looks identical to a working entry from the outside.
    pytest.importorskip("are.simulation")
    census = _live_census()
    for app_name, keys in _UNREACHABLE_STATE_KEYS.items():
        assert app_name in census, f"{app_name} is not an ARE app"
        assert keys <= set(census[app_name]), f"{app_name}: {keys - set(census[app_name])}"
