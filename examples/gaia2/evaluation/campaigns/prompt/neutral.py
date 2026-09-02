from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from examples.gaia2.evaluation.campaigns.prompt.contracts import SuiteResult
from sora.memory import _parse_plan_pending, _parse_plan_steps


@dataclass(frozen=True)
class NeutralCase:
    case_id: str
    topic: str
    variant: Literal["ordinary", "adversarial"]
    fixed_response: str
    expected_actions: tuple[str, ...] = ()
    expected_pending: int = 0
    expect_parse_failure: bool = False


def _plan(*steps: Mapping[str, object], pending: list[Mapping[str, object]] | None = None) -> str:
    value: dict[str, object] = {"steps": [dict(step) for step in steps]}
    if pending is not None:
        value["pending"] = pending
    return json.dumps(value, separators=(",", ":"))


_SEARCH = {"tool_id": "catalog", "operation_name": "search", "params": {"query": "blue"}}
_DETAIL = {
    "tool_id": "catalog",
    "operation_name": "details",
    "params": {"id": {"$from": "search", "path": "items.0.id"}},
}
_SEND = {"action": "send", "to": "user", "content": {"text": "done"}}
_WINDOW = {
    "watch": {"signal": "state_changed", "source": "calendar", "path": "events", "kind": "updated"},
    "when": "the event changes",
    "then": "reconcile the event",
    "until": {"text": "the event starts", "seconds": 3600},
}


NEUTRAL_CASES = (
    NeutralCase("lookup-ordinary", "lookup", "ordinary", _plan(_SEARCH), ("invoke",)),
    NeutralCase(
        "lookup-adversarial", "lookup", "adversarial", _plan(_SEARCH, _DETAIL), ("invoke", "invoke")
    ),
    NeutralCase(
        "joins-ordinary", "joins", "ordinary", _plan(_SEARCH, _DETAIL), ("invoke", "invoke")
    ),
    NeutralCase(
        "joins-adversarial",
        "joins",
        "adversarial",
        _plan(_SEARCH, _DETAIL, _SEND),
        ("invoke", "invoke", "send"),
    ),
    NeutralCase(
        "dates-ordinary",
        "dates",
        "ordinary",
        _plan({"tool_id": "clock", "operation_name": "now", "params": {}}),
        ("invoke",),
    ),
    NeutralCase(
        "dates-adversarial",
        "dates",
        "adversarial",
        _plan(
            {
                "tool_id": "calendar",
                "operation_name": "create",
                "params": {"start": {"$from": "now", "path": "next_day"}},
            }
        ),
        ("invoke",),
    ),
    NeutralCase(
        "fanout-ordinary",
        "fan-out",
        "ordinary",
        _plan(
            {
                "action": "subgoal",
                "goal": "notify each",
                "mode": "mechanical",
                "collection": {"$from": "search", "path": "items"},
                "as": "item",
                "template": _SEND,
            }
        ),
        ("subgoal",),
    ),
    NeutralCase(
        "fanout-adversarial",
        "fan-out",
        "adversarial",
        _plan(
            _SEARCH,
            {
                "action": "subgoal",
                "goal": "notify every exact match",
                "mode": "mechanical",
                "collection": {"$from": "search", "path": "items"},
                "as": "item",
                "template": _SEND,
            },
        ),
        ("invoke", "subgoal"),
    ),
    NeutralCase(
        "communication-ordinary", "communication-authorization", "ordinary", _plan(_SEND), ("send",)
    ),
    NeutralCase(
        "communication-adversarial",
        "communication-authorization",
        "adversarial",
        _plan(_SEARCH),
        ("invoke",),
    ),
    NeutralCase("replanning-ordinary", "replanning", "ordinary", _plan(_SEARCH), ("invoke",)),
    NeutralCase(
        "replanning-adversarial",
        "replanning",
        "adversarial",
        _plan({"tool_id": "catalog", "operation_name": "list_all", "params": {}}),
        ("invoke",),
    ),
    NeutralCase(
        "windows-ordinary",
        "multiple-windows",
        "ordinary",
        _plan(_SEARCH, pending=[_WINDOW]),
        ("invoke",),
        1,
    ),
    NeutralCase(
        "windows-adversarial",
        "multiple-windows",
        "adversarial",
        _plan(
            _SEARCH,
            pending=[
                _WINDOW,
                {**_WINDOW, "when": "the event is cancelled", "then": "notify the user"},
            ],
        ),
        ("invoke",),
        2,
    ),
    NeutralCase(
        "malformed-ordinary",
        "malformed-output",
        "ordinary",
        f"```json\n{_plan(_SEARCH)}\n```",
        ("invoke",),
    ),
    NeutralCase(
        "malformed-adversarial",
        "malformed-output",
        "adversarial",
        '{"steps":[',
        expect_parse_failure=True,
    ),
)


def _passes(case: NeutralCase) -> bool:
    try:
        steps = _parse_plan_steps(case.fixed_response)
        pending = _parse_plan_pending(case.fixed_response)
    except ValueError:
        return case.expect_parse_failure
    if case.expect_parse_failure:
        return False
    return (
        tuple(step.next_action for step in steps) == case.expected_actions
        and len(pending) == case.expected_pending
    )


def run_neutral_suite() -> SuiteResult:
    failures = tuple(case.case_id for case in NEUTRAL_CASES if not _passes(case))
    return SuiteResult(len(NEUTRAL_CASES) - len(failures), len(failures), failures)
