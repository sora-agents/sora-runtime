from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sora.memory import (
    _load_json_object,
    _parse_condition_verdict,
    _parse_keep,
    _parse_params,
    _parse_plan_pending,
    _parse_plan_steps,
    _parse_relevance,
    _parse_verdict,
    pending_from_raw,
    step_from_raw,
)


@dataclass(frozen=True)
class SuiteResult:
    passed: int
    failed: int
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Case:
    name: str
    check: Callable[[], bool]


def _raises(exc_type: type[BaseException], call: Callable[[], object]) -> bool:
    try:
        call()
    except exc_type:
        return True
    return False


def _pending(raw: dict[str, Any]) -> bool:
    return pending_from_raw(raw) is not None


def _contract_cases() -> tuple[_Case, ...]:
    valid_plan = '{"steps":[{"tool_id":"mail","operation_name":"search","params":{}}]}'
    cases = (
        _Case("valid plan envelope", lambda: len(_parse_plan_steps(valid_plan)) == 1),
        _Case(
            "missing steps rejected", lambda: _raises(ValueError, lambda: _parse_plan_steps("{}"))
        ),
        _Case(
            "non-list steps rejected",
            lambda: _raises(ValueError, lambda: _parse_plan_steps('{"steps":1}')),
        ),
        _Case(
            "prose wrapped object repaired",
            lambda: _load_json_object(f"answer: {valid_plan}")["steps"] != [],
        ),
        _Case(
            "fenced object repaired",
            lambda: _load_json_object(f"```json\n{valid_plan}\n```")["steps"] != [],
        ),
        _Case(
            "unbalanced object rejected",
            lambda: _raises(ValueError, lambda: _load_json_object('{"steps":[')),
        ),
        _Case("ground params valid", lambda: _parse_params('{"params":{"id":"a"}}') == {"id": "a"}),
        _Case(
            "ground params wrong type rejected",
            lambda: _raises(ValueError, lambda: _parse_params('{"params":[]}')),
        ),
        _Case("selection keeps in range", lambda: _parse_keep('{"keep":[2,0]}', 3) == [2, 0]),
        _Case(
            "selection drops invalid indices",
            lambda: _parse_keep('{"keep":[true,-1,5,1]}', 2) == [1],
        ),
        _Case("selection duplicate repaired", lambda: _parse_keep('{"keep":[1,1]}', 2) == [1]),
        _Case(
            "malformed selection rejected",
            lambda: _raises(ValueError, lambda: _parse_keep('{"keep":1}', 2)),
        ),
        _Case(
            "valid reference remains opaque",
            lambda: (
                step_from_raw(
                    {
                        "tool_id": "t",
                        "operation_name": "o",
                        "params": {"id": {"$from": "o", "path": "id"}},
                    }
                ).params["id"]
                == {"$from": "o", "path": "id"}
            ),
        ),
        _Case(
            "unknown subgoal mode rejected",
            lambda: _raises(
                ValueError,
                lambda: step_from_raw({"action": "subgoal", "goal": "g", "mode": "wide"}),
            ),
        ),
        _Case(
            "mechanical subgoal accepted",
            lambda: (
                str(
                    step_from_raw({"action": "subgoal", "goal": "g", "mode": "mechanical"}).params[
                        "mode"
                    ]
                )
                == "mechanical"
            ),
        ),
        _Case(
            "pending condition accepted",
            lambda: _pending({"watch": {"signal": "changed"}, "when": "x", "then": "y"}),
        ),
        _Case("pending without watch dropped", lambda: not _pending({"when": "x", "then": "y"})),
        _Case(
            "pending malformed field drops only condition",
            lambda: _parse_plan_pending('{"steps":[],"pending":[{"when":"x"}]}') == (),
        ),
        _Case(
            "condition verdict valid",
            lambda: _parse_condition_verdict('{"fired":[0],"retired":[1]}', 2).fired == (0,),
        ),
        _Case(
            "condition verdict conservative malformed",
            lambda: _parse_condition_verdict("nonsense", 2).fired == (),
        ),
        _Case("revalidate malformed defaults valid", lambda: _parse_verdict("nonsense") is True),
        _Case("revalidate false stays false", lambda: _parse_verdict('{"valid":false}') is False),
        _Case(
            "relevance malformed defaults none",
            lambda: _parse_relevance("nonsense", [{"goal": "g"}]) is None,
        ),
        _Case(
            "relevance out of range defaults none",
            lambda: _parse_relevance('{"episode":5,"reason":"x"}', [{"goal": "g"}]) is None,
        ),
    )
    return cases


def run_contract_suite() -> SuiteResult:
    failures: list[str] = []
    for case in _contract_cases():
        try:
            passed = case.check()
        except Exception as exc:  # the suite reports a contract failure instead of aborting CI
            failures.append(f"{case.name}: {type(exc).__name__}: {exc}")
            continue
        if not passed:
            failures.append(case.name)
    return SuiteResult(len(_contract_cases()) - len(failures), len(failures), tuple(failures))
