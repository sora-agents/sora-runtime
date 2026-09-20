"""Turn one run's timing probe plus its ``llm_calls`` rows into a per-gated-event table.

The question this answers is how a Time scenario's post-event tolerance is actually spent.  Three
terms compete for it, and only the first is under the harness's control:

``delivery``  the charged-clock gap between a world event's scheduled instant and the instant it
              was dispatched.  A freeze that swallowed a scheduled event would show up here, and
              nowhere else: the judge only ever sees the agent's action time.
``reaction``  the charged-clock gap between that dispatch and the agent's matching action.
``charge``    how much of the reaction was billed inference rather than simulated idling, summed
              from the round-trip windows that started inside the reaction span.

Pairing a gated oracle event with its trigger is done by time, not by the oracle graph, and that is
a real limitation worth stating: ARE keeps only oracle-to-oracle dependencies, so the world event
the agent actually reacted to is not a node the judge's graph names.  The nearest environment
dispatch preceding the agent's action is the best available attribution, and it is reported with
the gap so an implausible pairing is visible rather than silently averaged in.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Dispatch:
    env_class: str | None
    env_instance: int | None
    event_id: str | None
    event_type: str
    action: str | None
    scheduled_time: float
    delivered_charged_time: float

    @property
    def delivery_lateness(self) -> float:
        return self.delivered_charged_time - self.scheduled_time


@dataclass(frozen=True)
class ChargeWindow:
    started: float
    charged: float
    label: str | None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_probe(path: Path) -> tuple[list[Dispatch], list[dict[str, Any]]]:
    dispatches: list[Dispatch] = []
    gates: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        kind = row.get("probe")
        if kind == "dispatch":
            scheduled = row.get("scheduled_time")
            delivered = row.get("delivered_charged_time")
            if scheduled is None or delivered is None:
                continue
            dispatches.append(
                Dispatch(
                    env_class=row.get("env_class"),
                    env_instance=row.get("env_instance"),
                    event_id=row.get("event_id"),
                    event_type=str(row.get("event_type")),
                    action=row.get("action"),
                    scheduled_time=float(scheduled),
                    delivered_charged_time=float(delivered),
                )
            )
        elif kind == "gate":
            gates.append(row)
    return dispatches, gates


def _load_charges(path: Path) -> list[ChargeWindow]:
    windows: list[ChargeWindow] = []
    for row in _read_jsonl(path):
        label = row.get("semantic_label")
        for window in row.get("round_trip_windows") or ():
            started = window.get("charged_started_at")
            if started is None:
                continue
            windows.append(
                ChargeWindow(
                    started=float(started),
                    charged=float(window.get("charged_seconds") or 0.0),
                    label=label,
                )
            )
    return sorted(windows, key=lambda w: w.started)


def _by_pass(dispatches: list[Dispatch]) -> list[tuple[str, list[Dispatch]]]:
    """Split dispatches by environment instance, in first-seen order.

    A scored run has two passes and only the later one runs under the freeze; reporting a single
    pooled median would average the frozen run against an unfrozen oracle replay and report the
    result as if it described either.
    """
    order: list[int | None] = []
    groups: dict[int | None, list[Dispatch]] = {}
    for d in dispatches:
        if d.env_instance not in groups:
            groups[d.env_instance] = []
            order.append(d.env_instance)
    for d in dispatches:
        groups[d.env_instance].append(d)
    if len(order) == 1 and order[0] is None:
        return [("all dispatches (probe recorded no environment identity)", dispatches)]
    out = []
    for index, key in enumerate(order):
        group = groups[key]
        name = group[0].env_class or "unknown"
        out.append((f"pass {index + 1} ({name})", group))
    return out


def _trigger_for(dispatches: list[Dispatch], action_time: float) -> Dispatch | None:
    """Nearest environment dispatch at or before the agent's action."""
    candidates = [
        d for d in dispatches if d.delivered_charged_time <= action_time and "ENV" in d.event_type
    ]
    return max(candidates, key=lambda d: d.delivered_charged_time) if candidates else None


def _fmt(value: float | None, width: int = 8, places: int = 2) -> str:
    return "-".rjust(width) if value is None else f"{value:>{width}.{places}f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--llm-calls", type=Path)
    args = parser.parse_args(argv)

    dispatches, gates = _load_probe(args.probe)
    charges = _load_charges(args.llm_calls) if args.llm_calls else []

    print(f"dispatched events        : {len(dispatches)}")
    env = [d for d in dispatches if "ENV" in d.event_type]
    print(f"  of which environment   : {len(env)}")
    print(f"timing-gate evaluations  : {len(gates)}")
    gated = [g for g in gates if g.get("gated")]
    print(f"  of which time-gated    : {len(gated)}")
    print(f"charged round-trips      : {len(charges)}")
    print()

    if dispatches:
        print("Delivery lateness on the charged axis (dispatched - scheduled), seconds")
        for label, group in _by_pass(dispatches):
            lateness = [d.delivery_lateness for d in group]
            over = [d for d in group if d.delivery_lateness > 1.0]
            print(f"  {label}: {len(group)} events")
            print(f"    max     {max(lateness):.4f}")
            print(f"    median  {statistics.median(lateness):.4f}")
            print(f"    over 1s {len(over)} of {len(group)}")
            for d in sorted(over, key=lambda d: -d.delivery_lateness)[:10]:
                print(f"      {d.delivery_lateness:8.2f}s  {d.action}  ({d.event_id})")
        print()

    if not gated:
        print("No time-gated oracle events in this run.")
        return

    print("Per gated oracle event, all times in charged seconds")
    print()
    header = (
        f"{'oracle tool':<34}{'due':>10}{'action':>10}{'late':>8}"
        f"{'budget':>9}{'deliv':>9}{'react':>9}{'charge':>9}{'calls':>6}  verdict"
    )
    print(header)
    print("-" * len(header))

    for gate in gated:
        agent_time = gate.get("agent_event_time")
        oracle_axis = gate.get("oracle_axis_time")
        parent = gate.get("max_parent_agent_event_time") or 0.0
        due = None if oracle_axis is None else parent + float(oracle_axis)
        trigger = _trigger_for(dispatches, float(agent_time)) if agent_time is not None else None
        delivery_lateness = trigger.delivery_lateness if trigger else None
        reaction = (
            None
            if (trigger is None or agent_time is None)
            else float(agent_time) - trigger.delivered_charged_time
        )
        in_window = (
            [
                w
                for w in charges
                if trigger is not None
                and agent_time is not None
                and trigger.delivered_charged_time <= w.started <= float(agent_time)
            ]
            if charges
            else []
        )
        charged = sum(w.charged for w in in_window) if in_window else None
        tool = str(gate.get("oracle_tool"))[:33]
        print(
            f"{tool:<34}"
            f"{_fmt(due, 10, 1)}"
            f"{_fmt(agent_time, 10, 1)}"
            f"{_fmt(gate.get('lateness_seconds'))}"
            f"{_fmt(gate.get('remaining_budget_seconds'), 9)}"
            f"{_fmt(delivery_lateness, 9, 4)}"
            f"{_fmt(reaction, 9)}"
            f"{_fmt(charged, 9)}"
            f"{len(in_window):>6}"
            f"  {'PASS' if gate.get('verdict') else 'FAIL'}"
        )

    print()
    reactions = []
    charged_totals = []
    for gate in gated:
        agent_time = gate.get("agent_event_time")
        if agent_time is None:
            continue
        trigger = _trigger_for(dispatches, float(agent_time))
        if trigger is None:
            continue
        reactions.append(float(agent_time) - trigger.delivered_charged_time)
        charged_totals.append(
            sum(
                w.charged
                for w in charges
                if trigger.delivered_charged_time <= w.started <= float(agent_time)
            )
        )
    if reactions:
        print(
            f"reaction span   median {statistics.median(reactions):.2f}s  max {max(reactions):.2f}s"
        )
    if charged_totals:
        print(
            f"charge in span  median {statistics.median(charged_totals):.2f}s  "
            f"max {max(charged_totals):.2f}s"
        )
    budgets = [
        g["remaining_budget_seconds"]
        for g in gated
        if g.get("remaining_budget_seconds") is not None
    ]
    if budgets:
        print(
            f"remaining budget  min {min(budgets):.2f}s  median {statistics.median(budgets):.2f}s"
        )


if __name__ == "__main__":
    main()
