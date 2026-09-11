from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.gaia2.evaluation.core import (
    SCHEMA_VERSION,
    load_judge_profile,
    load_profiles,
    sha256_text,
)
from sora.activity import Activity
from sora.llm import CompletionRequest
from sora.memory import (
    FileMemoryBackend,
    PerceptionChannels,
    PerceptSnapshot,
    ProceduralMemory,
)
from sora.types import Change, PendingCondition, Plan, SignalWait, Step, Until

PROMPT_LABELS = (
    "plan",
    "ground",
    "select",
    "revalidate",
    "condition",
    "retirement",
    "relevance",
)
PROMPT_SOURCES = {
    "plan": ("src/sora/memory.py", "PLAN_SYSTEM_PROMPT", "default_plan_prompt"),
    "ground": ("src/sora/memory.py", "GROUND_SYSTEM_PROMPT", "default_ground_prompt"),
    "select": ("src/sora/memory.py", "SELECT_SYSTEM_PROMPT", "ProceduralMemory.select"),
    "revalidate": (
        "src/sora/memory.py",
        "REVALIDATE_SYSTEM_PROMPT",
        "ProceduralMemory.revalidate",
    ),
    "condition": (
        "src/sora/memory.py",
        "CONDITION_SYSTEM_PROMPT",
        "ProceduralMemory.evaluate_conditions",
    ),
    "retirement": (
        "src/sora/memory.py",
        "RETIREMENT_SYSTEM_PROMPT",
        "ProceduralMemory.judge_retirement",
    ),
    "relevance": (
        "src/sora/memory.py",
        "RELEVANCE_SYSTEM_PROMPT",
        "ProceduralMemory.judge_relevance",
    ),
}
PERCEPTION_TIERS = (
    (1, PerceptionChannels(properties=False, signals=False)),
    (2, PerceptionChannels(properties=False, signals=True)),
    (3, PerceptionChannels(properties=True, signals=True)),
)


class _CaptureClient:
    model = "snapshot-no-provider"

    def __init__(self) -> None:
        self.requests: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> str:
        self.requests.append(request)
        responses = {
            "plan": '{"steps":[]}',
            "ground": '{"params":{"query":"blue"}}',
            "select": '{"keep":[0]}',
            "revalidate": '{"valid":true}',
            "condition": '{"fired":[],"retired":[]}',
            "retirement": '{"retired":[]}',
            "relevance": '{"relevant":false}',
        }
        return responses[request.semantic_label]


async def _capture_requests(channels: PerceptionChannels) -> list[CompletionRequest]:
    client = _CaptureClient()
    observed = PerceptSnapshot(channels=channels)
    with tempfile.TemporaryDirectory(prefix="sora-prompt-snapshot-") as tmp:
        memory = ProceduralMemory(FileMemoryBackend(Path(tmp)), llm=client)
        activity = Activity(id="snapshot-activity", goal="Find the requested record", context={})
        await memory.infer(activity, {}, observed=observed)
        await memory.ground(
            activity,
            "search",
            None,
            {"query": {"$bind": "requested_query"}},
            observed=observed,
        )
        await memory.select(
            activity,
            [{"id": "item-1", "label": "blue"}, {"id": "item-2", "label": "red"}],
            "the item whose label is blue",
            observed=observed,
        )
        activity.plan = Plan(
            id="snapshot-plan",
            goal=activity.goal,
            steps=[Step("send", {"to": "user", "content": {"text": "done"}})],
        )
        await memory.revalidate(activity, observed=observed)
        condition = PendingCondition(
            watch=SignalWait(
                signal_name="state_changed",
                source="records",
                path="items",
                kind="updated",
            ),
            when="the requested record changes",
            then="check the record again",
            until=Until("the review window closes", seconds=3600),
        )
        await memory.evaluate_conditions(
            activity,
            [condition],
            [("records", Change(path="items", updated=("item-1",)))],
            observed,
        )
        await memory.judge_retirement(activity, [condition], observed)
        await memory.judge_relevance(
            [
                {
                    "activity_id": "snapshot-finished",
                    "goal": "Find the requested record",
                    "succeeded": True,
                    "summary": "The record was found and reported.",
                }
            ],
            [("records", Change(path="items", updated=("item-1",)))],
            observed,
        )
    return client.requests


def _prompt_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for perception_tier, channels in PERCEPTION_TIERS:
        requests = asyncio.run(_capture_requests(channels))
        if tuple(request.semantic_label for request in requests) != PROMPT_LABELS:
            raise ValueError("runtime semantic prompt inventory no longer matches the frozen seven")
        for request in requests:
            source_file, system_symbol, renderer = PROMPT_SOURCES[request.semantic_label]
            rows.append(
                {
                    "semantic_label": request.semantic_label,
                    "prompt_version": request.prompt_version,
                    "perception_tier": perception_tier,
                    "perception_channels": asdict(channels),
                    "source": {
                        "file": source_file,
                        "system_symbol": system_symbol,
                        "renderer": renderer,
                    },
                    "system": request.system,
                    "user": request.user,
                    "system_sha256": sha256_text(request.system),
                    "user_sha256": sha256_text(request.user),
                    "sections": [asdict(section) for section in request.sections],
                    "request_hints": {
                        "max_output_tokens": (
                            request.profile.max_output_tokens
                            if request.profile is not None
                            else None
                        ),
                        "reasoning": (
                            request.profile.reasoning if request.profile is not None else None
                        ),
                    },
                }
            )
    return rows


def build_prompt_snapshot(*, source_revision: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "source_revision": source_revision,
            # The isolated baseline was captured before any prompt source changed. Task-owned
            # harness or planning-document changes do not make the source prompt dirty.
            "prompt_source_dirty_diff_sha256": None,
        },
        "prompts": _prompt_rows(),
    }


def build_frozen_baseline(
    *,
    source_revision: str,
    root: Path,
    source_dirty_diff_sha256: str | None = None,
) -> dict[str, Any]:
    snapshot = build_prompt_snapshot(source_revision=source_revision)
    snapshot["provenance"]["source_dirty_diff_sha256"] = source_dirty_diff_sha256
    profiles = load_profiles(root / "profiles.json")
    snapshot["pre_task_gaia_agent_settings"] = {
        "config": "examples/gaia2/agent.yaml",
        "client": "sora.adapters.anthropic_llm.AnthropicLLMClient",
        "model": "claude-opus-4-8",
        "max_output_tokens": 32000,
        "thinking": "adaptive",
        "instrument": True,
        "request_profile_applied_per_semantic_call": False,
    }
    snapshot["evaluation_profiles"] = [profiles[name].to_dict() for name in sorted(profiles)]
    snapshot["judge_profile"] = load_judge_profile(
        root / "campaigns" / "prompt" / "judge.json"
    ).to_dict()
    snapshot["notes"] = {
        "campaigns": ["prompt", "aamas2027"],
        "contains_live_model_output": False,
        "contains_gaia_payloads": False,
        "contains_oracles": False,
        "contains_credentials": False,
        "reasoning_profile_mapping_deferred_to_item": 6,
        "statistical_bootstrap_seed": 20260831,
        "gaia_logical_agent_llm_call_limit": 200,
        "gaia_limit_unit": "logical_agent_llm_call",
        "provider_retries_consume_additional_call_admissions": False,
        "parser_repair_consumes_additional_call_admissions": False,
        "decision_cycles_are_benchmark_steps": False,
        "reported_architecture_diagnostics": [
            "external_actions",
            "decision_cycles",
            "prop_reads",
            "distinct_prop_reads",
        ],
        "prompt_profile": "gpt-5.4-medium-prompt",
        "cross_family_profile": "kimi-k2.5-prompt",
        "paper_transfer_profile": "gpt-5.4-high-paper",
        "gpt_5_4_temperature": (
            "intentionally omitted; the reasoning profiles support only the default value 1"
        ),
        "kimi_snapshot_status": "stable alias; no dated OpenRouter snapshot",
        "kimi_provider": (
            "Venice endpoint pinned after DeepInfra retired its route; chosen by measurement, "
            "because the endpoint fixes the decode rate and the output-cap semantics that the "
            "latency grid depends on, and neither is visible in what a provider declares"
        ),
        "kimi_reasoning": "OpenRouter unified reasoning enabled with provider pinned",
    }
    return snapshot


def load_frozen_snapshot(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported prompt snapshot schema in {path}")
    return raw


def request_as_dict(request: CompletionRequest) -> dict[str, Any]:
    """Useful to downstream tools inspecting a capture without exposing provider state."""
    return asdict(request)
