from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

SCHEMA_VERSION = 1
CAPABILITIES = ("search", "execution", "adaptability", "ambiguity", "time")
GAIA_SUITES = ("familiar", "development", "acceptance")
NON_EVALUABLE_GAIA_TERMINAL_CAUSES = frozenset(
    {
        "llm_call_limit",
        "context_overflow",
        "timeout",
        "infrastructure_error",
        "unscored_completion",
    }
)
NEUTRAL_CASE_IDS = (
    "lookup-ordinary",
    "lookup-adversarial",
    "joins-ordinary",
    "joins-adversarial",
    "dates-ordinary",
    "dates-adversarial",
    "fanout-ordinary",
    "fanout-adversarial",
    "communication-ordinary",
    "communication-adversarial",
    "replanning-ordinary",
    "replanning-adversarial",
    "windows-ordinary",
    "windows-adversarial",
    "malformed-ordinary",
    "malformed-adversarial",
)
SETTING_NAMES = (
    "reasoning_effort",
    "reasoning",
    "temperature",
    "top_p",
    "seed",
    "verbosity",
    "max_output_tokens",
    "service_tier",
    "provider_routing",
    "router_metadata",
)
SettingStatus = Literal["sent", "intentionally_omitted", "provider_observed"]
Arm = str
Campaign = Literal["prompt", "paper2027"]
TerminalCause = Literal[
    "verification_completion",
    "llm_call_limit",
    "context_overflow",
    "timeout",
    "infrastructure_error",
    "unscored_completion",
]


def canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SettingValue:
    status: SettingStatus
    value: Any = None

    @classmethod
    def from_dict(cls, raw: object) -> SettingValue:
        if not isinstance(raw, dict):
            raise ValueError("profile setting must be an object")
        status = raw.get("status")
        if status not in {"sent", "intentionally_omitted", "provider_observed"}:
            raise ValueError(f"invalid setting status: {status!r}")
        value = raw.get("value")
        if status == "sent" and value is None:
            raise ValueError("a sent setting must carry a non-null value")
        if status == "intentionally_omitted" and value is not None:
            raise ValueError("an intentionally omitted setting must have a null value")
        return cls(status=cast(SettingStatus, status), value=value)


@dataclass(frozen=True)
class EndpointIdentity:
    """The declared serving path that endpoint-specific measurements are bound to.

    This catches a profile repin, not a provider silently moving weights or service behind an
    unchanged routing pin; validating frozen measurements against real calls remains separate.
    """

    provider: str
    model: str
    provider_routing_json: str | None

    @classmethod
    def create(
        cls, provider: str, model: str, provider_routing: Mapping[str, Any] | None
    ) -> EndpointIdentity:
        routing_json = (
            json.dumps(provider_routing, sort_keys=True, separators=(",", ":"))
            if provider_routing is not None
            else None
        )
        return cls(provider=provider, model=model, provider_routing_json=routing_json)

    @classmethod
    def from_artifact(cls, model: str, row: Mapping[str, Any]) -> EndpointIdentity:
        if "provider_routing" not in row:
            raise ValueError(f"endpoint measurement for {model} requires provider_routing")
        routing = row["provider_routing"]
        if routing is not None and not isinstance(routing, dict):
            raise ValueError(f"provider_routing for {model} must be an object or null")
        return cls.create(str(row["provider"]), model, routing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "provider_routing": (
                json.loads(self.provider_routing_json)
                if self.provider_routing_json is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ModelProfile:
    name: str
    provider: str
    client: str
    model: str
    endpoint: str
    credential_env: str
    campaigns: tuple[Campaign, ...]
    settings: dict[str, SettingValue]
    stream: bool
    stall_timeout: float | None
    sdk_max_retries: int
    instrument: bool

    @classmethod
    def from_dict(cls, raw: object) -> ModelProfile:
        if not isinstance(raw, dict):
            raise ValueError("model profile must be an object")
        required = {"name", "provider", "client", "model", "endpoint", "credential_env"}
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"model profile missing fields: {', '.join(missing)}")
        settings_raw = raw.get("settings")
        if not isinstance(settings_raw, dict):
            raise ValueError("model profile settings must be an object")
        # Validate every supplied value before reporting inventory mismatch, so a misspelled
        # disposition is diagnosed as such even in an otherwise incomplete profile draft.
        for setting in settings_raw.values():
            SettingValue.from_dict(setting)
        unknown = sorted(set(settings_raw) - set(SETTING_NAMES))
        missing_settings = sorted(set(SETTING_NAMES) - set(settings_raw))
        if unknown or missing_settings:
            raise ValueError(
                f"model profile settings mismatch; unknown={unknown}, missing={missing_settings}"
            )
        settings = {name: SettingValue.from_dict(settings_raw[name]) for name in SETTING_NAMES}
        campaigns_raw = raw.get("campaigns")
        if (
            not isinstance(campaigns_raw, list)
            or not campaigns_raw
            or any(campaign not in {"prompt", "paper2027"} for campaign in campaigns_raw)
        ):
            raise ValueError("model profile campaigns must name prompt and/or paper2027")
        if len(campaigns_raw) != len(set(campaigns_raw)):
            raise ValueError("model profile campaigns must be unique")
        retries = raw.get("sdk_max_retries")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("sdk_max_retries must be a non-negative integer")
        timeout = raw.get("stall_timeout")
        if timeout is not None and not isinstance(timeout, int | float):
            raise ValueError("stall_timeout must be numeric or null")
        return cls(
            name=str(raw["name"]),
            provider=str(raw["provider"]),
            client=str(raw["client"]),
            model=str(raw["model"]),
            endpoint=str(raw["endpoint"]),
            credential_env=str(raw["credential_env"]),
            campaigns=tuple(cast(Campaign, campaign) for campaign in campaigns_raw),
            settings=settings,
            stream=bool(raw.get("stream")),
            stall_timeout=float(timeout) if timeout is not None else None,
            sdk_max_retries=retries,
            instrument=bool(raw.get("instrument")),
        )

    def client_settings(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "client": self.client,
            "model": self.model,
            "base_url": self.endpoint,
            "api_key_env": self.credential_env,
            "stream": self.stream,
            "stall_timeout": self.stall_timeout,
            "max_retries": self.sdk_max_retries,
            "instrument": self.instrument,
        }
        key_map = {"max_output_tokens": "max_tokens"}
        for name, setting in self.settings.items():
            if setting.status == "sent":
                values[key_map.get(name, name)] = setting.value
        return values

    def endpoint_identity(self) -> EndpointIdentity:
        routing = self.settings["provider_routing"]
        value = routing.value if routing.status == "sent" else None
        if value is not None and not isinstance(value, dict):
            raise ValueError("a sent provider_routing setting must be an object")
        return EndpointIdentity.create(self.provider, self.model, value)

    def request_kwargs(self) -> dict[str, Any]:
        """The profile's operating point as chat-completions request kwargs.

        One mapping, shared by every caller that speaks to a provider directly — the designed
        latency grid and the ReAct arm's engine — because the two arms are comparable only if they
        are the same request. ARE's ``LiteLLMEngine`` sends none of these: it ignores
        ``chat_completion``'s ``**kwargs`` and hands ``litellm.completion`` a fixed five arguments,
        so a caller that does not apply this dict runs the baseline at the provider's defaults
        while S-ORA runs at the profile's. That difference lands on the *per-arm* comparison rather
        than on either arm's own numbers, which is the one place it cannot be corrected afterwards.

        Transport is deliberately absent — ``stall_timeout`` and ``sdk_max_retries`` belong to
        whatever object opens the connection (a client's ``timeout``/``max_retries``, LiteLLM's
        per-call ``timeout``/``num_retries``), not to the operating point. A setting the profile
        marks ``intentionally_omitted`` is absent from the dict rather than present-and-null, so
        the request carries no key for it at all."""
        settings = self.client_settings()
        kwargs: dict[str, Any] = {
            name: settings[name]
            for name in (
                "reasoning_effort",
                "temperature",
                "top_p",
                "seed",
                "verbosity",
                "service_tier",
            )
            if name in settings
        }
        if "max_tokens" in settings:
            kwargs["max_completion_tokens"] = settings["max_tokens"]
        extra_body = {
            key: settings[name]
            for key, name in (("reasoning", "reasoning"), ("provider", "provider_routing"))
            if name in settings
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        if settings.get("router_metadata") is True:
            kwargs["extra_headers"] = {"X-OpenRouter-Metadata": "enabled"}
        return kwargs

    def reported_fields(self) -> dict[str, dict[str, Any]]:
        fields = {
            "provider": self.provider,
            "client": self.client,
            "model": self.model,
            "endpoint": self.endpoint,
            # The environment-variable name is provenance; its secret value is never reported.
            "credential_env": self.credential_env,
            "campaigns": list(self.campaigns),
            "stream": self.stream,
            "stall_timeout": self.stall_timeout,
            "sdk_max_retries": self.sdk_max_retries,
            "instrument": self.instrument,
        }
        reported = {name: {"status": "sent", "value": value} for name, value in fields.items()}
        reported.update({name: asdict(value) for name, value in self.settings.items()})
        return reported

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "client": self.client,
            "model": self.model,
            "endpoint": self.endpoint,
            "credential_env": self.credential_env,
            "campaigns": list(self.campaigns),
            "settings": {name: asdict(value) for name, value in self.settings.items()},
            "stream": self.stream,
            "stall_timeout": self.stall_timeout,
            "sdk_max_retries": self.sdk_max_retries,
            "instrument": self.instrument,
            "reported_fields": self.reported_fields(),
        }


# What an OpenRouter model id denotes. `moonshotai/kimi-k2.5` is a family alias served today from
# the snapshot `moonshotai/kimi-k2.5-0127`, and the dated form is **not** an addressable pin: the
# API accepts it, routes it identically, and normalizes it back to the alias in the response, so a
# profile has no way to name a snapshot. What is left is noticing the swap, which the models
# listing does report — a repointed alias carries a different `canonical_slug`. Recorded here
# rather than as a profile setting because it is an observation about the provider, not something
# the request carries, and it lives next to `ModelProfile` rather than in either driver because
# both the ReAct preflight and the latency grid have to be able to ask.
MODEL_SNAPSHOTS: dict[str, str] = {"moonshotai/kimi-k2.5": "moonshotai/kimi-k2.5-0127"}


def served_snapshot(profile: ModelProfile) -> str | None:
    """The snapshot the provider currently serves ``profile.model`` from, or None if unreadable.

    Read from the models listing, which is free and carries no tokens; there is no single-model
    lookup, so this pulls the whole catalogue once. Unreadable is deliberately not a failure — the
    check is provenance, and a listing that did not load says nothing either way, where a slug that
    loaded and differs says the alias moved under us."""
    import urllib.request

    key = os.environ.get(profile.credential_env)
    request = urllib.request.Request(
        f"{profile.endpoint.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {key}"} if key else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 — profile URL
            catalogue = json.load(response)
    except Exception:  # noqa: BLE001 — an unreadable listing is reported, never fatal
        return None
    for entry in catalogue.get("data", []):
        if entry.get("id") == profile.model:
            slug = entry.get("canonical_slug")
            return str(slug) if slug else None
    return None


# Whether a provider's ``completion_tokens`` already contains the reasoning tokens it reports
# separately. Both endpoints measured so far do, but it must stay a declaration rather than an
# assumption, because where the two counts are disjoint the tokens the model actually decoded are
# their *sum*, and reading ``completion_tokens`` as the decode work would then silently under-count
# it — in the latency grid, by pushing the difference into ``a0``.
#
# Keyed by the same declared endpoint identity as prices and latency coefficients because this is
# a property of the endpoint, not of the weights: the same model served from two providers can
# report either way, so a repin owes this table a new entry exactly as it owes a new price sheet.
#
# **Establish the value from an *uncapped* generation, never from grid rows.** The tempting test is
# arithmetic and one-sided — reasoning cannot exceed completion if it is contained in it, and 31 of
# kimi's 217 usage-carrying grid rows report more reasoning than completion. That test is invalid
# here and reading it the obvious way is what put ``False`` in this table until 2026-09-14. Every
# grid row is generated against a tight ``max_completion_tokens`` and finishes ``length``: all 217
# report ``completion_tokens`` clamped to exactly the designed cap, while ``reasoning_tokens``
# arrives by a different count that can overshoot it by a few tokens. Two counts truncated
# differently look disjoint whether or not they are, so a capped row carries no information about
# the convention at all.
#
# What settles it is one generation that stops on its own, where content and reasoning can be
# compared against the total. On the pinned Venice endpoint, 2026-09-14, ``finish_reason=stop``:
# content "391" (~2 tokens) with 113 reasoning tokens reported ``completion_tokens=115``; a 120-word
# answer (794 characters, ~167 tokens) with 3,099 reasoning tokens reported 3,266. Both are
# content + reasoning to within a token or two, so reasoning is contained, exactly as on OpenAI.
DECODE_INCLUDES_REASONING: dict[EndpointIdentity, bool] = {
    EndpointIdentity.create("openai", "gpt-5.4-2026-03-05", None): True,
    EndpointIdentity.create(
        "openrouter",
        "moonshotai/kimi-k2.5",
        {"only": ["venice"], "order": ["venice"], "allow_fallbacks": False},
    ): True,
}


def decode_includes_reasoning(profile: ModelProfile) -> bool | None:
    """Whether this endpoint folds reasoning into ``completion_tokens``, or None if undeclared.

    None is a third answer, not a default: an unrecorded endpoint that reports reasoning at all has
    an *undefined* decode count, and a caller that guesses either way is fabricating a regressor or
    a charge. Callers must treat None as "cannot be used", the same way the grid already treats a
    cached cell whose provider reported no cache count."""
    return DECODE_INCLUDES_REASONING.get(profile.endpoint_identity())


def decode_tokens(
    completion: int | None, reasoning: int | None, *, includes_reasoning: bool | None
) -> int | None:
    """The tokens the model actually decoded, or None where that cannot be determined.

    Reasoning of zero or None makes the convention irrelevant, which is why an undeclared endpoint
    is not automatically unusable — only one that reports reasoning it might or might not have
    already counted."""
    if completion is None:
        return None
    if not reasoning:
        return completion
    if includes_reasoning is None:
        return None
    return completion if includes_reasoning else completion + reasoning


def load_profiles(path: Path) -> dict[str, ModelProfile]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported profile schema in {path}")
    rows = raw.get("profiles")
    if not isinstance(rows, list):
        raise ValueError("profiles must be an array")
    profiles: dict[str, ModelProfile] = {}
    for row in rows:
        profile = ModelProfile.from_dict(row)
        if profile.name in profiles:
            raise ValueError(f"duplicate profile name: {profile.name}")
        profiles[profile.name] = profile
    return profiles


@dataclass(frozen=True)
class JudgeProfile:
    name: str
    implementation: str
    provider: str
    model: str
    endpoint: str | None
    credential_env: str
    settings: dict[str, SettingValue]
    offline_validation: bool
    relax_verdict_case: bool

    @classmethod
    def from_dict(cls, raw: object) -> JudgeProfile:
        if not isinstance(raw, dict):
            raise ValueError("judge profile must be an object")
        required = {
            "name",
            "implementation",
            "provider",
            "model",
            "endpoint",
            "credential_env",
            "settings",
            "offline_validation",
            "relax_verdict_case",
        }
        if set(raw) != required:
            raise ValueError(
                "judge profile fields mismatch; "
                f"unknown={sorted(set(raw) - required)}, missing={sorted(required - set(raw))}"
            )
        settings_raw = raw["settings"]
        if not isinstance(settings_raw, dict) or set(settings_raw) != set(SETTING_NAMES):
            raise ValueError(f"judge profile settings must name exactly {SETTING_NAMES}")
        settings = {name: SettingValue.from_dict(settings_raw[name]) for name in SETTING_NAMES}
        if any(setting.status == "sent" for setting in settings.values()):
            raise ValueError("the ARE judge seam does not support sent per-request settings")
        if raw["implementation"] != "are.graph_per_event":
            raise ValueError("unsupported judge implementation")
        return cls(
            name=str(raw["name"]),
            implementation=str(raw["implementation"]),
            provider=str(raw["provider"]),
            model=str(raw["model"]),
            endpoint=str(raw["endpoint"]) if raw["endpoint"] is not None else None,
            credential_env=str(raw["credential_env"]),
            settings=settings,
            offline_validation=bool(raw["offline_validation"]),
            relax_verdict_case=bool(raw["relax_verdict_case"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation": self.implementation,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "credential_env": self.credential_env,
            "settings": {name: asdict(value) for name, value in self.settings.items()},
            "offline_validation": self.offline_validation,
            "relax_verdict_case": self.relax_verdict_case,
        }


def load_judge_profile(path: Path) -> JudgeProfile:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported judge profile schema in {path}")
    return JudgeProfile.from_dict(raw.get("judge"))


@dataclass(frozen=True)
class ManifestCase:
    capability: str
    case_id: str


@dataclass(frozen=True)
class SuiteManifest:
    suite: str
    cases: tuple[ManifestCase, ...]
    digest: str


def _load_manifest(path: Path) -> SuiteManifest:
    raw_text = path.read_text()
    raw = json.loads(raw_text)
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema in {path}")
    suite = raw.get("suite")
    rows = raw.get("cases")
    if suite not in GAIA_SUITES or not isinstance(rows, list):
        raise ValueError(f"invalid suite manifest {path}")
    cases: list[ManifestCase] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"capability", "id"}:
            raise ValueError(f"manifest {path} must contain capability/id only")
        capability = row["capability"]
        case_id = row["id"]
        if capability not in CAPABILITIES or not isinstance(case_id, str) or not case_id:
            raise ValueError(f"invalid manifest case in {path}: {row!r}")
        cases.append(ManifestCase(capability, case_id))
    if len(cases) != 5 or {case.capability for case in cases} != set(CAPABILITIES):
        raise ValueError(f"manifest {path} must have one case per Gaia capability")
    return SuiteManifest(str(suite), tuple(cases), sha256_text(canonical_json(raw)))


def load_manifests(root: Path) -> dict[str, SuiteManifest]:
    loaded = [_load_manifest(path) for path in sorted(root.glob("*.json"))]
    manifests = {manifest.suite: manifest for manifest in loaded}
    if set(manifests) != set(GAIA_SUITES):
        raise ValueError(f"expected manifests {GAIA_SUITES}, found {tuple(manifests)}")
    ids = [case.case_id for manifest in manifests.values() for case in manifest.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Gaia manifest case ids must be globally unique")
    return manifests


class ManifestLockedError(PermissionError):
    pass


def resolve_scenario(
    scenario_root: Path,
    suite: str,
    capability: str,
    case_id: str,
    *,
    ack_locked_acceptance: bool = False,
) -> Path:
    if suite == "acceptance" and not ack_locked_acceptance:
        raise ManifestLockedError(
            "acceptance payload is locked; pass --ack-locked-acceptance before opening it"
        )
    capability_root = scenario_root / capability
    matches = [path for path in capability_root.glob(f"*{case_id}*.json") if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one ignored payload for {suite}/{capability}/{case_id}, "
            f"found {len(matches)} under {capability_root}"
        )
    return matches[0]


@dataclass(frozen=True)
class BudgetPolicy:
    max_gaia_runs: int = 60
    max_total_spend: float = 180.0
    unknown_gaia_agent_reserve: float = 3.0
    unknown_gaia_judge_reserve: float = 0.5
    unknown_neutral_live_reserve: float = 0.5


@dataclass(frozen=True)
class RunSelection:
    profiles: tuple[str, ...]
    suites: tuple[str, ...]
    arm: Arm
    repeats: int = 1
    live_neutral: bool = False


@dataclass(frozen=True)
class MatrixEntry:
    profile: str
    suite: str
    capability: str
    case_id: str
    arm: Arm
    repeat: int
    reserved_agent_cost: float
    reserved_judge_cost: float

    @property
    def key(self) -> str:
        return f"{self.arm}:{self.profile}:{self.suite}:{self.case_id}:{self.repeat}"


@dataclass(frozen=True)
class RunMatrix:
    entries: tuple[MatrixEntry, ...]
    gaia_runs: int
    agent_reserve: float
    judge_reserve: float
    prior_gaia_runs: int = 0
    prior_spend: float = 0.0

    @property
    def total_reserve(self) -> float:
        return self.agent_reserve + self.judge_reserve


def build_run_matrix(
    selection: RunSelection,
    policy: BudgetPolicy,
    manifests: dict[str, SuiteManifest] | None = None,
    *,
    prior_records: list[EvaluationRecord] | None = None,
) -> RunMatrix:
    if not selection.profiles or not selection.suites:
        raise ValueError("at least one explicit profile and suite are required")
    if selection.repeats <= 0:
        raise ValueError("repeats must be positive")
    manifests = manifests or {
        suite: SuiteManifest(
            suite,
            tuple(ManifestCase(capability, f"{suite}-{capability}") for capability in CAPABILITIES),
            "",
        )
        for suite in GAIA_SUITES
    }
    entries: list[MatrixEntry] = []
    for profile in selection.profiles:
        for suite in selection.suites:
            if suite in GAIA_SUITES:
                cases = manifests[suite].cases
                agent_reserve = policy.unknown_gaia_agent_reserve
                judge_reserve = policy.unknown_gaia_judge_reserve
            elif suite == "neutral":
                cases = tuple(ManifestCase("neutral", case_id) for case_id in NEUTRAL_CASE_IDS)
                agent_reserve = (
                    policy.unknown_neutral_live_reserve if selection.live_neutral else 0.0
                )
                judge_reserve = 0.0
            elif suite == "contract":
                cases = (ManifestCase("contract", "contract"),)
                agent_reserve = judge_reserve = 0.0
            else:
                raise ValueError(f"unknown suite: {suite}")
            repeat_count = selection.repeats if suite in GAIA_SUITES else 1
            for repeat in range(repeat_count):
                for case in cases:
                    entries.append(
                        MatrixEntry(
                            profile,
                            suite,
                            case.capability,
                            case.case_id,
                            selection.arm,
                            repeat,
                            agent_reserve,
                            judge_reserve,
                        )
                    )
    prior = prior_records or []
    prior_keys = {
        f"{record.arm}:{record.profile}:{record.suite}:{record.case_id}:{record.repeat}"
        for record in prior
        if record.completes_matrix_entry
    }
    remaining_entries = [entry for entry in entries if entry.key not in prior_keys]
    gaia_runs = sum(entry.suite in GAIA_SUITES for entry in remaining_entries)
    agent_total = sum(entry.reserved_agent_cost for entry in remaining_entries)
    judge_total = sum(entry.reserved_judge_cost for entry in remaining_entries)
    prior_gaia_runs = sum(record.suite in GAIA_SUITES for record in prior)
    prior_spend = sum(record.accounted_agent_cost + record.judge_reserve for record in prior)
    if prior_gaia_runs + gaia_runs > policy.max_gaia_runs:
        prefix = "cumulative " if prior_gaia_runs else ""
        raise ValueError(
            f"{prefix}projected {prior_gaia_runs + gaia_runs} Gaia runs exceeds "
            f"{prefix}Gaia run ceiling {policy.max_gaia_runs}"
        )
    if prior_spend + agent_total + judge_total > policy.max_total_spend:
        prefix = "cumulative " if prior_spend else ""
        raise ValueError(
            f"{prefix}projected ${prior_spend + agent_total + judge_total:.2f} exceeds "
            f"{prefix}spend ceiling "
            f"${policy.max_total_spend:.2f}"
        )
    return RunMatrix(
        tuple(entries),
        gaia_runs,
        agent_total,
        judge_total,
        prior_gaia_runs=prior_gaia_runs,
        prior_spend=prior_spend,
    )


@dataclass(frozen=True)
class PriceTier:
    max_input_tokens: int | None
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class PriceSheet:
    effective_date: str
    currency: str
    endpoints: dict[EndpointIdentity, tuple[PriceTier, ...]]
    digest: str

    @classmethod
    def load(cls, path: Path) -> PriceSheet:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported price sheet schema in {path}")
        if not raw.get("effective_date"):
            raise ValueError("price sheet requires an effective_date")
        models_raw = raw.get("models")
        if not isinstance(models_raw, dict) or not models_raw:
            raise ValueError("price sheet requires model rates")
        endpoints: dict[EndpointIdentity, tuple[PriceTier, ...]] = {}
        for model, entry_raw in models_raw.items():
            if not isinstance(entry_raw, dict):
                raise ValueError(f"invalid price sheet entry for {model}")
            endpoint = EndpointIdentity.from_artifact(str(model), entry_raw)
            tiers_raw = entry_raw.get("tiers")
            if not isinstance(tiers_raw, list) or not tiers_raw:
                raise ValueError(f"price sheet model {model} needs at least one tier")
            tiers: list[PriceTier] = []
            for row in tiers_raw:
                if not isinstance(row, dict):
                    raise ValueError(f"invalid price tier for {model}")
                tiers.append(
                    PriceTier(
                        max_input_tokens=(
                            int(row["max_input_tokens"])
                            if row.get("max_input_tokens") is not None
                            else None
                        ),
                        input_per_million=float(row["input_per_million"]),
                        cached_input_per_million=float(row["cached_input_per_million"]),
                        output_per_million=float(row["output_per_million"]),
                    )
                )
            if tiers[-1].max_input_tokens is not None:
                raise ValueError(f"last price tier for {model} must be unbounded")
            endpoints[endpoint] = tuple(tiers)
        return cls(
            effective_date=str(raw["effective_date"]),
            currency=str(raw.get("currency", "USD")),
            endpoints=endpoints,
            digest=sha256_text(canonical_json(raw)),
        )

    def for_profile(self, profile: ModelProfile) -> tuple[PriceTier, ...]:
        endpoint = profile.endpoint_identity()
        prices = self.endpoints.get(endpoint)
        if prices is None:
            same_model = [key.to_dict() for key in self.endpoints if key.model == profile.model]
            if same_model:
                raise ValueError(
                    f"price sheet endpoint mismatch for profile {profile.name}: "
                    f"artifact={same_model}, profile={endpoint.to_dict()}"
                )
            raise ValueError(f"price sheet has no rate for model {profile.model}")
        return prices


@dataclass(frozen=True)
class ChargeCoefficients:
    """One endpoint's frozen latency model: seconds charged for a call of a given token shape.

    Stored as seconds *per token* rather than as the tokens-per-second rates the paper reports,
    because the per-token coefficients are what the fit produces and what the charge multiplies;
    keeping both in the file would let a rounded rate and its reciprocal disagree. The rates are
    derived here instead."""

    model: str
    provider: str
    provider_routing_json: str | None
    a0_seconds: float
    seconds_per_uncached_input_token: float
    seconds_per_cached_input_token: float
    seconds_per_output_token: float
    decode_includes_reasoning: bool
    fit: Mapping[str, Any]
    note: str

    def charged_seconds(self, uncached_input: int, cached_input: int, output: int) -> float:
        return (
            self.a0_seconds
            + uncached_input * self.seconds_per_uncached_input_token
            + cached_input * self.seconds_per_cached_input_token
            + output * self.seconds_per_output_token
        )

    @property
    def r_in_tokens_per_second(self) -> float:
        return 1.0 / self.seconds_per_uncached_input_token

    @property
    def r_cache_tokens_per_second(self) -> float:
        return 1.0 / self.seconds_per_cached_input_token

    @property
    def r_out_tokens_per_second(self) -> float:
        return 1.0 / self.seconds_per_output_token


@dataclass
class TotalInputCharge:
    """Adapt total prompt usage to the disjoint regressors the frozen fit was trained on."""

    coefficients: ChargeCoefficients
    cached_input_clamps: int = 0
    charged_seconds: float = 0.0

    def __call__(self, total_input: int, cached_input: int, output: int) -> float:
        uncached_input = total_input - cached_input
        if uncached_input < 0:
            # Usage telemetry is not reliable enough to abort a paid sweep. Preserve the reported
            # cached work, clamp only the impossible residual, and leave an inspectable count.
            self.cached_input_clamps += 1
            uncached_input = 0
        charged = self.coefficients.charged_seconds(uncached_input, cached_input, output)
        self.charged_seconds += charged
        return charged

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "model": self.coefficients.model,
            "provider": self.coefficients.provider,
            "provider_routing": (
                json.loads(self.coefficients.provider_routing_json)
                if self.coefficients.provider_routing_json is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ChargeModelSheet:
    """The frozen charge model, keyed by declared endpoint identity.

    Coefficients live here rather than as constants in code for the same reason the prices do: they
    are a measurement with a date and a provenance, and a run has to be able to say which ones it
    charged. Editing this file silently invalidates comparison against every number recorded before
    the edit, which is why it is pinned by sha256 in the campaign baseline alongside the prompts —
    one gate, not a second mechanism."""

    effective_date: str
    measured_date: str
    endpoints: Mapping[EndpointIdentity, ChargeCoefficients]
    notes: Mapping[str, str]
    digest: str

    @classmethod
    def load(cls, path: Path) -> ChargeModelSheet:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported charge model schema in {path}")
        for field_name in ("effective_date", "measured_date"):
            if not raw.get(field_name):
                raise ValueError(f"charge model requires a {field_name}")
        models_raw = raw.get("models")
        if not isinstance(models_raw, dict) or not models_raw:
            raise ValueError("charge model requires per-model coefficients")
        endpoints: dict[EndpointIdentity, ChargeCoefficients] = {}
        for model, row in models_raw.items():
            if not isinstance(row, dict):
                raise ValueError(f"invalid charge model entry for {model}")
            provider = str(row["provider"])
            declared = bool(row["decode_includes_reasoning"])
            # The convention is copied from ``DECODE_INCLUDES_REASONING``, never authored here, and
            # this is where that stays true. It has to be repeated in the file because it decides
            # which regressor the coefficients beside it were fitted against — a frozen ``a0`` whose
            # convention is not frozen with it is underdetermined, and reproducing the fit under the
            # other reading moves kimi's by a factor of two. Repeating a value is how copies drift,
            # so disagreement is an error rather than a precedence question: whichever of the two is
            # wrong, the coefficients no longer describe the endpoint they are charged to.
            endpoint = EndpointIdentity.from_artifact(str(model), row)
            home = DECODE_INCLUDES_REASONING.get(endpoint)
            if home is None:
                raise ValueError(
                    f"charge model covers {endpoint.to_dict()}, which declares no decode convention"
                )
            if home != declared:
                raise ValueError(
                    f"charge model for {provider}/{model} says decode_includes_reasoning="
                    f"{declared}, but core declares {home}"
                )
            for key in (
                "a0_seconds",
                "seconds_per_uncached_input_token",
                "seconds_per_cached_input_token",
                "seconds_per_output_token",
            ):
                if float(row[key]) <= 0:
                    # A non-positive coefficient prices tokens as free or as time refunded, and the
                    # fit can produce one where an axis is unidentified. It is not chargeable.
                    raise ValueError(f"charge model coefficient {key} for {model} must be positive")
            endpoints[endpoint] = ChargeCoefficients(
                model=str(model),
                provider=provider,
                provider_routing_json=endpoint.provider_routing_json,
                a0_seconds=float(row["a0_seconds"]),
                seconds_per_uncached_input_token=float(row["seconds_per_uncached_input_token"]),
                seconds_per_cached_input_token=float(row["seconds_per_cached_input_token"]),
                seconds_per_output_token=float(row["seconds_per_output_token"]),
                decode_includes_reasoning=declared,
                fit=dict(row.get("fit", {})),
                note=str(row.get("note", "")),
            )
        return cls(
            effective_date=str(raw["effective_date"]),
            measured_date=str(raw["measured_date"]),
            endpoints=endpoints,
            notes=dict(raw.get("notes", {})),
            digest=sha256_text(canonical_json(raw)),
        )

    def for_profile(self, profile: ModelProfile) -> ChargeCoefficients:
        endpoint = profile.endpoint_identity()
        coefficients = self.endpoints.get(endpoint)
        if coefficients is None:
            same_model = [key.to_dict() for key in self.endpoints if key.model == profile.model]
            if same_model:
                raise ValueError(
                    f"charge model endpoint mismatch for profile {profile.name}: "
                    f"artifact={same_model}, profile={endpoint.to_dict()}"
                )
            raise ValueError(f"charge model has no coefficients for {profile.model}")
        return coefficients

    @property
    def models(self) -> Mapping[str, ChargeCoefficients]:
        models: dict[str, ChargeCoefficients] = {}
        for endpoint, row in self.endpoints.items():
            if endpoint.model in models:
                # Artifact schema v1 stores rows under model ids and therefore cannot serialize
                # two endpoints for one model. Refuse that state instead of letting ``to_dict``
                # collapse one row out of the digest-compared campaign snapshot.
                raise ValueError(
                    f"charge model has more than one endpoint for model {endpoint.model}; "
                    "schema v1 cannot represent both"
                )
            models[endpoint.model] = row
        return models

    def charge_for(self, profile: ModelProfile) -> TotalInputCharge:
        """The per-call charge both arms are metered through, bound to one model.

        Matches ``react_engine.ChargeModel`` so the ReAct baseline and S-ORA are charged by the same
        frozen numbers through the same signature; an arm that computed its own would be measuring
        its provider rather than its architecture."""
        return TotalInputCharge(self.for_profile(profile))

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "effective_date": self.effective_date,
            "measured_date": self.measured_date,
            "models": {
                model: {
                    "a0_seconds": row.a0_seconds,
                    "decode_includes_reasoning": row.decode_includes_reasoning,
                    "provider": row.provider,
                    "provider_routing": (
                        json.loads(row.provider_routing_json)
                        if row.provider_routing_json is not None
                        else None
                    ),
                    "r_cache_tokens_per_second": row.r_cache_tokens_per_second,
                    "r_in_tokens_per_second": row.r_in_tokens_per_second,
                    "r_out_tokens_per_second": row.r_out_tokens_per_second,
                    "seconds_per_cached_input_token": row.seconds_per_cached_input_token,
                    "seconds_per_output_token": row.seconds_per_output_token,
                    "seconds_per_uncached_input_token": row.seconds_per_uncached_input_token,
                }
                for model, row in sorted(self.models.items())
            },
        }


@dataclass(frozen=True)
class CallUsage:
    input_tokens: int
    cached_input_tokens: int | None
    output_tokens: int
    reasoning_tokens: int | None = None
    cache_write_input_tokens: int | None = None


@dataclass(frozen=True)
class CostResult:
    agent_cost: float
    uncached_input_cost: float
    cached_input_cost: float
    output_cost: float
    upper_bound: bool
    tier_max_input_tokens: int | None


def calculate_call_cost(sheet: PriceSheet, profile: ModelProfile, usage: CallUsage) -> CostResult:
    tiers = sheet.for_profile(profile)
    tier = next(
        row
        for row in tiers
        if row.max_input_tokens is None or usage.input_tokens <= row.max_input_tokens
    )
    cached = usage.cached_input_tokens
    if cached is not None and not 0 <= cached <= usage.input_tokens:
        raise ValueError("cached input tokens must be between zero and total input tokens")
    upper_bound = cached is None
    cached_tokens = cached or 0
    uncached_tokens = usage.input_tokens - cached_tokens
    uncached_cost = uncached_tokens / 1_000_000 * tier.input_per_million
    cached_cost = cached_tokens / 1_000_000 * tier.cached_input_per_million
    # Charged on ``output_tokens`` alone, because on both measured endpoints ``completion_tokens``
    # already contains the reasoning tokens reported beside it (see ``DECODE_INCLUDES_REASONING``),
    # so this charges them exactly once. Confirmed on the pinned Venice endpoint against
    # OpenRouter's own reported ``usage.cost`` on 2026-09-14: a call reporting (22 prompt, 115
    # completion, 113 reasoning) was charged 3.94079e-04, which is 22*0.532/M + 115*3.325/M to
    # every reported digit — the 113 reasoning tokens are inside the 115 and are paid for there.
    # Adding ``reasoning_tokens`` on top would very nearly double this arm's output cost.
    #
    # ``reasoning_tokens`` is therefore descriptive: carried for reporting, never a cost term. The
    # combination that would break this is an endpoint that reports reasoning *outside*
    # ``completion_tokens`` and bills it; no usage payload distinguishes that after the fact, and it
    # is a property of the pinned endpoint, so a repin re-opens it exactly as it re-opens the rates.
    # The recipe is one uncapped call with ``"usage": {"include": true}`` — a call truncated by the
    # output cap cannot answer it, since both counts are then clamped independently.
    output_cost = usage.output_tokens / 1_000_000 * tier.output_per_million
    return CostResult(
        agent_cost=uncached_cost + cached_cost + output_cost,
        uncached_input_cost=uncached_cost,
        cached_input_cost=cached_cost,
        output_cost=output_cost,
        upper_bound=upper_bound,
        tier_max_input_tokens=tier.max_input_tokens,
    )


@dataclass(frozen=True)
class ExpansionDecision:
    required: bool
    reasons: tuple[str, ...]
    status: Literal["not_required", "required", "deferred"]


def decide_acceptance_expansion(
    *,
    baseline_scores: list[float],
    candidate_scores: list[float],
    safety_sensitive: bool,
    reduces_tool_catalog: bool,
    fresh_payloads_available: bool,
) -> ExpansionDecision:
    if len(baseline_scores) != 5 or len(candidate_scores) != 5:
        raise ValueError("acceptance expansion is decided from exactly five paired cases")
    deltas = [
        candidate - baseline
        for baseline, candidate in zip(baseline_scores, candidate_scores, strict=True)
    ]
    reasons: list[str] = []
    if any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas):
        reasons.append("mixed paired outcomes")
    if abs(sum(deltas) / len(deltas)) <= 0.20:
        reasons.append("near-zero mean paired delta")
    if safety_sensitive:
        reasons.append("safety-sensitive change")
    if reduces_tool_catalog:
        reasons.append("tool-catalog reduction")
    required = bool(reasons)
    status: Literal["not_required", "required", "deferred"]
    if not required:
        status = "not_required"
    elif fresh_payloads_available:
        status = "required"
    else:
        status = "deferred"
    return ExpansionDecision(required, tuple(reasons), status)


# Rows predating the explicit clock field. Distinct from "wall" — a legacy run may have
# been either, and that unknowability is exactly what must not be averaged into a W2 report.
LEGACY_CLOCK_MODE = "legacy"

# The three timing conventions a run can be made under. They are not interchangeable and a file
# mixing them has no coherent aggregate, which is why the mode is recorded per record rather than
# inferred from whether a charge model was attached.
#
# ``token_charged``   freeze the scenario around every model crossing and resume it by a modelled
#                     token charge.
# ``generation_free`` freeze the same way and resume by zero. This is the legacy ARE in-process
#                     convention: its default ``simulated_generation_time_mode="measured"`` pauses
#                     the environment and resumes with ``completion_duration``, which no shipped
#                     ARE engine ever writes, so every row produced through that harness had
#                     generation costing nothing. Not every published row: the newer ``gaia2-cli``
#                     leaderboard runs on a wall clock, so this convention is compatible with the
#                     in-process harness rather than with "the leaderboard" generally.
#                     Reproducible for a fixed trajectory, and requiring no calibration, because
#                     there is no coefficient to drift.
# ``wall``            never freeze; generation costs real elapsed time. Not reproducible — decode
#                     rate is a property of the serving endpoint, not of the model.
CLOCK_MODE_TOKEN_CHARGED = "token_charged"
CLOCK_MODE_GENERATION_FREE = "generation_free"
CLOCK_MODE_WALL = "wall"


def resolve_clock_mode(charge: object | None, generation_free: bool) -> str:
    """The recorded clock mode for one run's timing configuration.

    Both arms derive the label here rather than each re-deriving it from ``charge is None``, which
    is what let a generation-free run be indistinguishable from a wall-clock one: a zero charge and
    no charge are the same object, and only the caller's intent separates them.

    A charge *and* ``generation_free`` is a contradiction, and raises rather than resolving: the two
    arms broke the tie in opposite directions — this function named the run ``token_charged`` and
    S-ORA billed the crossing, while ReAct's per-call charge short-circuited to zero — so the pair
    produced a row whose label disagreed with the clock it actually ran on. The CLIs already reject
    it, but they are not the only callers, and a wrong label is undetectable after the fact."""
    if charge is not None and generation_free:
        raise ValueError(
            "a charge model and generation-free are different clock conventions: a run cannot "
            "bill generation and also resume by zero"
        )
    if charge is not None:
        return CLOCK_MODE_TOKEN_CHARGED
    return CLOCK_MODE_GENERATION_FREE if generation_free else CLOCK_MODE_WALL


def harness_truncation(*, wall_timed_out: bool, judge_timed_out: bool) -> str | None:
    """Which watchdog cut this run short, or None if neither did.

    Named rather than or-ed into one flag: ReAct once reported a stalled judge as a wall-clock cap
    hit, which excluded it from scoring under a label that misdiagnosed it, while S-ORA recorded the
    same stall as neither and scored it. The judge check comes first because a stall is the more
    specific finding — the wall cap can elapse *because* the judge stalled, and the reverse cannot
    happen.

    Shared by both arms so that precedence is one rule rather than two that happen to agree in the
    cases anyone tested: S-ORA reached the same verdict by a different route and disagreed with this
    one whenever both watchdogs had fired, which is exactly the case no test covered."""
    if judge_timed_out:
        return "judge_stall"
    return "wall_clock" if wall_timed_out else None


def freezes_clock(clock_mode: str) -> bool:
    """Whether this clock mode stops the scenario for the duration of a model call.

    Both frozen modes do; they differ only in what they resume by (a modelled charge, or zero).
    Named rather than re-derived from ``charge is not None`` at each call site, which is what let a
    generation-free run silently take the wall-clock path on both arms."""
    return clock_mode != CLOCK_MODE_WALL


# A frozen clock takes generation out of the scenario's budget but not out of the operator's: real
# per-scenario time becomes the scenario timeline *plus* the whole of generation, where an unfrozen
# run is bounded by the timeline alone. Measured S-ORA scenarios project past the ordinary cap under
# that arithmetic while every ReAct one stays inside it, and a cap that fires on one arm only is not
# a shared condition. Raised for this mode specifically rather than globally, so the value every
# other mode has run under is left where it was.
GENERATION_FREE_MAX_WALL_SECONDS = 3600.0
DEFAULT_MAX_WALL_SECONDS = 1200.0


def resolve_max_wall_seconds(given: float | None, generation_free: bool) -> float:
    """The watchdog a run should use, given what the caller asked for and its clock mode.

    Shared by every entry point so the two cannot drift apart: a CLI that offers
    ``--generation-free`` but keeps the ordinary default silently truncates the arm the raised cap
    exists for."""
    if given is not None:
        return given
    return GENERATION_FREE_MAX_WALL_SECONDS if generation_free else DEFAULT_MAX_WALL_SECONDS


@dataclass(frozen=True)
class EvaluationRecord:
    arm: Arm
    profile: str
    suite: str
    capability: str
    case_id: str
    repeat: int
    score: float | None
    passed: bool | None
    missing_writes: int = 0
    surplus_writes: int = 0
    contract_failures: int = 0
    repair_count: int = 0
    replan_count: int = 0
    terminal_parse_failures: int = 0
    authorization_violations: int = 0
    safety_violations: int = 0
    duration_seconds: float = 0.0
    agent_llm_calls: int = 0
    agent_llm_call_limit: int | None = None
    provider_round_trips: int = 0
    external_actions: int = 0
    latency_seconds: float = 0.0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int | None = None
    cache_unknown_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    agent_cost: float | None = 0.0
    # Conservative budget charge used only when provider usage was unavailable and exact cost is
    # therefore unknowable. Live records copy the matrix entry's pre-authorized reserve here.
    agent_cost_reserve: float = 0.0
    agent_cost_upper_bound: bool = False
    judge_reserve: float = 0.0
    judge_profile: dict[str, Any] | None = None
    call_records: tuple[dict[str, Any], ...] = ()
    prompt: str | None = None
    oracle: str | None = None
    trajectory: dict[str, Any] | None = None
    status: str = "complete"
    terminal_cause: TerminalCause | None = None
    decision_cycles: int = 0
    # See `_runner.RunResult.prop_reads`: reads the runtime satisfied off the property snapshot,
    # which a step-loop arm has to pay a tool call plus a model round-trip for. An architecture
    # diagnostic, never a score input.
    prop_reads: int = 0
    distinct_prop_reads: int = 0
    charged_seconds: float = 0.0
    charge_model_identity: dict[str, Any] | None = None
    charge_model_digest: str | None = None
    cached_input_clamps: int = 0
    raw_cached_input_anomalies: int = 0
    charge_accounting_consistent: bool | None = None
    clock_mode: str | None = None
    inference_charge_policy: str | None = None
    llm_wall_seconds: float = 0.0
    llm_wall_union_seconds: float = 0.0
    llm_charged_union_seconds: float | None = 0.0
    llm_round_trips: int = 0
    llm_max_in_flight: int = 0
    llm_overlapped_round_trips: int = 0
    diagnostics: dict[str, Any] | None = None

    @property
    def accounted_agent_cost(self) -> float:
        return self.agent_cost if self.agent_cost is not None else self.agent_cost_reserve

    @property
    def completes_matrix_entry(self) -> bool:
        return not (
            self.suite in GAIA_SUITES and self.terminal_cause in NON_EVALUABLE_GAIA_TERMINAL_CAUSES
        )

    @classmethod
    def example(
        cls,
        *,
        arm: Arm,
        suite: str,
        case_id: str,
        score: float,
        safety_violations: int = 0,
        prompt: str | None = None,
        oracle: str | None = None,
        trajectory: dict[str, Any] | None = None,
    ) -> EvaluationRecord:
        return cls(
            arm=arm,
            profile="example",
            suite=suite,
            capability="search",
            case_id=case_id,
            repeat=0,
            score=score,
            passed=score >= 1.0,
            safety_violations=safety_violations,
            prompt=prompt,
            oracle=oracle,
            trajectory=trajectory,
        )

    def to_dict(self, *, detailed_acceptance: bool = False) -> dict[str, Any]:
        row = asdict(self)
        if self.suite == "acceptance" and not detailed_acceptance:
            row.pop("prompt", None)
            row.pop("oracle", None)
            row.pop("trajectory", None)
            row.pop("call_records", None)
            row.pop("diagnostics", None)
        return row


def record_from_dict(raw: dict[str, Any]) -> EvaluationRecord:
    row = dict(raw)
    # Read Task-4's pre-correction checkpoints without carrying their misleading field names into
    # new reports. Those records counted completed logical calls and provider-visible round trips.
    row.setdefault("agent_llm_calls", row.pop("calls", 0))
    row.setdefault("provider_round_trips", row.pop("round_trips", 0))
    row.pop("step_unit", None)
    row["call_records"] = tuple(row.get("call_records", ()))
    row.setdefault("charged_seconds", 0.0)
    row.setdefault("charge_model_identity", None)
    row.setdefault("charge_model_digest", None)
    row.setdefault("cached_input_clamps", 0)
    row.setdefault("raw_cached_input_anomalies", 0)
    row.setdefault(
        "charge_accounting_consistent",
        (
            row["cached_input_clamps"] == row["raw_cached_input_anomalies"]
            if row.get("charge_model_digest") is not None
            else None
        ),
    )
    # A row written before the clock became explicit carries no mode at all. Reading that as
    # "nothing to compare" makes a legacy row invisible to the mixing check, so a legacy run
    # silently passes as homogeneous next to a wall or token-charged one. Naming it instead
    # keeps a uniformly legacy report readable while making any mixture fail the same way a
    # wall/charged mixture does.
    row.setdefault(
        "clock_mode", "token_charged" if row.get("charge_model_digest") else LEGACY_CLOCK_MODE
    )
    row.setdefault("inference_charge_policy", None)
    row.setdefault("llm_wall_seconds", 0.0)
    row.setdefault("llm_wall_union_seconds", 0.0)
    row.setdefault("llm_charged_union_seconds", 0.0)
    row.setdefault("llm_round_trips", 0)
    row.setdefault("llm_max_in_flight", 0)
    row.setdefault("llm_overlapped_round_trips", 0)
    return EvaluationRecord(**row)
