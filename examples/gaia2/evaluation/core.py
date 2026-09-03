from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

SCHEMA_VERSION = 1
CAPABILITIES = ("search", "execution", "adaptability", "ambiguity", "time")
GAIA_SUITES = ("familiar", "development", "acceptance")
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
Campaign = Literal["prompt", "aamas2027"]
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
            or any(campaign not in {"prompt", "aamas2027"} for campaign in campaigns_raw)
        ):
            raise ValueError("model profile campaigns must name prompt and/or aamas2027")
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
    models: dict[str, tuple[PriceTier, ...]]
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
        models: dict[str, tuple[PriceTier, ...]] = {}
        for model, tiers_raw in models_raw.items():
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
            models[str(model)] = tuple(tiers)
        return cls(
            effective_date=str(raw["effective_date"]),
            currency=str(raw.get("currency", "USD")),
            models=models,
            digest=sha256_text(canonical_json(raw)),
        )


@dataclass(frozen=True)
class CallUsage:
    input_tokens: int
    cached_input_tokens: int | None
    output_tokens: int
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class CostResult:
    agent_cost: float
    uncached_input_cost: float
    cached_input_cost: float
    output_cost: float
    upper_bound: bool
    tier_max_input_tokens: int | None


def calculate_call_cost(sheet: PriceSheet, model: str, usage: CallUsage) -> CostResult:
    tiers = sheet.models.get(model)
    if tiers is None:
        raise ValueError(f"price sheet has no rate for model {model}")
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
    # Provider usage already includes reasoning in output. ``reasoning_tokens`` is descriptive and
    # must not be charged again.
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

    @property
    def accounted_agent_cost(self) -> float:
        return self.agent_cost if self.agent_cost is not None else self.agent_cost_reserve

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
        return row


def record_from_dict(raw: dict[str, Any]) -> EvaluationRecord:
    row = dict(raw)
    # Read Task-4's pre-correction checkpoints without carrying their misleading field names into
    # new reports. Those records counted completed logical calls and provider-visible round trips.
    row.setdefault("agent_llm_calls", row.pop("calls", 0))
    row.setdefault("provider_round_trips", row.pop("round_trips", 0))
    row.pop("step_unit", None)
    row["call_records"] = tuple(row.get("call_records", ()))
    return EvaluationRecord(**row)
