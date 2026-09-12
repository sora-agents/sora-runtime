"""Typed prompt modules and channel-aware assembly."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sora.llm import PromptSection


class PromptId(StrEnum):
    PLAN = "plan"
    GROUND = "ground"
    SELECT = "select"
    REVALIDATE = "revalidate"
    CONDITION = "condition"
    RETIREMENT = "retirement"
    RELEVANCE = "relevance"


@dataclass(frozen=True)
class PerceptionChannels:
    """Perception affordances declared by the tools visible to one model call."""

    properties: bool = False
    signals: bool = False

    @property
    def rich(self) -> bool:
        return self.properties and self.signals

    @property
    def any(self) -> bool:
        return self.properties or self.signals


OPERATIONS_ONLY = PerceptionChannels()
SIGNALS_ONLY = PerceptionChannels(signals=True)
PROPERTIES_ONLY = PerceptionChannels(properties=True)
RICH_CHANNELS = PerceptionChannels(properties=True, signals=True)
SUPPORTED_PERCEPTION_CHANNELS = frozenset(
    (OPERATIONS_ONLY, SIGNALS_ONLY, PROPERTIES_ONLY, RICH_CHANNELS)
)


def fitted_channels(channels: PerceptionChannels | None) -> PerceptionChannels:
    """Normalize the legacy ``None`` compatibility value at the rendering boundary."""
    fitted = RICH_CHANNELS if channels is None else channels
    if fitted not in SUPPORTED_PERCEPTION_CHANNELS:
        raise ValueError(f"unsupported perception channels: {fitted!r}")
    return fitted


@dataclass(frozen=True)
class PromptVariant:
    channels: PerceptionChannels
    text: str


@dataclass(frozen=True)
class PromptModule:
    name: str
    text: str
    dynamic: bool = False
    variants: tuple[PromptVariant, ...] = ()

    def render(self, channels: PerceptionChannels) -> str:
        for variant in self.variants:
            if variant.channels == channels:
                return variant.text
        return self.text


@dataclass(frozen=True)
class PromptRendering:
    system: str
    user: str
    sections: tuple[PromptSection, ...]
    semantic_label: PromptId
    prompt_version: str

    def pair(self) -> tuple[str, str]:
        return self.system, self.user


@dataclass(frozen=True)
class PromptManifest:
    semantic_label: PromptId
    prompt_version: str
    system_modules: tuple[PromptModule, ...]

    @property
    def system_prompt(self) -> str:
        """The rich compatibility prompt exposed through the historical public constants."""
        return "".join(module.text for module in self.system_modules)

    def render(
        self,
        user: str,
        channels: PerceptionChannels | None,
    ) -> PromptRendering:
        fitted = fitted_channels(channels)
        selected: list[PromptModule] = []
        user_module = PromptModule("context", user, dynamic=True)
        for module in (*self.system_modules, user_module):
            rendered = module.render(fitted)
            if rendered:
                selected.append(PromptModule(module.name, rendered, module.dynamic))
        system = "".join(module.text for module in selected if not module.dynamic)
        user = "".join(module.text for module in selected if module.dynamic)
        sections = tuple(
            PromptSection(
                name=("user." if module.dynamic else "system.") + module.name,
                characters=len(module.text),
                dynamic=module.dynamic,
            )
            for module in selected
        )
        return PromptRendering(
            system,
            user,
            sections,
            semantic_label=self.semantic_label,
            prompt_version=self.prompt_version,
        )

    def render_custom(self, system: str, user: str) -> PromptRendering:
        return PromptRendering(
            system,
            user,
            (),
            semantic_label=self.semantic_label,
            prompt_version=self.prompt_version,
        )
