"""Named prompt modules and channel-aware assembly for the built-in prompt suite."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sora.llm import PromptSection


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


@dataclass(frozen=True)
class PromptModule:
    name: str
    text: str
    dynamic: bool = False


@dataclass(frozen=True)
class PromptRendering:
    system: str
    user: str
    sections: tuple[PromptSection, ...]

    def pair(self) -> tuple[str, str]:
        return self.system, self.user


ModuleAdapter = Callable[[PromptModule, PerceptionChannels | None], str]


@dataclass(frozen=True)
class PromptManifest:
    semantic_label: str
    system_modules: tuple[PromptModule, ...]

    @classmethod
    def split(
        cls,
        semantic_label: str,
        text: str,
        anchors: Sequence[tuple[str, str]],
    ) -> PromptManifest:
        """Partition existing text at stable logical anchors without changing a byte."""
        modules: list[PromptModule] = []
        start = 0
        for index, (_name, anchor) in enumerate(anchors):
            position = text.index(anchor, start)
            if position > start:
                previous_name = "role" if index == 0 else anchors[index - 1][0]
                modules.append(PromptModule(previous_name, text[start:position]))
            start = position
        final_name = anchors[-1][0] if anchors else "instructions"
        modules.append(PromptModule(final_name, text[start:]))
        return cls(semantic_label, tuple(module for module in modules if module.text))

    def render(
        self,
        user_modules: Sequence[PromptModule],
        channels: PerceptionChannels | None,
        *,
        adapt: ModuleAdapter | None = None,
    ) -> PromptRendering:
        selected: list[PromptModule] = []
        for module in self.system_modules:
            rendered = module.text if adapt is None else adapt(module, channels)
            if rendered:
                selected.append(PromptModule(module.name, rendered))
        selected.extend(module for module in user_modules if module.text)
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
        return PromptRendering(system, user, sections)
