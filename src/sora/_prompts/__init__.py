"""Named prompt modules and channel-aware assembly for the built-in prompt suite."""

from sora._prompts.core import (
    OPERATIONS_ONLY,
    PROPERTIES_ONLY,
    RICH_CHANNELS,
    SIGNALS_ONLY,
    SUPPORTED_PERCEPTION_CHANNELS,
    PerceptionChannels,
    PromptId,
    PromptManifest,
    PromptModule,
    PromptRendering,
    PromptVariant,
    fitted_channels,
)
from sora._prompts.ground import GROUND_PROMPT, GROUND_SYSTEM_PROMPT
from sora._prompts.judgments import (
    CONDITION_PROMPT,
    CONDITION_SYSTEM_PROMPT,
    RELEVANCE_PROMPT,
    RELEVANCE_SYSTEM_PROMPT,
    RETIREMENT_PROMPT,
    RETIREMENT_SYSTEM_PROMPT,
    REVALIDATE_PROMPT,
    REVALIDATE_SYSTEM_PROMPT,
    SELECT_PROMPT,
    SELECT_SYSTEM_PROMPT,
)
from sora._prompts.plan import PLAN_PROMPT, PLAN_SYSTEM_PROMPT

__all__ = [
    "CONDITION_PROMPT",
    "CONDITION_SYSTEM_PROMPT",
    "GROUND_PROMPT",
    "GROUND_SYSTEM_PROMPT",
    "OPERATIONS_ONLY",
    "PLAN_PROMPT",
    "PLAN_SYSTEM_PROMPT",
    "PROPERTIES_ONLY",
    "PerceptionChannels",
    "PromptId",
    "PromptManifest",
    "PromptModule",
    "PromptRendering",
    "PromptVariant",
    "RELEVANCE_PROMPT",
    "RELEVANCE_SYSTEM_PROMPT",
    "RETIREMENT_PROMPT",
    "RETIREMENT_SYSTEM_PROMPT",
    "REVALIDATE_PROMPT",
    "REVALIDATE_SYSTEM_PROMPT",
    "RICH_CHANNELS",
    "SELECT_PROMPT",
    "SELECT_SYSTEM_PROMPT",
    "SIGNALS_ONLY",
    "SUPPORTED_PERCEPTION_CHANNELS",
    "fitted_channels",
]
