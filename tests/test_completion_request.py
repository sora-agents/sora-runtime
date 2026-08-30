"""Provider-neutral completion request contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sora.llm import CompletionProfile, CompletionRequest, PromptSection


def test_completion_request_carries_call_description_and_transport_hints() -> None:
    profile = CompletionProfile(max_output_tokens=64, reasoning="low")
    sections = (
        PromptSection("instructions", characters=12, dynamic=False),
        PromptSection("observations", characters=34, dynamic=True),
    )

    request = CompletionRequest(
        system="system text",
        user="user text",
        semantic_label="plan",
        prompt_version="1",
        profile=profile,
        sections=sections,
    )

    assert request.system == "system text"
    assert request.user == "user text"
    assert request.semantic_label == "plan"
    assert request.prompt_version == "1"
    assert request.profile is profile
    assert request.sections == sections
    assert not hasattr(request, "response_contract")


def test_completion_request_defaults_preserve_unprofiled_unsectioned_calls() -> None:
    request = CompletionRequest(
        system="system text",
        user="user text",
        semantic_label="custom",
        prompt_version="1",
    )

    assert request.profile is None
    assert request.sections == ()


def test_completion_request_values_are_immutable() -> None:
    request = CompletionRequest("system", "user", "plan", "1")

    with pytest.raises(FrozenInstanceError):
        request.user = "changed"  # type: ignore[misc]
