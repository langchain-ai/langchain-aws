"""Tests for AWS model profiles."""

import pytest

from langchain_aws import ChatBedrockConverse

_CLAUDE_5_MODELS = (
    "claude-opus-5",
    "claude-opus-5-5",
    "claude-sonnet-5",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
)


@pytest.mark.parametrize(
    "model_id",
    [
        f"{prefix}anthropic.{model}"
        for model in _CLAUDE_5_MODELS
        for prefix in ("", "au.", "eu.", "global.", "jp.", "us.")
        if not (prefix in ("au.", "jp.") and model == "claude-sonnet-5-5")
    ],
)
def test_claude_5_structured_output_profiles(model_id: str) -> None:
    """Claude 5 chat models expose native structured output support."""
    model = ChatBedrockConverse(model=model_id, region_name="us-west-2")
    assert model.profile
    assert model.profile["structured_output"] is True


@pytest.mark.parametrize(
    "model_id", ["in.anthropic.claude-opus-5", "in.anthropic.claude-sonnet-5"]
)
def test_claude_5_india_profiles_lack_structured_output(model_id: str) -> None:
    """Bedrock rejects output_config.format on the India inference profiles."""
    model = ChatBedrockConverse(model=model_id, region_name="us-west-2")
    assert model.profile
    assert model.profile.get("structured_output") is not True
