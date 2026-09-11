"""Tests for AWS model profiles."""

import pytest

from langchain_aws import ChatBedrockConverse


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-opus-5",
        "au.anthropic.claude-opus-5",
        "eu.anthropic.claude-opus-5",
        "global.anthropic.claude-opus-5",
        "jp.anthropic.claude-opus-5",
        "us.anthropic.claude-opus-5",
        "anthropic.claude-sonnet-5",
        "au.anthropic.claude-sonnet-5",
        "eu.anthropic.claude-sonnet-5",
        "global.anthropic.claude-sonnet-5",
        "jp.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
    ],
)
def test_claude_5_structured_output_profiles(model_id: str) -> None:
    """Claude 5 chat models expose native structured output support."""
    model = ChatBedrockConverse(model=model_id, region_name="us-west-2")
    assert model.profile
    assert model.profile["structured_output"] is True
