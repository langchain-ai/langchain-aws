"""Standard LangChain integration tests for ChatOpenAIMantle.

These hit the live Amazon Bedrock Mantle endpoint, so they are skipped unless a
Bedrock API key is available via ``AWS_BEARER_TOKEN_BEDROCK``. Unskip in CI once
the test account has ``bedrock-mantle`` API permissions configured.

Run locally with::

    AWS_REGION=us-east-1 AWS_BEARER_TOKEN_BEDROCK=... \\
        uv run --group test --group test_integration \\
        pytest tests/integration_tests/chat_models/test_openai.py -v
"""

import os
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_aws import ChatOpenAIMantle

# Skip the whole module until a Bedrock Mantle API key (and, in CI, the
# corresponding account permissions) is available.
pytestmark = pytest.mark.skipif(
    not os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
    reason=(
        "Requires a Bedrock Mantle API key in AWS_BEARER_TOKEN_BEDROCK. "
        "Unskip in CI once the test account has bedrock-mantle API permissions."
    ),
)

# Open-weight model served on the default Mantle base path (/v1) via the
# Chat Completions API.
MODEL_NAME = "openai.gpt-oss-120b"


class TestOpenAIMantleIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAIMantle

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": MODEL_NAME,
            "region_name": os.getenv("AWS_REGION", "us-east-1"),
            # Enable usage on streamed responses for this test run. Not set as a
            # class default: BaseChatOpenAI leaves stream_usage off for custom
            # base URLs since not every OpenAI-compatible endpoint/model supports
            # stream_options.include_usage.
            "stream_usage": True,
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {"max_tokens": 1000, "temperature": 0}

    # NOTE: capability flags below are conservative defaults for a Mantle
    # open-weight model. Validate/adjust them against a live run — e.g. gpt-oss
    # rejects tool_choice="required" and strict structured output on Mantle, so
    # those may need to be disabled or xfail-marked per model.
    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        # gpt-oss on Mantle does not honor forced tool_choice (returns no
        # tool_calls / 400 on tool_choice="required"), so the standard
        # test_tool_choice is skipped. Revisit per-model when adding frontier
        # models that support forced tool choice.
        return False

    @property
    def has_structured_output(self) -> bool:
        return False
