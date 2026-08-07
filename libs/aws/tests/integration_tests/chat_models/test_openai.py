"""Standard LangChain integration tests for ChatOpenAIMantle.

These hit the live Amazon Bedrock Mantle endpoint and are skipped by default.
Two independent live checks are provided:

1. ``TestOpenAIMantleIntegration`` — the standard suite, run with a static
   Bedrock API key::

       AWS_REGION=us-east-1 AWS_BEARER_TOKEN_BEDROCK=... \\
           uv run --group test --group test_integration \\
           pytest tests/integration_tests/chat_models/test_openai.py -v

2. ``test_credential_derived_auth_live`` — the credential-derived path, which
   mints and refreshes a short-term key from ordinary AWS credentials (no static
   token needed). Run with an opt-in flag plus any AWS credentials for a
   Mantle-enabled account::

       MANTLE_CREDS_E2E=1 AWS_REGION=us-east-1 \\
           uv run --group test --group test_integration \\
           pytest tests/integration_tests/chat_models/test_openai.py \\
           -k credential_derived -v
"""

import os
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_aws import ChatOpenAIMantle
from langchain_aws.utils import _BedrockApiKeyProvider

# Static-key path: skip until a Bedrock Mantle API key (and, in CI, the
# corresponding account permissions) is available.
requires_static_key = pytest.mark.skipif(
    not os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
    reason=(
        "Requires a Bedrock Mantle API key in AWS_BEARER_TOKEN_BEDROCK. "
        "Unskip in CI once the test account has bedrock-mantle API permissions."
    ),
)

# Credential-derived path: opt-in via a simple flag. Needs only ambient AWS
# credentials + region for a Mantle-enabled account (no static bearer token).
requires_aws_creds = pytest.mark.skipif(
    not os.getenv("MANTLE_CREDS_E2E"),
    reason=(
        "Set MANTLE_CREDS_E2E=1 (with AWS credentials + region for a "
        "Mantle-enabled account) to run the credential-derived auth check."
    ),
)

# Open-weight model served on the default Mantle base path (/v1) via the
# Chat Completions API.
MODEL_NAME = "openai.gpt-oss-120b"


@requires_static_key
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


@requires_aws_creds
def test_credential_derived_auth_live() -> None:
    """End-to-end: derive a short-term Bedrock key from AWS creds and invoke.

    With no static ``AWS_BEARER_TOKEN_BEDROCK``, ``ChatOpenAIMantle`` installs a
    ``_BedrockApiKeyProvider`` that mints (and transparently refreshes) a
    short-term Bedrock API key from the ambient AWS credential chain (env,
    profile, assumed role, IRSA). This is the credential-derived counterpart to
    the static-key ``TestOpenAIMantleIntegration`` suite above.
    """
    assert not os.getenv("AWS_BEARER_TOKEN_BEDROCK"), (
        "Unset AWS_BEARER_TOKEN_BEDROCK to exercise the credential-derived path."
    )

    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )

    # The creds-derived provider (not a static key) must be installed and must
    # mint a real short-term key from the ambient AWS credentials.
    provider = model.openai_api_key
    assert isinstance(provider, _BedrockApiKeyProvider)
    token = provider()
    assert isinstance(token, str) and token

    response = model.invoke("What is 2 + 2? Reply with just the number.")
    assert isinstance(response.content, str)
    assert "4" in response.content
