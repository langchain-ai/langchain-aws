"""Standard LangChain integration tests for ChatOpenAIBedrock.

These hit the live Amazon Bedrock ``bedrock-runtime`` OpenAI-compatible endpoint
and are skipped by default. Two independent live checks are provided:

1. ``TestOpenAIBedrockIntegration`` — the standard suite, run with a static
   Bedrock API key::

       AWS_REGION=us-west-2 AWS_BEARER_TOKEN_BEDROCK=... \\
           uv run --group test --group test_integration \\
           pytest tests/integration_tests/chat_models/test_openai_bedrock.py -v

2. ``test_credential_derived_auth_live`` — the credential-derived path, which
   mints and refreshes a short-term key from ordinary AWS credentials (no static
   token needed). Run with an opt-in flag plus any AWS credentials for an
   account with access to the OpenAI models on bedrock-runtime::

       BEDROCK_OAI_CREDS_E2E=1 AWS_REGION=us-west-2 \\
           uv run --group test --group test_integration \\
           pytest tests/integration_tests/chat_models/test_openai_bedrock.py \\
           -k credential_derived -v
"""

import os
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_aws import ChatOpenAIBedrock
from langchain_aws.utils import _BedrockApiKeyProvider

# Static-key path: skip until a Bedrock API key is available.
requires_static_key = pytest.mark.skipif(
    not os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
    reason=(
        "Requires a Bedrock API key in AWS_BEARER_TOKEN_BEDROCK for an account "
        "with access to OpenAI models on the bedrock-runtime endpoint."
    ),
)

# Credential-derived path: opt-in via a simple flag. Needs only ambient AWS
# credentials + region for a bedrock-runtime OpenAI-enabled account.
requires_aws_creds = pytest.mark.skipif(
    not os.getenv("BEDROCK_OAI_CREDS_E2E"),
    reason=(
        "Set BEDROCK_OAI_CREDS_E2E=1 (with AWS credentials + region for an "
        "account with OpenAI models on bedrock-runtime) to run the "
        "credential-derived auth check."
    ),
)

# bedrock-runtime requires a cross-Region inference-profile id (e.g. "us."),
# not a bare foundation-model id.
MODEL_NAME = os.getenv("BEDROCK_OAI_MODEL", "us.openai.gpt-5.6-sol")


@requires_static_key
class TestOpenAIBedrockIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAIBedrock

    @property
    def chat_model_params(self) -> dict:
        # Use the Responses API: GPT-5.x reasoning models reject function tools
        # on the Chat Completions path while reasoning is active (they require
        # /v1/responses or reasoning_effort="none"), so the standard tool and
        # structured-output tests need the Responses path.
        return {
            "model": MODEL_NAME,
            "region_name": os.getenv("AWS_REGION", "us-west-2"),
            "use_responses_api": True,
            "stream_usage": True,
        }

    @property
    def standard_chat_model_params(self) -> dict:
        # GPT-5.x on bedrock-runtime rejects ``max_tokens`` and requires
        # ``max_completion_tokens`` (handled by BaseChatOpenAI), and only
        # supports the default temperature.
        return {"max_tokens": 1000}

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_structured_output(self) -> bool:
        return True


@requires_aws_creds
def test_credential_derived_auth_live() -> None:
    """End-to-end: derive a short-term Bedrock key from AWS creds and invoke.

    With no static ``AWS_BEARER_TOKEN_BEDROCK``, ``ChatOpenAIBedrock`` installs a
    ``_BedrockApiKeyProvider`` that mints (and transparently refreshes) a
    short-term Bedrock API key from the ambient AWS credential chain (env,
    profile, assumed role, IRSA).
    """
    assert not os.getenv("AWS_BEARER_TOKEN_BEDROCK"), (
        "Unset AWS_BEARER_TOKEN_BEDROCK to exercise the credential-derived path."
    )

    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name=os.getenv("AWS_REGION", "us-west-2"),
    )

    provider = model.openai_api_key
    assert isinstance(provider, _BedrockApiKeyProvider)
    token = provider()
    assert isinstance(token, str) and token

    response = model.invoke("What is 2 + 2? Reply with just the number.")
    assert isinstance(response.content, str)
    assert "4" in response.content


@requires_aws_creds
def test_cross_region_global_profile_live() -> None:
    """A global cross-Region inference profile works via the model id alone."""
    model = ChatOpenAIBedrock(
        model=os.getenv("BEDROCK_OAI_GLOBAL_MODEL", "global.openai.gpt-5.6-sol"),
        region_name=os.getenv("AWS_REGION", "us-west-2"),
    )
    response = model.invoke("What is 2 + 2? Reply with just the number.")
    assert isinstance(response.content, str)
    assert "4" in response.content


@requires_aws_creds
def test_responses_api_live() -> None:
    """The Responses API path works on bedrock-runtime."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name=os.getenv("AWS_REGION", "us-west-2"),
        use_responses_api=True,
    )
    response = model.invoke("Say OK.")
    assert isinstance(response.content, str) or isinstance(response.content, list)
