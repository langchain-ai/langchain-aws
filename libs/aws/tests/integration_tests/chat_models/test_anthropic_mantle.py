"""Standard LangChain integration tests for ChatAnthropicMantle.

These hit the live Amazon Bedrock Mantle Anthropic Messages endpoint, so they
are skipped unless a Bedrock API key is available via
``AWS_BEARER_TOKEN_BEDROCK``. Unskip in CI once the test account has
``bedrock-mantle`` API permissions.

Run locally with::

    AWS_REGION=us-east-1 AWS_BEARER_TOKEN_BEDROCK=... \\
        uv run --group test --group test_integration \\
        pytest tests/integration_tests/chat_models/test_anthropic_mantle.py -v

Override the model with ``ANTHROPIC_MANTLE_MODEL`` if needed, e.g.::

    ANTHROPIC_MANTLE_MODEL=anthropic.claude-opus-5 ...

To discover the exact model ids your account/region can call on Mantle, list
the catalog with your Bedrock API key::

    curl -s https://bedrock-mantle.$AWS_REGION.api.aws/anthropic/v1/models \\
        -H "x-api-key: $AWS_BEARER_TOKEN_BEDROCK" \\
        -H "anthropic-version: 2023-06-01"
"""

import os
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_aws import ChatAnthropicMantle

# Skip the whole module until a Bedrock Mantle API key (and, in CI, the
# corresponding account permissions) is available.
pytestmark = pytest.mark.skipif(
    not os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
    reason=(
        "Requires a Bedrock Mantle API key in AWS_BEARER_TOKEN_BEDROCK. "
        "Unskip in CI once the test account has bedrock-mantle API permissions."
    ),
)

# Claude model served on the Mantle Anthropic Messages API. Mantle catalog ids
# use the ``anthropic.`` prefix (no ``us.`` cross-region prefix — Mantle handles
# routing). Override with ANTHROPIC_MANTLE_MODEL, and use the catalog-listing
# curl in the module docstring to confirm what your account/region can call.
MODEL_NAME = os.getenv("ANTHROPIC_MANTLE_MODEL", "anthropic.claude-sonnet-5")


class TestAnthropicMantleIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAnthropicMantle

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": MODEL_NAME,
            "region_name": os.getenv("AWS_REGION", "us-east-1"),
        }

    @property
    def standard_chat_model_params(self) -> dict:
        # Claude Sonnet 5 / Opus 5 deprecate ``temperature`` (they use adaptive
        # thinking / output_config.effort instead), so it is intentionally not
        # set here — sending it returns a 400 invalid_request_error.
        return {"max_tokens": 1000}

    @property
    def supports_image_inputs(self) -> bool:
        # Claude supports vision; the Mantle Messages API accepts image content
        # blocks just like the native Anthropic API.
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return True
