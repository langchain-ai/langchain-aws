"""Standard LangChain integration tests for ChatAnthropicMantle.

These hit the live Amazon Bedrock Mantle Anthropic Messages endpoint.
Authentication uses the ambient AWS credentials (SigV4), or a Bedrock API
key if one is set via ``AWS_BEARER_TOKEN_BEDROCK``.

Run locally with::

    AWS_REGION=us-east-1 \\
        uv run --group test --group test_integration \\
        pytest tests/integration_tests/chat_models/test_anthropic_mantle.py -v

Override the model with ``ANTHROPIC_MANTLE_MODEL`` if needed, e.g.::

    ANTHROPIC_MANTLE_MODEL=anthropic.claude-opus-5 ...

To discover the exact model ids your account/region can call on Mantle, list
the catalog::

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

# Claude model served on the Mantle Anthropic Messages API. Mantle catalog ids
# use the ``anthropic.`` prefix (no ``us.`` cross-region prefix — Mantle handles
# routing). Override with ANTHROPIC_MANTLE_MODEL, and use the catalog-listing
# curl in the module docstring to confirm what your account/region can call.
MODEL_NAME = os.getenv("ANTHROPIC_MANTLE_MODEL", "anthropic.claude-sonnet-5")

# Sonnet 5 has always-on adaptive thinking and may emit a signature-only thinking
# block on any request, which stochastically fails the streaming tests.
STREAM_MODEL_NAME = "anthropic.claude-haiku-4-5"


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

    @pytest.fixture
    def model(self, request: pytest.FixtureRequest) -> BaseChatModel:
        extra_init_params = getattr(request, "param", None) or {}
        params = {
            **self.standard_chat_model_params,
            **self.chat_model_params,
            **extra_init_params,
        }
        if request.node.originalname in ("test_stream", "test_astream"):
            params["model"] = STREAM_MODEL_NAME
        return self.chat_model_class(**params)


def _aws_credentials_available() -> bool:
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:
        return False


@pytest.mark.skipif(
    not _aws_credentials_available(),
    reason="Requires resolvable AWS credentials to generate Bedrock API key "
    "and SigV4 sign.",
)
def test_auth_mode_sigv4_with_env_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aws_bedrock_token_generator import provide_token

    region = os.getenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", provide_token(region=region))
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model_name=MODEL_NAME,
        region_name=region,
        auth_mode="sigv4",
        max_tokens=50,
    )
    assert model._client._use_sigv4 is True
    response = model.invoke("Say OK and nothing else.")
    assert isinstance(response.content, (str, list))
    assert response.content


@pytest.mark.skipif(
    not _aws_credentials_available(),
    reason="Requires resolvable AWS credentials to mint a Bedrock API key.",
)
def test_auth_mode_api_key_live(monkeypatch: pytest.MonkeyPatch) -> None:
    from aws_bedrock_token_generator import provide_token

    region = os.getenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", provide_token(region=region))
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model_name=MODEL_NAME,
        region_name=region,
        auth_mode="api_key",
        max_tokens=50,
    )
    assert model._client._use_sigv4 is False
    response = model.invoke("Say OK and nothing else.")
    assert isinstance(response.content, (str, list))
    assert response.content
