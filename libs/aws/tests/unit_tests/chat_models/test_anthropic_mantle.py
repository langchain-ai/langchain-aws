"""ChatAnthropicMantle unit tests."""

from typing import Tuple, Type, cast

import pytest
from langchain_core.language_models import BaseChatModel, ModelProfile
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr
from pytest import MonkeyPatch

from langchain_aws import ChatAnthropicMantle

MODEL_NAME = "anthropic.claude-sonnet-5"


class TestAnthropicMantleStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatAnthropicMantle

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": MODEL_NAME,
            "region_name": "us-east-1",
            "bedrock_api_key": "test-bedrock-key",
        }

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Env vars, init args, and expected attrs for env-based initialization."""
        return (
            {
                "AWS_BEARER_TOKEN_BEDROCK": "env-bedrock-key",
                "AWS_REGION": "us-west-2",
            },
            {"model": MODEL_NAME},
            {"bedrock_api_key": "env-bedrock-key"},
        )


def test_initialization() -> None:
    """Explicit params are stored on the model."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.model == MODEL_NAME
    assert model.region_name == "us-east-1"
    assert cast("SecretStr", model.bedrock_api_key).get_secret_value() == "test-key"


def test_default_base_url_from_region() -> None:
    """base_url defaults to the region's Bedrock Mantle Anthropic endpoint."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert (
        model.anthropic_api_url == "https://bedrock-mantle.us-east-1.api.aws/anthropic"
    )


def test_explicit_base_url_is_respected() -> None:
    """An explicit base_url overrides the region-derived default."""
    custom = "https://my-proxy.example.com/anthropic"
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        base_url=custom,
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.anthropic_api_url == custom


def test_region_and_key_from_env() -> None:
    """Region and bedrock_api_key are read from the environment."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_DEFAULT_REGION", raising=False)
        m.setenv("AWS_REGION", "eu-west-1")
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-key")
        model = ChatAnthropicMantle(model=MODEL_NAME)  # type: ignore[call-arg]
        assert (
            model.anthropic_api_url
            == "https://bedrock-mantle.eu-west-1.api.aws/anthropic"
        )
        assert cast("SecretStr", model.bedrock_api_key).get_secret_value() == "env-key"


def test_bedrock_api_key_routed_to_anthropic_key() -> None:
    """The Bedrock API key is used as the Anthropic client api_key (x-api-key)."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("bearer-abc"),
    )
    assert model.anthropic_api_key.get_secret_value() == "bearer-abc"


def test_ls_params_provider() -> None:
    """Tracing provider is reported as anthropic-mantle."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    ls_params = model._get_ls_params()
    assert ls_params["ls_provider"] == "anthropic-mantle"
    assert ls_params["ls_model_name"] == MODEL_NAME


def test_llm_type() -> None:
    """The _llm_type identifies the Mantle Anthropic surface."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model._llm_type == "anthropic-mantle-chat"


def test_lc_secrets() -> None:
    """The bedrock_api_key maps to its environment variable."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.lc_secrets["bedrock_api_key"] == "AWS_BEARER_TOKEN_BEDROCK"


def test_get_lc_namespace() -> None:
    """The LangChain namespace identifies the Mantle Anthropic module."""
    assert ChatAnthropicMantle.get_lc_namespace() == [
        "langchain",
        "chat_models",
        "anthropic_mantle",
    ]


@pytest.mark.parametrize(
    "model_name",
    [
        "anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
        "global.anthropic.claude-sonnet-5",
        "us-gov.anthropic.claude-sonnet-5",
        "apac.anthropic.claude-sonnet-5",
    ],
)
def test_model_profile(model_name: str) -> None:
    """Model profile is resolved from the model name across id formats."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=model_name,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.profile
    assert "max_input_tokens" in model.profile


def test_explicit_profile_is_respected() -> None:
    """An explicitly supplied profile is not overwritten."""
    profile = ModelProfile(max_input_tokens=123)
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
        profile=profile,
    )
    assert model.profile == profile


def test_inherits_anthropic_features() -> None:
    """Key ChatAnthropic methods are inherited unchanged."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    for attr in (
        "bind_tools",
        "with_structured_output",
        "_stream",
        "_astream",
        "_generate",
        "_agenerate",
    ):
        assert hasattr(model, attr)
