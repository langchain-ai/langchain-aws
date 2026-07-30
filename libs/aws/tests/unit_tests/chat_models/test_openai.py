"""ChatOpenAIMantle unit tests."""

from typing import Tuple, Type, cast

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr
from pytest import MonkeyPatch

from langchain_aws import ChatOpenAIMantle

MODEL_NAME = "openai.gpt-oss-120b"


class TestOpenAIMantleStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAIMantle

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
    """Explicit params are stored and the model is an OpenAI-compatible model."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.model_name == MODEL_NAME
    assert model.region_name == "us-east-1"
    assert cast("SecretStr", model.bedrock_api_key).get_secret_value() == "test-key"


def test_default_base_url_from_region() -> None:
    """base_url defaults to the region's Bedrock Mantle endpoint."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.openai_api_base == "https://bedrock-mantle.us-east-1.api.aws/v1"


def test_explicit_base_url_is_respected() -> None:
    """An explicit base_url overrides the region-derived default."""
    custom = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        base_url=custom,
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.openai_api_base == custom


def test_region_and_key_from_env() -> None:
    """Region and bedrock_api_key are read from the environment."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_DEFAULT_REGION", raising=False)
        m.setenv("AWS_REGION", "eu-west-1")
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-key")
        model = ChatOpenAIMantle(model=MODEL_NAME)
        assert model.openai_api_base == "https://bedrock-mantle.eu-west-1.api.aws/v1"
        assert cast("SecretStr", model.bedrock_api_key).get_secret_value() == "env-key"


def test_bedrock_api_key_routed_to_openai_key() -> None:
    """The Bedrock API key is used as the OpenAI client api_key."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("bearer-abc"),
    )
    assert cast("SecretStr", model.openai_api_key).get_secret_value() == "bearer-abc"


def test_ls_params_provider() -> None:
    """Tracing provider is reported as openai-mantle."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    ls_params = model._get_ls_params()
    assert ls_params["ls_provider"] == "openai-mantle"
    assert ls_params["ls_model_name"] == MODEL_NAME


def test_lc_secrets() -> None:
    """The bedrock_api_key maps to its environment variable."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.lc_secrets["bedrock_api_key"] == "AWS_BEARER_TOKEN_BEDROCK"


def test_profile_resolved_from_model_name() -> None:
    """The model profile is populated from static profile data by default."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.profile is not None
    assert model.profile.get("tool_calling") is True


def test_explicit_profile_is_respected() -> None:
    """A caller-supplied profile is not overwritten by the default lookup."""
    custom = {"tool_calling": False}
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
        profile=custom,  # type: ignore[arg-type]
    )
    assert model.profile == custom


def test_profile_empty_for_unknown_model() -> None:
    """An unknown model resolves to an empty profile rather than raising."""
    model = ChatOpenAIMantle(
        model="openai.does-not-exist",
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.profile == {}


def test_inherits_openai_features() -> None:
    """Key BaseChatOpenAI methods are inherited unchanged."""
    model = ChatOpenAIMantle(
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
