"""ChatOpenAIBedrock unit tests (bedrock-runtime OpenAI-compatible endpoint)."""

from collections.abc import AsyncIterator
from typing import Tuple, Type, cast
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel, SecretStr
from pytest import MonkeyPatch

from langchain_aws import ChatOpenAIBedrock
from langchain_aws.utils import _BedrockApiKeyProvider

# bedrock-runtime requires a cross-Region inference-profile id, not a bare id.
MODEL_NAME = "us.openai.gpt-5.6-sol"
RUNTIME_BASE_URL = "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"


class TestOpenAIBedrockStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAIBedrock

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": MODEL_NAME,
            "region_name": "us-west-2",
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
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.model_name == MODEL_NAME
    assert model.region_name == "us-west-2"
    assert cast("SecretStr", model.bedrock_api_key).get_secret_value() == "test-key"


def test_default_base_url_from_region() -> None:
    """base_url defaults to the region's bedrock-runtime OpenAI endpoint."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.openai_api_base == RUNTIME_BASE_URL


def test_explicit_base_url_is_respected() -> None:
    """An explicit base_url overrides the region-derived default."""
    custom = "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        base_url=custom,
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.openai_api_base == custom


def test_missing_region_and_base_url_raises() -> None:
    """Fail locally instead of routing the Bedrock key to the default OpenAI host."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_REGION", raising=False)
        m.delenv("AWS_DEFAULT_REGION", raising=False)
        with pytest.raises(ValueError, match="region"):
            ChatOpenAIBedrock(model=MODEL_NAME, bedrock_api_key=SecretStr("test-key"))


def test_explicit_base_url_without_region_ok() -> None:
    """An explicit base_url bypasses the region requirement."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_REGION", raising=False)
        m.delenv("AWS_DEFAULT_REGION", raising=False)
        model = ChatOpenAIBedrock(
            model=MODEL_NAME,
            base_url=RUNTIME_BASE_URL,
            bedrock_api_key=SecretStr("test-key"),
        )
    assert model.openai_api_base == RUNTIME_BASE_URL


def test_region_and_key_from_env() -> None:
    """Region and bedrock_api_key are read from the environment."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_DEFAULT_REGION", raising=False)
        m.setenv("AWS_REGION", "eu-west-1")
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-key")
        model = ChatOpenAIBedrock(model=MODEL_NAME)
        assert (
            model.openai_api_base
            == "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1"
        )
        assert cast("SecretStr", model.bedrock_api_key).get_secret_value() == "env-key"


def test_bedrock_api_key_routed_to_openai_key() -> None:
    """The Bedrock API key is used as the OpenAI client api_key."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("bearer-abc"),
    )
    assert cast("SecretStr", model.openai_api_key).get_secret_value() == "bearer-abc"


def test_ls_params_provider() -> None:
    """Tracing provider is reported as openai-bedrock."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
    )
    ls_params = model._get_ls_params()
    assert ls_params["ls_provider"] == "openai-bedrock"
    assert ls_params["ls_model_name"] == MODEL_NAME


def test_llm_type_and_namespace() -> None:
    """Endpoint-specific llm_type and serialization namespace."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model._llm_type == "openai-bedrock-chat"
    assert model.get_lc_namespace() == ["langchain", "chat_models", "openai_bedrock"]


def test_lc_secrets() -> None:
    """The bedrock_api_key maps to its environment variable."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.lc_secrets["bedrock_api_key"] == "AWS_BEARER_TOKEN_BEDROCK"


def test_explicit_profile_is_respected() -> None:
    """A caller-supplied profile is not overwritten by the default lookup."""
    custom = {"tool_calling": False}
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
        profile=custom,  # type: ignore[arg-type]
    )
    assert model.profile == custom


def test_profile_empty_for_unknown_model() -> None:
    """An unknown model resolves to an empty profile rather than raising."""
    model = ChatOpenAIBedrock(
        model="us.openai.does-not-exist",
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.profile == {}


def test_profile_resolved_through_cross_region_prefix() -> None:
    """A geographic inference-profile id inherits the base model's profile.

    ``us.openai.gpt-5.6-sol`` is not registered directly, but the cross-Region
    prefix is stripped so it resolves to the base ``openai.gpt-5.6-sol`` profile.
    """
    model = ChatOpenAIBedrock(
        model="us.openai.gpt-5.6-sol",
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
    )
    base = ChatOpenAIBedrock(
        model="openai.gpt-5.6-sol",
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.profile != {}
    assert model.profile == base.profile


def test_stream_routes_to_responses_api() -> None:
    """_stream delegates to the Responses API path when it is active."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
        use_responses_api=True,
    )
    with (
        patch.object(
            BaseChatOpenAI, "_stream_responses", return_value=iter([])
        ) as responses,
        patch.object(BaseChatOpenAI, "_stream", return_value=iter([])) as completions,
    ):
        list(model._stream([HumanMessage("hi")]))
    responses.assert_called_once()
    completions.assert_not_called()


def test_stream_routes_to_chat_completions_when_disabled() -> None:
    """_stream falls back to Chat Completions when the Responses API is off."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
        use_responses_api=False,
    )
    with (
        patch.object(
            BaseChatOpenAI, "_stream_responses", return_value=iter([])
        ) as responses,
        patch.object(BaseChatOpenAI, "_stream", return_value=iter([])) as completions,
    ):
        list(model._stream([HumanMessage("hi")]))
    completions.assert_called_once()
    responses.assert_not_called()


async def _empty_async_stream(*args: object, **kwargs: object) -> AsyncIterator[None]:
    """An async generator that yields nothing (stand-in for a streamed call)."""
    for _ in ():
        yield


async def test_astream_routes_to_responses_api() -> None:
    """_astream delegates to the Responses API path when it is active."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
        use_responses_api=True,
    )
    with (
        patch.object(
            BaseChatOpenAI, "_astream_responses", side_effect=_empty_async_stream
        ) as responses,
        patch.object(
            BaseChatOpenAI, "_astream", side_effect=_empty_async_stream
        ) as completions,
    ):
        async for _ in model._astream([HumanMessage("hi")]):
            pass
    responses.assert_called_once()
    completions.assert_not_called()


async def test_astream_routes_to_chat_completions_when_disabled() -> None:
    """_astream falls back to Chat Completions when the Responses API is off."""
    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
        use_responses_api=False,
    )
    with (
        patch.object(
            BaseChatOpenAI, "_astream_responses", side_effect=_empty_async_stream
        ) as responses,
        patch.object(
            BaseChatOpenAI, "_astream", side_effect=_empty_async_stream
        ) as completions,
    ):
        async for _ in model._astream([HumanMessage("hi")]):
            pass
    completions.assert_called_once()
    responses.assert_not_called()


def test_with_structured_output_defaults_to_json_schema() -> None:
    """with_structured_output defaults method to json_schema."""

    class Schema(BaseModel):
        answer: str

    model = ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
    )
    with patch.object(BaseChatOpenAI, "with_structured_output") as parent:
        model.with_structured_output(Schema)
    assert parent.call_args.kwargs["method"] == "json_schema"


# ---------------------------------------------------------------------------
# Credential-derived (short-term key) authentication path
# ---------------------------------------------------------------------------


def test_provider_installed_when_no_static_key() -> None:
    """With no static bearer key, a _BedrockApiKeyProvider is used as api_key."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.setenv("AWS_REGION", "us-west-2")
        model = ChatOpenAIBedrock(
            model=MODEL_NAME,
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
            aws_session_token=SecretStr("TOKEN_TEST"),
        )

    provider = model.openai_api_key
    assert isinstance(provider, _BedrockApiKeyProvider)
    assert model.openai_api_base == RUNTIME_BASE_URL
    assert provider._region == "us-west-2"
    assert provider._aws_access_key_id == "AKIA_TEST"
    assert provider._aws_secret_access_key == "SECRET_TEST"
    assert provider._aws_session_token == "TOKEN_TEST"


def test_static_key_takes_precedence_over_creds() -> None:
    """A static bedrock_api_key wins over AWS credentials."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.setenv("AWS_REGION", "us-west-2")
        model = ChatOpenAIBedrock(
            model=MODEL_NAME,
            bedrock_api_key=SecretStr("bearer-abc"),
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
        )
    assert cast("SecretStr", model.openai_api_key).get_secret_value() == "bearer-abc"


def test_explicit_api_key_not_overridden_by_provider() -> None:
    """An explicit api_key is left untouched (no provider installed)."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.setenv("AWS_REGION", "us-west-2")
        model = ChatOpenAIBedrock(model=MODEL_NAME, api_key=SecretStr("explicit"))
    assert cast("SecretStr", model.openai_api_key).get_secret_value() == "explicit"


def test_installed_provider_mints_token_when_called() -> None:
    """The installed provider returns a minted short-term key when invoked."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.setenv("AWS_REGION", "us-west-2")
        model = ChatOpenAIBedrock(
            model=MODEL_NAME,
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
        )
    provider = cast("_BedrockApiKeyProvider", model.openai_api_key)
    with patch("aws_bedrock_token_generator.provide_token", return_value="tok-xyz"):
        assert provider() == "tok-xyz"


def test_provider_forwards_profile_and_ttl() -> None:
    """Profile name and requested TTL are forwarded; default chain otherwise."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.delenv("AWS_ACCESS_KEY_ID", raising=False)
        m.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        m.setenv("AWS_REGION", "us-west-2")
        model = ChatOpenAIBedrock(
            model=MODEL_NAME,
            credentials_profile_name="my-profile",
            api_key_ttl_seconds=1800,
        )
    provider = model.openai_api_key
    assert isinstance(provider, _BedrockApiKeyProvider)
    assert provider._credentials_profile_name == "my-profile"
    assert provider._ttl_seconds == 1800
    assert provider._aws_access_key_id is None


# ---------------------------------------------------------------------------
# Guardrails: rejected on the OpenAI-compatible path (model-gated on runtime)
# ---------------------------------------------------------------------------


def _make_model(**kwargs: object) -> ChatOpenAIBedrock:
    return ChatOpenAIBedrock(
        model=MODEL_NAME,
        region_name="us-west-2",
        bedrock_api_key=SecretStr("test-key"),
        **kwargs,  # type: ignore[arg-type]
    )


def test_guardrail_config_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="Guardrails are not supported"):
        _make_model(guardrails={"guardrailIdentifier": "gr-1", "guardrailVersion": "1"})


def test_guardrail_default_headers_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="Guardrails are not supported"):
        _make_model(
            default_headers={
                "X-Amzn-Bedrock-GuardrailIdentifier": "gr-1",
                "X-Amzn-Bedrock-GuardrailVersion": "1",
            },
        )


def test_guardrail_extra_headers_rejected_per_request() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="Guardrails are not supported"):
        model._get_request_payload(
            "hello",
            extra_headers={"X-Amzn-Bedrock-GuardrailIdentifier": "gr-1"},
        )


def test_non_guardrail_headers() -> None:
    model = _make_model(default_headers={"X-Custom-Header": "ok"})
    payload = model._get_request_payload(
        "hello", extra_headers={"X-Another-Header": "ok"}
    )
    assert payload["extra_headers"] == {"X-Another-Header": "ok"}


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_inherits_openai_features() -> None:
    """Key BaseChatOpenAI methods are inherited unchanged."""
    model = _make_model()
    for attr in (
        "bind_tools",
        "with_structured_output",
        "_stream",
        "_astream",
        "_generate",
        "_agenerate",
    ):
        assert hasattr(model, attr)


def test_base_class_not_instantiable() -> None:
    """The shared base class cannot be constructed directly."""
    from langchain_aws.chat_models.openai import _BaseChatBedrockOpenAI

    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        _BaseChatBedrockOpenAI(
            model=MODEL_NAME,
            region_name="us-west-2",
            bedrock_api_key=SecretStr("test-key"),
        )
