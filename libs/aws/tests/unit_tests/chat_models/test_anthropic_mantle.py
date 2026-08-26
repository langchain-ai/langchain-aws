"""ChatAnthropicMantle unit tests."""

from collections.abc import Mapping
from typing import Any, Tuple, Type, cast
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel, ModelProfile
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr
from pytest import MonkeyPatch

from langchain_aws import ChatAnthropicMantle

MODEL_NAME = "anthropic.claude-sonnet-5"


def _constructed_client_params(
    model: ChatAnthropicMantle,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    with (
        patch(
            "langchain_aws.chat_models.anthropic.AnthropicBedrockMantle"
        ) as sync_client,
        patch(
            "langchain_aws.chat_models.anthropic.AsyncAnthropicBedrockMantle"
        ) as async_client,
    ):
        _ = model._client
        _ = model._async_client

    return sync_client.call_args.kwargs, async_client.call_args.kwargs


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
    """The client resolves the region's Bedrock Mantle Anthropic endpoint."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model._client_params["aws_region"] == "us-east-1"
    assert str(model._client.base_url).rstrip("/") == (
        "https://bedrock-mantle.us-east-1.api.aws/anthropic"
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
        assert model._client_params["aws_region"] == "eu-west-1"
        assert str(model._client.base_url).rstrip("/") == (
            "https://bedrock-mantle.eu-west-1.api.aws/anthropic"
        )
        assert cast("SecretStr", model.bedrock_api_key).get_secret_value() == "env-key"


def test_bedrock_api_key_routed_to_client() -> None:
    """The Bedrock API key is passed to the Mantle client (bearer auth mode)."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("bearer-abc"),
    )
    assert model._client_params["api_key"] == "bearer-abc"


def test_sigv4_credentials_routed_to_client() -> None:
    """Without a Bedrock API key, AWS credentials flow to the client (SigV4)."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=None,
        aws_access_key_id=SecretStr("AKIA-test"),
        aws_secret_access_key=SecretStr("secret-test"),
        aws_session_token=SecretStr("token-test"),
    )
    params = model._client_params
    assert "api_key" not in params
    assert params["aws_access_key"] == "AKIA-test"
    assert params["aws_secret_key"] == "secret-test"
    assert params["aws_session_token"] == "token-test"


def test_credentials_profile_routed_to_client() -> None:
    """A named AWS profile flows to the client for SigV4 auth."""
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=None,
        credentials_profile_name="my-profile",
    )
    assert model._client_params["aws_profile"] == "my-profile"


@pytest.mark.parametrize(
    ("credential_environment", "sigv4_params", "expected_client_params"),
    [
        (
            {},
            {"credentials_profile_name": "my-profile"},
            {"aws_profile": "my-profile"},
        ),
        (
            {},
            {
                "aws_access_key_id": SecretStr("AKIA-test"),
                "aws_secret_access_key": SecretStr("secret-test"),
                "aws_session_token": SecretStr("token-test"),
            },
            {
                "aws_access_key": "AKIA-test",
                "aws_secret_key": "secret-test",
                "aws_session_token": "token-test",
            },
        ),
        (
            {"AWS_SECRET_ACCESS_KEY": "secret-from-env"},
            {"aws_access_key_id": SecretStr("AKIA-explicit")},
            {
                "aws_access_key": "AKIA-explicit",
                "aws_secret_key": "secret-from-env",
            },
        ),
        (
            {"AWS_ACCESS_KEY_ID": "AKIA-from-env"},
            {"aws_secret_access_key": SecretStr("secret-explicit")},
            {
                "aws_access_key": "AKIA-from-env",
                "aws_secret_key": "secret-explicit",
            },
        ),
    ],
    ids=[
        "profile",
        "explicit-keys",
        "explicit-access-key",
        "explicit-secret-key",
    ],
)
def test_explicit_sigv4_credentials_outrank_ambient_api_key(
    credential_environment: dict[str, str],
    sigv4_params: dict[str, Any],
    expected_client_params: dict[str, str],
) -> None:
    """An ambient bearer token does not override explicit SigV4 credentials."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_ACCESS_KEY_ID", raising=False)
        m.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        m.delenv("AWS_SESSION_TOKEN", raising=False)
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "ambient-key")
        for name, value in credential_environment.items():
            m.setenv(name, value)
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model=MODEL_NAME,
            region_name="us-east-1",
            **sigv4_params,
        )

        client_params_by_type = _constructed_client_params(model)

    for client_params in client_params_by_type:
        for name, value in expected_client_params.items():
            assert client_params[name] == value
        assert "api_key" not in client_params


def test_explicit_bedrock_api_key_outranks_sigv4_credentials() -> None:
    """An explicitly passed bearer key keeps precedence over SigV4 signals."""
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "ambient-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model=MODEL_NAME,
            region_name="us-east-1",
            bedrock_api_key=SecretStr("explicit-key"),
            credentials_profile_name="my-profile",
        )

        client_params_by_type = _constructed_client_params(model)

    for client_params in client_params_by_type:
        assert client_params["api_key"] == "explicit-key"


def test_ambient_api_key_is_forwarded_without_explicit_sigv4_credentials() -> None:
    """Ambient bearer authentication remains the default without SigV4 signals."""
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "ambient-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model=MODEL_NAME,
            region_name="us-east-1",
        )

        client_params_by_type = _constructed_client_params(model)

    for client_params in client_params_by_type:
        assert client_params["api_key"] == "ambient-key"


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


def _make_model(**kwargs: Any) -> ChatAnthropicMantle:
    return ChatAnthropicMantle(  # type: ignore[call-arg]
        model_name=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
        **kwargs,
    )


def test_guardrail_default_headers_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="not supported on the bedrock-mantle"):
        _make_model(
            default_headers={
                "X-Amzn-Bedrock-GuardrailIdentifier": "gr-1",
                "X-Amzn-Bedrock-GuardrailVersion": "1",
            },
        )


def test_guardrail_extra_headers_rejected_per_request() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="not supported on the bedrock-mantle"):
        model._get_request_payload(
            "hello",
            extra_headers={"X-Amzn-Bedrock-GuardrailIdentifier": "gr-1"},
        )


def test_non_guardrail_headers_still_allowed() -> None:
    model = _make_model(default_headers={"X-Custom-Header": "ok"})
    payload = model._get_request_payload(
        "hello", extra_headers={"X-Another-Header": "ok"}
    )
    assert payload["extra_headers"] == {"X-Another-Header": "ok"}


def test_explicit_sigv4_credentials_select_sigv4_at_sdk_level() -> None:
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME,
            region_name="us-east-1",
            aws_access_key_id=SecretStr("key-id"),
            aws_secret_access_key=SecretStr("sec-key"),
        )
        client = model._client
        assert client._use_sigv4 is True
        assert client.api_key is None


def test_env_sigv4_credentials_do_not_outrank_ambient_api_key() -> None:
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
        m.setenv("AWS_ACCESS_KEY_ID", "key-id")
        m.setenv("AWS_SECRET_ACCESS_KEY", "sec-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME, region_name="us-east-1"
        )

        client_params_by_type = _constructed_client_params(model)

    for client_params in client_params_by_type:
        assert client_params["api_key"] == "api-key"


def test_auth_mode_default_is_auto() -> None:
    model = ChatAnthropicMantle(  # type: ignore[call-arg]
        model_name=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.auth_mode == "auto"


def test_auth_mode_api_key_requires_key() -> None:
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        with pytest.raises(ValueError, match="requires a Bedrock API key"):
            ChatAnthropicMantle(  # type: ignore[call-arg]
                model_name=MODEL_NAME,
                region_name="us-east-1",
                auth_mode="api_key",
            )


def test_auth_mode_api_key_from_env_ok() -> None:
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME,
            region_name="us-east-1",
            auth_mode="api_key",
        )
        assert model._client_params["api_key"] == "api-key"


def test_auth_mode_sigv4_ignores_env_bearer_token() -> None:
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME,
            region_name="us-east-1",
            auth_mode="sigv4",
        )
        # the env-derived key must not reach the SDK client
        assert "api_key" not in model._client_params
        client = model._client
        assert client._use_sigv4 is True
        assert client.api_key is None


def test_auth_mode_sigv4_async_client_pinned_too() -> None:
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME,
            region_name="us-east-1",
            auth_mode="sigv4",
        )
        client = model._async_client
        assert client._use_sigv4 is True
        assert client.api_key is None


def test_auth_mode_sigv4_conflicts_with_explicit_key() -> None:
    with pytest.raises(ValueError, match="conflicts with an explicitly provided"):
        ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME,
            region_name="us-east-1",
            auth_mode="sigv4",
            bedrock_api_key=SecretStr("explicit-key"),
        )


def test_auth_mode_sigv4_with_explicit_credentials() -> None:
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME,
            region_name="us-east-1",
            auth_mode="sigv4",
            aws_access_key_id=SecretStr("key-id"),
            aws_secret_access_key=SecretStr("sec-key"),
        )
        params = model._client_params
        assert "api_key" not in params
        assert params["aws_access_key"] == "key-id"
        assert model._client._use_sigv4 is True


def test_auth_mode_auto_env_bearer_wins() -> None:
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME, region_name="us-east-1"
        )
        assert model._client._use_sigv4 is False
        assert model._client.api_key == "api-key"


def test_auth_mode_api_key_overrides_explicit_sigv4_precedence() -> None:
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME,
            region_name="us-east-1",
            auth_mode="api_key",
            aws_access_key_id=SecretStr("key-id"),
            aws_secret_access_key=SecretStr("sec-key"),
        )
        client = model._client
        assert client._use_sigv4 is False
        assert client.api_key == "api-key"


def test_auth_mode_api_key_with_explicit_profile() -> None:
    with MonkeyPatch().context() as m:
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
        model = ChatAnthropicMantle(  # type: ignore[call-arg]
            model_name=MODEL_NAME,
            region_name="us-east-1",
            auth_mode="api_key",
            credentials_profile_name="my-profile",
        )
        client = model._client
        assert client._use_sigv4 is False
        assert client.api_key == "api-key"
