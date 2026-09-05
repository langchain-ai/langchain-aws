import os
import time
from datetime import timedelta
from typing import Any, Dict, Generator, List, Tuple
from unittest import mock

import pytest
from botocore.config import Config
from botocore.exceptions import UnknownServiceError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import SecretStr

from langchain_aws.utils import (
    _BEDROCK_API_KEY_MAX_TTL_SECONDS,
    _BedrockApiKeyProvider,
    _StaticCredentialProvider,
    count_tokens_api_supported_for_model,
    create_aws_client,
    parse_model_provider,
    reasoning_effort_additional_fields,
    thinking_disabled_in_params,
    thinking_enabled_in_params,
    thinking_forced_tool_use_unsupported,
    thinking_in_params,
    thinking_on_by_default,
    trim_message_whitespace,
)


@pytest.fixture
def mock_boto3() -> Generator[
    Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock], None, None
]:
    with (
        mock.patch("boto3.Session") as m_session,
        mock.patch("boto3.client") as m_client,
    ):
        mock_session_instance = mock.MagicMock()
        m_session.return_value = mock_session_instance
        mock_session_instance.region_name = "us-west-2"

        mock_client_instance = mock.MagicMock()
        mock_session_instance.client.return_value = mock_client_instance
        m_client.return_value = mock_client_instance

        yield m_session, m_client, mock_client_instance


@pytest.mark.parametrize(
    "creds",
    [
        {"aws_access_key_id": SecretStr("test_key")},
        {"aws_secret_access_key": SecretStr("test_secret")},
        {"aws_session_token": SecretStr("test_token")},
    ],
)
def test_invalid_creds(
    creds: Dict[str, SecretStr],
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    with pytest.raises(
        ValueError,
        match="both aws_access_key_id and aws_secret_access_key must be specified",
    ):
        create_aws_client("bedrock-runtime", **creds)  # type: ignore


def test_valid_creds(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    client = create_aws_client(
        "bedrock-runtime",
        aws_access_key_id=SecretStr("test_key"),
        aws_secret_access_key=SecretStr("test_secret"),
    )

    session_mock.assert_called_once_with(
        aws_access_key_id="test_key", aws_secret_access_key="test_secret"
    )
    client_mock.assert_not_called()
    assert client == client_instance


def test_valid_creds_with_session_token(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    client = create_aws_client(
        "bedrock-runtime",
        aws_access_key_id=SecretStr("test_key"),
        aws_secret_access_key=SecretStr("test_secret"),
        aws_session_token=SecretStr("test_token"),
    )

    session_mock.assert_called_once_with(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
        aws_session_token="test_token",
    )
    client_mock.assert_not_called()
    assert client == client_instance


def test_creds_from_profile_name(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    client = create_aws_client(
        "bedrock-runtime", credentials_profile_name="test_profile"
    )

    session_mock.assert_called_once_with(profile_name="test_profile")
    client_mock.assert_not_called()
    assert client == client_instance


def test_creds_default(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client("bedrock-runtime")

    session_mock.assert_not_called()
    client_mock.assert_called_once_with(service_name="bedrock-runtime")
    assert client == client_instance


@pytest.mark.parametrize(
    "env_var,env_value,expected_region",
    [
        ("AWS_REGION", "us-west-2", "us-west-2"),
        ("AWS_DEFAULT_REGION", "us-east-1", "us-east-1"),
    ],
)
def test_region_from_env_vars(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
    env_var: str,
    env_value: str,
    expected_region: str,
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    # Clear other AWS region env vars to test only the specified one
    env_patch = {env_var: env_value}
    for var in ["AWS_REGION", "AWS_DEFAULT_REGION"]:
        if var != env_var:
            env_patch[var] = ""

    with mock.patch.dict(os.environ, env_patch):
        client = create_aws_client("bedrock-runtime")

    session_mock.assert_not_called()

    client_mock.assert_called_once_with(
        service_name="bedrock-runtime", region_name=expected_region
    )
    assert client == client_instance


def test_endpoint_url(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        )

    session_mock.assert_not_called()
    client_mock.assert_called_once_with(
        service_name="bedrock-runtime",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
    )
    assert client == client_instance


def test_with_config(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    boto_config = Config(max_pool_connections=10)

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client("bedrock-runtime", config=boto_config)

    session_mock.assert_not_called()
    client_mock.assert_called_once_with(
        service_name="bedrock-runtime", config=boto_config
    )
    assert client == client_instance


def test_endpoint_url_with_creds(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3
    session_instance = session_mock.return_value

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            aws_access_key_id=SecretStr("test_key"),
            aws_secret_access_key=SecretStr("test_secret"),
            endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
        )

    session_mock.assert_called_once_with(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
    )
    session_instance.client.assert_called_once_with(
        service_name="bedrock-runtime",
        region_name="us-west-2",
        endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
    )
    client_mock.assert_not_called()
    assert client == client_instance


def test_region_with_creds(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3
    session_instance = session_mock.return_value

    client = create_aws_client(
        "bedrock-runtime",
        aws_access_key_id=SecretStr("test_key"),
        aws_secret_access_key=SecretStr("test_secret"),
        region_name="us-east-1",
    )

    session_mock.assert_called_once_with(
        aws_access_key_id="test_key",
        aws_secret_access_key="test_secret",
    )
    session_instance.client.assert_called_once_with(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )
    client_mock.assert_not_called()
    assert client == client_instance


def test_session_region_fallback(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3
    session_instance = session_mock.return_value

    session_instance.region_name = "us-west-2"

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            aws_access_key_id=SecretStr("test_key"),
            aws_secret_access_key=SecretStr("test_secret"),
        )

    session_mock.assert_called_once()
    session_instance.client.assert_called_once_with(
        service_name="bedrock-runtime", region_name="us-west-2"
    )
    assert client == client_instance


@pytest.fixture
def mock_boto3_with_imports() -> Generator[
    Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock, type[UnknownServiceError]],
    None,
    None,
]:
    with (
        mock.patch("boto3.Session") as m_session,
        mock.patch("boto3.client") as m_client,
        mock.patch(
            "botocore.exceptions.UnknownServiceError", UnknownServiceError
        ) as m_error,
    ):
        mock_session_instance = mock.MagicMock()
        m_session.return_value = mock_session_instance
        mock_session_instance.region_name = "us-west-2"

        mock_client_instance = mock.MagicMock()
        mock_session_instance.client.return_value = mock_client_instance
        m_client.return_value = mock_client_instance

        yield m_session, m_client, mock_client_instance, m_error


def test_bad_service_error_with_session(
    mock_boto3_with_imports: Tuple[
        mock.MagicMock, mock.MagicMock, mock.MagicMock, type[UnknownServiceError]
    ],
) -> None:
    session_mock, _, _, error_class = mock_boto3_with_imports
    session_instance = session_mock.return_value

    session_instance.client.side_effect = error_class(
        service_name="not-a-service", known_service_names=["bedrock-runtime"]
    )

    with pytest.raises(
        ModuleNotFoundError,
        match="Ensure that you have installed the latest boto3 package",
    ):
        create_aws_client(
            "not-a-service",
            aws_access_key_id=SecretStr("test_key"),
            aws_secret_access_key=SecretStr("test_secret"),
        )


def test_bad_service_error_with_direct_client(
    mock_boto3_with_imports: Tuple[
        mock.MagicMock, mock.MagicMock, mock.MagicMock, type[UnknownServiceError]
    ],
) -> None:
    _, client_mock, _, error_class = mock_boto3_with_imports

    client_mock.side_effect = error_class(
        service_name="not-a-service", known_service_names=["bedrock-runtime"]
    )

    with pytest.raises(
        ModuleNotFoundError,
        match="Ensure that you have installed the latest boto3 package",
    ):
        create_aws_client("not-a-service")


def test_boto3_error_with_session(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, _, _ = mock_boto3
    session_instance = session_mock.return_value

    session_instance.client.side_effect = ValueError("Service error")

    with pytest.raises(ValueError, match="Error raised by service"):
        create_aws_client(
            "bedrock-runtime",
            aws_access_key_id=SecretStr("test_key"),
            aws_secret_access_key=SecretStr("test_secret"),
        )


def test_boto3_error_with_direct_client(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    _, client_mock, _ = mock_boto3

    client_mock.side_effect = ValueError("Service error")

    with pytest.raises(ValueError, match="Error raised by service"):
        create_aws_client("bedrock-runtime")


def test_generic_error_with_session(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, _, _ = mock_boto3
    session_instance = session_mock.return_value

    session_instance.client.side_effect = Exception("Generic error")

    with pytest.raises(ValueError, match="Error raised by service:\n\nGeneric error"):
        create_aws_client(
            "bedrock-runtime",
            aws_access_key_id=SecretStr("test_key"),
            aws_secret_access_key=SecretStr("test_secret"),
        )


def test_generic_error_with_direct_client(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    _, client_mock, _ = mock_boto3

    client_mock.side_effect = Exception("Generic error")

    with pytest.raises(ValueError, match="Error raised by service:\n\nGeneric error"):
        create_aws_client("bedrock-runtime")


def test_trim_message_whitespace_final_ai_message() -> None:
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there!   \n  ")]

    result = trim_message_whitespace(messages)

    assert result[0].content == "Hello"
    assert result[1].content == "Hi there!"

    messages = [
        HumanMessage(content="Hello"),
        AIMessage(
            content=[
                {"type": "text", "text": "First response.   \n  "},
                {"type": "text", "text": "Second response.\t  "},
            ]
        ),
    ]

    result = trim_message_whitespace(messages)

    assert result[1].content[0]["text"] == "First response."
    assert result[1].content[1]["text"] == "Second response."


def test_trim_message_whitespace_final_nonai_message() -> None:
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!   \n  "),
        HumanMessage(content="How are you?   \n  "),
    ]

    result = trim_message_whitespace(messages)

    assert result[0].content == "Hello"
    assert result[1].content == "Hi there!   \n  "
    assert result[2].content == "How are you?   \n  "


def test_trim_message_whitespace_no_ai_messages() -> None:
    messages = [
        HumanMessage(content="Hello   \n  "),
        HumanMessage(content="How are you?\t  "),
    ]

    result = trim_message_whitespace(messages)

    assert result[0].content == "Hello   \n  "
    assert result[1].content == "How are you?\t  "


def test_trim_message_whitespace_with_empty_messages() -> None:
    messages: List[BaseMessage] = []

    result = trim_message_whitespace(messages)

    assert result == messages


@pytest.mark.parametrize(
    "model_id,expected_result",
    [
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", True),
        ("us.anthropic.claude-opus-4-20250514-v1:0", True),
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", True),
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", True),
        ("us.anthropic.claude-3-5-sonnet-20240620-v1:0", True),
        ("us.anthropic.claude-fable-5", False),
        ("us.anthropic.claude-sonnet-5", False),
        ("global.anthropic.claude-opus-5", False),
        ("us.anthropic.claude-3-sonnet-20240229-v1:0", False),
        ("us.meta.llama4-scout-17b-instruct-v1:0", False),
        ("us.amazon.nova-pro-v1:0", False),
        ("xai.grok-4.6", False),
    ],
)
def test_count_tokens_api_supported_for_model(
    model_id: str, expected_result: bool
) -> None:
    result = count_tokens_api_supported_for_model(model_id)

    assert result == expected_result


@pytest.mark.parametrize(
    "model_id,expected_result",
    [
        ("anthropic.claude-3-7-sonnet-20250219-v1:0", True),
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", True),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("anthropic.claude-sonnet-4-6", True),
        ("us.anthropic.claude-opus-4-20250514-v1:0", True),
        ("anthropic.claude-opus-4-6-v1", True),
        ("global.anthropic.claude-opus-4-7", True),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", True),
        ("global.anthropic.claude-opus-4-8", False),
        ("us.anthropic.claude-sonnet-5", False),
        ("global.anthropic.claude-opus-5", False),
        ("global.anthropic.claude-fable-5", False),
        ("deepseek.v3-v1:0", True),
        ("deepseek.v3.2", True),
    ],
)
def test_thinking_forced_tool_use_unsupported(
    model_id: str, expected_result: bool
) -> None:
    assert thinking_forced_tool_use_unsupported(model_id) == expected_result


@pytest.mark.parametrize(
    "model_id,params,expected_result",
    [
        ("anthropic.claude-sonnet-4-6", {"thinking": {"type": "enabled"}}, True),
        ("anthropic.claude-sonnet-4-6", {"reasoning_effort": "high"}, False),
        ("deepseek.v3.2", {"reasoning_effort": "high"}, True),
        ("deepseek.v3.2", {"reasoning_effort": "low"}, False),
        ("deepseek.v3.2", {"thinking": {"type": "enabled"}}, False),
        ("deepseek.v3.2", {}, False),
    ],
)
def test_thinking_enabled_in_params(
    model_id: str, params: Dict[str, Any], expected_result: bool
) -> None:
    assert thinking_enabled_in_params(model_id, params) == expected_result


@pytest.mark.parametrize(
    "model_id,expected_result",
    [
        ("us.anthropic.claude-sonnet-5", True),
        ("global.anthropic.claude-opus-5", True),
        ("global.anthropic.claude-fable-5", True),
        ("global.anthropic.claude-opus-4-8", False),
        ("anthropic.claude-sonnet-4-6", False),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", False),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", False),
    ],
)
def test_thinking_on_by_default(model_id: str, expected_result: bool) -> None:
    assert thinking_on_by_default(model_id) == expected_result


@pytest.mark.parametrize(
    "params,expected_result",
    [
        ({"thinking": {"type": "disabled"}}, True),
        ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, False),
        ({"thinking": {"type": "adaptive"}}, False),
        ({}, False),
    ],
)
def test_thinking_disabled_in_params(params: dict, expected_result: bool) -> None:
    assert thinking_disabled_in_params(params) == expected_result


@pytest.mark.parametrize(
    "base_model,effort,expected_fields",
    [
        (
            "us.anthropic.claude-sonnet-5",
            "high",
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        ),
        (
            "anthropic.claude-opus-5",
            "xhigh",
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
        ),
        (
            "amazon.nova-2-lite-v1:0",
            "medium",
            {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}},
        ),
        (
            "openai.gpt-oss-120b-1:0",
            "low",
            {"reasoning_effort": "low"},
        ),
        (
            "moonshot.kimi-k2-thinking",
            "max",
            {"reasoning_effort": "max"},
        ),
        (
            "moonshotai.kimi-k2.5",
            "low",
            {"reasoning_effort": "low"},
        ),
        ("amazon.titan-text-express-v1", "high", {}),
        ("meta.llama3-1-70b-instruct-v1:0", "high", {}),
        # Native GPT-5.x models are bedrock-mantle-only
        ("openai.gpt-5.5", "medium", {}),
    ],
)
def test_reasoning_effort_additional_fields(
    base_model: str, effort: str, expected_fields: dict
) -> None:
    assert reasoning_effort_additional_fields(base_model, effort) == expected_fields


def test_api_key_uses_token_provider(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """Test that api_key uses token provider injection instead of env var."""
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            api_key=SecretStr("test-api-key"),
        )
        # Should NOT set the environment variable anymore
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") is None

    # The new implementation uses boto3.Session with a custom botocore_session
    session_mock.assert_called_once()
    # Verify botocore_session was passed (contains the token provider)
    call_kwargs = session_mock.call_args[1]
    assert "botocore_session" in call_kwargs

    # The client should be created via session.client() with bearer auth config
    session_instance = session_mock.return_value
    session_instance.client.assert_called_once()
    client_call_kwargs = session_instance.client.call_args[1]
    assert client_call_kwargs["service_name"] == "bedrock-runtime"
    # Check that auth_scheme_preference is set in config
    assert hasattr(client_call_kwargs.get("config"), "auth_scheme_preference")
    assert client == client_instance


def test_api_key_token_provider_actually_injected() -> None:
    """Verify the token provider is actually injected and returns the token."""

    # Create a client with api_key - this will create a real botocore session
    # but we mock the client creation to avoid needing real credentials
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch("boto3.Session") as mock_session,
    ):
        mock_session_instance = mock.MagicMock()
        mock_session_instance.region_name = "us-west-2"
        mock_client_instance = mock.MagicMock()
        mock_session_instance.client.return_value = mock_client_instance
        mock_session.return_value = mock_session_instance

        create_aws_client(
            "bedrock-runtime",
            api_key=SecretStr("my-test-bearer-token"),
        )

        # Verify that boto3.Session was called with a botocore_session
        session_call_kwargs = mock_session.call_args[1]
        assert "botocore_session" in session_call_kwargs
        bc_session = session_call_kwargs["botocore_session"]

        # Verify the token provider was injected
        token_provider_chain = bc_session.get_component("token_provider")
        providers = token_provider_chain._providers

        # The first provider should be our static token provider
        assert len(providers) > 0
        static_provider = providers[0]

        # Verify it returns the correct token
        from botocore.tokens import FrozenAuthToken

        token = static_provider.load_token()
        assert isinstance(token, FrozenAuthToken)
        assert token.token == "my-test-bearer-token"


def test_api_key_multi_tenant_isolation() -> None:
    """Test that multiple clients with different API keys do not interfere."""
    with mock.patch.dict(os.environ, {}, clear=True):
        # Mock boto3.Session to capture the botocore_session passed to it
        sessions_created: list[dict[str, Any]] = []

        def capture_session(*args: Any, **kwargs: Any) -> mock.MagicMock:
            mock_session_instance = mock.MagicMock()
            mock_session_instance.region_name = "us-west-2"
            mock_client_instance = mock.MagicMock()
            mock_session_instance.client.return_value = mock_client_instance
            sessions_created.append(
                {"args": args, "kwargs": kwargs, "instance": mock_session_instance}
            )
            return mock_session_instance

        with mock.patch("boto3.Session", side_effect=capture_session):
            # Create first client with first API key
            create_aws_client(
                "bedrock-runtime",
                api_key=SecretStr("tenant-a-api-key"),
            )

            # Create second client with different API key
            create_aws_client(
                "bedrock-runtime",
                api_key=SecretStr("tenant-b-api-key"),
            )

        # Verify two separate sessions were created
        assert len(sessions_created) == 2

        # Verify each session has its own token provider with the correct token
        for i, (session_data, expected_token) in enumerate(
            zip(sessions_created, ["tenant-a-api-key", "tenant-b-api-key"])
        ):
            bc_session = session_data["kwargs"].get("botocore_session")
            assert bc_session is not None, f"Session {i} missing botocore_session"

            token_provider_chain = bc_session.get_component("token_provider")
            static_provider = token_provider_chain._providers[0]

            token = static_provider.load_token()
            assert token.token == expected_token, f"Session {i} has wrong token"

        # Verify environment variable was never set
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") is None


def test_api_key_with_region(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """Test that api_key works with region_name."""
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            region_name="us-west-2",
            api_key=SecretStr("test-api-key"),
        )

    # The new implementation uses boto3.Session with a custom botocore_session
    session_mock.assert_called_once()
    session_instance = session_mock.return_value
    session_instance.client.assert_called_once()
    client_call_kwargs = session_instance.client.call_args[1]
    assert client_call_kwargs["service_name"] == "bedrock-runtime"
    assert client_call_kwargs["region_name"] == "us-west-2"
    assert client == client_instance


@pytest.mark.parametrize(
    "conflicting_creds",
    [
        {
            "aws_access_key_id": SecretStr("key"),
            "aws_secret_access_key": SecretStr("secret"),
        },
        {"credentials_profile_name": "my-profile"},
    ],
)
def test_api_key_takes_precedence_over_creds(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
    conflicting_creds: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that api_key takes precedence over AWS credentials."""
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            api_key=SecretStr("test-api-key"),
            **conflicting_creds,
        )
        # Should NOT set the environment variable anymore
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") is None

    # Verify warning was logged
    assert "Both api_key and AWS credentials were provided" in caplog.text

    # The new implementation uses boto3.Session with a custom botocore_session
    session_mock.assert_called_once()
    session_instance = session_mock.return_value
    session_instance.client.assert_called_once()
    client_call_kwargs = session_instance.client.call_args[1]
    assert client_call_kwargs["service_name"] == "bedrock-runtime"
    assert client == client_instance


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"thinking": {"type": "enabled", "budget_tokens": 5000}}, True),
        ({"thinking": {"type": "adaptive"}}, True),
        ({"thinking": {"type": "disabled"}}, False),
        ({"thinking": {}}, False),
        ({}, False),
        ({"other_param": "value"}, False),
    ],
)
def test_thinking_in_params(params: dict, expected: bool) -> None:
    assert thinking_in_params(params) == expected


@pytest.mark.parametrize(
    "model_id,expected_provider",
    [
        ("anthropic.claude-sonnet-5", "anthropic"),
        ("global.anthropic.claude-fable-5", "anthropic"),
        ("us-gov.anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic"),
        ("minimax.minimax-m2.5", "minimax"),
        ("us.minimax.minimax-m2.5", "minimax"),
        ("moonshotai.kimi-k2.5", "moonshotai"),
        ("moonshot.kimi-k2-thinking", "moonshot"),
    ],
)
def test_parse_model_provider(model_id: str, expected_provider: str) -> None:
    assert parse_model_provider(model_id) == expected_provider


@pytest.mark.parametrize("api_key", [SecretStr(""), None])
def test_empty_or_none_api_key_is_ignored(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
    api_key: SecretStr | None,
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            api_key=api_key,
        )

    session_mock.assert_not_called()
    client_mock.assert_called_once_with(service_name="bedrock-runtime")
    assert client == client_instance


def test_api_key_from_env_var_preserved_when_not_provided(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(
        os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "env-api-key"}, clear=True
    ):
        client = create_aws_client("bedrock-runtime")
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "env-api-key"

    session_mock.assert_not_called()
    client_mock.assert_called_once_with(service_name="bedrock-runtime")
    assert client == client_instance


def test_api_key_overrides_existing_env_var(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """Test that api_key is used for auth without modifying env var."""
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(
        os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "old-env-key"}, clear=True
    ):
        client = create_aws_client(
            "bedrock-runtime",
            api_key=SecretStr("new-api-key"),
        )
        # Should NOT modify the environment variable anymore
        # The token is injected via the token provider chain instead
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "old-env-key"

    # The new implementation uses boto3.Session with a custom botocore_session
    session_mock.assert_called_once()
    session_instance = session_mock.return_value
    session_instance.client.assert_called_once()
    assert client == client_instance


# ---------------------------------------------------------------------------
# create_aws_bedrock_runtime_client tests
# ---------------------------------------------------------------------------

# Mock the smithy SDK modules so tests work without installing nova-sonic deps
_mock_client_instance = mock.MagicMock()
_mock_client_instance._ensure_setup = mock.AsyncMock()


@pytest.fixture
def mock_bedrock_runtime_sdk() -> Generator[
    Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock], None, None
]:
    """Mock aws_sdk_bedrock_runtime and the smithy transport/identity modules."""
    mock_client_cls = mock.MagicMock(return_value=_mock_client_instance)
    mock_transport_cls = mock.MagicMock()
    mock_identity_cls = mock.MagicMock()

    mock_client_module = mock.MagicMock(spec=["AsyncBedrockRuntimeClient"])
    mock_client_module.AsyncBedrockRuntimeClient = mock_client_cls
    mock_crt_module = mock.MagicMock(spec=["AWSCRTHTTPClient"])
    mock_crt_module.AWSCRTHTTPClient = mock_transport_cls
    mock_identity_module = mock.MagicMock(spec=["AWSCredentialsIdentity"])
    mock_identity_module.AWSCredentialsIdentity = mock_identity_cls

    with (
        mock.patch.dict(
            "sys.modules",
            {
                "aws_sdk_bedrock_runtime": mock.MagicMock(),
                "aws_sdk_bedrock_runtime.client": mock_client_module,
                "smithy_http": mock.MagicMock(),
                "smithy_http.aio": mock.MagicMock(),
                "smithy_http.aio.crt": mock_crt_module,
                "smithy_aws_core": mock.MagicMock(),
                "smithy_aws_core.identity": mock.MagicMock(),
                "smithy_aws_core.identity.components": mock_identity_module,
            },
        ),
    ):
        yield mock_client_cls, mock_transport_cls, mock_identity_cls


def _apply_config_plugin(client_cls: mock.MagicMock) -> mock.MagicMock:
    """Run the client's config plugin against a bare config and return it."""
    plugins = client_cls.call_args[1]["plugins"]
    assert len(plugins) == 1
    config = mock.MagicMock(
        spec=[
            "endpoint_uri",
            "region",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_credentials_identity_resolver",
            "transport",
        ]
    )
    config.aws_access_key_id = None
    config.aws_secret_access_key = None
    config.aws_credentials_identity_resolver = None
    plugins[0](config)
    return config


def _create_client(**kwargs: Any) -> Any:
    from langchain_aws.utils import create_aws_bedrock_runtime_client

    return create_aws_bedrock_runtime_client(**kwargs)


def test_bedrock_runtime_default_no_creds(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """No credentials provided — endpoint built from region, awscrt transport."""
    client_cls, transport_cls, _ = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        client = _create_client(region_name="us-east-1")

    client_cls.assert_called_once()
    assert client == _mock_client_instance
    _mock_client_instance._ensure_setup.assert_awaited()

    config = _apply_config_plugin(client_cls)
    assert config.endpoint_uri == "https://bedrock-runtime.us-east-1.amazonaws.com"
    assert config.region == "us-east-1"
    transport_cls.assert_called_once()
    assert config.transport == transport_cls.return_value


def test_bedrock_runtime_explicit_keys(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """Explicit keys become a credentials identity resolver."""
    client_cls, _, identity_cls = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        _create_client(
            region_name="us-west-2",
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
        )

    identity_cls.assert_called_once_with(
        access_key_id="AKIA_TEST",
        secret_access_key="SECRET_TEST",
        session_token=None,
    )
    config = _apply_config_plugin(client_cls)
    assert config.endpoint_uri == "https://bedrock-runtime.us-west-2.amazonaws.com"
    assert config.region == "us-west-2"
    assert config.aws_credentials_identity_resolver is not None
    assert config.aws_access_key_id is None
    assert config.aws_secret_access_key is None


def test_bedrock_runtime_explicit_keys_with_session_token(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """Session token is forwarded when provided alongside access keys."""
    _, _, identity_cls = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        _create_client(
            region_name="eu-west-1",
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
            aws_session_token=SecretStr("TOKEN_TEST"),
        )

    identity_cls.assert_called_once_with(
        access_key_id="AKIA_TEST",
        secret_access_key="SECRET_TEST",
        session_token="TOKEN_TEST",
    )


def test_bedrock_runtime_profile_name(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """Profile name resolves credentials via boto3.Session."""
    client_cls, _, identity_cls = mock_bedrock_runtime_sdk

    mock_creds = mock.MagicMock()
    mock_creds.access_key = "PROFILE_KEY"
    mock_creds.secret_key = "PROFILE_SECRET"
    mock_creds.token = "PROFILE_TOKEN"

    mock_session = mock.MagicMock()
    mock_session.get_credentials.return_value.get_frozen_credentials.return_value = (
        mock_creds
    )
    mock_session.region_name = "ap-southeast-1"

    with (
        mock.patch("boto3.Session", return_value=mock_session) as session_cls,
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        _create_client(credentials_profile_name="my-profile")

    session_cls.assert_called_once_with(profile_name="my-profile")
    identity_cls.assert_called_once_with(
        access_key_id="PROFILE_KEY",
        secret_access_key="PROFILE_SECRET",
        session_token="PROFILE_TOKEN",
    )
    config = _apply_config_plugin(client_cls)
    assert config.endpoint_uri == "https://bedrock-runtime.ap-southeast-1.amazonaws.com"
    assert config.region == "ap-southeast-1"
    assert config.aws_credentials_identity_resolver is not None


def test_bedrock_runtime_profile_no_credentials_raises(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """Profile that returns no credentials raises ValueError."""
    mock_session = mock.MagicMock()
    mock_session.get_credentials.return_value = None

    with (
        mock.patch("boto3.Session", return_value=mock_session),
        mock.patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="Could not load credentials"),
    ):
        _create_client(credentials_profile_name="bad-profile")


@pytest.mark.parametrize(
    "creds",
    [
        {"aws_access_key_id": SecretStr("only_key")},
        {"aws_secret_access_key": SecretStr("only_secret")},
        {"aws_session_token": SecretStr("only_token")},
    ],
)
def test_bedrock_runtime_invalid_creds(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
    creds: Dict[str, SecretStr],
) -> None:
    """Partial credentials (missing key or secret) raise ValueError."""
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        pytest.raises(
            ValueError,
            match="both aws_access_key_id and aws_secret_access_key must be specified",
        ),
    ):
        _create_client(**creds)


def test_bedrock_runtime_custom_endpoint_url(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """Custom endpoint_url is passed through without modification."""
    client_cls, _, _ = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        _create_client(
            region_name="us-east-1",
            endpoint_url="https://custom.endpoint.example.com",
        )

    config = _apply_config_plugin(client_cls)
    assert config.endpoint_uri == "https://custom.endpoint.example.com"
    assert config.region == "us-east-1"


@pytest.mark.parametrize(
    "env_var,env_value",
    [
        ("AWS_REGION", "eu-central-1"),
        ("AWS_DEFAULT_REGION", "ap-northeast-1"),
    ],
)
def test_bedrock_runtime_region_from_env(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
    env_var: str,
    env_value: str,
) -> None:
    """Region falls back to AWS_REGION / AWS_DEFAULT_REGION env vars."""
    client_cls, _, _ = mock_bedrock_runtime_sdk

    env_patch = {env_var: env_value}
    for var in ["AWS_REGION", "AWS_DEFAULT_REGION"]:
        if var != env_var:
            env_patch[var] = ""

    with mock.patch.dict(os.environ, env_patch):
        _create_client()

    config = _apply_config_plugin(client_cls)
    assert config.region == env_value
    assert env_value in config.endpoint_uri


def test_bedrock_runtime_api_key_sets_env(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """api_key sets AWS_BEARER_TOKEN_BEDROCK env var and skips AWS creds."""
    client_cls, _, identity_cls = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        _create_client(
            region_name="us-east-1",
            api_key=SecretStr("my-api-key"),
        )
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "my-api-key"

    # When api_key is used, no credentials resolver is built
    identity_cls.assert_not_called()
    config = _apply_config_plugin(client_cls)
    assert config.aws_credentials_identity_resolver is None
    assert config.aws_access_key_id is None
    assert config.aws_secret_access_key is None


def test_bedrock_runtime_api_key_with_creds_warns(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """Both api_key and AWS creds logs a warning; api_key wins."""
    client_cls, _, identity_cls = mock_bedrock_runtime_sdk

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch("langchain_aws.utils.logger") as mock_logger,
    ):
        _create_client(
            region_name="us-east-1",
            api_key=SecretStr("my-api-key"),
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
        )

    mock_logger.warning.assert_called_once()
    assert "Both api_key and AWS credentials" in mock_logger.warning.call_args[0][0]

    # api_key wins — no credentials resolver is built
    identity_cls.assert_not_called()
    config = _apply_config_plugin(client_cls)
    assert config.aws_credentials_identity_resolver is None


def test_bedrock_runtime_session_region_fallback(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    """When no region_name and profile provides one, use session region."""
    client_cls, _, _ = mock_bedrock_runtime_sdk

    mock_creds = mock.MagicMock()
    mock_creds.access_key = "KEY"
    mock_creds.secret_key = "SECRET"
    mock_creds.token = None

    mock_session = mock.MagicMock()
    mock_session.get_credentials.return_value.get_frozen_credentials.return_value = (
        mock_creds
    )
    mock_session.region_name = "sa-east-1"

    with (
        mock.patch("boto3.Session", return_value=mock_session),
        mock.patch.dict(os.environ, {}, clear=True),
    ):
        _create_client(credentials_profile_name="regional-profile")

    config = _apply_config_plugin(client_cls)
    assert config.region == "sa-east-1"


def test_bedrock_runtime_missing_awscrt_raises() -> None:
    """Missing awscrt transport raises a helpful error."""
    mock_client_module = mock.MagicMock(spec=["AsyncBedrockRuntimeClient"])
    mock_client_module.AsyncBedrockRuntimeClient = mock.MagicMock()

    mock_session = mock.MagicMock()
    mock_session.get_credentials.return_value = None

    with (
        mock.patch.dict(
            "sys.modules",
            {
                "aws_sdk_bedrock_runtime": mock.MagicMock(),
                "aws_sdk_bedrock_runtime.client": mock_client_module,
                "smithy_http": None,
                "smithy_http.aio": None,
                "smithy_http.aio.crt": None,
            },
        ),
        mock.patch("boto3.Session", return_value=mock_session),
        mock.patch.dict(os.environ, {}, clear=True),
        pytest.raises(ModuleNotFoundError, match="awscrt"),
    ):
        _create_client(region_name="us-east-1")


# ---------------------------------------------------------------------------
# _StaticCredentialProvider / _BedrockApiKeyProvider tests
# ---------------------------------------------------------------------------


def _make_provider(**overrides: Any) -> _BedrockApiKeyProvider:
    """Build a provider with explicit static creds unless overridden."""
    params: Dict[str, Any] = {
        "region": "us-east-1",
        "aws_access_key_id": "AKIA_TEST",
        "aws_secret_access_key": "SECRET_TEST",
    }
    params.update(overrides)
    return _BedrockApiKeyProvider(**params)


def test_static_credential_provider_load() -> None:
    creds = object()
    provider = _StaticCredentialProvider(creds)
    assert provider.load() is creds
    assert provider.METHOD == "explicit"


def test_bedrock_api_key_provider_generates_and_caches() -> None:
    with mock.patch(
        "aws_bedrock_token_generator.provide_token", return_value="tok-1"
    ) as m_provide:
        provider = _make_provider(aws_session_token="TOKEN_TEST")

        # First call mints the token.
        assert provider() == "tok-1"
        # Second call (well within TTL) returns the cached token, no re-mint.
        assert provider() == "tok-1"
        m_provide.assert_called_once()

        kwargs = m_provide.call_args[1]
        assert kwargs["region"] == "us-east-1"
        assert kwargs["expiry"] == timedelta(seconds=_BEDROCK_API_KEY_MAX_TTL_SECONDS)

        cred_provider = kwargs["aws_credentials_provider"]
        assert isinstance(cred_provider, _StaticCredentialProvider)
        loaded = cred_provider.load()
        assert loaded.access_key == "AKIA_TEST"
        assert loaded.secret_key == "SECRET_TEST"
        assert loaded.token == "TOKEN_TEST"


def test_bedrock_api_key_provider_profile_resolution() -> None:
    fake_creds = object()
    fake_session = mock.MagicMock()
    fake_session.get_credentials.return_value = fake_creds

    with (
        mock.patch(
            "aws_bedrock_token_generator.provide_token", return_value="tok"
        ) as m_provide,
        mock.patch("botocore.session.Session", return_value=fake_session) as m_session,
    ):
        provider = _BedrockApiKeyProvider(
            region="us-west-2", credentials_profile_name="my-profile"
        )
        assert provider() == "tok"

    m_session.assert_called_once_with(profile="my-profile")
    cred_provider = m_provide.call_args[1]["aws_credentials_provider"]
    assert isinstance(cred_provider, _StaticCredentialProvider)
    assert cred_provider.load() is fake_creds


def test_bedrock_api_key_provider_default_chain_when_unresolvable() -> None:
    """If botocore can't resolve creds, provider is None (default chain)."""
    fake_session = mock.MagicMock()
    fake_session.get_credentials.return_value = None

    with (
        mock.patch(
            "aws_bedrock_token_generator.provide_token", return_value="tok"
        ) as m_provide,
        mock.patch("botocore.session.Session", return_value=fake_session),
    ):
        provider = _BedrockApiKeyProvider(region="us-east-1")
        assert provider() == "tok"

    assert m_provide.call_args[1]["aws_credentials_provider"] is None


def test_bedrock_api_key_provider_refreshes_when_near_expiry() -> None:
    with mock.patch(
        "aws_bedrock_token_generator.provide_token",
        side_effect=["tok-1", "tok-2"],
    ) as m_provide:
        provider = _make_provider(ttl_seconds=1000)
        assert provider() == "tok-1"

        # Force the cached token into the mandatory-refresh window.
        provider._expires_at = time.monotonic() + 100
        assert provider() == "tok-2"
        assert m_provide.call_count == 2


def test_bedrock_api_key_provider_caps_ttl_to_cred_lifetime() -> None:
    with (
        mock.patch(
            "aws_bedrock_token_generator.provide_token", return_value="tok"
        ) as m_provide,
        mock.patch.object(
            _BedrockApiKeyProvider,
            "_credentials_seconds_remaining",
            return_value=300,
        ),
    ):
        provider = _make_provider(ttl_seconds=10000)
        provider()

    assert m_provide.call_args[1]["expiry"] == timedelta(seconds=300)


def test_bedrock_api_key_provider_advisory_failure_keeps_token() -> None:
    with mock.patch(
        "aws_bedrock_token_generator.provide_token",
        side_effect=["tok-1", RuntimeError("boom")],
    ):
        provider = _make_provider(ttl_seconds=1000)
        assert provider() == "tok-1"

        # Advisory window (needs refresh) but not yet mandatory: 700s remaining.
        provider._expires_at = time.monotonic() + 700
        # Refresh fails but is non-mandatory, so the stale token is returned.
        assert provider() == "tok-1"


def test_bedrock_api_key_provider_mandatory_failure_raises() -> None:
    with mock.patch(
        "aws_bedrock_token_generator.provide_token",
        side_effect=["tok-1", RuntimeError("boom")],
    ):
        provider = _make_provider(ttl_seconds=1000)
        assert provider() == "tok-1"

        provider._expires_at = time.monotonic() + 100  # mandatory window
        with pytest.raises(RuntimeError, match="boom"):
            provider()


async def test_bedrock_api_key_provider_async_call() -> None:
    with mock.patch("aws_bedrock_token_generator.provide_token", return_value="tok"):
        provider = _make_provider()
        assert await provider.async_call() == "tok"
