import os
from typing import Any, Dict, Generator, List, Tuple
from unittest import mock

import pytest
from botocore.config import Config
from botocore.exceptions import UnknownServiceError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import SecretStr

from langchain_aws.utils import (
    count_tokens_api_supported_for_model,
    create_aws_client,
    thinking_in_params,
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
        ("us.anthropic.claude-3-sonnet-20240229-v1:0", False),
        ("us.meta.llama4-scout-17b-instruct-v1:0", False),
        ("us.amazon.nova-pro-v1:0", False),
    ],
)
def test_count_tokens_api_supported_for_model(
    model_id: str, expected_result: bool
) -> None:
    result = count_tokens_api_supported_for_model(model_id)

    assert result == expected_result


def test_api_key_sets_env_var(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            api_key=SecretStr("test-api-key"),
        )
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "test-api-key"

    session_mock.assert_not_called()
    client_mock.assert_called_once_with(service_name="bedrock-runtime")
    assert client == client_instance


def test_api_key_with_region(
    mock_boto3: Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock],
) -> None:
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            region_name="us-west-2",
            api_key=SecretStr("test-api-key"),
        )

    session_mock.assert_not_called()
    client_mock.assert_called_once_with(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )
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
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(os.environ, {}, clear=True):
        client = create_aws_client(
            "bedrock-runtime",
            api_key=SecretStr("test-api-key"),
            **conflicting_creds,
        )
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "test-api-key"

    # Verify warning was logged
    assert "Both api_key and AWS credentials were provided" in caplog.text

    session_mock.assert_not_called()
    client_mock.assert_called_once_with(service_name="bedrock-runtime")
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
    session_mock, client_mock, client_instance = mock_boto3

    with mock.patch.dict(
        os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "old-env-key"}, clear=True
    ):
        client = create_aws_client(
            "bedrock-runtime",
            api_key=SecretStr("new-api-key"),
        )
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "new-api-key"

    session_mock.assert_not_called()
    client_mock.assert_called_once_with(service_name="bedrock-runtime")
    assert client == client_instance


# ---------------------------------------------------------------------------
# create_aws_bedrock_runtime_client tests
# ---------------------------------------------------------------------------

# Mock the smithy SDK modules so tests work without installing nova-sonic deps
_mock_config_instance = mock.MagicMock()
_mock_client_instance = mock.MagicMock()


@pytest.fixture
def mock_bedrock_runtime_sdk() -> Generator[
    Tuple[mock.MagicMock, mock.MagicMock], None, None
]:
    """Mock aws_sdk_bedrock_runtime so create_aws_bedrock_runtime_client works."""
    with (
        mock.patch.dict(
            "sys.modules",
            {
                "aws_sdk_bedrock_runtime": mock.MagicMock(),
                "aws_sdk_bedrock_runtime.client": mock.MagicMock(),
                "aws_sdk_bedrock_runtime.config": mock.MagicMock(),
            },
        ),
        mock.patch(
            "langchain_aws.utils.create_aws_bedrock_runtime_client.__module__",
            create=True,
        ),
    ):
        # We need to patch at the point of import inside the function
        mock_config_cls = mock.MagicMock(return_value=_mock_config_instance)
        mock_client_cls = mock.MagicMock(return_value=_mock_client_instance)

        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "aws_sdk_bedrock_runtime": mock.MagicMock(),
                    "aws_sdk_bedrock_runtime.client": mock.MagicMock(
                        BedrockRuntimeClient=mock_client_cls
                    ),
                    "aws_sdk_bedrock_runtime.config": mock.MagicMock(
                        Config=mock_config_cls
                    ),
                },
            ),
        ):
            yield mock_config_cls, mock_client_cls


def _create_client(**kwargs: Any) -> Any:
    from langchain_aws.utils import create_aws_bedrock_runtime_client

    return create_aws_bedrock_runtime_client(**kwargs)


def test_bedrock_runtime_default_no_creds(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
) -> None:
    """No credentials provided — keys should be None, endpoint built from region."""
    config_cls, client_cls = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        _create_client(region_name="us-east-1")

    config_cls.assert_called_once_with(
        endpoint_uri="https://bedrock-runtime.us-east-1.amazonaws.com",
        region="us-east-1",
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
    )
    client_cls.assert_called_once()


def test_bedrock_runtime_explicit_keys(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
) -> None:
    """Explicit access key + secret key are passed through as plain strings."""
    config_cls, client_cls = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        _create_client(
            region_name="us-west-2",
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
        )

    config_cls.assert_called_once_with(
        endpoint_uri="https://bedrock-runtime.us-west-2.amazonaws.com",
        region="us-west-2",
        aws_access_key_id="AKIA_TEST",
        aws_secret_access_key="SECRET_TEST",
        aws_session_token=None,
    )


def test_bedrock_runtime_explicit_keys_with_session_token(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
) -> None:
    """Session token is forwarded when provided alongside access keys."""
    config_cls, _ = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        _create_client(
            region_name="eu-west-1",
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
            aws_session_token=SecretStr("TOKEN_TEST"),
        )

    config_cls.assert_called_once_with(
        endpoint_uri="https://bedrock-runtime.eu-west-1.amazonaws.com",
        region="eu-west-1",
        aws_access_key_id="AKIA_TEST",
        aws_secret_access_key="SECRET_TEST",
        aws_session_token="TOKEN_TEST",
    )


def test_bedrock_runtime_profile_name(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
) -> None:
    """Profile name resolves credentials via boto3.Session."""
    config_cls, _ = mock_bedrock_runtime_sdk

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
    config_cls.assert_called_once_with(
        endpoint_uri="https://bedrock-runtime.ap-southeast-1.amazonaws.com",
        region="ap-southeast-1",
        aws_access_key_id="PROFILE_KEY",
        aws_secret_access_key="PROFILE_SECRET",
        aws_session_token="PROFILE_TOKEN",
    )


def test_bedrock_runtime_profile_no_credentials_raises(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
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
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
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
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
) -> None:
    """Custom endpoint_url is passed through without modification."""
    config_cls, _ = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        _create_client(
            region_name="us-east-1",
            endpoint_url="https://custom.endpoint.example.com",
        )

    config_cls.assert_called_once_with(
        endpoint_uri="https://custom.endpoint.example.com",
        region="us-east-1",
        aws_access_key_id=None,
        aws_secret_access_key=None,
        aws_session_token=None,
    )


@pytest.mark.parametrize(
    "env_var,env_value",
    [
        ("AWS_REGION", "eu-central-1"),
        ("AWS_DEFAULT_REGION", "ap-northeast-1"),
    ],
)
def test_bedrock_runtime_region_from_env(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
    env_var: str,
    env_value: str,
) -> None:
    """Region falls back to AWS_REGION / AWS_DEFAULT_REGION env vars."""
    config_cls, _ = mock_bedrock_runtime_sdk

    env_patch = {env_var: env_value}
    for var in ["AWS_REGION", "AWS_DEFAULT_REGION"]:
        if var != env_var:
            env_patch[var] = ""

    with mock.patch.dict(os.environ, env_patch):
        _create_client()

    call_kwargs = config_cls.call_args[1]
    assert call_kwargs["region"] == env_value
    assert env_value in call_kwargs["endpoint_uri"]


def test_bedrock_runtime_api_key_sets_env(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
) -> None:
    """api_key sets AWS_BEARER_TOKEN_BEDROCK env var."""
    config_cls, _ = mock_bedrock_runtime_sdk

    with mock.patch.dict(os.environ, {}, clear=True):
        _create_client(
            region_name="us-east-1",
            api_key=SecretStr("my-api-key"),
        )
        assert os.environ.get("AWS_BEARER_TOKEN_BEDROCK") == "my-api-key"

    # When api_key is used, credentials should be None (default chain)
    call_kwargs = config_cls.call_args[1]
    assert call_kwargs["aws_access_key_id"] is None
    assert call_kwargs["aws_secret_access_key"] is None


def test_bedrock_runtime_api_key_with_creds_warns(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
) -> None:
    """Both api_key and AWS creds logs a warning; api_key wins."""
    config_cls, _ = mock_bedrock_runtime_sdk

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

    # api_key wins — credentials should be None
    call_kwargs = config_cls.call_args[1]
    assert call_kwargs["aws_access_key_id"] is None


def test_bedrock_runtime_session_region_fallback(
    mock_bedrock_runtime_sdk: Tuple[mock.MagicMock, mock.MagicMock],
) -> None:
    """When no region_name and profile provides one, use session region."""
    config_cls, _ = mock_bedrock_runtime_sdk

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

    call_kwargs = config_cls.call_args[1]
    assert call_kwargs["region"] == "sa-east-1"
