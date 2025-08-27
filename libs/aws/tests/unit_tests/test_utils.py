import os
from typing import Dict, Generator, Tuple
from unittest import mock

import pytest
from botocore.config import Config
from botocore.exceptions import UnknownServiceError
from pydantic import SecretStr

from langchain_aws.utils import create_aws_client


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

    with mock.patch.dict(os.environ, {env_var: env_value}):
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
    Tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock, mock.MagicMock], None, None
]:
    with (
        mock.patch("boto3.Session") as m_session,
        mock.patch("boto3.client") as m_client,
        mock.patch("botocore.exceptions.UnknownServiceError", UnknownServiceError),
    ):
        mock_session_instance = mock.MagicMock()
        m_session.return_value = mock_session_instance
        mock_session_instance.region_name = "us-west-2"

        mock_client_instance = mock.MagicMock()
        mock_session_instance.client.return_value = mock_client_instance
        m_client.return_value = mock_client_instance

        yield m_session, m_client, mock_client_instance, UnknownServiceError


def test_bad_service_error_with_session(
    mock_boto3_with_imports: Tuple[
        mock.MagicMock, mock.MagicMock, mock.MagicMock, mock.MagicMock
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
        mock.MagicMock, mock.MagicMock, mock.MagicMock, mock.MagicMock
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
