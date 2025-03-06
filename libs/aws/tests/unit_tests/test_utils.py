import os
from unittest import mock

import pytest
from botocore.config import Config
from botocore.exceptions import UnknownServiceError
from pydantic import SecretStr
from utils import get_aws_client


@pytest.fixture
def mock_session():
    with mock.patch('boto3.Session') as m:
        mock_session_instance = mock.MagicMock()
        m.return_value = mock_session_instance
        mock_session_instance.region_name = 'us-west-2'
        mock_client = mock.MagicMock()
        mock_session_instance.client.return_value = mock_client
        yield m, mock_session_instance, mock_client


@pytest.mark.parametrize(
    "creds",
    [
        {"aws_access_key_id": SecretStr("test_key")},
        {"aws_secret_access_key": SecretStr("test_secret")},
        {"aws_session_token": SecretStr("test_token")}
    ]
)
def test_invalid_creds(creds):
    with pytest.raises(ValueError, match="both aws_access_key_id and aws_secret_access_key must be specified"):
        get_aws_client('bedrock-runtime', **creds)


def test_valid_creds(mock_session):
    session_mock, session_instance, client_mock = mock_session

    client = get_aws_client(
        'bedrock-runtime',
        aws_access_key_id=SecretStr('test_key'),
        aws_secret_access_key=SecretStr('test_secret')
    )

    session_mock.assert_called_once_with(
        aws_access_key_id='test_key',
        aws_secret_access_key='test_secret'
    )
    session_instance.client.assert_called_once_with('bedrock-runtime', region_name='us-west-2')
    assert client == client_mock


def test_valid_creds_with_session_token(mock_session):
    session_mock, session_instance, client_mock = mock_session

    client = get_aws_client(
        'bedrock-runtime',
        aws_access_key_id=SecretStr('test_key'),
        aws_secret_access_key=SecretStr('test_secret'),
        aws_session_token=SecretStr('test_token')
    )

    session_mock.assert_called_once_with(
        aws_access_key_id='test_key',
        aws_secret_access_key='test_secret',
        aws_session_token='test_token'
    )
    assert client == client_mock


def test_creds_from_profile_name(mock_session):
    session_mock, session_instance, client_mock = mock_session
    client = get_aws_client('bedrock-runtime', credentials_profile_name='test_profile')
    session_mock.assert_called_once_with(profile_name='test_profile')
    assert client == client_mock


def test_creds_default(mock_session):
    session_mock, session_instance, client_mock = mock_session
    client = get_aws_client('bedrock-runtime')
    session_mock.assert_called_once_with()
    session_instance.client.assert_called_once_with('bedrock-runtime', region_name='us-west-2')
    assert client == client_mock


@pytest.mark.parametrize(
    "env_var,env_value,expected_region",
    [
        ("AWS_REGION", "us-west-2", "us-west-2"),
        ("AWS_DEFAULT_REGION", "us-east-1", "us-east-1")
    ]
)
def test_region_from_env_vars(mock_session, env_var, env_value, expected_region):
    _, session_instance, client_mock = mock_session

    with mock.patch.dict(os.environ, {env_var: env_value}):
        client = get_aws_client('bedrock-runtime')

    session_instance.client.assert_called_once_with('bedrock-runtime', region_name=expected_region)
    assert client == client_mock


def test_endpoint_url(mock_session):
    _, session_instance, client_mock = mock_session

    client = get_aws_client(
        'bedrock-runtime',
        endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com'
    )

    session_instance.client.assert_called_once_with(
        'bedrock-runtime',
        region_name='us-west-2',
        endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com'
    )
    assert client == client_mock


def test_with_config(mock_session):
    _, session_instance, client_mock = mock_session

    boto_config = Config(
        max_pool_connections=10
    )

    client = get_aws_client('bedrock-runtime', config=boto_config)

    session_instance.client.assert_called_once_with(
        'bedrock-runtime',
        region_name='us-west-2',
        config=boto_config
    )
    assert client == client_mock


def test_bad_service_error(mock_session):
    _, session_instance, _ = mock_session
    session_instance.client.side_effect = UnknownServiceError(
        service_name='not-a-service',
        known_service_names=['bedrock-runtime']
    )

    with pytest.raises(ModuleNotFoundError, match="Ensure that you have installed the latest boto3 package"):
        get_aws_client('not-a-service')


def test_internal_service_error(mock_session):
    _, session_instance, _ = mock_session
    session_instance.client.side_effect = ValueError()

    with pytest.raises(ValueError, match="Error raised by service"):
        get_aws_client('bedrock-runtime')


def test_generic_error(mock_session):
    _, session_instance, _ = mock_session
    session_instance.client.side_effect = Exception()

    with pytest.raises(ValueError, match="Could not load credentials to authenticate with AWS client"):
        get_aws_client('bedrock-runtime')