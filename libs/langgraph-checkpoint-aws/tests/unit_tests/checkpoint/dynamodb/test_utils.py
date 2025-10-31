"""Tests for utility functions."""

from unittest.mock import Mock, patch

from botocore.config import Config

from langgraph_checkpoint_aws.checkpoint.dynamodb.utils import (
    create_client_config,
    create_dynamodb_client,
    process_aws_client_args,
)


class TestProcessAWSClientArgs:
    """Test process_aws_client_args function."""

    def test_process_with_all_parameters(self):
        """Test processing with all parameters."""
        custom_config = Config(connect_timeout=10)

        session_kwargs, client_kwargs = process_aws_client_args(
            region_name="us-east-1",
            endpoint_url="http://localhost:8000",
            boto_config=custom_config,
        )

        assert session_kwargs["region_name"] == "us-east-1"
        assert client_kwargs["endpoint_url"] == "http://localhost:8000"
        assert "config" in client_kwargs

    def test_process_with_no_parameters(self):
        """Test processing with no parameters."""
        session_kwargs, client_kwargs = process_aws_client_args()

        assert session_kwargs == {}
        assert "config" in client_kwargs
        assert "endpoint_url" not in client_kwargs


class TestCreateClientConfig:
    """Test create_client_config function."""

    def test_create_config_with_and_without_existing(self):
        """Test creating config with and without existing config."""
        # Without existing config
        config = create_client_config()
        assert config.user_agent_extra is not None
        assert "langgraph-dynamodb" in config.user_agent_extra
        assert "x-client-framework" in config.user_agent_extra

        # With existing user agent - should merge
        existing_config = Config(user_agent_extra="my-app/1.0")
        config = create_client_config(existing_config)
        assert "my-app/1.0" in config.user_agent_extra
        assert "langgraph-dynamodb" in config.user_agent_extra


class TestCreateDynamoDBClient:
    """Test create_dynamodb_client function."""

    @patch("langgraph_checkpoint_aws.checkpoint.dynamodb.utils.boto3.Session")
    def test_create_client_comprehensive(self, mock_session_class):
        """Test creating client with all parameters and user agent."""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.meta.service_model.service_name = "dynamodb"
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        custom_config = Config(connect_timeout=10)

        create_dynamodb_client(
            region_name="us-east-1",
            endpoint_url="http://localhost:8000",
            boto_config=custom_config,
        )

        # Verify session was created with correct parameters
        mock_session_class.assert_called_once()
        session_kwargs = mock_session_class.call_args[1]
        assert session_kwargs["region_name"] == "us-east-1"

        # Verify client was created with correct parameters and user agent
        mock_session.client.assert_called_once()
        client_kwargs = mock_session.client.call_args[1]
        assert client_kwargs["endpoint_url"] == "http://localhost:8000"
        assert "config" in client_kwargs
        assert "langgraph-dynamodb" in client_kwargs["config"].user_agent_extra
