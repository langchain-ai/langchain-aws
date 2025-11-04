import uuid
from unittest.mock import Mock, patch

import boto3
import pytest
from botocore.config import Config
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langgraph_checkpoint_aws import SDK_USER_AGENT
from langgraph_checkpoint_aws.models import (
    CreateSessionRequest,
)
from langgraph_checkpoint_aws.utils import (
    construct_checkpoint_tuple,
    deserialize_from_base64,
    generate_checkpoint_id,
    generate_deterministic_uuid,
    generate_write_id,
    process_aws_client_args,
    serialize_to_base64,
    to_boto_params,
)


class TestUtils:
    @pytest.fixture
    def json_serializer(self):
        return JsonPlusSerializer()

    def test_session_models_conversion(self, sample_metadata, sample_kms_key_arn):
        request = CreateSessionRequest(
            session_metadata=sample_metadata,
            encryption_key_arn=sample_kms_key_arn,
        )

        result = to_boto_params(request)

        assert result == {
            "encryptionKeyArn": sample_kms_key_arn,
            "sessionMetadata": {"key1": "value1", "key2": "value2"},
        }

        # Test without optional fields
        request = CreateSessionRequest()
        result = to_boto_params(request)
        assert result == {}

    @pytest.mark.parametrize(
        "test_case",
        [
            ("", "d41d8cd9-8f00-b204-e980-0998ecf8427e"),
            ("test-string", "661f8009-fa8e-56a9-d0e9-4a0a644397d7"),
            (
                "checkpoint$abc|def$11111111-1111-1111-1111-111111111111",
                "321a564a-a10d-4ffe-ae32-b32c1131af27",
            ),
        ],
    )
    def test_generate_deterministic_uuid(self, test_case):
        input_string, expected_uuid = test_case
        input_string_bytes = input_string.encode("utf-8")
        result_as_str = generate_deterministic_uuid(input_string)
        result_as_bytes = generate_deterministic_uuid(input_string_bytes)

        assert isinstance(result_as_str, uuid.UUID)
        assert isinstance(result_as_bytes, uuid.UUID)
        # Test deterministic behavior
        assert str(result_as_str) == expected_uuid
        assert str(result_as_bytes) == expected_uuid

    def test__generate_checkpoint_id_success(self):
        input_str = "test_namespace"
        result = generate_checkpoint_id(input_str)
        assert result == "72f4457f-e6bb-e1db-49ee-06cd9901904f"

    def test__generate_write_id_success(self):
        checkpoint_ns = "test_namespace"
        checkpoint_id = "test_checkpoint"
        result = generate_write_id(checkpoint_ns, checkpoint_id)
        assert result == "f75c463a-a608-0629-401e-f4d270073c0c"

    def test_serialize_deserialize_base64_success(self, json_serializer):
        sample_dict = {"key": "value"}
        serialized = serialize_to_base64(json_serializer, sample_dict)
        deserialized = deserialize_from_base64(json_serializer, *serialized)
        assert deserialized == sample_dict

    @patch("langgraph_checkpoint_aws.utils.deserialize_from_base64")
    def test__construct_checkpoint_tuple(
        self,
        mock_deserialize_from_base64,
        sample_session_checkpoint,
        sample_session_pending_write,
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"

        serde = Mock(spec=SerializerProtocol)
        mock_deserialize_from_base64.return_value = {}

        # Act
        result = construct_checkpoint_tuple(
            thread_id,
            checkpoint_ns,
            sample_session_checkpoint,
            [sample_session_pending_write],
            [],
            serde,
        )

        # Assert
        assert isinstance(result, CheckpointTuple)
        assert result.config.get("configurable", {})["thread_id"] == thread_id
        assert result.config.get("configurable", {})["checkpoint_ns"] == checkpoint_ns


@patch("boto3.Session.client")
@patch("botocore.client.BaseClient._make_request")
def test_process_aws_client_args_user_agent(mock_make_request, mock_client):
    # Setup
    config = Config(user_agent_extra="existing_agent")

    # Process args
    session_kwargs, client_kwargs = process_aws_client_args(
        region_name="us-west-2", config=config
    )

    # Create session
    session = boto3.Session(**session_kwargs)

    # Mock client instance to avoid network calls
    mock_client_instance = Mock()
    mock_client.return_value = mock_client_instance

    # Create client (mocked) - we don't use the client but need to call this to test
    session.client("bedrock-agent-runtime", **client_kwargs)

    # Verify that the config contains our user agent
    assert "config" in client_kwargs
    config_obj = client_kwargs["config"]
    assert hasattr(config_obj, "user_agent_extra")
    assert "existing_agent" in config_obj.user_agent_extra
    assert SDK_USER_AGENT in config_obj.user_agent_extra
