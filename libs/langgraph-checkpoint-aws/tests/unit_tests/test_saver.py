import datetime
import json
from unittest.mock import ANY, Mock, patch

import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple
from pydantic import SecretStr

from langgraph_checkpoint_aws.models import (
    GetInvocationStepResponse,
    InvocationStep,
    ListInvocationStepsResponse,
)
from langgraph_checkpoint_aws.saver import (
    BedrockAgentRuntimeSessionClient,
    BedrockSessionSaver,
)


class TestBedrockSessionSaver:
    @pytest.fixture
    def session_saver(self, mock_boto_client):
        with patch("boto3.Session") as mock_boto3_session:
            mock_boto3_session.return_value.client.return_value = mock_boto_client
            yield BedrockSessionSaver()

    @pytest.fixture
    def runnable_config(self):
        return RunnableConfig(
            configurable={
                "thread_id": "test_thread_id",
                "checkpoint_ns": "test_namespace",
            }
        )

    def test_init_with_default_client(self, mock_boto_client):
        with patch("boto3.Session") as mock_boto3_session:
            mock_boto3_session.return_value.client.return_value = mock_boto_client
            saver = BedrockSessionSaver()
            assert isinstance(saver.session_client, BedrockAgentRuntimeSessionClient)
            assert saver.session_client.client == mock_boto_client

    def test_init_with_all_parameters(self, mock_boto_client):
        with patch("boto3.Session") as mock_boto3_session:
            mock_boto3_session.return_value.client.return_value = mock_boto_client

            config = Config(retries={"max_attempts": 5, "mode": "standard"})
            endpoint_url = "https://custom-endpoint.amazonaws.com"

            BedrockSessionSaver(
                region_name="us-west-2",
                credentials_profile_name="test-profile",
                aws_access_key_id=SecretStr("test-access-key"),
                aws_secret_access_key=SecretStr("test-secret-key"),
                aws_session_token=SecretStr("test-session-token"),
                endpoint_url=endpoint_url,
                config=config,
            )

            mock_boto3_session.assert_called_with(
                region_name="us-west-2",
                profile_name="test-profile",
                aws_access_key_id="test-access-key",
                aws_secret_access_key="test-secret-key",
                aws_session_token="test-session-token",
            )

            mock_boto3_session.return_value.client.assert_called_with(
                "bedrock-agent-runtime", endpoint_url=endpoint_url, config=ANY
            )

    def test_init_with_custom_session(self, mock_boto_client):
        """Test BedrockSessionSaver initialization with custom session"""
        # Arrange
        mock_custom_session = Mock()
        mock_custom_session.client.return_value = mock_boto_client

        # Act
        saver = BedrockSessionSaver(session=mock_custom_session)

        # Assert
        mock_custom_session.client.assert_called_once_with(
            "bedrock-agent-runtime", config=ANY
        )
        assert saver.session_client.client == mock_boto_client

    def test_init_with_pre_configured_client(self):
        """Test initialization with a pre-configured bedrock-agent-runtime client."""
        # Arrange
        mock_client = Mock()
        mock_client.meta.service_model.service_name = "bedrock-agent-runtime"

        # Act
        saver = BedrockSessionSaver(client=mock_client)

        # Assert
        assert saver.session_client.client == mock_client

    def test_init_with_wrong_service_client(self):
        """Test that client for wrong service raises ValueError."""
        # Arrange
        s3_client = Mock()
        s3_client.meta.service_model.service_name = "s3"

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Invalid client: expected 'bedrock-agent-runtime' client, "
            "got 's3' client. Please provide a bedrock-agent-runtime client.",
        ):
            BedrockSessionSaver(client=s3_client)

    def test__create_session_invocation_success(
        self, mock_boto_client, session_saver, sample_create_invocation_response
    ):
        # Arrange
        thread_id = "test_thread_id"
        invocation_id = "test_invocation_id"
        mock_boto_client.create_invocation.return_value = (
            sample_create_invocation_response
        )

        # Act
        session_saver._create_session_invocation(thread_id, invocation_id)

        # Assert
        mock_boto_client.create_invocation.assert_called_once_with(
            sessionIdentifier=thread_id,
            invocationId=invocation_id,
        )

    def test__create_session_invocation_conflict(self, mock_boto_client, session_saver):
        # Arrange
        mock_boto_client.create_invocation.side_effect = ClientError(
            error_response={"Error": {"Code": "ConflictException"}},
            operation_name="CreateInvocation",
        )
        thread_id = "test_thread_id"
        invocation_id = "test_invocation_id"

        # Act
        session_saver._create_session_invocation(thread_id, invocation_id)

        # Assert
        mock_boto_client.create_invocation.assert_called_once_with(
            sessionIdentifier=thread_id,
            invocationId=invocation_id,
        )

    def test__create_session_invocation_raises_error(
        self, mock_boto_client, session_saver
    ):
        # Arrange
        thread_id = "test_thread_id"
        invocation_id = "test_invocation_id"

        mock_boto_client.create_invocation.side_effect = ClientError(
            error_response={"Error": {"Code": "SomeOtherError"}},
            operation_name="CreateInvocation",
        )

        # Act & Assert
        with pytest.raises(ClientError) as exc_info:
            session_saver._create_session_invocation(thread_id, invocation_id)

        assert exc_info.value.response.get("Error", {}).get("Code") == "SomeOtherError"
        mock_boto_client.create_invocation.assert_called_once_with(
            sessionIdentifier=thread_id,
            invocationId=invocation_id,
        )

    def test__get_checkpoint_pending_writes_success(
        self,
        mock_boto_client,
        session_saver,
        sample_session_pending_write,
        sample_list_invocation_steps_response,
        sample_get_invocation_step_response,
    ):
        # Arrange
        thread_id = "test_thread"
        checkpoint_ns = "test_namespace"
        checkpoint_id = "test_checkpoint"

        # serialize payload
        sample_get_invocation_step_response["invocationStep"]["payload"][
            "contentBlocks"
        ][0]["text"] = sample_session_pending_write.model_dump_json()
        mock_boto_client.list_invocation_steps.return_value = (
            sample_list_invocation_steps_response
        )
        mock_boto_client.get_invocation_step.return_value = (
            sample_get_invocation_step_response
        )

        # Act
        result = session_saver._get_checkpoint_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

        # Assert
        assert len(result) == 1
        mock_boto_client.list_invocation_steps.assert_called_once_with(
            sessionIdentifier="test_thread",
            invocationIdentifier="f75c463a-a608-0629-401e-f4d270073c0c",
            maxResults=1,
        )
        mock_boto_client.get_invocation_step.assert_called_once_with(
            sessionIdentifier="test_thread",
            invocationIdentifier=ANY,
            invocationStepId=sample_list_invocation_steps_response[
                "invocationStepSummaries"
            ][0]["invocationStepId"],
        )

    def test__get_checkpoint_pending_writes_no_invocation_steps(
        self,
        mock_boto_client,
        session_saver,
        sample_list_invocation_steps_response,
    ):
        sample_list_invocation_steps_response["invocationStepSummaries"] = []
        mock_boto_client.list_invocation_steps.return_value = (
            sample_list_invocation_steps_response
        )
        result = session_saver._get_checkpoint_pending_writes(
            "thread_id", "ns", "checkpoint_id"
        )
        assert result == []

    def test__get_checkpoint_pending_writes_resource_not_found(
        self, mock_boto_client, session_saver
    ):
        mock_boto_client.list_invocation_steps.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "Resource not found",
                }
            },
            "ListInvocationSteps",
        )
        result = session_saver._get_checkpoint_pending_writes(
            "thread_id", "ns", "checkpoint_id"
        )
        assert result == []

    def test__get_checkpoint_pending_writes_client_error(
        self, mock_boto_client, session_saver, sample_invocation_step_payload
    ):
        mock_boto_client.list_invocation_steps.side_effect = ClientError(
            {"Error": {"Code": "SomeError", "Message": "Error occurred"}},
            "PutInvocationStep",
        )

        with pytest.raises(ClientError):
            session_saver._get_checkpoint_pending_writes(
                "thread_id", "ns", "checkpoint_id"
            )

    def test__save_invocation_step_success(
        self,
        mock_boto_client,
        session_saver,
        sample_invocation_step_payload,
        sample_put_invocation_step_response,
    ):
        # Arrange
        thread_id = "test_thread_id"
        invocation_identifier = "test_invocation_identifier"
        invocation_step_id = "test_invocation_step_id"
        invocation_step_time = datetime.datetime.now(datetime.timezone.utc)
        mock_boto_client.put_invocation_step.return_value = (
            sample_put_invocation_step_response
        )

        # Act
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = invocation_step_time
            session_saver._save_invocation_step(
                thread_id,
                invocation_identifier,
                invocation_step_id,
                sample_invocation_step_payload,
            )

        # Assert
        mock_boto_client.put_invocation_step.assert_called_once_with(
            sessionIdentifier=thread_id,
            invocationIdentifier=invocation_identifier,
            invocationStepId=invocation_step_id,
            invocationStepTime=invocation_step_time,
            payload={"contentBlocks": [{"text": "sample text"}]},
        )

    def test__save_invocation_step_client_error(
        self, mock_boto_client, session_saver, sample_invocation_step_payload
    ):
        mock_boto_client.put_invocation_step.side_effect = ClientError(
            {"Error": {"Code": "SomeError", "Message": "Error occurred"}},
            "PutInvocationStep",
        )

        with pytest.raises(ClientError):
            session_saver._save_invocation_step(
                "thread_id", "inv_id", "step_id", sample_invocation_step_payload
            )

    def test__find_most_recent_checkpoint_step_success(
        self,
        mock_boto_client,
        session_saver,
        sample_session_checkpoint,
        sample_list_invocation_steps_response,
        sample_get_invocation_step_response,
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"

        # serialize payload
        sample_get_invocation_step_response["invocationStep"]["payload"][
            "contentBlocks"
        ][0]["text"] = sample_session_checkpoint.model_dump_json()
        mock_boto_client.list_invocation_steps.return_value = (
            sample_list_invocation_steps_response
        )
        mock_boto_client.get_invocation_step.return_value = (
            sample_get_invocation_step_response
        )

        # Act
        result = session_saver._find_most_recent_checkpoint_step(
            thread_id, checkpoint_ns
        )

        # Assert
        assert result is not None
        mock_boto_client.list_invocation_steps.assert_called_once_with(
            sessionIdentifier=thread_id,
            invocationIdentifier=checkpoint_ns,
        )
        mock_boto_client.get_invocation_step.assert_called_once_with(
            sessionIdentifier=thread_id,
            invocationIdentifier=ANY,
            invocationStepId=sample_list_invocation_steps_response[
                "invocationStepSummaries"
            ][0]["invocationStepId"],
        )

    def test__find_most_recent_checkpoint_step_skips_writes(
        self,
        mock_boto_client,
        session_saver,
        sample_session_pending_write,
        sample_list_invocation_steps_response,
        sample_get_invocation_step_response,
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"

        # serialize payload
        sample_get_invocation_step_response["invocationStep"]["payload"][
            "contentBlocks"
        ][0]["text"] = sample_session_pending_write.model_dump_json()
        mock_boto_client.list_invocation_steps.return_value = (
            sample_list_invocation_steps_response
        )
        mock_boto_client.get_invocation_step.return_value = (
            sample_get_invocation_step_response
        )

        # Act
        result = session_saver._find_most_recent_checkpoint_step(
            thread_id, checkpoint_ns
        )

        # Assert
        assert result is None
        mock_boto_client.list_invocation_steps.assert_called_once_with(
            sessionIdentifier=thread_id,
            invocationIdentifier=checkpoint_ns,
        )
        mock_boto_client.get_invocation_step.assert_called_once_with(
            sessionIdentifier=thread_id,
            invocationIdentifier=ANY,
            invocationStepId=sample_list_invocation_steps_response[
                "invocationStepSummaries"
            ][0]["invocationStepId"],
        )

    def test__find_most_recent_checkpoint_step_no_invocation_steps(
        self,
        mock_boto_client,
        session_saver,
        sample_list_invocation_steps_response,
    ):
        sample_list_invocation_steps_response["invocationStepSummaries"] = []
        mock_boto_client.list_invocation_steps.return_value = (
            sample_list_invocation_steps_response
        )
        result = session_saver._find_most_recent_checkpoint_step("thread_id", "ns")
        assert result is None

    def test__get_checkpoint_step_with_checkpoint_id(
        self,
        mock_boto_client,
        session_saver,
        sample_get_invocation_step_response,
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"
        checkpoint_id = "test_checkpoint_id"
        session_saver._find_most_recent_checkpoint_step = Mock()
        mock_boto_client.get_invocation_step.return_value = (
            sample_get_invocation_step_response
        )

        # Act
        session_saver._get_checkpoint_step(thread_id, checkpoint_ns, checkpoint_id)

        # Assert
        session_saver._find_most_recent_checkpoint_step.assert_not_called()
        mock_boto_client.get_invocation_step.assert_called_once_with(
            sessionIdentifier=thread_id,
            invocationIdentifier=checkpoint_ns,
            invocationStepId=checkpoint_id,
        )

    def test__get_checkpoint_step_without_checkpoint_id(
        self,
        mock_boto_client,
        session_saver,
        sample_invocation_step_payload,
        sample_get_invocation_step_response,
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"
        session_saver._find_most_recent_checkpoint_step = Mock(
            return_value=sample_invocation_step_payload
        )

        # Act
        result = session_saver._get_checkpoint_step(thread_id, checkpoint_ns)

        # Assert
        assert result == sample_invocation_step_payload
        session_saver._find_most_recent_checkpoint_step.assert_called_once_with(
            thread_id,
            checkpoint_ns,
        )
        mock_boto_client.get_invocation_step.assert_not_called()

    def test__get_checkpoint_step_empty_without_checkpoint_id(
        self,
        mock_boto_client,
        session_saver,
        sample_invocation_step_payload,
        sample_get_invocation_step_response,
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"
        session_saver._find_most_recent_checkpoint_step = Mock(return_value=None)

        # Act
        result = session_saver._get_checkpoint_step(thread_id, checkpoint_ns)

        # Assert
        assert result is None
        session_saver._find_most_recent_checkpoint_step.assert_called_once_with(
            thread_id,
            checkpoint_ns,
        )
        mock_boto_client.get_invocation_step.assert_not_called()

    def test__get_task_sends_without_parent_checkpoint_id(
        self, session_saver, sample_session_checkpoint
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"

        # Act
        result = session_saver._get_task_sends(thread_id, checkpoint_ns, None)

        # Assert
        assert result == []

    def test__get_task_sends(
        self, session_saver, sample_session_pending_write_with_sends
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"
        parent_checkpoint_id = "test_parent_checkpoint_id"

        session_saver._get_checkpoint_pending_writes = Mock(
            return_value=sample_session_pending_write_with_sends
        )

        # Act
        result = session_saver._get_task_sends(
            thread_id, checkpoint_ns, parent_checkpoint_id
        )

        # Assert
        assert result == [
            ["2", "__pregel_tasks", ["json", b"eyJrMiI6ICJ2MiJ9"], "/test2/path2", 1],
            ["3", "__pregel_tasks", ["json", b"eyJrMyI6ICJ2MyJ9"], "/test3/path3", 1],
        ]

    def test__get_task_sends_empty(self, session_saver):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"
        parent_checkpoint_id = "test_parent_checkpoint_id"

        session_saver._get_checkpoint_pending_writes = Mock(return_value=[])

        # Act
        result = session_saver._get_task_sends(
            thread_id, checkpoint_ns, parent_checkpoint_id
        )

        # Assert
        assert result == []

    @patch("langgraph_checkpoint_aws.saver.construct_checkpoint_tuple")
    def test_get_tuple_success(
        self,
        mock_construct_checkpoint,
        session_saver,
        runnable_config,
        sample_get_invocation_step_response,
        sample_session_pending_write_with_sends,
        sample_session_checkpoint,
    ):
        # Arrange
        sample_get_invocation_step_response["invocationStep"]["payload"][
            "contentBlocks"
        ][0]["text"] = sample_session_checkpoint.model_dump_json()

        # Mock all required internal methods
        session_saver._generate_checkpoint_id = Mock(return_value="test_checkpoint_id")
        session_saver._get_checkpoint_step = Mock(
            return_value=InvocationStep(
                **sample_get_invocation_step_response["invocationStep"]
            )
        )
        session_saver._get_checkpoint_pending_writes = Mock(
            return_value=sample_session_pending_write_with_sends
        )
        session_saver._get_task_sends = Mock(return_value=[])
        mock_construct_checkpoint.return_value = Mock(spec=CheckpointTuple)

        # Act
        result = session_saver.get_tuple(runnable_config)

        # Assert
        assert isinstance(result, CheckpointTuple)

    def test_get_tuple_success_empty(self, session_saver, runnable_config):
        # Arrange
        session_saver._get_checkpoint_step = Mock(return_value=None)

        # Act
        result = session_saver.get_tuple(runnable_config)

        # Assert
        assert result is None

    def test_get_tuple_resource_not_found_error(self, session_saver, runnable_config):
        # Arrange
        session_saver._get_checkpoint_step = Mock(
            side_effect=ClientError(
                {
                    "Error": {
                        "Code": "ResourceNotFoundException",
                        "Message": "Resource not found",
                    }
                },
                "ListInvocationSteps",
            )
        )

        # Act
        result = session_saver.get_tuple(runnable_config)

        # Assert
        assert result is None

    def test_get_tuple_error(self, session_saver, runnable_config):
        # Arrange
        session_saver._get_checkpoint_step = Mock(
            side_effect=ClientError(
                {"Error": {"Code": "SomeOtherError"}},
                "ListInvocationSteps",
            )
        )

        # Act and Assert
        with pytest.raises(ClientError):
            session_saver.get_tuple(runnable_config)

    def test_put_success(
        self,
        session_saver,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
    ):
        # Arrange
        session_saver._create_session_invocation = Mock()
        session_saver._save_invocation_step = Mock()

        # Act
        session_saver.put(
            runnable_config, sample_checkpoint, sample_checkpoint_metadata, {}
        )

        # Assert
        session_saver._create_session_invocation.assert_called_once_with(
            runnable_config["configurable"]["thread_id"],
            "72f4457f-e6bb-e1db-49ee-06cd9901904f",
        )
        session_saver._save_invocation_step.assert_called_once_with(
            runnable_config["configurable"]["thread_id"],
            "72f4457f-e6bb-e1db-49ee-06cd9901904f",
            "checkpoint_123",
            ANY,
        )

    def test_put_writes_success(
        self,
        session_saver,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
    ):
        # Arrange
        task_id = "test_task_id"
        task_path = "test_task_path"
        writes = [("__pregel_pull", "__start__"), ("__pregel_pull", "add_one")]
        runnable_config["configurable"]["checkpoint_id"] = "test_checkpoint_id"

        session_saver._create_session_invocation = Mock()
        session_saver._save_invocation_step = Mock()
        session_saver._get_checkpoint_pending_writes = Mock(return_value=[])

        # Act
        session_saver.put_writes(runnable_config, writes, task_id, task_path)

        # Assert
        session_saver._create_session_invocation.assert_called_once_with(
            runnable_config["configurable"]["thread_id"],
            "ea473b95-7b9c-fe52-df2c-3a7353d3148b",
        )
        session_saver._save_invocation_step.assert_called_once_with(
            runnable_config["configurable"]["thread_id"],
            "ea473b95-7b9c-fe52-df2c-3a7353d3148b",
            None,
            ANY,
        )

    def test_put_writes_skip_existing_writes(
        self,
        session_saver,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
        sample_session_pending_write,
    ):
        # Arrange
        task_id = "test_task_id"
        task_path = "test_task_path"
        writes = [("__pregel_pull", "__start__")]
        runnable_config["configurable"]["checkpoint_id"] = "test_checkpoint_id"

        session_saver._create_session_invocation = Mock()
        session_saver._save_invocation_step = Mock()

        sample_session_pending_write.task_id = task_id
        sample_session_pending_write.write_idx = 0

        session_saver._get_checkpoint_pending_writes = Mock(
            return_value=[sample_session_pending_write]
        )

        # Act
        session_saver.put_writes(runnable_config, writes, task_id, task_path)

        # Assert
        session_saver._create_session_invocation.assert_called_once_with(
            runnable_config["configurable"]["thread_id"],
            "ea473b95-7b9c-fe52-df2c-3a7353d3148b",
        )
        session_saver._save_invocation_step.assert_not_called()

    def test_put_writes_override_existing_writes(
        self,
        session_saver,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
        sample_session_pending_write,
    ):
        # Arrange
        task_id = "test_task_id"
        task_path = "test_task_path"
        writes = [("__error__", "__start__")]
        runnable_config["configurable"]["checkpoint_id"] = "test_checkpoint_id"

        session_saver._create_session_invocation = Mock()
        session_saver._save_invocation_step = Mock()

        sample_session_pending_write.task_id = task_id
        sample_session_pending_write.write_idx = 0

        session_saver._get_checkpoint_pending_writes = Mock(
            return_value=[sample_session_pending_write]
        )

        # Act
        session_saver.put_writes(runnable_config, writes, task_id, task_path)

        # Assert
        session_saver._create_session_invocation.assert_called_once_with(
            runnable_config["configurable"]["thread_id"],
            "ea473b95-7b9c-fe52-df2c-3a7353d3148b",
        )
        session_saver._save_invocation_step.assert_called_once_with(
            runnable_config["configurable"]["thread_id"],
            "ea473b95-7b9c-fe52-df2c-3a7353d3148b",
            None,
            ANY,
        )

    @patch("langgraph_checkpoint_aws.saver.construct_checkpoint_tuple")
    def test_list_success(
        self,
        mock_construct_checkpoint,
        session_saver,
        runnable_config,
        sample_session_checkpoint,
        sample_list_invocation_steps_response,
        sample_get_invocation_step_response,
    ):
        # Arrange
        sample_get_invocation_step_response["invocationStep"]["payload"][
            "contentBlocks"
        ][0]["text"] = sample_session_checkpoint.model_dump_json()

        # Mock all required internal methods
        session_saver._generate_checkpoint_id = Mock(return_value="test_checkpoint_id")
        session_saver.session_client.get_invocation_step = Mock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        session_saver.session_client.list_invocation_steps = Mock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )
        session_saver._get_checkpoint_pending_writes = Mock(return_value=[])
        session_saver._get_task_sends = Mock(return_value=[])
        mock_construct_checkpoint.return_value = Mock(spec=CheckpointTuple)

        # Act
        result = session_saver.list(runnable_config)

        # Assert
        assert len(list(result)) == 1

    def test_list_skips_writes(
        self,
        session_saver,
        runnable_config,
        sample_session_pending_write,
        sample_list_invocation_steps_response,
        sample_get_invocation_step_response,
    ):
        # Arrange
        sample_get_invocation_step_response["invocationStep"]["payload"][
            "contentBlocks"
        ][0]["text"] = sample_session_pending_write.model_dump_json()

        # Mock all required internal methods
        session_saver._generate_checkpoint_id = Mock(return_value="test_checkpoint_id")
        session_saver.session_client.get_invocation_step = Mock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        session_saver.session_client.list_invocation_steps = Mock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )

        # Act
        result = session_saver.list(runnable_config)

        # Assert
        assert len(list(result)) == 0

    @patch("langgraph_checkpoint_aws.saver.construct_checkpoint_tuple")
    def test_list_with_limit(
        self,
        mock_construct_checkpoint,
        session_saver,
        runnable_config,
        sample_session_checkpoint,
        sample_list_invocation_steps_response,
        sample_get_invocation_step_response,
    ):
        # Arrange
        sample_get_invocation_step_response["invocationStep"]["payload"][
            "contentBlocks"
        ][0]["text"] = sample_session_checkpoint.model_dump_json()

        # Mock all required internal methods
        session_saver._generate_checkpoint_id = Mock(return_value="test_checkpoint_id")
        session_saver.session_client.get_invocation_step = Mock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        # Duplicate list response
        sample_list_invocation_steps_response["invocationStepSummaries"] *= 10
        session_saver.session_client.list_invocation_steps = Mock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )
        session_saver._get_checkpoint_pending_writes = Mock(return_value=[])
        session_saver._get_task_sends = Mock(return_value=[])
        mock_construct_checkpoint.return_value = Mock(spec=CheckpointTuple)

        # Act
        result = session_saver.list(runnable_config, limit=3)

        # Assert
        assert len(list(result)) == 3

    def test_list_with_filter(
        self,
        session_saver,
        runnable_config,
        sample_session_checkpoint,
        sample_list_invocation_steps_response,
        sample_get_invocation_step_response,
    ):
        # Arrange
        sample_get_invocation_step_response["invocationStep"]["payload"][
            "contentBlocks"
        ][0]["text"] = sample_session_checkpoint.model_dump_json()

        # Mock all required internal methods
        session_saver._generate_checkpoint_id = Mock(return_value="test_checkpoint_id")
        session_saver.session_client.get_invocation_step = Mock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        session_saver.session_client.list_invocation_steps = Mock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )
        session_saver._get_checkpoint_pending_writes = Mock(return_value=[])
        session_saver._get_task_sends = Mock(return_value=[])
        session_saver._construct_checkpoint_tuple = Mock(
            return_value=Mock(spec=CheckpointTuple)
        )

        # Act
        result = session_saver.list(runnable_config, filter={"key": "value1"})

        # Assert
        assert len(list(result)) == 0

    def test_list_with_before(
        self,
        session_saver,
        runnable_config,
        sample_session_checkpoint,
        sample_list_invocation_steps_response,
        sample_get_invocation_step_response,
    ):
        # Arrange
        before = RunnableConfig(
            configurable={
                "checkpoint_id": sample_get_invocation_step_response["invocationStep"][
                    "invocationStepId"
                ]
            }
        )
        sample_session_checkpoint.metadata = json.dumps(
            sample_session_checkpoint.metadata
        )
        sample_get_invocation_step_response["invocationStep"]["payload"][
            "contentBlocks"
        ][0]["text"] = sample_session_checkpoint.model_dump_json()

        # Mock all required internal methods
        session_saver._generate_checkpoint_id = Mock(return_value="test_checkpoint_id")
        session_saver.session_client.get_invocation_step = Mock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        session_saver.session_client.list_invocation_steps = Mock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )

        # Act
        result = session_saver.list(runnable_config, before=before)

        # Assert
        assert len(list(result)) == 0

    def test_list_empty_response(
        self,
        session_saver,
        runnable_config,
    ):
        # Arrange
        session_saver.session_client.list_invocation_steps = Mock(
            return_value=ListInvocationStepsResponse(invocation_step_summaries=[])
        )

        # Act
        result = session_saver.list(runnable_config)

        # Assert
        assert len(list(result)) == 0

    def test_list_returns_empty_on_resource_not_found(
        self,
        session_saver,
        runnable_config,
    ):
        # Arrange
        session_saver.session_client.list_invocation_steps = Mock(
            side_effect=ClientError(
                {
                    "Error": {
                        "Code": "ResourceNotFoundException",
                        "Message": "Resource not found",
                    }
                },
                "ListInvocationSteps",
            )
        )

        # Act
        result = session_saver.list(runnable_config)

        # Assert
        assert len(list(result)) == 0

    def test_list_error(
        self,
        session_saver,
        runnable_config,
    ):
        # Arrange
        session_saver.session_client.list_invocation_steps = Mock(
            side_effect=ClientError(
                {"Error": {"Code": "SomeOtherError"}},
                "ListInvocationSteps",
            )
        )

        # Act and Assert
        with pytest.raises(ClientError):
            next(session_saver.list(runnable_config))
