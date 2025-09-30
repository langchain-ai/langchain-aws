import datetime
import json
import sys
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointTuple

from langgraph_checkpoint_aws.async_saver import (
    AsyncBedrockSessionSaver,
)
from langgraph_checkpoint_aws.models import (
    GetInvocationStepResponse,
    InvocationStep,
    ListInvocationStepsResponse,
)


class TestAsyncBedrockSessionSaver:
    @pytest.fixture
    def session_saver(self, mock_boto_client):
        with patch("boto3.Session") as mock_aioboto_session:
            mock_aioboto_session.return_value.client.return_value = mock_boto_client
            yield AsyncBedrockSessionSaver()

    @pytest.fixture
    def runnable_config(self):
        return RunnableConfig(
            configurable={
                "thread_id": "test_thread_id",
                "checkpoint_ns": "test_namespace",
            }
        )

    def test_init_with_custom_session(self, mock_boto_client):
        """Test AsyncBedrockSessionSaver initialization with custom session"""
        # Arrange
        mock_custom_session = Mock()
        mock_custom_session.client.return_value = mock_boto_client

        # Act
        saver = AsyncBedrockSessionSaver(session=mock_custom_session)

        # Assert
        assert saver.session_client.session == mock_custom_session
        assert saver.session_client.client == mock_boto_client

    def test_init_with_pre_configured_client(self):
        """Test AsyncBedrockSessionSaver initialization with pre-configured client"""
        # Arrange
        mock_client = Mock()
        mock_client.meta.service_model.service_name = "bedrock-agent-runtime"

        # Act
        saver = AsyncBedrockSessionSaver(client=mock_client)

        # Assert
        assert saver.session_client.client == mock_client

    def test_init_with_wrong_service_client(self):
        """Test AsyncBedrockSessionSaver raises error with wrong service client"""
        # Arrange
        mock_wrong_client = Mock()
        mock_wrong_client.meta.service_model.service_name = "some-other-service"

        # Act & Assert
        with pytest.raises(
            ValueError,
            match="Invalid client: expected 'bedrock-agent-runtime' client, got "
            "'some-other-service' client. Please provide a bedrock-agent-runtime "
            "client.",
        ):
            AsyncBedrockSessionSaver(client=mock_wrong_client)

    @pytest.mark.asyncio
    async def test__create_session_invocation_success(
        self, mock_boto_client, session_saver, sample_create_invocation_response
    ):
        # Arrange
        thread_id = "test_thread_id"
        invocation_id = "test_invocation_id"
        mock_boto_client.create_invocation.return_value = (
            sample_create_invocation_response
        )

        # Act
        await session_saver._create_session_invocation(thread_id, invocation_id)

        # Assert
        mock_boto_client.create_invocation.assert_called_once()

    @pytest.mark.asyncio
    async def test__create_session_invocation_conflict(
        self, mock_boto_client, session_saver
    ):
        # Arrange
        error_response = {"Error": {"Code": "ConflictException", "Message": "Conflict"}}
        mock_boto_client.create_invocation.side_effect = ClientError(
            error_response=error_response,
            operation_name="CreateInvocation",
        )
        thread_id = "test_thread_id"
        invocation_id = "test_invocation_id"

        # Act - should not raise an exception
        await session_saver._create_session_invocation(thread_id, invocation_id)

        # Assert
        mock_boto_client.create_invocation.assert_called_once()

    @pytest.mark.asyncio
    async def test__create_session_invocation_raises_error(
        self, mock_boto_client, session_saver
    ):
        # Arrange
        thread_id = "test_thread_id"
        invocation_id = "test_invocation_id"

        error_response = {"Error": {"Code": "SomeOtherError", "Message": "Other error"}}
        mock_boto_client.create_invocation.side_effect = ClientError(
            error_response=error_response,
            operation_name="CreateInvocation",
        )

        # Act & Assert
        with pytest.raises(ClientError) as exc_info:
            await session_saver._create_session_invocation(thread_id, invocation_id)

        assert exc_info.value.response["Error"]["Code"] == "SomeOtherError"
        mock_boto_client.create_invocation.assert_called_once()

    @pytest.mark.asyncio
    async def test__get_checkpoint_pending_writes_success(
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
        result = await session_saver._get_checkpoint_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

        # Assert
        assert len(result) == 1
        mock_boto_client.list_invocation_steps.assert_called_once()
        mock_boto_client.get_invocation_step.assert_called_once()

    @pytest.mark.asyncio
    async def test__get_checkpoint_pending_writes_no_invocation_steps(
        self,
        mock_boto_client,
        session_saver,
        sample_list_invocation_steps_response,
    ):
        # Arrange
        sample_list_invocation_steps_response["invocationStepSummaries"] = []
        mock_boto_client.list_invocation_steps.return_value = (
            sample_list_invocation_steps_response
        )

        # Act
        result = await session_saver._get_checkpoint_pending_writes(
            "thread_id", "ns", "checkpoint_id"
        )

        # Assert
        assert result == []
        mock_boto_client.list_invocation_steps.assert_called_once()

    @pytest.mark.asyncio
    async def test__get_checkpoint_pending_writes_resource_not_found(
        self, mock_boto_client, session_saver
    ):
        # Arrange
        error_response = {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "Resource not found",
            }
        }
        mock_boto_client.list_invocation_steps.side_effect = ClientError(
            error_response=error_response,
            operation_name="ListInvocationSteps",
        )

        # Act
        result = await session_saver._get_checkpoint_pending_writes(
            "thread_id", "ns", "checkpoint_id"
        )

        # Assert
        assert result == []
        mock_boto_client.list_invocation_steps.assert_called_once()

    @pytest.mark.asyncio
    async def test__get_checkpoint_pending_writes_client_error(
        self, mock_boto_client, session_saver, sample_invocation_step_payload
    ):
        # Arrange
        error_response = {"Error": {"Code": "SomeError", "Message": "Error occurred"}}
        mock_boto_client.list_invocation_steps.side_effect = ClientError(
            error_response=error_response,
            operation_name="ListInvocationSteps",
        )

        # Act & Assert
        with pytest.raises(ClientError):
            await session_saver._get_checkpoint_pending_writes(
                "thread_id", "ns", "checkpoint_id"
            )

        mock_boto_client.list_invocation_steps.assert_called_once()

    @pytest.mark.asyncio
    async def test__save_invocation_step_success(
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
        mock_boto_client.put_invocation_step.return_value = (
            sample_put_invocation_step_response
        )

        # Act
        with patch("datetime.datetime") as mock_datetime:
            invocation_step_time = datetime.datetime.now(datetime.timezone.utc)
            mock_datetime.now.return_value = invocation_step_time
            await session_saver._save_invocation_step(
                thread_id,
                invocation_identifier,
                invocation_step_id,
                sample_invocation_step_payload,
            )

        # Assert
        mock_boto_client.put_invocation_step.assert_called_once()

    @pytest.mark.asyncio
    async def test__save_invocation_step_client_error(
        self, mock_boto_client, session_saver, sample_invocation_step_payload
    ):
        # Arrange
        error_response = {"Error": {"Code": "SomeError", "Message": "Error occurred"}}
        mock_boto_client.put_invocation_step.side_effect = ClientError(
            error_response=error_response,
            operation_name="PutInvocationStep",
        )

        # Act & Assert
        with pytest.raises(ClientError):
            await session_saver._save_invocation_step(
                "thread_id", "inv_id", "step_id", sample_invocation_step_payload
            )

        mock_boto_client.put_invocation_step.assert_called_once()

    @pytest.mark.asyncio
    async def test__find_most_recent_checkpoint_step_success(
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
        result = await session_saver._find_most_recent_checkpoint_step(
            thread_id, checkpoint_ns
        )

        # Assert
        assert result is not None
        mock_boto_client.list_invocation_steps.assert_called_once()
        mock_boto_client.get_invocation_step.assert_called_once()

    @pytest.mark.asyncio
    async def test__find_most_recent_checkpoint_step_skips_writes(
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
        result = await session_saver._find_most_recent_checkpoint_step(
            thread_id, checkpoint_ns
        )

        # Assert
        assert result is None
        mock_boto_client.list_invocation_steps.assert_called_once()
        mock_boto_client.get_invocation_step.assert_called_once()

    @pytest.mark.asyncio
    async def test__find_most_recent_checkpoint_step_no_invocation_steps(
        self,
        mock_boto_client,
        session_saver,
        sample_list_invocation_steps_response,
    ):
        # Arrange
        sample_list_invocation_steps_response["invocationStepSummaries"] = []
        mock_boto_client.list_invocation_steps.return_value = (
            sample_list_invocation_steps_response
        )

        # Act
        result = await session_saver._find_most_recent_checkpoint_step(
            "thread_id", "ns"
        )

        # Assert
        assert result is None
        mock_boto_client.list_invocation_steps.assert_called_once()

    @pytest.mark.asyncio
    async def test__get_checkpoint_step_with_checkpoint_id(
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
        await session_saver._get_checkpoint_step(
            thread_id, checkpoint_ns, checkpoint_id
        )

        # Assert
        session_saver._find_most_recent_checkpoint_step.assert_not_called()
        mock_boto_client.get_invocation_step.assert_called_once()

    @pytest.mark.asyncio
    async def test__get_checkpoint_step_without_checkpoint_id(
        self,
        mock_boto_client,
        session_saver,
        sample_invocation_step_payload,
        sample_get_invocation_step_response,
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"
        session_saver._find_most_recent_checkpoint_step = AsyncMock(
            return_value=sample_invocation_step_payload
        )

        # Act
        result = await session_saver._get_checkpoint_step(thread_id, checkpoint_ns)

        # Assert
        assert result == sample_invocation_step_payload
        session_saver._find_most_recent_checkpoint_step.assert_called_once_with(
            thread_id,
            checkpoint_ns,
        )
        mock_boto_client.get_invocation_step.assert_not_called()

    @pytest.mark.asyncio
    async def test__get_checkpoint_step_empty_without_checkpoint_id(
        self,
        mock_boto_client,
        session_saver,
        sample_invocation_step_payload,
        sample_get_invocation_step_response,
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"
        session_saver._find_most_recent_checkpoint_step = AsyncMock(return_value=None)

        # Act
        result = await session_saver._get_checkpoint_step(thread_id, checkpoint_ns)

        # Assert
        assert result is None
        session_saver._find_most_recent_checkpoint_step.assert_called_once_with(
            thread_id,
            checkpoint_ns,
        )
        mock_boto_client.get_invocation_step.assert_not_called()

    @pytest.mark.asyncio
    async def test__get_task_sends_without_parent_checkpoint_id(
        self, session_saver, sample_session_checkpoint
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"

        # Act
        result = await session_saver._get_task_sends(thread_id, checkpoint_ns, None)

        # Assert
        assert result == []

    @pytest.mark.asyncio
    async def test__get_task_sends(
        self, session_saver, sample_session_pending_write_with_sends
    ):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"
        parent_checkpoint_id = "test_parent_checkpoint_id"

        session_saver._get_checkpoint_pending_writes = AsyncMock(
            return_value=sample_session_pending_write_with_sends
        )

        # Act
        result = await session_saver._get_task_sends(
            thread_id, checkpoint_ns, parent_checkpoint_id
        )

        # Assert
        assert result == [
            ["2", "__pregel_tasks", ["json", b"eyJrMiI6ICJ2MiJ9"], "/test2/path2", 1],
            ["3", "__pregel_tasks", ["json", b"eyJrMyI6ICJ2MyJ9"], "/test3/path3", 1],
        ]
        session_saver._get_checkpoint_pending_writes.assert_called_once_with(
            thread_id, checkpoint_ns, parent_checkpoint_id
        )

    @pytest.mark.asyncio
    async def test__get_task_sends_empty(self, session_saver):
        # Arrange
        thread_id = "test_thread_id"
        checkpoint_ns = "test_namespace"
        parent_checkpoint_id = "test_parent_checkpoint_id"

        session_saver._get_checkpoint_pending_writes = AsyncMock(return_value=[])

        # Act
        result = await session_saver._get_task_sends(
            thread_id, checkpoint_ns, parent_checkpoint_id
        )

        # Assert
        assert result == []
        session_saver._get_checkpoint_pending_writes.assert_called_once_with(
            thread_id, checkpoint_ns, parent_checkpoint_id
        )

    @pytest.mark.asyncio
    @patch("langgraph_checkpoint_aws.async_saver.construct_checkpoint_tuple")
    async def test_aget_tuple_success(
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
        session_saver._generate_checkpoint_id = AsyncMock(
            return_value="test_checkpoint_id"
        )
        session_saver._get_checkpoint_step = AsyncMock(
            return_value=InvocationStep(
                **sample_get_invocation_step_response["invocationStep"]
            )
        )
        session_saver._get_checkpoint_pending_writes = AsyncMock(
            return_value=sample_session_pending_write_with_sends
        )
        session_saver._get_task_sends = AsyncMock(return_value=[])
        mock_construct_checkpoint.return_value = AsyncMock(spec=CheckpointTuple)

        # Act
        result = await session_saver.aget_tuple(runnable_config)

        # Assert
        assert isinstance(result, CheckpointTuple)

    @pytest.mark.asyncio
    async def test_aget_tuple_success_empty(self, session_saver, runnable_config):
        # Arrange
        session_saver._get_checkpoint_step = AsyncMock(return_value=None)

        # Act
        result = await session_saver.aget_tuple(runnable_config)

        # Assert
        assert result is None
        session_saver._get_checkpoint_step.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_tuple_resource_not_found_error(
        self, session_saver, runnable_config
    ):
        # Arrange
        error_response = {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "Resource not found",
            }
        }
        session_saver._get_checkpoint_step = AsyncMock(
            side_effect=ClientError(
                error_response=error_response,
                operation_name="ListInvocationSteps",
            )
        )

        # Act
        result = await session_saver.aget_tuple(runnable_config)

        # Assert
        assert result is None
        session_saver._get_checkpoint_step.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_tuple_error(self, session_saver, runnable_config):
        # Arrange
        error_response = {
            "Error": {"Code": "SomeOtherError", "Message": "Some other error"}
        }
        session_saver._get_checkpoint_step = AsyncMock(
            side_effect=ClientError(
                error_response=error_response,
                operation_name="ListInvocationSteps",
            )
        )

        # Act and Assert
        with pytest.raises(ClientError):
            await session_saver.aget_tuple(runnable_config)

        session_saver._get_checkpoint_step.assert_called_once()

    @pytest.mark.asyncio
    async def test_aput_success(
        self,
        session_saver,
        runnable_config,
        sample_checkpoint,
        sample_checkpoint_metadata,
    ):
        # Arrange
        session_saver._create_session_invocation = AsyncMock()
        session_saver._save_invocation_step = AsyncMock()

        # Act
        await session_saver.aput(
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

    @pytest.mark.asyncio
    async def test_aput_writes_success(
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

        session_saver._create_session_invocation = AsyncMock()
        session_saver._save_invocation_step = AsyncMock()
        session_saver._get_checkpoint_pending_writes = AsyncMock(return_value=[])

        # Act
        await session_saver.aput_writes(runnable_config, writes, task_id, task_path)

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

    @pytest.mark.asyncio
    async def test_aput_writes_skip_existing_writes(
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

        session_saver._create_session_invocation = AsyncMock()
        session_saver._save_invocation_step = AsyncMock()

        sample_session_pending_write.task_id = task_id
        sample_session_pending_write.write_idx = 0

        session_saver._get_checkpoint_pending_writes = AsyncMock(
            return_value=[sample_session_pending_write]
        )

        # Act
        await session_saver.aput_writes(runnable_config, writes, task_id, task_path)

        # Assert
        session_saver._create_session_invocation.assert_called_once_with(
            runnable_config["configurable"]["thread_id"],
            "ea473b95-7b9c-fe52-df2c-3a7353d3148b",
        )
        session_saver._save_invocation_step.assert_not_called()

    @pytest.mark.asyncio
    async def test_aput_writes_override_existing_writes(
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
        writes = [(sys.intern("__error__"), "__start__")]
        runnable_config["configurable"]["checkpoint_id"] = "test_checkpoint_id"

        session_saver._create_session_invocation = AsyncMock()
        session_saver._save_invocation_step = AsyncMock()

        sample_session_pending_write.task_id = task_id
        sample_session_pending_write.write_idx = 0

        session_saver._get_checkpoint_pending_writes = AsyncMock(
            return_value=[sample_session_pending_write]
        )

        # Act
        await session_saver.aput_writes(runnable_config, writes, task_id, task_path)

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

    @pytest.mark.asyncio
    @patch("langgraph_checkpoint_aws.async_saver.construct_checkpoint_tuple")
    async def test_alist_success(
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
        session_saver._generate_checkpoint_id = AsyncMock(
            return_value="test_checkpoint_id"
        )
        session_saver.session_client.get_invocation_step = AsyncMock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        session_saver.session_client.list_invocation_steps = AsyncMock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )
        session_saver._get_checkpoint_pending_writes = AsyncMock(return_value=[])
        session_saver._get_task_sends = AsyncMock(return_value=[])
        mock_construct_checkpoint.return_value = AsyncMock(spec=CheckpointTuple)

        # Act
        result = [
            checkpoint async for checkpoint in session_saver.alist(runnable_config)
        ]

        # Assert
        assert len(list(result)) == 1

    @pytest.mark.asyncio
    async def test_alist_skips_writes(
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
        session_saver._generate_checkpoint_id = AsyncMock(
            return_value="test_checkpoint_id"
        )
        session_saver.session_client.get_invocation_step = AsyncMock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        session_saver.session_client.list_invocation_steps = AsyncMock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )

        # Act
        result = [
            checkpoint async for checkpoint in session_saver.alist(runnable_config)
        ]

        # Assert
        assert len(list(result)) == 0

    @pytest.mark.asyncio
    @patch("langgraph_checkpoint_aws.async_saver.construct_checkpoint_tuple")
    async def test_alist_with_limit(
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
        session_saver._generate_checkpoint_id = AsyncMock(
            return_value="test_checkpoint_id"
        )
        session_saver.session_client.get_invocation_step = AsyncMock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        # Duplicate list response
        sample_list_invocation_steps_response["invocationStepSummaries"] *= 10
        session_saver.session_client.list_invocation_steps = AsyncMock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )
        session_saver._get_checkpoint_pending_writes = AsyncMock(return_value=[])
        session_saver._get_task_sends = AsyncMock(return_value=[])
        mock_construct_checkpoint.return_value = AsyncMock(spec=CheckpointTuple)

        # Act
        result = [
            checkpoint
            async for checkpoint in session_saver.alist(runnable_config, limit=3)
        ]

        # Assert
        assert len(list(result)) == 3

    @pytest.mark.asyncio
    async def test_alist_with_filter(
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
        session_saver._generate_checkpoint_id = AsyncMock(
            return_value="test_checkpoint_id"
        )
        session_saver.session_client.get_invocation_step = AsyncMock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        session_saver.session_client.list_invocation_steps = AsyncMock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )
        session_saver._get_checkpoint_pending_writes = AsyncMock(return_value=[])
        session_saver._get_task_sends = AsyncMock(return_value=[])
        session_saver._construct_checkpoint_tuple = AsyncMock(
            return_value=AsyncMock(spec=CheckpointTuple)
        )

        # Act
        result = [
            checkpoint
            async for checkpoint in session_saver.alist(
                runnable_config, filter={"key": "value1"}
            )
        ]

        # Assert
        assert len(list(result)) == 0

    @pytest.mark.asyncio
    async def test_alist_with_before(
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
        session_saver._generate_checkpoint_id = AsyncMock(
            return_value="test_checkpoint_id"
        )
        session_saver.session_client.get_invocation_step = AsyncMock(
            return_value=GetInvocationStepResponse(
                **sample_get_invocation_step_response
            )
        )
        session_saver.session_client.list_invocation_steps = AsyncMock(
            return_value=ListInvocationStepsResponse(
                **sample_list_invocation_steps_response
            )
        )

        # Act
        result = [
            checkpoint
            async for checkpoint in session_saver.alist(runnable_config, before=before)
        ]

        # Assert
        assert len(list(result)) == 0

    @pytest.mark.asyncio
    async def test_alist_empty_response(
        self,
        session_saver,
        runnable_config,
    ):
        # Arrange
        session_saver.session_client.list_invocation_steps = AsyncMock(
            return_value=ListInvocationStepsResponse(invocation_step_summaries=[])
        )

        # Act
        result = [
            checkpoint async for checkpoint in session_saver.alist(runnable_config)
        ]

        # Assert
        assert len(result) == 0
        session_saver.session_client.list_invocation_steps.assert_called_once()

    @pytest.mark.asyncio
    async def test_alist_returns_empty_on_resource_not_found(
        self,
        session_saver,
        runnable_config,
    ):
        # Arrange
        error_response = {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "Resource not found",
            }
        }
        session_saver.session_client.list_invocation_steps = AsyncMock(
            side_effect=ClientError(
                error_response=error_response,
                operation_name="ListInvocationSteps",
            )
        )

        # Act
        result = [
            checkpoint async for checkpoint in session_saver.alist(runnable_config)
        ]

        # Assert
        assert len(result) == 0
        session_saver.session_client.list_invocation_steps.assert_called_once()

    @pytest.mark.asyncio
    async def test_alist_error(
        self,
        session_saver,
        runnable_config,
    ):
        # Arrange
        error_response = {
            "Error": {"Code": "SomeOtherError", "Message": "Some other error"}
        }
        session_saver.session_client.list_invocation_steps = AsyncMock(
            side_effect=ClientError(
                error_response=error_response,
                operation_name="ListInvocationSteps",
            )
        )

        # Act and Assert
        with pytest.raises(ClientError):
            async for _ in session_saver.alist(runnable_config):
                pass

        session_saver.session_client.list_invocation_steps.assert_called_once()
