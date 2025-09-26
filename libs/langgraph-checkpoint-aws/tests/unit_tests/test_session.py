from unittest.mock import ANY, Mock, patch

import pytest

from langgraph_checkpoint_aws.models import (
    CreateInvocationRequest,
    CreateInvocationResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    DeleteSessionRequest,
    EndSessionRequest,
    EndSessionResponse,
    GetInvocationStepRequest,
    GetInvocationStepResponse,
    GetSessionRequest,
    GetSessionResponse,
    ListInvocationsRequest,
    ListInvocationsResponse,
    ListInvocationStepsRequest,
    ListInvocationStepsResponse,
    PutInvocationStepRequest,
    PutInvocationStepResponse,
)
from langgraph_checkpoint_aws.saver import BedrockAgentRuntimeSessionClient


class TestBedrockAgentRuntimeSessionClient:
    @pytest.fixture
    def mock_session_client(self, mock_boto_client):
        with patch("boto3.Session") as mock_boto3_session:
            mock_boto3_session.return_value.client.return_value = mock_boto_client
            yield BedrockAgentRuntimeSessionClient()

    def test_init_with_custom_session(self, mock_boto_client):
        """Test initialization with a custom boto3 session"""
        # Arrange
        mock_custom_session = Mock()
        mock_custom_session.client.return_value = mock_boto_client

        # Act
        client = BedrockAgentRuntimeSessionClient(session=mock_custom_session)

        # Assert
        mock_custom_session.client.assert_called_once_with(
            "bedrock-agent-runtime", config=ANY
        )
        assert client.client == mock_boto_client

    def test_init_without_session(self, mock_boto_client):
        """Test initialization without custom session (default behavior)"""
        # Arrange & Act
        with patch("boto3.Session") as mock_boto3_session:
            mock_boto3_session.return_value.client.return_value = mock_boto_client
            client = BedrockAgentRuntimeSessionClient()

            # Assert
            mock_boto3_session.assert_called_once()
            assert client.client == mock_boto_client

    class TestSession:
        def test_create_session(
            self, mock_session_client, mock_boto_client, sample_create_session_response
        ):
            # Arrange
            mock_boto_client.create_session.return_value = (
                sample_create_session_response
            )
            request = CreateSessionRequest()

            # Act
            response = mock_session_client.create_session(request)

            # Assert
            assert isinstance(response, CreateSessionResponse)
            mock_boto_client.create_session.assert_called_once()

        def test_create_session_with_user_attr(
            self, mock_session_client, mock_boto_client, sample_create_session_response
        ):
            # Arrange
            mock_boto_client.create_session.return_value = (
                sample_create_session_response
            )
            request = CreateSessionRequest(
                session_metadata={"key": "value"},
                encryption_key_arn="test-arn",
                tags={"tag1": "value1"},
            )

            # Act
            response = mock_session_client.create_session(request)

            # Assert
            assert isinstance(response, CreateSessionResponse)
            mock_boto_client.create_session.assert_called_once()

        def test_get_session(
            self,
            mock_session_client,
            mock_boto_client,
            sample_get_session_response,
            sample_session_id,
        ):
            # Arrange
            mock_boto_client.get_session.return_value = sample_get_session_response
            request = GetSessionRequest(session_identifier=sample_session_id)

            # Act
            response = mock_session_client.get_session(request)

            # Assert
            assert isinstance(response, GetSessionResponse)
            mock_boto_client.get_session.assert_called_once()

        def test_end_session(
            self,
            mock_session_client,
            mock_boto_client,
            sample_get_session_response,
            sample_session_id,
        ):
            # Arrange
            mock_boto_client.end_session.return_value = sample_get_session_response
            request = EndSessionRequest(session_identifier=sample_session_id)

            # Act
            response = mock_session_client.end_session(request)

            # Assert
            assert isinstance(response, EndSessionResponse)
            mock_boto_client.end_session.assert_called_once()

        def test_delete_session(
            self, mock_session_client, mock_boto_client, sample_session_id
        ):
            # Arrange
            request = DeleteSessionRequest(session_identifier=sample_session_id)

            # Act
            mock_session_client.delete_session(request)

            # Assert
            mock_boto_client.delete_session.assert_called_once()

    class TestInvocation:
        def test_create_invocation(
            self,
            mock_session_client,
            mock_boto_client,
            sample_session_id,
            sample_create_invocation_response,
        ):
            # Arrange
            mock_boto_client.create_invocation.return_value = (
                sample_create_invocation_response
            )
            request = CreateInvocationRequest(session_identifier=sample_session_id)

            # Act
            response = mock_session_client.create_invocation(request)

            # Assert
            assert isinstance(response, CreateInvocationResponse)
            mock_boto_client.create_invocation.assert_called_once()

        def test_create_invocation_with_user_attr(
            self,
            mock_session_client,
            mock_boto_client,
            sample_session_id,
            sample_invocation_id,
            sample_create_invocation_response,
        ):
            # Arrange
            mock_boto_client.create_invocation.return_value = (
                sample_create_invocation_response
            )
            request = CreateInvocationRequest(
                session_identifier=sample_session_id,
                invocation_id=sample_invocation_id,
                description="Test invocation description",
            )

            # Act
            response = mock_session_client.create_invocation(request)

            # Assert
            assert isinstance(response, CreateInvocationResponse)
            mock_boto_client.create_invocation.assert_called_once()

        def test_list_invocation(
            self,
            mock_session_client,
            mock_boto_client,
            sample_session_id,
            sample_invocation_id,
            sample_list_invocation_response,
        ):
            # Arrange
            mock_boto_client.list_invocations.return_value = (
                sample_list_invocation_response
            )
            request = ListInvocationsRequest(
                session_identifier=sample_session_id, max_results=1
            )

            # Act
            response = mock_session_client.list_invocations(request)

            # Assert
            assert isinstance(response, ListInvocationsResponse)
            mock_boto_client.list_invocations.assert_called_once()

    class TestInvocationStep:
        def test_put_invocation_step(
            self,
            mock_session_client,
            mock_boto_client,
            sample_session_id,
            sample_invocation_id,
            sample_invocation_step_id,
            sample_timestamp,
            sample_invocation_step_payload,
            sample_put_invocation_step_response,
        ):
            # Arrange
            mock_boto_client.put_invocation_step.return_value = (
                sample_put_invocation_step_response
            )
            request = PutInvocationStepRequest(
                session_identifier=sample_session_id,
                invocation_identifier=sample_invocation_id,
                invocation_step_id=sample_invocation_step_id,
                invocation_step_time=sample_timestamp,
                payload=sample_invocation_step_payload,
            )

            # Act
            response = mock_session_client.put_invocation_step(request)

            # Assert
            assert isinstance(response, PutInvocationStepResponse)
            mock_boto_client.put_invocation_step.assert_called_once()

        def test_get_invocation_step(
            self,
            mock_session_client,
            mock_boto_client,
            sample_session_id,
            sample_invocation_id,
            sample_invocation_step_id,
            sample_get_invocation_step_response,
        ):
            # Arrange
            mock_boto_client.get_invocation_step.return_value = (
                sample_get_invocation_step_response
            )
            request = GetInvocationStepRequest(
                session_identifier=sample_session_id,
                invocation_identifier=sample_invocation_id,
                invocation_step_id=sample_invocation_step_id,
            )

            # Act
            response = mock_session_client.get_invocation_step(request)

            # Assert
            assert isinstance(response, GetInvocationStepResponse)
            mock_boto_client.get_invocation_step.assert_called_once()

        def test_list_invocation_steps(
            self,
            mock_session_client,
            mock_boto_client,
            sample_session_id,
            sample_invocation_id,
            sample_list_invocation_steps_response,
        ):
            # Arrange
            mock_boto_client.list_invocation_steps.return_value = (
                sample_list_invocation_steps_response
            )
            request = ListInvocationStepsRequest(
                session_identifier=sample_session_id,
                max_results=1,
            )

            # Act
            response = mock_session_client.list_invocation_steps(request)

            # Assert
            assert isinstance(response, ListInvocationStepsResponse)
            mock_boto_client.list_invocation_steps.assert_called_once()

        def test_list_invocation_steps_by_invocation(
            self,
            mock_session_client,
            mock_boto_client,
            sample_session_id,
            sample_invocation_id,
            sample_list_invocation_steps_response,
        ):
            # Arrange
            mock_boto_client.list_invocation_steps.return_value = (
                sample_list_invocation_steps_response
            )
            request = ListInvocationStepsRequest(
                session_identifier=sample_session_id,
                invocation_identifier=sample_invocation_id,
                max_results=1,
            )

            # Act
            response = mock_session_client.list_invocation_steps(request)

            # Assert
            assert isinstance(response, ListInvocationStepsResponse)
            mock_boto_client.list_invocation_steps.assert_called_once()
