from typing import Any, Optional

import boto3
from botocore.config import Config
from pydantic import SecretStr

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
from langgraph_checkpoint_aws.utils import (
    _validate_bedrock_client,
    process_aws_client_args,
    to_boto_params,
)


class BedrockAgentRuntimeSessionClient:
    """
    Client for AWS Bedrock Agent Runtime API

    This class provides an interface to interact with AWS Bedrock Agent Runtime service.
    It handles session management, invocations and invocation steps through the Bedrock Agent Runtime API.

    The client supports operations like:
    - Session management (create, get, end, delete)
    - Invocation management (create, list)
    - Invocation step management (put, get, list)

    Attributes:
        client: Boto3 client for bedrock-agent-runtime service
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        session: Optional[boto3.Session] = None,
        region_name: Optional[str] = None,
        credentials_profile_name: Optional[str] = None,
        aws_access_key_id: Optional[SecretStr] = None,
        aws_secret_access_key: Optional[SecretStr] = None,
        aws_session_token: Optional[SecretStr] = None,
        endpoint_url: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize BedrockAgentRuntime with AWS configuration

        Args:
            client: Pre-configured bedrock-agent-runtime client instance
            session: Pre-configured boto3 session instance
            region_name: AWS region (e.g., us-west-2)
            credentials_profile_name: AWS credentials profile name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token
            endpoint_url: Custom endpoint URL
            config: Boto3 config object
        """
        if client is not None:
            # Use provided client directly
            _validate_bedrock_client(client)
            self.client = client
        else:
            _session_kwargs, _client_kwargs = process_aws_client_args(
                region_name,
                credentials_profile_name,
                aws_access_key_id,
                aws_secret_access_key,
                aws_session_token,
                endpoint_url,
                config,
            )

            if session is not None:
                # Use provided session directly
                self.session = session
            else:
                # Create a standard boto3 session
                self.session = boto3.Session(**_session_kwargs)

            self.client = self.session.client("bedrock-agent-runtime", **_client_kwargs)

    def create_session(
        self, request: Optional[CreateSessionRequest] = None
    ) -> CreateSessionResponse:
        """
        Create a new session

        Args:
            request (CreateSessionRequest): Optional object containing session creation details

        Returns:
            CreateSessionResponse: Response object containing session identifier and metadata
        """

        response = self.client.create_session(
            **to_boto_params(request) if request else {}
        )
        return CreateSessionResponse(**response)

    def get_session(self, request: GetSessionRequest) -> GetSessionResponse:
        """
        Get details of an existing session

        Args:
            request (GetSessionRequest): Object containing session identifier

        Returns:
            GetSessionResponse: Response object containing session details and metadata
        """
        response = self.client.get_session(**to_boto_params(request))
        return GetSessionResponse(**response)

    def end_session(self, request: EndSessionRequest) -> EndSessionResponse:
        """
        End an existing session

        Args:
            request (EndSessionRequest): Object containing session identifier

        Returns:
            EndSessionResponse: Response object containing the ended session details
        """
        response = self.client.end_session(**to_boto_params(request))
        return EndSessionResponse(**response)

    def delete_session(self, request: DeleteSessionRequest) -> None:
        """
        Delete an existing session

        Args:
            request (DeleteSessionRequest): Object containing session identifier
        """
        self.client.delete_session(**to_boto_params(request))

    def create_invocation(
        self, request: CreateInvocationRequest
    ) -> CreateInvocationResponse:
        """
        Create a new invocation

        Args:
            request (CreateInvocationRequest): Object containing invocation details

        Returns:
            CreateInvocationResponse: Response object containing invocation identifier and metadata
        """
        response = self.client.create_invocation(**to_boto_params(request))
        return CreateInvocationResponse(**response)

    def list_invocations(
        self, request: ListInvocationsRequest
    ) -> ListInvocationsResponse:
        """
        List invocations for a session

        Args:
            request (ListInvocationsRequest): Object containing session identifier

        Returns:
            ListInvocationsResponse: Response object containing list of invocations and pagination token
        """
        response = self.client.list_invocations(**to_boto_params(request))
        return ListInvocationsResponse(**response)

    def put_invocation_step(
        self, request: PutInvocationStepRequest
    ) -> PutInvocationStepResponse:
        """
        Put a step in an invocation

        Args:
            request (PutInvocationStepRequest): Object containing invocation identifier and step payload

        Returns:
            PutInvocationStepResponse: Response object containing invocation step identifier
        """
        response = self.client.put_invocation_step(**to_boto_params(request))
        return PutInvocationStepResponse(**response)

    def get_invocation_step(
        self, request: GetInvocationStepRequest
    ) -> GetInvocationStepResponse:
        """
        Get a step in an invocation

        Args:
            request (GetInvocationStepRequest): Object containing invocation and step identifiers

        Returns:
            GetInvocationStepResponse: Response object containing invocation step identifier and payload
        """
        response = self.client.get_invocation_step(**to_boto_params(request))
        return GetInvocationStepResponse(**response)

    def list_invocation_steps(
        self, request: ListInvocationStepsRequest
    ) -> ListInvocationStepsResponse:
        """
        List steps in an invocation

        Args:
            request (ListInvocationStepsRequest): Object containing invocation step id and pagination token

        Returns:
            ListInvocationStepsResponse: Response object containing list of invocation steps and pagination token
        """
        response = self.client.list_invocation_steps(**to_boto_params(request))
        return ListInvocationStepsResponse(**response)
