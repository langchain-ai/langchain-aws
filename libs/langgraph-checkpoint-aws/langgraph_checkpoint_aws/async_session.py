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
    run_boto3_in_executor,
    to_boto_params,
)


class AsyncBedrockAgentRuntimeSessionClient:
    """
    Asynchronous client for AWS Bedrock Agent Runtime API using standard boto3 with async executor.
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
        Initialize AsyncBedrockAgentRuntime with AWS configuration

        Args:
            client: Pre-configured bedrock-agent-runtime client instance
            session: Pre-configured boto3.Session instance
            region_name: AWS region (e.g., us-west-2)
            credentials_profile_name: AWS credentials profile name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            aws_session_token: AWS session token
            endpoint_url: Custom endpoint URL
            config: Boto3 config object
        """
        if client is not None:
            # Use provided client
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

            # Pre-create the client to avoid creating it for each operation
            self.client = self.session.client("bedrock-agent-runtime", **_client_kwargs)

    async def create_session(
        self, request: Optional[CreateSessionRequest] = None
    ) -> CreateSessionResponse:
        """Create a new session asynchronously"""
        params = to_boto_params(request) if request else {}
        response = await run_boto3_in_executor(self.client.create_session, **params)
        return CreateSessionResponse(**response)

    async def get_session(self, request: GetSessionRequest) -> GetSessionResponse:
        """Get details of an existing session asynchronously"""
        response = await run_boto3_in_executor(
            self.client.get_session, **to_boto_params(request)
        )
        return GetSessionResponse(**response)

    async def end_session(self, request: EndSessionRequest) -> EndSessionResponse:
        """End an existing session asynchronously"""
        response = await run_boto3_in_executor(
            self.client.end_session, **to_boto_params(request)
        )
        return EndSessionResponse(**response)

    async def delete_session(self, request: DeleteSessionRequest) -> None:
        """Delete an existing session asynchronously"""
        await run_boto3_in_executor(
            self.client.delete_session, **to_boto_params(request)
        )

    async def create_invocation(
        self, request: CreateInvocationRequest
    ) -> CreateInvocationResponse:
        """Create a new invocation asynchronously"""
        response = await run_boto3_in_executor(
            self.client.create_invocation, **to_boto_params(request)
        )
        return CreateInvocationResponse(**response)

    async def list_invocations(
        self, request: ListInvocationsRequest
    ) -> ListInvocationsResponse:
        """List invocations for a session asynchronously"""
        response = await run_boto3_in_executor(
            self.client.list_invocations, **to_boto_params(request)
        )
        return ListInvocationsResponse(**response)

    async def put_invocation_step(
        self, request: PutInvocationStepRequest
    ) -> PutInvocationStepResponse:
        """Put a step in an invocation asynchronously"""
        response = await run_boto3_in_executor(
            self.client.put_invocation_step, **to_boto_params(request)
        )
        return PutInvocationStepResponse(**response)

    async def get_invocation_step(
        self, request: GetInvocationStepRequest
    ) -> GetInvocationStepResponse:
        """Get a step in an invocation asynchronously"""
        response = await run_boto3_in_executor(
            self.client.get_invocation_step, **to_boto_params(request)
        )
        return GetInvocationStepResponse(**response)

    async def list_invocation_steps(
        self, request: ListInvocationStepsRequest
    ) -> ListInvocationStepsResponse:
        """List steps in an invocation asynchronously"""
        response = await run_boto3_in_executor(
            self.client.list_invocation_steps, **to_boto_params(request)
        )
        return ListInvocationStepsResponse(**response)
