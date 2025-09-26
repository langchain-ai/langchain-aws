import datetime
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from pydantic import SecretStr

from langgraph_checkpoint_aws.async_session import AsyncBedrockAgentRuntimeSessionClient
from langgraph_checkpoint_aws.constants import CHECKPOINT_PREFIX
from langgraph_checkpoint_aws.models import (
    BedrockSessionContentBlock,
    CreateInvocationRequest,
    GetInvocationStepRequest,
    InvocationStep,
    InvocationStepPayload,
    ListInvocationStepsRequest,
    PutInvocationStepRequest,
    SessionCheckpoint,
    SessionPendingWrite,
)
from langgraph_checkpoint_aws.utils import (
    construct_checkpoint_tuple,
    create_session_checkpoint,
    deserialize_data,
    generate_checkpoint_id,
    generate_write_id,
    process_write_operations,
    process_writes_invocation_content_blocks,
    transform_pending_task_writes,
)


class AsyncBedrockSessionSaver(BaseCheckpointSaver):
    """Asynchronously saves and retrieves checkpoints using Amazon Bedrock Agent Runtime sessions.

    This class provides async functionality to persist checkpoint data and writes to Bedrock Agent Runtime sessions.
    It handles creating invocations, managing checkpoint data, and tracking pending writes.

    Args:
        client: Pre-configured bedrock-agent-runtime client instance
        session: Pre-configured boto3 session instance for custom credential
        region_name: AWS region name
        credentials_profile_name: AWS credentials profile name
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        aws_session_token: AWS session token
        endpoint_url: Custom endpoint URL for the Bedrock service
        config: Botocore config object
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
    ) -> None:
        super().__init__()
        self.session_client = AsyncBedrockAgentRuntimeSessionClient(
            client=client,
            session=session,
            region_name=region_name,
            credentials_profile_name=credentials_profile_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            endpoint_url=endpoint_url,
            config=config,
        )

    async def _create_session_invocation(self, thread_id: str, invocation_id: str):
        """Asynchronously create a new invocation if one doesn't already exist.

        Args:
            thread_id: The session identifier
            invocation_id: The unique invocation identifier

        Raises:
            ClientError: If creation fails for reasons other than the invocation already existing
        """
        try:
            await self.session_client.create_invocation(
                CreateInvocationRequest(
                    session_identifier=thread_id,
                    invocation_id=invocation_id,
                )
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConflictException":
                raise e

    async def _get_checkpoint_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[SessionPendingWrite]:
        """Asynchronously retrieve pending write operations for a given checkpoint from the Bedrock session.

        This method retrieves any pending write operations that were stored for a specific checkpoint.
        It first gets the most recent invocation step, then retrieves the full details of that step,
        and finally parses the content blocks to reconstruct the PendingWrite objects.

        Args:
            thread_id: Session thread identifier used to locate the checkpoint data
            checkpoint_ns: Namespace that groups related checkpoints together
            checkpoint_id: Unique identifier for the specific checkpoint to retrieve

        Returns:
            List of PendingWrite objects containing task_id, channel, value, task_path and write_idx.
            Returns empty list if no pending writes are found.
        """
        # Generate unique ID for the write operation
        writes_id = generate_write_id(checkpoint_ns, checkpoint_id)

        try:
            # Retrieve most recent invocation step (limit 1) for this writes_id
            invocation_steps = await self.session_client.list_invocation_steps(
                ListInvocationStepsRequest(
                    session_identifier=thread_id,
                    invocation_identifier=writes_id,
                    max_results=1,
                )
            )
            invocation_step_summaries = invocation_steps.invocation_step_summaries

            # Return empty list if no steps found
            if len(invocation_step_summaries) == 0:
                return []

            # Get complete details for the most recent step
            invocation_step = await self.session_client.get_invocation_step(
                GetInvocationStepRequest(
                    session_identifier=thread_id,
                    invocation_identifier=writes_id,
                    invocation_step_id=invocation_step_summaries[0].invocation_step_id,
                )
            )

            return process_writes_invocation_content_blocks(
                invocation_step.invocation_step.payload.content_blocks, self.serde
            )

        except ClientError as e:
            # Return empty list if resource not found, otherwise re-raise error
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return []
            raise e

    async def _save_invocation_step(
        self,
        thread_id: str,
        invocation_identifier: str,
        invocation_step_id: Optional[str],
        payload: InvocationStepPayload,
    ) -> None:
        """Asynchronously persist an invocation step and its payload to the Bedrock session store.

        This method stores a single invocation step along with its associated payload data
        in the Bedrock session. The step is timestamped with the current UTC time.

        Args:
            thread_id: Unique identifier for the session thread
            invocation_identifier: Identifier for the specific invocation
            invocation_step_id: Unique identifier for this step within the invocation
            payload: InvocationStepPayload object containing the content blocks to store

        Returns:
            None
        """
        await self.session_client.put_invocation_step(
            PutInvocationStepRequest(
                session_identifier=thread_id,
                invocation_identifier=invocation_identifier,
                invocation_step_id=invocation_step_id,
                invocation_step_time=datetime.datetime.now(datetime.timezone.utc),
                payload=payload,
            )
        )

    async def _find_most_recent_checkpoint_step(
        self, thread_id: str, invocation_id: str
    ) -> Optional[InvocationStep]:
        """Asynchronously retrieve the most recent checkpoint step from a session's invocation history.

        Iterates through all invocation steps in reverse chronological order until it finds
        a step with a checkpoint payload type. Uses pagination to handle large result sets.

        Args:
            thread_id: The unique identifier for the session thread
            invocation_id: The identifier for the specific invocation to search

        Returns:
            InvocationStep object if a checkpoint is found, None otherwise
        """
        next_token = None
        while True:
            # Get batch of invocation steps using pagination token if available
            invocation_steps = await self.session_client.list_invocation_steps(
                ListInvocationStepsRequest(
                    session_identifier=thread_id,
                    invocation_identifier=invocation_id,
                    next_token=next_token,
                )
            )

            # Return None if no steps found in this batch
            if len(invocation_steps.invocation_step_summaries) == 0:
                return None

            # Check each step in the batch for checkpoint type
            for invocation_step_summary in invocation_steps.invocation_step_summaries:
                invocation_step = await self.session_client.get_invocation_step(
                    GetInvocationStepRequest(
                        session_identifier=thread_id,
                        invocation_identifier=invocation_id,
                        invocation_step_id=invocation_step_summary.invocation_step_id,
                    )
                )

                # Parse the step payload and check if it's a checkpoint
                step_payload = json.loads(
                    invocation_step.invocation_step.payload.content_blocks[0].text
                )
                if step_payload["step_type"] == CHECKPOINT_PREFIX:
                    return invocation_step.invocation_step

            # Get token for next batch of results
            next_token = invocation_steps.next_token
            if next_token is None:
                return None

    async def _get_checkpoint_step(
        self, thread_id: str, invocation_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[InvocationStep]:
        """Asynchronously retrieve checkpoint step data.

        Args:
            thread_id: Session thread identifier
            invocation_id: Invocation identifier
            checkpoint_id: Optional checkpoint identifier

        Returns:
            InvocationStep if found, None otherwise
        """
        if checkpoint_id is None:
            step = await self._find_most_recent_checkpoint_step(
                thread_id, invocation_id
            )
            if step is None:
                return None
            return step

        response = await self.session_client.get_invocation_step(
            GetInvocationStepRequest(
                session_identifier=thread_id,
                invocation_identifier=invocation_id,
                invocation_step_id=checkpoint_id,
            )
        )
        return response.invocation_step

    async def _get_task_sends(
        self, thread_id: str, checkpoint_ns: str, parent_checkpoint_id: Optional[str]
    ) -> list:
        """Asynchronously get sorted task sends for parent checkpoint.

        Args:
            thread_id: Session thread identifier
            checkpoint_ns: Checkpoint namespace
            parent_checkpoint_id: Parent checkpoint identifier

        Returns:
            Sorted list of task sends
        """
        if not parent_checkpoint_id:
            return []

        pending_writes = await self._get_checkpoint_pending_writes(
            thread_id, checkpoint_ns, parent_checkpoint_id
        )
        return transform_pending_task_writes(pending_writes)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously retrieve a checkpoint tuple from the Bedrock session.

        This function retrieves checkpoint data from the session, processes it and returns
        a structured CheckpointTuple containing the checkpoint state and metadata.

        Args:
            config (RunnableConfig): Configuration containing thread_id and optional checkpoint_ns.

        Returns:
            Optional[CheckpointTuple]: Structured checkpoint data if found, None otherwise.
        """
        session_thread_id = config["configurable"]["thread_id"]
        checkpoint_namespace = config["configurable"].get("checkpoint_ns", "")
        checkpoint_identifier = get_checkpoint_id(config)

        invocation_id = generate_checkpoint_id(checkpoint_namespace)

        try:
            invocation_step = await self._get_checkpoint_step(
                session_thread_id, invocation_id, checkpoint_identifier
            )
            if invocation_step is None:
                return None

            session_checkpoint = SessionCheckpoint(
                **json.loads(invocation_step.payload.content_blocks[0].text)
            )

            pending_write_ops = await self._get_checkpoint_pending_writes(
                session_thread_id,
                checkpoint_namespace,
                invocation_step.invocation_step_id,
            )

            task_sends = await self._get_task_sends(
                session_thread_id,
                checkpoint_namespace,
                session_checkpoint.parent_checkpoint_id,
            )

            return construct_checkpoint_tuple(
                session_thread_id,
                checkpoint_namespace,
                session_checkpoint,
                pending_write_ops,
                task_sends,
                self.serde,
            )

        except ClientError as err:
            if err.response["Error"]["Code"] != "ResourceNotFoundException":
                raise err
            return None

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Asynchronously store a new checkpoint in the Bedrock session.

        This method persists checkpoint data and metadata to a Bedrock Agent Runtime session.
        It serializes the checkpoint data, creates a session invocation, and saves an invocation
        step containing the checkpoint information.

        Args:
            config (RunnableConfig): Configuration containing thread_id and checkpoint namespace
            checkpoint (Checkpoint): The checkpoint data to store, containing state and channel values
            metadata (CheckpointMetadata): Metadata associated with the checkpoint like timestamps
            new_versions (ChannelVersions): Version information for communication channels

        Returns:
            RunnableConfig: Updated configuration with thread_id, checkpoint_ns and checkpoint_id
        """
        session_checkpoint = create_session_checkpoint(
            checkpoint, config, metadata, self.serde, new_versions
        )

        # Create session invocation to store checkpoint
        checkpoint_invocation_identifier = generate_checkpoint_id(
            session_checkpoint.checkpoint_ns
        )
        await self._create_session_invocation(
            session_checkpoint.thread_id, checkpoint_invocation_identifier
        )
        await self._save_invocation_step(
            session_checkpoint.thread_id,
            checkpoint_invocation_identifier,
            session_checkpoint.checkpoint_id,
            InvocationStepPayload(
                content_blocks=[
                    BedrockSessionContentBlock(
                        text=session_checkpoint.model_dump_json()
                    ),
                ]
            ),
        )

        return RunnableConfig(
            configurable={
                "thread_id": session_checkpoint.thread_id,
                "checkpoint_ns": session_checkpoint.checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Asynchronously store write operations in the Bedrock session.

        This method handles storing write operations by:
        1. Creating a new invocation for the writes
        2. Retrieving existing pending writes
        3. Building new content blocks for writes that don't exist
        4. Preserving existing writes that aren't being updated
        5. Saving all content blocks in a new invocation step

        Args:
            config (RunnableConfig): Configuration containing thread_id, checkpoint_ns and checkpoint_id
            writes (Sequence[tuple[str, Any]]): Sequence of (channel, value) tuples to write
            task_id (str): Identifier for the task performing the writes
            task_path (str, optional): Path information for the task. Defaults to empty string.

        Returns:
            None
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        # Generate unique identifier for this write operation
        writes_invocation_identifier = generate_write_id(checkpoint_ns, checkpoint_id)

        # Create new session invocation
        await self._create_session_invocation(thread_id, writes_invocation_identifier)

        # Get existing pending writes for this checkpoint
        current_pending_writes = await self._get_checkpoint_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

        content_blocks, new_writes = process_write_operations(
            writes,
            task_id,
            current_pending_writes,
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_path,
            self.serde,
        )

        # Save content blocks if any exist
        if content_blocks and new_writes:
            await self._save_invocation_step(
                thread_id,
                writes_invocation_identifier,
                None,  # Let service generate the step id
                InvocationStepPayload(content_blocks=content_blocks),
            )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously list checkpoints matching the given criteria.

        Args:
            config: Optional configuration to filter by
            filter: Optional dictionary of filter criteria
            before: Optional configuration to get checkpoints before
            limit: Optional maximum number of checkpoints to return

        Returns:
            AsyncIterator of matching CheckpointTuple objects
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns")

        invocation_identifier = None

        # Get invocation ID only if checkpoint_ns is provided
        if checkpoint_ns is not None:
            invocation_identifier = generate_checkpoint_id(checkpoint_ns)

        # List all invocation steps with pagination
        matching_checkpoints = []
        next_token = None

        while True:
            try:
                response = await self.session_client.list_invocation_steps(
                    ListInvocationStepsRequest(
                        session_identifier=thread_id,
                        invocation_identifier=invocation_identifier,
                        next_token=next_token,
                    )
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    return
                else:
                    raise e

            # Check if there are more pages
            next_token = response.next_token

            # Process current page
            for step in response.invocation_step_summaries:
                if before and step.invocation_step_id >= get_checkpoint_id(before):
                    continue

                # Get full step details to access metadata
                step_detail = await self.session_client.get_invocation_step(
                    GetInvocationStepRequest(
                        session_identifier=thread_id,
                        invocation_identifier=step.invocation_id,
                        invocation_step_id=step.invocation_step_id,
                    )
                )

                payload = json.loads(
                    step_detail.invocation_step.payload.content_blocks[0].text
                )

                # Append checkpoints and ignore writes
                if payload["step_type"] != CHECKPOINT_PREFIX:
                    continue

                session_checkpoint = SessionCheckpoint(**payload)

                # Apply metadata filter
                if filter:
                    metadata = (
                        deserialize_data(self.serde, session_checkpoint.metadata)
                        if session_checkpoint.metadata
                        else {}
                    )
                    if not all(metadata.get(k) == v for k, v in filter.items()):
                        continue

                # Append checkpoints
                matching_checkpoints.append(session_checkpoint)

                if limit and len(matching_checkpoints) >= limit:
                    next_token = None
                    break

            if next_token is None:
                break

        # Yield checkpoint tuples
        for checkpoint in matching_checkpoints:
            pending_write_ops = await self._get_checkpoint_pending_writes(
                thread_id,
                checkpoint.checkpoint_ns,
                checkpoint.checkpoint_id,
            )

            task_sends = await self._get_task_sends(
                thread_id, checkpoint.checkpoint_ns, checkpoint.parent_checkpoint_id
            )

            yield construct_checkpoint_tuple(
                thread_id,
                checkpoint.checkpoint_ns,
                checkpoint,
                pending_write_ops,
                task_sends,
                self.serde,
            )
