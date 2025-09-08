import datetime
from collections.abc import Iterator, Sequence
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

from langgraph_checkpoint_aws.constants import CHECKPOINT_PREFIX
from langgraph_checkpoint_aws.utils import (
    construct_checkpoint_tuple,
    deserialize_from_base64,
    generate_checkpoint_id,
    generate_write_id,
    process_aws_client_args,
    serialize_to_base64,
    transform_pending_task_writes,
)


class BedrockAgentCoreEventsSaver(BaseCheckpointSaver):
    """Saves and retrieves checkpoints using Amazon Bedrock AgentCore Events API.

    This class provides functionality to persist checkpoint data and writes using the
    Bedrock AgentCore Events API for memory management.

    Args:
        memory_id: The memory store identifier
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
        memory_id: str,
        region_name: Optional[str] = None,
        credentials_profile_name: Optional[str] = None,
        aws_access_key_id: Optional[SecretStr] = None,
        aws_secret_access_key: Optional[SecretStr] = None,
        aws_session_token: Optional[SecretStr] = None,
        endpoint_url: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__()
        self.memory_id = memory_id

        _session_kwargs, _client_kwargs = process_aws_client_args(
            region_name,
            credentials_profile_name,
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
            endpoint_url,
            config,
        )
        session = boto3.Session(**_session_kwargs)
        self.client = session.client("bedrock-agentcore", **_client_kwargs)

    def _create_event(
        self,
        actor_id: str,
        session_id: str,
        payload_data: dict[str, Any],
        branch: Optional[str] = None,
        use_blob: bool = False,
    ) -> dict[str, Any]:
        """Create an event in the memory store.

        Args:
            actor_id: Identifier for the event's actor
            session_id: Identifier for the event's session
            payload_data: Event content (single payload object)
            branch: Optional branch to organize events
            use_blob: Whether to store payload as binary blob instead of conversational

        Returns:
            Created event response
        """
        request_params = {
            "memoryId": self.memory_id,
            "actorId": actor_id,
            "sessionId": session_id,
            "eventTimestamp": datetime.datetime.now(datetime.timezone.utc),
        }

        if use_blob:
            # Convert payload to binary blob format
            import base64
            import json

            payload_json = json.dumps(payload_data)
            payload_blob = base64.b64encode(payload_json.encode("utf-8")).decode(
                "utf-8"
            )
            request_params["payload"] = [{"blob": payload_blob}]
        else:
            # Use conversational payload type
            request_params["payload"] = [{"conversational": payload_data}]

        if branch:
            request_params["branch"] = branch

        return self.client.create_event(**request_params)

    def _decode_event_payload(self, event: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Decode event payload, handling both blob and conversational formats.

        Args:
            event: Event object from the API

        Returns:
            Decoded payload as dictionary, or None if no payload
        """
        if not event.get("payload") or not isinstance(event["payload"], list):
            return None

        # Payload is an array, get the first item
        payload_array = event["payload"]
        if not payload_array:
            return None
            
        payload_item = payload_array[0]

        if "blob" in payload_item:
            # Decode blob payload
            import base64
            import json

            blob_data = base64.b64decode(payload_item["blob"].encode("utf-8"))
            payload_json = blob_data.decode("utf-8")
            return json.loads(payload_json)
        elif "conversational" in payload_item:
            # Return conversational payload directly
            return payload_item["conversational"]
        else:
            # Fallback for unknown payload types
            return payload_item

    def _get_event(
        self, actor_id: str, session_id: str, event_id: str
    ) -> Optional[dict[str, Any]]:
        """Get an event from the memory store.

        Args:
            actor_id: Identifier for the event's actor
            session_id: Identifier for the event's session
            event_id: Specific event identifier

        Returns:
            Event data if found, None otherwise
        """
        try:
            response = self.client.get_event(
                memoryId=self.memory_id,
                actorId=actor_id,
                sessionId=session_id,
                eventId=event_id,
            )
            return response
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return None
            raise e

    def _list_events(
        self,
        actor_id: str,
        session_id: str,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
        include_payloads: bool = True,
    ) -> dict[str, Any]:
        """List events from the memory store.

        Args:
            actor_id: Identifier for the event's actor
            session_id: Identifier for the event's session
            max_results: Maximum number of results to return
            next_token: Pagination token
            include_payloads: Whether to include event payloads

        Returns:
            List events response
        """
        request_params = {
            "memoryId": self.memory_id,
            "actorId": actor_id,
            "sessionId": session_id,
            "includePayloads": include_payloads,
        }

        if max_results:
            request_params["maxResults"] = max_results
        if next_token:
            request_params["nextToken"] = next_token

        return self.client.list_events(**request_params)

    def _get_checkpoint_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[dict[str, Any]]:
        """Retrieve pending write operations for a given checkpoint from Events API.

        Args:
            thread_id: Session thread identifier
            checkpoint_ns: Namespace that groups related checkpoints
            checkpoint_id: Unique identifier for the specific checkpoint

        Returns:
            List of pending write dictionaries
        """
        actor_id = generate_write_id(checkpoint_ns, checkpoint_id)

        try:
            events_response = self._list_events(
                actor_id=actor_id,
                session_id=thread_id,
                max_results=100,  # Get all writes for this checkpoint
                include_payloads=True,
            )

            events = events_response.get("events", [])
            pending_writes = []

            # Events are returned in reverse chronological order, process them
            for event in events:
                payload_data = self._decode_event_payload(event)
                if payload_data and "writes" in payload_data:
                    # Handle new format where writes are stored as a collection
                    for write_item in payload_data["writes"]:
                        if write_item.get("step_type") == "write":
                            pending_writes.append(
                                {
                                    "step_type": write_item["step_type"],
                                    "thread_id": write_item["thread_id"],
                                    "checkpoint_ns": write_item["checkpoint_ns"],
                                    "checkpoint_id": write_item["checkpoint_id"],
                                    "task_id": write_item["task_id"],
                                    "channel": write_item["channel"],
                                    "value": deserialize_from_base64(
                                        self.serde, *write_item["value"]
                                    ),
                                    "task_path": write_item["task_path"],
                                    "write_idx": write_item["write_idx"],
                                }
                            )
                elif payload_data and payload_data.get("step_type") == "write":
                    # Handle legacy format for backward compatibility
                    pending_writes.append(
                        {
                            "step_type": payload_data["step_type"],
                            "thread_id": payload_data["thread_id"],
                            "checkpoint_ns": payload_data["checkpoint_ns"],
                            "checkpoint_id": payload_data["checkpoint_id"],
                            "task_id": payload_data["task_id"],
                            "channel": payload_data["channel"],
                            "value": deserialize_from_base64(
                                self.serde, *payload_data["value"]
                            ),
                            "task_path": payload_data["task_path"],
                            "write_idx": payload_data["write_idx"],
                        }
                    )

            return pending_writes

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return []
            raise e

    def _get_task_sends(
        self, thread_id: str, checkpoint_ns: str, parent_checkpoint_id: Optional[str]
    ) -> list:
        """Get sorted task sends for parent checkpoint.

        Args:
            thread_id: Session thread identifier
            checkpoint_ns: Checkpoint namespace
            parent_checkpoint_id: Parent checkpoint identifier

        Returns:
            Sorted list of task sends
        """
        if not parent_checkpoint_id:
            return []

        pending_writes = self._get_checkpoint_pending_writes(
            thread_id, checkpoint_ns, parent_checkpoint_id
        )

        # Convert to SessionPendingWrite objects for processing
        session_pending_writes = []
        for write in pending_writes:
            session_pending_writes.append(type("SessionPendingWrite", (), write)())

        return transform_pending_task_writes(session_pending_writes)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Retrieve a checkpoint tuple from the Bedrock AgentCore Events.

        Args:
            config: Configuration containing thread_id and optional checkpoint_ns

        Returns:
            Structured checkpoint data if found, None otherwise
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        actor_id = generate_checkpoint_id(checkpoint_ns)

        try:
            if checkpoint_id:
                # Get specific checkpoint event
                event_response = self._get_event(actor_id, thread_id, checkpoint_id)
                if not event_response:
                    return None
                event = event_response["event"]
            else:
                # Get most recent checkpoint event (events are in reverse chronological order)
                events_response = self._list_events(
                    actor_id, thread_id, max_results=1, include_payloads=True
                )
                events = events_response.get("events", [])
                if not events:
                    return None
                # First event is the most recent due to reverse order
                event = events[0]

            # Parse checkpoint data from event payload
            checkpoint_data = self._decode_event_payload(event)
            if (
                not checkpoint_data
                or checkpoint_data.get("step_type") != CHECKPOINT_PREFIX
            ):
                return None

            # Create session checkpoint object
            session_checkpoint = type(
                "SessionCheckpoint",
                (),
                {
                    "step_type": checkpoint_data["step_type"],
                    "thread_id": checkpoint_data["thread_id"],
                    "checkpoint_ns": checkpoint_data["checkpoint_ns"],
                    "checkpoint_id": event["eventId"],
                    "checkpoint": checkpoint_data["checkpoint"],
                    "metadata": checkpoint_data["metadata"],
                    "parent_checkpoint_id": checkpoint_data.get(
                        "parent_checkpoint_id"
                    ),
                    "channel_values": checkpoint_data["channel_values"],
                    "version": checkpoint_data.get("version"),
                },
            )()

            # Get pending writes and task sends
            pending_writes = self._get_checkpoint_pending_writes(
                thread_id, checkpoint_ns, event["eventId"]
            )

            task_sends = self._get_task_sends(
                thread_id, checkpoint_ns, checkpoint_data.get("parent_checkpoint_id")
            )

            # Convert pending writes to the expected format
            pending_writes_objects = []
            for write in pending_writes:
                pending_writes_objects.append(type("SessionPendingWrite", (), write)())

            return construct_checkpoint_tuple(
                thread_id,
                checkpoint_ns,
                session_checkpoint,
                pending_writes_objects,
                task_sends,
                self.serde,
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return None
            raise e

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a new checkpoint in the Bedrock AgentCore Events.

        Args:
            config: Configuration containing thread_id and checkpoint namespace
            checkpoint: The checkpoint data to store
            metadata: Metadata associated with the checkpoint
            new_versions: Version information for communication channels

        Returns:
            Updated configuration with checkpoint details
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        actor_id = generate_checkpoint_id(checkpoint_ns)

        # Create event payload with checkpoint data using proper serialization
        payload_data = {
            "step_type": CHECKPOINT_PREFIX,
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint": serialize_to_base64(
                self.serde,
                {k: v for k, v in checkpoint.items() if k != "pending_sends"},
            ),
            "metadata": self.serde.dumps(metadata).decode() if metadata else None,
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "channel_values": serialize_to_base64(
                self.serde, checkpoint.get("channel_values", {})
            ),
            "version": serialize_to_base64(self.serde, new_versions),
        }

        response = self._create_event(
            actor_id=actor_id,
            session_id=thread_id,
            payload_data=payload_data,
            branch=checkpoint_ns,
            use_blob=True,  # Use blob for more efficient storage of binary data
        )

        return RunnableConfig(
            configurable={
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": response["event"]["eventId"],
            }
        )

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store write operations in the Bedrock AgentCore Events.

        Args:
            config: Configuration containing thread_id, checkpoint_ns and checkpoint_id
            writes: Sequence of (channel, value) tuples to write
            task_id: Identifier for the task performing the writes
            task_path: Path information for the task
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        actor_id = generate_write_id(checkpoint_ns, checkpoint_id)

        # Create payload for writes using proper serialization
        # Since we can only have one payload per event, we'll store writes as a collection
        writes_data = []
        for idx, (channel, value) in enumerate(writes):
            writes_data.append(
                {
                    "step_type": "write",
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "channel": channel,
                    "value": serialize_to_base64(self.serde, value),
                    "task_path": task_path,
                    "write_idx": idx,
                }
            )

        if writes_data:
            payload_data = {
                "writes": writes_data,
                "write_count": len(writes_data),
            }

            self._create_event(
                actor_id=actor_id,
                session_id=thread_id,
                payload_data=payload_data,
                branch=checkpoint_ns,
                use_blob=True,  # Use blob for storage
            )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching the given criteria.

        Args:
            config: Optional configuration to filter by
            filter: Optional dictionary of filter criteria
            before: Optional configuration to get checkpoints before
            limit: Optional maximum number of checkpoints to return

        Returns:
            Iterator of matching CheckpointTuple objects
        """
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        actor_id = generate_checkpoint_id(checkpoint_ns)

        try:
            next_token = None
            yielded_count = 0

            while True:
                events_response = self._list_events(
                    actor_id=actor_id,
                    session_id=thread_id,
                    max_results=min(limit or 100, 100),
                    next_token=next_token,
                    include_payloads=True,
                )

                events = events_response.get("events", [])
                if not events:
                    break

                # Events are returned in reverse chronological order
                for event in events:
                    if limit and yielded_count >= limit:
                        return

                    if before and event["eventId"] >= get_checkpoint_id(before):
                        continue

                    # Parse checkpoint from event payload
                    checkpoint_data = self._decode_event_payload(event)
                    if (
                        not checkpoint_data
                        or checkpoint_data.get("step_type") != CHECKPOINT_PREFIX
                    ):
                        continue

                    # Apply metadata filter if provided
                    if filter:
                        metadata = (
                            self.serde.loads(checkpoint_data["metadata"].encode())
                            if checkpoint_data.get("metadata")
                            else {}
                        )
                        if not all(metadata.get(k) == v for k, v in filter.items()):
                            continue

                    # Create session checkpoint object
                    session_checkpoint = type(
                        "SessionCheckpoint",
                        (),
                        {
                            "step_type": checkpoint_data["step_type"],
                            "thread_id": checkpoint_data["thread_id"],
                            "checkpoint_ns": checkpoint_data["checkpoint_ns"],
                            "checkpoint_id": event["eventId"],
                            "checkpoint": checkpoint_data["checkpoint"],
                            "metadata": checkpoint_data["metadata"],
                            "parent_checkpoint_id": checkpoint_data.get(
                                "parent_checkpoint_id"
                            ),
                            "channel_values": checkpoint_data["channel_values"],
                            "version": checkpoint_data.get("version"),
                        },
                    )()

                    # Get pending writes and task sends for this checkpoint
                    pending_writes = self._get_checkpoint_pending_writes(
                        thread_id, checkpoint_ns, event["eventId"]
                    )

                    task_sends = self._get_task_sends(
                        thread_id,
                        checkpoint_ns,
                        checkpoint_data.get("parent_checkpoint_id"),
                    )

                    # Convert pending writes to the expected format
                    pending_writes_objects = []
                    for write in pending_writes:
                        pending_writes_objects.append(
                            type("SessionPendingWrite", (), write)()
                        )

                    yield construct_checkpoint_tuple(
                        thread_id,
                        checkpoint_ns,
                        session_checkpoint,
                        pending_writes_objects,
                        task_sends,
                        self.serde,
                    )

                    yielded_count += 1

                next_token = events_response.get("nextToken")
                if not next_token:
                    break

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return
            raise e
