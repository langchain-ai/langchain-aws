"""
Helper classes for AgentCore Memory Checkpoint Saver.
"""

from __future__ import annotations

import base64
import datetime
import json
import logging
import warnings
from collections import defaultdict
from typing import Any, cast

import boto3
from botocore.config import Config
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    RunnableConfig,
    SerializerProtocol,
)

from langgraph_checkpoint_aws.agentcore.constants import (
    EMPTY_CHANNEL_VALUE,
    EventDecodingError,
)
from langgraph_checkpoint_aws.agentcore.models import (
    ChannelDataEvent,
    CheckpointerConfig,
    CheckpointEvent,
    WriteItem,
    WritesEvent,
)

logger = logging.getLogger(__name__)

# Union type for all events
EventType = CheckpointEvent | ChannelDataEvent | WritesEvent


class EventSerializer:
    """Handles serialization and deserialization of events to store in AgentCore Memory."""  # noqa: E501

    def __init__(self, serde: SerializerProtocol):
        self.serde = serde

    def serialize_value(self, value: Any) -> dict[str, Any]:
        """Serialize a value using the serde protocol."""
        type_tag, binary_data = self.serde.dumps_typed(value)
        return {"type": type_tag, "data": base64.b64encode(binary_data).decode("utf-8")}

    def deserialize_value(self, serialized: dict[str, Any]) -> Any:
        """Deserialize a value using the serde protocol."""
        try:
            type_tag = serialized["type"]
            binary_data = base64.b64decode(serialized["data"])
            return self.serde.loads_typed((type_tag, binary_data))
        except Exception as e:
            raise EventDecodingError(f"Failed to deserialize value: {e}") from e

    def serialize_event(self, event: EventType) -> str:
        """Serialize an event to JSON string."""

        # Create a custom serializer for Pydantic models
        def custom_serializer(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Get the base dictionary
        event_dict = event.model_dump(exclude_none=True)

        # Handle special serialization for specific fields
        if isinstance(event, CheckpointEvent):
            event_dict["checkpoint_data"] = self.serialize_value(event.checkpoint_data)
            event_dict["metadata"] = self.serialize_value(event.metadata)

        elif isinstance(event, ChannelDataEvent):
            if event.value != EMPTY_CHANNEL_VALUE:
                event_dict["value"] = self.serialize_value(event.value)

        elif isinstance(event, WritesEvent):
            event_dict["writes"] = [
                {
                    **write.model_dump(exclude_none=True),
                    "value": self.serialize_value(write.value),
                }
                for write in event.writes
            ]

        return json.dumps(event_dict, default=custom_serializer)

    def deserialize_event(self, data: str) -> EventType:
        """Deserialize JSON string to event."""
        try:
            event_dict = json.loads(data)
            event_type = event_dict.get("event_type")

            if event_type == "checkpoint":
                # Deserialize checkpoint data and metadata
                event_dict["checkpoint_data"] = self.deserialize_value(
                    event_dict["checkpoint_data"]
                )
                event_dict["metadata"] = self.deserialize_value(event_dict["metadata"])
                return CheckpointEvent(**event_dict)

            elif event_type == "channel_data":
                # Deserialize channel value if not empty
                if "value" in event_dict and isinstance(event_dict["value"], dict):
                    event_dict["value"] = self.deserialize_value(event_dict["value"])
                return ChannelDataEvent(**event_dict)

            elif event_type == "writes":
                # Deserialize write values
                for write in event_dict["writes"]:
                    if isinstance(write["value"], dict):
                        write["value"] = self.deserialize_value(write["value"])
                return WritesEvent(**event_dict)

            else:
                raise EventDecodingError(f"Unknown event type: {event_type}")

        except json.JSONDecodeError as e:
            raise EventDecodingError(f"Failed to parse JSON: {e}") from e
        except Exception as e:
            raise EventDecodingError(f"Failed to deserialize event: {e}") from e


class AgentCoreEventClient:
    """Handles low-level event storage and retrieval from AgentCore Memory for checkpoints."""  # noqa: E501

    def __init__(
        self, memory_id: str, serializer: EventSerializer | None = None, **boto3_kwargs
    ):
        self.memory_id = memory_id
        # mypy: need to set actual serializer if None
        if serializer is None:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

            self.serializer = EventSerializer(JsonPlusSerializer())
        else:
            self.serializer = serializer

        config = Config(
            user_agent_extra="x-client-framework:langgraph_agentcore_memory"
        )
        self.client = boto3.client("bedrock-agentcore", config=config, **boto3_kwargs)

    def store_blob_event(
        self, event: EventType, session_id: str, actor_id: str
    ) -> None:
        """Store an event in AgentCore Memory."""
        serialized = self.serializer.serialize_event(event)

        self.client.create_event(
            memoryId=self.memory_id,
            actorId=actor_id,
            sessionId=session_id,
            eventTimestamp=datetime.datetime.now(datetime.timezone.utc),
            payload=[{"blob": serialized}],
        )

    def store_blob_events_batch(
        self, events: list[EventType], session_id: str, actor_id: str
    ) -> None:
        """Store multiple events in a single API call to AgentCore Memory."""
        # Serialize all events into payload blobs
        payload = []
        timestamp = datetime.datetime.now(datetime.timezone.utc)

        for event in events:
            serialized = self.serializer.serialize_event(event)
            payload.append({"blob": serialized})

        # Store all events in a single create_event call
        self.client.create_event(
            memoryId=self.memory_id,
            actorId=actor_id,
            sessionId=session_id,
            eventTimestamp=timestamp,
            payload=payload,
        )

    def get_events(
        self,
        session_id: str,
        actor_id: str,
        limit: int | None = None,
        max_results: int | None = 100,
    ) -> list[EventType]:
        """Retrieve events from AgentCore Memory.

        Args:
            session_id: The session ID to retrieve events for
            actor_id: The actor ID to retrieve events for
            limit: The maximum number of events to parse from ListEvents
            max_results: Maximum number of results to retrieve. Defaults to 100.

        Returns:
            List of retrieved events
        """

        if max_results is not None and max_results <= 0:
            return []

        all_events = []
        next_token = None
        limit_reached = False

        while True:
            params = {
                "memoryId": self.memory_id,
                "actorId": actor_id,
                "sessionId": session_id,
                "maxResults": max_results,
                "includePayloads": True,
            }

            if next_token:
                params["nextToken"] = next_token

            response = self.client.list_events(**params)

            for event in response.get("events", []):
                for payload_item in event.get("payload", []):
                    blob = payload_item.get("blob")
                    if blob:
                        try:
                            parsed_event = self.serializer.deserialize_event(blob)
                            all_events.append(parsed_event)
                        except EventDecodingError as e:
                            logger.warning(f"Failed to decode event: {e}")

                        if limit is not None and len(all_events) >= limit:
                            limit_reached = True
                            break

                if limit_reached:
                    break

            next_token = response.get("nextToken")

            if limit_reached and next_token:
                warnings.warn(
                    f"Stopped retrieving events at limit of {limit}. "
                    f"There may be additional checkpoints that were not retrieved. "
                    f"Consider increasing the limit parameter, or set None for no "
                    f"limit.",
                    UserWarning,
                    stacklevel=2,
                )

            if limit_reached or not next_token:
                break

        return all_events

    def delete_events(self, session_id: str, actor_id: str) -> None:
        """Delete all events for a session."""
        params = {
            "memoryId": self.memory_id,
            "actorId": actor_id,
            "sessionId": session_id,
            "maxResults": 100,
            "includePayloads": False,
        }

        while True:
            response = self.client.list_events(**params)
            events = response.get("events", [])

            if not events:
                break

            for event in events:
                self.client.delete_event(
                    memoryId=self.memory_id,
                    sessionId=session_id,
                    eventId=event["eventId"],
                    actorId=actor_id,
                )

            next_token = response.get("nextToken")
            if not next_token:
                break
            params["nextToken"] = next_token


class EventProcessor:
    """Processes events into checkpoint data structures."""

    @staticmethod
    def process_events(
        events: list[EventType],
    ) -> tuple[
        dict[str, CheckpointEvent],
        dict[str, list[WriteItem]],
        dict[tuple[str, str], Any],
    ]:
        """Process events into organized data structures."""
        checkpoints = {}
        writes_by_checkpoint = defaultdict(list)
        channel_data_by_version = {}

        for event in events:
            if isinstance(event, CheckpointEvent):
                checkpoints[event.checkpoint_id] = event

            elif isinstance(event, WritesEvent):
                writes_by_checkpoint[event.checkpoint_id].extend(event.writes)

            elif isinstance(event, ChannelDataEvent):
                if event.value != EMPTY_CHANNEL_VALUE:
                    channel_data_by_version[(event.channel, event.version)] = (
                        event.value
                    )

        return checkpoints, writes_by_checkpoint, channel_data_by_version

    @staticmethod
    def build_checkpoint_tuple(
        checkpoint_event: CheckpointEvent,
        writes: list[WriteItem],
        channel_data: dict[tuple[str, str], Any],
        config: CheckpointerConfig,
    ) -> CheckpointTuple:
        """Build a CheckpointTuple from processed data."""
        # Build pending writes
        pending_writes = [
            (write.task_id, write.channel, write.value) for write in writes
        ]

        # Build parent config
        parent_config = None
        if checkpoint_event.parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": config.thread_id,
                    "actor_id": config.actor_id,
                    "checkpoint_ns": config.checkpoint_ns,
                    "checkpoint_id": checkpoint_event.parent_checkpoint_id,
                }
            }

        # Build checkpoint with channel values
        checkpoint = checkpoint_event.checkpoint_data.copy()
        channel_values = {}

        for channel, version in checkpoint.get("channel_versions", {}).items():
            if (channel, version) in channel_data:
                channel_values[channel] = channel_data[(channel, version)]

        # Clean orphan tool_calls from messages channel if present
        # This ensures messages loaded from checkpoints are valid for LLM providers
        if "messages" in channel_values:
            channel_values["messages"] = clean_orphan_tool_calls(
                channel_values["messages"]
            )

        checkpoint["channel_values"] = channel_values

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": config.thread_id,
                    "actor_id": config.actor_id,
                    "checkpoint_ns": config.checkpoint_ns,
                    "checkpoint_id": checkpoint_event.checkpoint_id,
                }
            },
            checkpoint=cast(Checkpoint, checkpoint),
            metadata=cast(CheckpointMetadata, checkpoint_event.metadata),
            parent_config=cast(RunnableConfig, parent_config)
            if parent_config
            else None,
            pending_writes=pending_writes,
        )


def clean_orphan_tool_calls(messages: list[Any]) -> list[Any]:
    """Remove tool_calls from AIMessages that don't have corresponding ToolMessages.

    This ensures messages loaded from checkpoints are valid for LLM providers
    like Bedrock that require tool_use blocks to be immediately followed by
    tool_result blocks.

    When a checkpoint is saved during tool execution (between AIMessage with
    tool_calls and the corresponding ToolMessage), the state becomes "incomplete"
    from the LLM provider's perspective. This function cleans up such orphaned
    tool_calls to make the message history valid.

    Args:
        messages: List of messages from checkpoint channel_values

    Returns:
        List of messages with orphaned tool_calls removed from AIMessages
    """
    if not messages:
        return messages

    # Build a set of all tool_call_ids that have corresponding ToolMessages
    resolved_tool_call_ids = {
        msg.tool_call_id
        for msg in messages
        if isinstance(msg, ToolMessage) and hasattr(msg, "tool_call_id")
    }

    # Clean up AIMessages with orphaned tool_calls
    cleaned_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Filter out tool_calls that don't have corresponding ToolMessages
            valid_tool_calls, removed_tool_calls = [], []

            for tc in msg.tool_calls:
                tc_id = tc.get("id")
                if tc_id in resolved_tool_call_ids:
                    valid_tool_calls.append(tc)
                else:
                    removed_tool_calls.append(tc_id)

            # If we removed some tool_calls, create a new message with cleaned
            # tool_calls
            if removed_tool_calls:
                logger.warning(
                    f"Removed {len(removed_tool_calls)} orphaned tool_call(s) "
                    f"from AIMessage during checkpoint load: {removed_tool_calls}"
                )
                cleaned_msg = msg.model_copy(update={"tool_calls": valid_tool_calls})
                cleaned_messages.append(cleaned_msg)
            else:
                cleaned_messages.append(msg)
        else:
            cleaned_messages.append(msg)

    return cleaned_messages


def convert_langchain_messages_to_event_messages(
    messages: list[BaseMessage],
) -> list[tuple[str, str]]:
    """Convert LangChain messages to Bedrock Agent Core events

    Args:
        messages: List of Langchain messages (BaseMessage)

    Returns:
        List of AgentCore event tuples (text, role)
    """
    converted_messages = []
    for msg in messages:
        # Skip if event already saved
        if msg.additional_kwargs.get("event_id") is not None:
            continue

        text = msg.text()
        if not text.strip():
            continue

        # Map LangChain roles to Bedrock Agent Core roles
        if msg.type == "human":
            role = "USER"
        elif msg.type == "ai":
            role = "ASSISTANT"
        elif msg.type == "tool":
            role = "TOOL"
        elif msg.type == "system":
            role = "OTHER"
        else:
            logger.warning(f"Skipping unsupported message type: {msg.type}")
            continue

        converted_messages.append((text, role))

    return converted_messages
