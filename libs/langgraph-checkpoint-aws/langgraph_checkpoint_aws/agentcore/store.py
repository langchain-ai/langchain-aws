"""
AgentCore Memory Store implementation following BaseStore pattern.

This implementation uses the ops pattern and batch function handlers,
directly calling the AgentCore Memory API for create_event and retrieve_memory_records.
"""

import logging
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_core.messages import BaseMessage
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from langgraph_checkpoint_aws.agentcore.helpers import (
    convert_langchain_messages_to_event_messages,
)

logger = logging.getLogger(__name__)


class AgentCoreMemoryStore(BaseStore):
    """
    AgentCore Memory Store implementation using BaseStore pattern.

    This store saves chat messages as conversational events in AgentCore Memory
    and retrieves processed memories through semantic search. The embedding and
    memory processing happens automatically in the AgentCore Memory service.

    Args:
        memory_id: The AgentCore Memory resource ID
        **boto3_kwargs: Additional arguments passed to boto3.client()

    Example:
        ```python
        store = AgentCoreMemoryStore(
            memory_id="memory_abc123",
            region_name="us-west-2"
        )

        # Store a message
        from langchain_core.messages import HumanMessage
        store.put(("user123", "session456"), "msg1", {
            "message": HumanMessage("I love coffee")
        })

        # Search for processed memories
        results = store.search(("facts", "user123"), query="user preferences")
        ```
    """

    supports_ttl: bool = False
    ttl_config = None

    def __init__(self, *, memory_id: str, **boto3_kwargs: Any):
        self.memory_id = memory_id

        config = Config(
            user_agent_extra="x-client-framework:langgraph_agentcore_memory_store",
            retries={"max_attempts": 4, "mode": "adaptive"},
        )
        self.client = boto3.client("bedrock-agentcore", config=config, **boto3_kwargs)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations in a single batch."""
        results = []

        for op in ops:
            if isinstance(op, PutOp):
                self._handle_put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                result = self._handle_search(op)
                results.append(result)
            elif isinstance(op, GetOp):
                result = self._handle_get(op)
                results.append(result)
            elif isinstance(op, ListNamespacesOp):
                # ListNamespacesOp not supported for AgentCore Memory
                results.append([])
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute multiple operations asynchronously (not implemented)."""
        raise NotImplementedError(
            "AgentCore Memory client does not support async operations yet"
        )

    def _handle_put(self, op: PutOp) -> None:
        """Handle PutOp by creating conversational events in AgentCore Memory."""
        if op.value is None:
            # TODO: Delete operation support - need to figure out if we are deleting events or records
            logger.warning("Delete operations not supported in AgentCore Memory")
            return

        message = op.value.get("message")
        if not isinstance(message, BaseMessage):
            raise ValueError(
                "Value must contain a 'message' key with a BaseMessage object"
            )

        # Convert namespace tuple to actor_id and session_id
        if len(op.namespace) != 2:
            raise ValueError("Namespace must be a tuple of (actor_id, session_id)")

        actor_id, session_id = op.namespace
        event_messages = convert_langchain_messages_to_event_messages([message])

        if not event_messages:
            logger.warning(
                f"No valid event messages to create for message type: {message.type}"
            )
            return

        conversational_payloads = []
        for text, role in event_messages:
            conversational_payloads.append(
                {"conversational": {"content": {"text": text}, "role": role}}
            )

        self.client.create_event(
            memoryId=self.memory_id,
            actorId=actor_id,
            sessionId=session_id,
            eventTimestamp=datetime.now(timezone.utc),
            payload=conversational_payloads,
        )
        logger.debug(f"Created event for message in namespace {op.namespace}")

    def _handle_get(self, op: GetOp) -> Optional[Item]:
        """Handle GetOp by retrieving a specific memory record from AgentCore Memory."""
        try:
            response = self.client.get_memory_record(
                memoryId=self.memory_id, memoryRecordId=op.key
            )

            memory_record = response.get("memoryRecord")
            if not memory_record:
                return None

            return self._convert_memory_record_to_item(memory_record, op.namespace)

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                # Memory record not found
                return None
            else:
                # Re-raise other client errors
                logger.error(f"Failed to get memory record: {e}")
                raise
        except Exception as e:
            logger.error(f"Failed to get memory record: {e}")
            raise

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """Handle SearchOp by retrieving memory records from AgentCore Memory."""
        if not op.query:
            logger.warning("Search requires a query for AgentCore Memory")
            return []

        namespace_str = self._convert_namespace_to_string(op.namespace_prefix)

        search_criteria = {"searchQuery": op.query, "topK": op.limit}

        response = self.client.retrieve_memory_records(
            memoryId=self.memory_id,
            namespace=namespace_str,
            searchCriteria=search_criteria,
            maxResults=op.limit,
        )

        memory_records = response.get("memoryRecordSummaries", [])
        return self._convert_memory_records_to_search_items(
            memory_records, op.namespace_prefix
        )

    def _convert_namespace_to_string(self, namespace_tuple: tuple[str, ...]) -> str:
        """Convert namespace tuple to AgentCore namespace string."""
        if not isinstance(namespace_tuple, tuple):
            raise TypeError("namespace_tuple must be a tuple")
        if not namespace_tuple:
            return "/"
        return "/" + "/".join(namespace_tuple)

    def _convert_memory_record_to_item(
        self, memory_record: dict, namespace: tuple[str, ...]
    ) -> Item:
        """Convert a single AgentCore memory record to an Item object."""
        # Extract content
        content = memory_record.get("content", {})
        text = content.get("text", "") if isinstance(content, dict) else str(content)

        # Extract metadata
        memory_record_id = memory_record.get("memoryRecordId", str(uuid.uuid4()))
        created_at = memory_record.get("createdAt")

        # Parse timestamp - API only provides createdAt, use it for both created_at and updated_at
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                created_at = datetime.now(timezone.utc)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return Item(
            namespace=namespace,
            key=memory_record_id,
            value={
                "content": text,
                "memory_strategy_id": memory_record.get("memoryStrategyId"),
                "namespaces": memory_record.get("namespaces", []),
            },
            created_at=created_at,
            updated_at=created_at,  # Memory records are not updated
        )

    def _convert_memory_records_to_search_items(
        self, memory_records: list, namespace: tuple[str, ...]
    ) -> list[SearchItem]:
        """Convert AgentCore memory records to SearchItem objects."""
        results = []

        for record in memory_records:
            content = record.get("content", {})
            text = (
                content.get("text", "") if isinstance(content, dict) else str(content)
            )

            memory_record_id = record.get("memoryRecordId", str(uuid.uuid4()))
            score = record.get("score")
            created_at = record.get("createdAt")

            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(
                        created_at.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    created_at = datetime.now(timezone.utc)
            elif created_at is None:
                created_at = datetime.now(timezone.utc)

            search_item = SearchItem(
                namespace=namespace,
                key=memory_record_id,
                value={
                    "content": text,
                    "memory_strategy_id": record.get("memoryStrategyId"),
                    "namespaces": record.get("namespaces", []),
                },
                created_at=created_at,
                updated_at=created_at,  # Memory records are not updated
                score=float(score) if score is not None else None,
            )
            results.append(search_item)

        return results
