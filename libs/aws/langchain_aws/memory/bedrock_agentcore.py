"""Module for AWS Bedrock Agent Core memory integration.

This module provides tools to allow agents to use the AWS Bedrock Agent Core
memory API to manage and search memories.
"""

import logging
from typing import List, Any, Dict

from bedrock_agentcore.memory import MemoryClient
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# TODO: Once Bedrock AgentCore introduces metadata to store the tool call ID,
# implement logic to properly save and load for ToolCall messages
TOOL_CALL_ID_PLACEHOLDER = "unknown"


def convert_langchain_messages_to_events(
    messages: List[BaseMessage], include_system_messages=False
) -> List[Dict[str, Any]]:
    """Convert LangChain messages to Bedrock Agent Core events

    Args:
        messages: List of Langchain messages (BaseMessage)
        include_system_messages: Flag for whether to include system messages in the conversion (as OTHER) or skip them

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
        elif msg.type == "system" and include_system_messages:
            role = "OTHER"
        else:
            logger.warning(f"Skipping unsupported message type: {msg.type}")
            continue

        converted_messages.append((text, role))

    return converted_messages


def convert_events_to_langchain_messages(
    events: List[Dict[str, Any]]
) -> List[BaseMessage]:
    """Convert Bedrock Agent Core events back to LangChain messages.

    Args:
        events: List of event dictionaries with 'payload' containing conversational data

    Returns:
        List of LangChain BaseMessage objects
    """
    messages = []

    for event in events:
        if "payload" not in event:
            continue

        for payload_item in event.get("payload", []):
            if "conversational" not in payload_item:
                continue

            conv = payload_item["conversational"]
            role = conv.get("role", "")
            content = conv.get("content", {}).get("text", "")

            if not content.strip():
                continue

            message = None
            if role == "USER":
                message = HumanMessage(content=content)
            elif role == "ASSISTANT":
                message = AIMessage(content=content)
            elif role == "TOOL":
                # As of now, the tool_call_id is not stored or returned by the Memory API
                message = ToolMessage(
                    content=content, tool_call_id=TOOL_CALL_ID_PLACEHOLDER
                )
            elif role == "OTHER":
                message = SystemMessage(content=content)
            else:
                logger.warning(f"Skipping unknown message role: {role}")
                continue

            # Preserve event metadata
            if message and "eventId" in event:
                message.additional_kwargs["event_id"] = event["eventId"]

            if message:
                messages.append(message)

    return messages


def store_agentcore_memory_events(
    memory_client: MemoryClient,
    messages: List[BaseMessage],
    memory_id: str,
    actor_id: str,
    session_id: str,
    include_system_messages: bool = False,
) -> str:
    """Stores Langchain Messages as Bedrock AgentCore Memory events in short term memory

    Args:
        memory_client: Initialized MemoryClient instance
        memory_id: Memory identifier (e.g., "test-memory-id")
        actor_id: Actor identifier (e.g., "user")
        session_id: Session identifier (e.g., "session-1")
        include_system_messages: Flag for whether to save system messages (as OTHER) or skip them

    Returns:
        The ID of the event that was created
    """

    if len(messages) == 0:
        raise ValueError("The messages field cannot be empty.")

    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id cannot be empty")
    if not actor_id or not actor_id.strip():
        raise ValueError("actor_id cannot be empty")
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")

    events_to_store = convert_langchain_messages_to_events(
        messages, include_system_messages
    )
    if not events_to_store:
        raise ValueError(
            "No valid messages to store. All messages were either empty, "
            "already stored, or filtered out."
        )

    response = memory_client.create_event(
        memory_id=memory_id,
        actor_id=actor_id,
        session_id=session_id,
        messages=events_to_store,
    )
    event_id = response.get("eventId")
    if not event_id:
        raise RuntimeError("AgentCore did not return an event ID")

    return event_id


def list_agentcore_memory_events(
    memory_client: MemoryClient,
    memory_id: str,
    actor_id: str,
    session_id: str,
    max_results: int = 100,
) -> List[BaseMessage]:
    """Lists the events in short term memory from Bedrock Agentcore Memory as Langchain Messages

    Args:
        memory_client: Initialized MemoryClient instance
        memory_id: Memory identifier (e.g., "test-memory-id")
        actor_id: Actor identifier (e.g., "user")
        session_id: Session identifier (e.g., "session-1")
        max_results: The maximum number of results to return

    Returns:
        A list of LangChain messages of previous events saved in short term memory
    """
    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id cannot be empty")
    if not actor_id or not actor_id.strip():
        raise ValueError("actor_id cannot be empty")
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")

    events = memory_client.list_events(
        memory_id=memory_id,
        actor_id=actor_id,
        session_id=session_id,
        max_results=max_results,
        include_payload=True,
    )

    return convert_events_to_langchain_messages(events)


def retrieve_agentcore_memories(
    memory_client: MemoryClient,
    memory_id: str,
    namespace_str: str,
    query: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Search for memories in AWS Bedrock Agentcore Memory

    Args:
        memory_client: The AgentCore memory client
        memory_id: The memory identifier in AgentCore
        namespace_str: The namespace to be searched
        query: The query to be embedded and used in the semantic search for memories
        limit: The limit for results to be retrieved

    Returns:
        A list of memory results with content, score, and metadata
    """
    if not memory_id or not memory_id.strip():
        raise ValueError("memory_id cannot be empty")
    if not namespace_str or not namespace_str.strip():
        raise ValueError("actor_id cannot be empty")
    if not query or not query.strip():
        raise ValueError("actor_id cannot be empty")

    memories = memory_client.retrieve_memories(
        memory_id=memory_id,
        namespace=namespace_str,
        query=query,
        top_k=limit,
    )

    results = []
    for item in memories:
        content = item.get("content", {}).get("text", "")
        result = {
            "content": content,
            "score": item.get("score", 0.0),
            "metadata": item.get("metadata", {}),
        }

        results.append(result)

    return results


class StoreMemoryEventsToolInput(BaseModel):
    """Input schema for storing memory events."""

    messages: List[BaseMessage] = Field(
        description="List of messages to store in memory"
    )


class ListMemoryEventsToolInput(BaseModel):
    """Input schema for listing memory events."""

    max_results: int = Field(
        default=100, description="Maximum number of events to retrieve"
    )


class SearchMemoryInput(BaseModel):
    """Input schema for searching memories."""

    query: str = Field(description="Search query to find relevant memories")
    limit: int = Field(
        default=3, description="Maximum number of search results to return"
    )


def create_store_memory_events_tool(
    memory_client: MemoryClient, memory_id: str, actor_id: str, session_id: str
) -> StructuredTool:
    """Factory function to create a memory storage tool with pre-configured connection details.

    Args:
        memory_client: Initialized MemoryClient instance
        memory_id: Memory identifier (e.g., "test-memory-id")
        actor_id: Actor identifier (e.g., "user")
        session_id: Session identifier (e.g., "session-1")

    Returns:
        StructuredTool to store events that only requires the 'messages' parameter
    """

    def _store_messages(messages: List[BaseMessage]) -> str:
        """Internal function with pre-bound connection details."""
        return store_agentcore_memory_events(
            memory_client=memory_client,
            messages=messages,
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
        )

    return StructuredTool.from_function(
        func=_store_messages,
        name="store_memory_events",
        description="Store conversation messages in AgentCore memory for later retrieval",
        args_schema=StoreMemoryEventsToolInput,
    )


def create_list_memory_events_tool(
    memory_client: MemoryClient, memory_id: str, actor_id: str, session_id: str
) -> StructuredTool:
    """Factory function to create a memory listing tool with pre-configured connection details.

    Args:
        memory_client: Initialized MemoryClient instance
        memory_id: Memory identifier (e.g., "test-memory-id")
        actor_id: Actor identifier (e.g., "user")
        session_id: Session identifier (e.g., "session-1")

    Returns:
        StructuredTool for listing events that only requires 'max_results' parameter
    """

    def _list_events(max_results: int = 100) -> List[BaseMessage]:
        """Internal function with pre-bound connection details."""
        return list_agentcore_memory_events(
            memory_client=memory_client,
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            max_results=max_results,
        )

    return StructuredTool.from_function(
        func=_list_events,
        name="list_memory_events",
        description="Retrieve recent conversation messages from AgentCore memory",
        args_schema=ListMemoryEventsToolInput,
    )


def create_retrieve_memory_tool(
    memory_client: MemoryClient,
    memory_id: str,
    namespace: str,
    tool_name: str = "retrieve_memory",
    tool_description: str = "Search for relevant memories using semantic similarity",
) -> StructuredTool:
    """Factory function to create a memory search tool with pre-configured connection details.

    Args:
        memory_client: Initialized MemoryClient instance
        memory_id: Memory identifier (e.g., "test-memory-id")
        namespace: Namespace for search (e.g., "/summaries/user/session-1")
        tool_name: Name of the tool, i.e. "search_user_preferences"
        tool_description: Description of the tool's purpose, i.e. "Use this tool to search for user preferences"

    Returns:
        StructuredTool to retrieve memories that only requires 'query' and 'limit' parameters
    """

    def _search_memories(query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Internal function with pre-bound connection details."""
        return retrieve_agentcore_memories(
            memory_client=memory_client,
            memory_id=memory_id,
            namespace_str=namespace,
            query=query,
            limit=limit,
        )

    return StructuredTool.from_function(
        func=_search_memories,
        name=tool_name,
        description=tool_description,
        args_schema=SearchMemoryInput,
    )
