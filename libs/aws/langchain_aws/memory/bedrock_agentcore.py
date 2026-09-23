"""Module for AWS Bedrock Agent Core memory integration.

This module provides integration between LangChain/LangGraph and AWS Bedrock Agent Core
memory API. It includes a memory store implementation and tools for managing and
searching memories.
"""

import json
import logging
from typing import List

from bedrock_agentcore.memory import MemoryClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


def create_store_messages_tool(
    memory_client: MemoryClient,
    name: str = "store_messages"
) -> StructuredTool:
    """Create a tool for storing messages directly with Bedrock Agent Core MemoryClient.

    This tool enables AI assistants to store messages in Bedrock Agent Core.
    The tool expects the following configuration values to be passed via RunnableConfig:
    - memory_id: The ID of the memory to store in
    - actor_id: (optional) The actor ID to use
    - session_id: (optional) The session ID to use

    Args:
        memory_client: The MemoryClient instance to use
        name: The name of the tool

    Returns:
        A structured tool for storing messages
    """

    instructions = (
        "Use this tool to store all messages from the user and AI model. These "
        "messages are processed to extract summary or facts of the conversation, "
        "which can be later retrieved using the search_memory tool."
    )

    def store_messages(
        messages: List[BaseMessage],
        config: RunnableConfig,
    ) -> str:
        """Stores conversation messages in AWS Bedrock Agent Core Memory.

        Args:
            messages: List of messages to store

        Returns:
            A confirmation message.
        """
        if not (configurable := config.get("configurable", None)):
            raise ValueError(
                "A runtime config containing memory_id, actor_id, and session_id is required."
            )
        
        if not (memory_id := configurable.get("memory_id", None)):
            raise ValueError(
                "Missing memory_id in the runtime config."
            )
        
        if not (session_id := configurable.get("session_id", None)):
            raise ValueError(
                "Missing session_id in the runtime config."
            )
        
        if not (actor_id := configurable.get("actor_id", None)):
            raise ValueError(
                "Missing actor_id in the runtime config."
            )
            
        # Convert BaseMessage list to list of (text, role) tuples
        # TODO: This should correctly convert to 
        converted_messages = []
        for msg in messages:
            
            # Skip if event already saved
            if msg.additional_kwargs.get("event_id", None) is not None:
                continue

            # Extract text content
            content = msg.content
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict) and content['type'] == 'text':
                text = content['text']
            else:
                continue
            
            # Map LangChain roles to Bedrock Agent Core roles
            # Available roles in Bedrock: USER, ASSISTANT, TOOL
            if msg.type == "human":
                role = "USER"
            elif msg.type == "ai":
                role = "ASSISTANT"
            elif msg.type == "tool":
                role = "TOOL"
            else:
                continue  # Skip unsupported message types
            
            converted_messages.append((text, role))
        
        # Create event with converted messages directly using the MemoryClient
        response = memory_client.create_event(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=converted_messages
        )
        
        return f"Memory created with ID: {response.get('eventId')}"

    # Create a StructuredTool with the custom name
    return StructuredTool.from_function(
        func=store_messages, name=name, description=instructions
    )


def create_list_messages_tool(
    memory_client: MemoryClient,
    name: str = "list_messages",
) -> StructuredTool:
    """Create a tool for listing conversation messages from Bedrock Agent Core Memory.

    This tool allows AI assistants to retrieve the message history from a conversation
    stored in Bedrock Agent Core Memory.
    
    The tool expects the following configuration values to be passed via RunnableConfig:
    - memory_id: The ID of the memory to retrieve from (required)
    - actor_id: The actor ID to use (required)
    - session_id: The session ID to use (required)

    Args:
        memory_client: The MemoryClient instance to use
        name: The name of the tool

    Returns:
        A structured tool for listing conversation messages
    """

    instructions = (
        "Use this tool to retrieve the conversation history from memory. "
        "This can help in understanding the context of the current conversation, "
        "or reviewing past interactions."
    )

    def list_messages(
        max_results: int = 100,
        config: RunnableConfig = None,
    ) -> List[BaseMessage]:
        """List conversation messages from AWS Bedrock Agent Core Memory.

        Args:
            max_results: Maximum number of messages to return
            config: RunnableConfig containing memory_id, actor_id, and session_id

        Returns:
            A list of LangChain message objects (HumanMessage, AIMessage, ToolMessage)
        """
        if not (configurable := config.get("configurable", None)):
            raise ValueError(
                "A runtime config with memory_id, actor_id, and session_id is required"
                " for list_messages tool."
            )
        
        if not (memory_id := configurable.get("memory_id", None)):
            raise ValueError(
                "Missing memory_id in the runtime config."
            )
            
        if not (actor_id := configurable.get("actor_id", None)):
            raise ValueError(
                "Missing actor_id in the runtime config."
            )
            
        if not (session_id := configurable.get("session_id", None)):
            raise ValueError(
                "Missing session_id in the runtime config."
            )
        
        events = memory_client.list_events(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            max_results=max_results,
            include_payload=True
        )
        
        # Extract and format messages as LangChain message objects
        messages = []
        for event in events:
            # Extract messages from event payload
            if "payload" in event:
                for payload_item in event.get("payload", []):
                    if "conversational" in payload_item:
                        conv = payload_item["conversational"]
                        role = conv.get("role", "")
                        content = conv.get("content", {}).get("text", "")
                        
                        # Convert to appropriate LangChain message type based on role
                        if role == "USER":
                            message = HumanMessage(content=content)
                        elif role == "ASSISTANT":
                            message = AIMessage(content=content)
                        elif role == "TOOL":
                            #message = ToolMessage(content=content, tool_call_id="unknown")
                            # skipping tool events as tool_call_id etc. will be missing
                            continue
                        else:
                            # Skip unknown message types
                            continue
                            
                        # Add metadata if available
                        if "eventId" in event:
                            message.additional_kwargs["event_id"] = event["eventId"]
                        if "eventTimestamp" in event:
                            pass
                            # Skip this, this currently not serialized correctly
                            # message.additional_kwargs["timestamp"] = event["eventTimestamp"]
                            
                        messages.append(message)
        
        return messages

    # Create a StructuredTool with the custom name
    return StructuredTool.from_function(
        func=list_messages, name=name, description=instructions
    )


def create_search_memory_tool(
    memory_client: MemoryClient,
    name: str = "search_memory",
) -> StructuredTool:
    """Create a tool for searching memories in AWS Bedrock Agent Core.

    This tool allows AI assistants to search through stored memories in AWS
    Bedrock Agent Core using semantic search.
    
    The tool expects the following configuration values to be passed via RunnableConfig:
    - memory_id: The ID of the memory to search in (required)
    - namespace: The namespace to search in (required)

    Args:
        memory_client: The MemoryClient instance to use
        name: The name of the tool

    Returns:
        A structured tool for searching memories.
    """

    instructions = (
        "Use this tool to search for helpful facts and preferences from the past "
        "conversations. Based on the namespace and configured memories, this will "
        "provide summaries, user preferences or semantic search for the session."
    )

    def search_memory(
        query: str,
        limit: int = 3,
        config: RunnableConfig = None,
    ) -> str:
        """Search for memories in AWS Bedrock Agent Core.

        Args:
            query: The search query to find relevant memories.
            limit: Maximum number of results to return.

        Returns:
            A string representation of the search results.
        """
        if not (configurable := config.get("configurable", None)):
            raise ValueError(
                "A runtime config with memory_id, namespace, and actor_id is required."
            )
        
        if not (memory_id := configurable.get("memory_id", None)):
            raise ValueError(
                "Missing memory_id in the runtime config."
            )
            
        # Namespace is required
        if not (namespace_val := configurable.get("namespace", None)):
            raise ValueError(
                "Missing namespace in the runtime config."
            )
            
        # Format the namespace
        if isinstance(namespace_val, tuple):
            # Join tuple elements with '/'
            namespace_str = "/" + "/".join(namespace_val)
        elif isinstance(namespace_val, str):
            # Ensure string starts with '/'
            namespace_str = namespace_val if namespace_val.startswith("/") else f"/{namespace_val}"
        else:
            raise ValueError(
                f"Namespace must be a string or tuple, got {type(namespace_val)}"
            )
                
        # Perform the search directly using the MemoryClient
        memories = memory_client.retrieve_memories(
            memory_id=memory_id,
            namespace=namespace_str,
            query=query,
            top_k=limit,
        )

        # Process and format results
        results = []
        for item in memories:
            # Extract content from the memory item
            content = item.get("content", {}).get("text", "")

            # Try to parse JSON content if it looks like JSON
            if content and content.startswith("{") and content.endswith("}"):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass

            results.append(content)

        return results
        

    # Create a StructuredTool with the custom name
    return StructuredTool.from_function(
        func=search_memory,
        name=name,
        description=instructions
    )
