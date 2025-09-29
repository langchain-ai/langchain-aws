"""
Tool factories for AgentCore Memory Store operations.

This module provides tool factories for creating LangChain tools that interact with
AgentCore Memory Store, following the pattern established by langmem but adapted
for the AgentCore service specifics.
"""

import functools
import logging
import typing
import uuid
from typing import Optional, Type

from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.store.base import BaseStore
from langgraph.utils.config import get_config, get_store

if typing.TYPE_CHECKING:
    from langchain_core.tools.base import ArgsSchema

try:
    from pydantic import ConfigDict
except ImportError:
    ConfigDict = None

logger = logging.getLogger(__name__)


class NamespaceTemplate:
    """Template for namespace configuration with runtime substitution."""

    def __init__(self, namespace: tuple[str, ...] | str):
        if isinstance(namespace, str):
            self.namespace_parts = (namespace,)
        else:
            self.namespace_parts = namespace

    def __call__(self, config: dict | None = None) -> tuple[str, ...]:
        """Format namespace with runtime configuration."""
        if not config:
            try:
                config = get_config()
            except RuntimeError:
                # If we're outside a runnable context, just return the template
                # This allows the tool to be created outside of a runnable context
                return self.namespace_parts

        configurable = config.get("configurable", {})
        formatted_parts = []

        for part in self.namespace_parts:
            if part.startswith("{") and part.endswith("}"):
                # Format with configurable values
                try:
                    formatted_part = part.format(**configurable)
                    formatted_parts.append(formatted_part)
                except KeyError as e:
                    raise ValueError(
                        f"Missing required configurable key for namespace: {e}"
                    )
            else:
                formatted_parts.append(part)

        return tuple(formatted_parts)


def create_manage_memory_tool(
    namespace: tuple[str, ...] | str,
    *,
    instructions: str = "Proactively call this tool when you:\n\n"
    "1. Identify a new USER preference.\n"
    "2. Receive an explicit USER request to remember something or otherwise alter your behavior.\n"
    "3. Are working and want to record important context.\n"
    "4. Identify that an existing MEMORY is incorrect or outdated.\n",
    schema: Type = BaseMessage,
    actions_permitted: Optional[
        tuple[typing.Literal["create", "update", "delete"], ...]
    ] = ("create",),
    store: Optional[BaseStore] = None,
    name: str = "manage_memory",
):
    """Create a tool for managing persistent memories in AgentCore Memory.

    This function creates a tool that allows AI assistants to create memories
    that are automatically processed by AgentCore Memory service. The tool stores
    conversational events that get transformed into searchable memories.

    Args:
        namespace: The namespace structure for organizing memories. For AgentCore,
            this should be a tuple of (actor_id, session_id). Uses runtime
            configuration with placeholders like `{actor_id}` and `{thread_id}`.
        instructions: Custom instructions for when to use the memory tool.
        schema: The schema for memory content. Defaults to BaseMessage for AgentCore.
        actions_permitted: Currently only "create" is supported for AgentCore Memory.
        store: The BaseStore to use. If not provided, uses the configured store.
        name: The name of the tool.

    Returns:
        A StructuredTool that can be used for memory management.

    Example:
        ```python
        from langmem import create_manage_memory_tool
        from langgraph.prebuilt import create_react_agent
        from langgraph_checkpoint_aws.agentcore import AgentCoreMemoryStore

        memory_tool = create_manage_memory_tool(
            namespace=("{actor_id}", "{thread_id}"),
        )

        store = AgentCoreMemoryStore(
            memory_id="memory_abc123",
            region_name="us-west-2"
        )

        agent = create_react_agent(
            "anthropic:claude-3-5-sonnet-latest",
            tools=[memory_tool],
            store=store,
        )
        ```
    """
    if actions_permitted != ("create",):
        logger.warning(
            "AgentCore Memory currently only supports 'create' action. "
            "Other actions will be ignored."
        )

    namespacer = NamespaceTemplate(namespace)
    initial_store = store

    async def amanage_memory(
        content: schema,  # type: ignore
        action: typing.Literal["create"] = "create",  # type: ignore
        *,
        id: Optional[uuid.UUID] = None,
    ):
        """Async version of manage_memory."""
        store = _get_store(initial_store)

        if action != "create":
            raise ValueError(
                "AgentCore Memory only supports 'create' action. "
                "Memory updates and deletions happen automatically through the service."
            )

        if id is not None:
            logger.warning(
                "AgentCore Memory does not support custom IDs. Ignoring provided ID."
            )

        namespace = namespacer()
        if len(namespace) != 2:
            raise ValueError(
                "AgentCore Memory requires namespace to be (actor_id, session_id)"
            )

        # Generate a unique key for this memory
        key = str(uuid.uuid4())

        # Ensure content is a BaseMessage
        if not isinstance(content, BaseMessage):
            raise ValueError("Content must be a BaseMessage for AgentCore Memory")

        # Store the message
        await store.aput(namespace, key, {"message": content})

        return f"Created memory event {key} in AgentCore Memory"

    def manage_memory(
        content: schema,  # type: ignore
        action: typing.Literal["create"] = "create",  # type: ignore
        *,
        id: Optional[uuid.UUID] = None,
    ):
        """Sync version of manage_memory."""
        store = _get_store(initial_store)

        if action != "create":
            raise ValueError(
                "AgentCore Memory only supports 'create' action. "
                "Memory updates and deletions happen automatically through the service."
            )

        if id is not None:
            logger.warning(
                "AgentCore Memory does not support custom IDs. Ignoring provided ID."
            )

        namespace = namespacer()
        if len(namespace) != 2:
            raise ValueError(
                "AgentCore Memory requires namespace to be (actor_id, session_id)"
            )

        # Generate a unique key for this memory
        key = str(uuid.uuid4())

        # Ensure content is a BaseMessage
        if not isinstance(content, BaseMessage):
            raise ValueError("Content must be a BaseMessage for AgentCore Memory")

        # Store the message
        store.put(namespace, key, {"message": content})

        return f"Created memory event {key} in AgentCore Memory"

    description = f"""Create a memory event in AgentCore Memory.
The memory will be automatically processed and made searchable by the AgentCore service.
{instructions}"""

    return _ToolWithRequired.from_function(
        manage_memory, amanage_memory, name=name, description=description
    )


def create_search_memory_tool(
    namespace: tuple[str, ...] | str,
    *,
    instructions: str = "Search for relevant memories and user preferences to provide context for your responses.",
    store: BaseStore | None = None,
    response_format: typing.Literal["content", "content_and_artifact"] = "content",
    name: str = "search_memory",
):
    """Create a tool for searching memories in AgentCore Memory Store.

    This function creates a tool that allows AI assistants to search through
    processed memories using semantic search powered by AgentCore Memory service.

    Args:
        namespace: The namespace for searching memories. For AgentCore, this is
            typically ("facts", "{actor_id}") for user facts/preferences.
        instructions: Custom instructions for when to use the search tool.
        store: The BaseStore to use. If not provided, uses the configured store.
        response_format: Whether to return just content or content with artifacts.
        name: The name of the tool.

    Returns:
        A StructuredTool for memory search.

    Example:
        ```python
        search_tool = create_search_memory_tool(
            namespace=("facts", "{actor_id}"),
        )
        ```
    """
    namespacer = NamespaceTemplate(namespace)
    initial_store = store

    async def asearch_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: Optional[dict] = None,
    ):
        """Async version of search_memory."""
        store = _get_store(initial_store)
        namespace = namespacer()

        memories = await store.asearch(
            namespace,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )

        if response_format == "content_and_artifact":
            return _format_search_results(memories), memories
        return _format_search_results(memories)

    def search_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: Optional[dict] = None,
    ):
        """Sync version of search_memory."""
        store = _get_store(initial_store)
        namespace = namespacer()

        memories = store.search(
            namespace,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )

        if response_format == "content_and_artifact":
            return _format_search_results(memories), memories
        return _format_search_results(memories)

    description = f"""Search AgentCore Memory for relevant information.
{instructions}"""

    # Create the tool with proper response format handling
    if response_format == "content_and_artifact":
        return _SearchToolWithArtifacts.from_function(
            search_memory,
            asearch_memory,
            name=name,
            description=description,
        )
    else:
        return StructuredTool.from_function(
            search_memory,
            asearch_memory,
            name=name,
            description=description,
        )


def _get_store(initial_store: BaseStore | None = None) -> BaseStore:
    """Get the store instance, either from parameter or configuration."""
    try:
        if initial_store is not None:
            return initial_store
        else:
            return get_store()
    except RuntimeError as e:
        raise RuntimeError(
            "Could not get store. Make sure a store is configured in your graph."
        ) from e


def _format_search_results(memories: list) -> str:
    """Format search results for display."""
    if not memories:
        return "No memories found."

    results = []
    for i, memory in enumerate(memories, 1):
        content = memory.value.get("content", "")
        score = memory.score
        memory_id = memory.key

        result_str = f"{i}. {content}"
        if score is not None:
            result_str += f" (relevance: {score:.2f})"
        result_str += f" [id: {memory_id}]"

        results.append(result_str)

    return "\n".join(results)


class _ToolWithRequired(StructuredTool):
    """Tool with required parameters properly configured."""

    @functools.cached_property
    def tool_call_schema(self) -> "ArgsSchema":
        tcs = super().tool_call_schema
        try:
            if tcs.model_config:
                tcs.model_config["json_schema_extra"] = _ensure_schema_contains_required
            elif ConfigDict is not None:
                tcs.model_config = ConfigDict(
                    json_schema_extra=_ensure_schema_contains_required
                )
        except Exception:
            pass
        return tcs


class _SearchToolWithArtifacts(StructuredTool):
    """Search tool that returns both content and artifacts as a tuple."""

    @functools.cached_property
    def tool_call_schema(self) -> "ArgsSchema":
        tcs = super().tool_call_schema
        try:
            if tcs.model_config:
                tcs.model_config["json_schema_extra"] = _ensure_schema_contains_required
            elif ConfigDict is not None:
                tcs.model_config = ConfigDict(
                    json_schema_extra=_ensure_schema_contains_required
                )
        except Exception:
            pass
        return tcs


def _ensure_schema_contains_required(schema: dict) -> None:
    """Ensure schema contains required fields."""
    schema.setdefault("required", [])


# Additional helper tool for direct event storage (AgentCore specific)
def create_store_event_tool(
    *,
    store: Optional[BaseStore] = None,
    name: str = "store_conversation_event",
):
    """Create a tool for storing conversation events directly in AgentCore Memory.

    This is an AgentCore-specific tool that allows storing conversation events
    that will be processed into memories by the AgentCore service.

    Args:
        store: The BaseStore to use. If not provided, uses the configured store.
        name: The name of the tool.

    Returns:
        A StructuredTool for storing conversation events.

    Example:
        ```python
        store_tool = create_store_event_tool()
        ```
    """
    initial_store = store

    async def astore_event(
        message: BaseMessage,
        actor_id: str,
        session_id: str,
    ):
        """Store a conversation event asynchronously."""
        store = _get_store(initial_store)
        namespace = (actor_id, session_id)
        key = str(uuid.uuid4())

        await store.aput(namespace, key, {"message": message})
        return f"Stored conversation event {key}"

    def store_event(
        message: BaseMessage,
        actor_id: str,
        session_id: str,
    ):
        """Store a conversation event synchronously."""
        store = _get_store(initial_store)
        namespace = (actor_id, session_id)
        key = str(uuid.uuid4())

        store.put(namespace, key, {"message": message})
        return f"Stored conversation event {key}"

    description = """Store a conversation event in AgentCore Memory.
This event will be automatically processed into searchable memories by the service."""

    return StructuredTool.from_function(
        store_event,
        astore_event,
        name=name,
        description=description,
    )
