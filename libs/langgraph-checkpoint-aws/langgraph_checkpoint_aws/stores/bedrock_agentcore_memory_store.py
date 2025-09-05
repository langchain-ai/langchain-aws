import logging
from collections.abc import Iterable
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Union,
)

from bedrock_agentcore.memory.client import MemoryClient
from bedrock_agentcore.memory.constants import MessageRole
from langchain_core.messages import (
    BaseMessage,
)
from langgraph.store.base import (
    BaseStore,
    Item,
    SearchItem,
    TTLConfig,
)

logger = logging.getLogger(__name__)

class AgentCoreRetrieveOp(NamedTuple):
    namespace: tuple[str, ...]  # (actor_id, session_id)
    query: str
    top_k: int = 10
    memory_strategy_id: str | None = None

class AgentCoreListOp(NamedTuple):
    namespace: tuple[str, ...]  # (actor_id, session_id)
    max_results: int = 100

class AgentCoreStoreOp(NamedTuple):
    """Operation to store a message event."""
    namespace: tuple[str, ...]  # (actor_id, session_id)
    key: str  # event identifier
    message: BaseMessage
    event_timestamp: datetime | None = None

class AgentCoreDeleteOp(NamedTuple):
    """Operation to delete a memory record."""
    memory_record_id: str

class AgentCoreGetOp(NamedTuple):
    """Operation to get a memory record."""
    memory_record_id: str

AgentCoreOp = Union[AgentCoreRetrieveOp, AgentCoreListOp, AgentCoreStoreOp, AgentCoreDeleteOp, AgentCoreGetOp]
AgentCoreResult = Union[Item, list[Item], list[SearchItem], list[tuple[str, ...]], None]

# Define missing constants and types
class NotProvided:
    pass

NOT_PROVIDED = NotProvided()
NamespacePath = tuple[str, ...]

class BedrockAgentCoreMemoryStore(BaseStore):
    """Bedrock AgentCore Memory Store support for storage of chat messages and retrieval of long term memories

    !!! example "Examples"
        Storing conversation messages:
            memory_client = MemoryClient(region="us-west-2")
            store = BedrockAgentCoreMemoryStore(memory_client)

    Stores enable persistence and memory that can be shared across threads,
    scoped to user IDs, assistant IDs, or other arbitrary namespaces.

    Note:
        This implementation depends on Amazon Bedrock AgentCore Memory to store and process
        messages then later retrieve the processed memories through semantic search. An example
        would be saving a conversation and then processing async user preferences for later
        search in a user preferences namespace.
    """

    supports_ttl: bool = False
    ttl_config: TTLConfig | None = None

    __slots__ = ("memory_client", "memory_id")

    def __init__(self, *, memory_id: str, memory_client: MemoryClient) -> None:

        # Bedrock AgentCore Memory Client
        self.memory_client: MemoryClient = memory_client
        self.memory_id = memory_id

    def batch(self, ops: Iterable[AgentCoreOp]) -> list[AgentCoreResult]:
        """Execute a batch of AgentCore operations synchronously."""
        results = []
        
        for op in ops:
            if isinstance(op, AgentCoreRetrieveOp):
                result = self._retrieve_memories(op)
                results.append(result)
            elif isinstance(op, AgentCoreListOp):
                result = self._list_memory_records(op)
                results.append(result)
            elif isinstance(op, AgentCoreGetOp):
                result = self._get_memory_record(op)
                results.append(result)
            elif isinstance(op, AgentCoreStoreOp):
                self._store_message(op)
                results.append(None)
            elif isinstance(op, AgentCoreDeleteOp):
                self._delete_memory_record(op)
                results.append(None)
            else:
                raise ValueError(f"Unknown AgentCore operation type: {type(op)}")
        
        return results

    async def abatch(self, ops: Iterable[AgentCoreOp]) -> list[AgentCoreResult]:
        """Execute a batch of AgentCore operations asynchronously."""
        raise NotImplementedError("The Bedrock AgentCore Memory client does not yet support async operations")

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Retrieve a single memory item.

        Args:
            namespace: (actor_id, session_id) indicating where the memory is stored
            key: Unique identifier for the memory event
            refresh_ttl: Not applicable for Bedrock AgentCore Memory

        Returns:
            Item with the individual record information retrieved
        """
        op = AgentCoreGetOp(memory_record_id=key)
        result = self.batch([op])[0]
        return result



    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
        # AgentCore-specific parameters
        memory_strategy_id: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchItem]:
        """Search for items within a namespace prefix.

        Args:
            namespace_prefix: the namespace tuple of which to search (actor_id, session_id)
            query: the query to search for in Bedrock AgentCore memory
            filter: Not supported by Bedrock AgentCore Memory (will be ignored)
            limit: Maximum number of items to return (used as top_k if not specified)
            offset: Not supported by Bedrock AgentCore Memory (will be ignored)
            refresh_ttl: Not applicable for Bedrock AgentCore Memory
            memory_strategy_id (optional): strategy ID to search for
            top_k (optional): the maximum number of top-scoring memory records to return

        Returns:
            List of items matching the search criteria.

        ???+ example "Examples"
            Basic listing of long term memories (no semantic search):
            ```python
            # List memory records in a namespace
            results = store.search(("user-1", "session-1"))
            ```

            Basic semantic searching for long term memories:
            ```python
            # Search for user preferences for a certain query
            results = store.search(
                ("user-1", "session-1"),
                query="favorite coffeeshops and past orders"
            )
            ```
        """
        
        if query:
            # Use semantic search
            op = AgentCoreRetrieveOp(
                namespace=namespace_prefix,
                query=query,
                top_k=top_k or limit,
                memory_strategy_id=memory_strategy_id
            )
        else:
            # Use list operation
            op = AgentCoreListOp(
                namespace=namespace_prefix,
                max_results=limit
            )
        
        return self.batch([op])[0] or []

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        """Store or update a message event in Bedrock AgentCore Memory

        Args:
            namespace: a tuple with actor id and session id as the arguments
                Example: ("actorId", "sessionId")
            key: the event identifier for the memory
            value: The message data containing a "message" key with a BaseMessage object
            index: Not supported - indexing is handled automatically by Bedrock AgentCore
            ttl: Not supported - TTL is handled by Bedrock AgentCore service

        Note:
            Async processing of messages in Bedrock AgentCore such as summarization or user
            preference abstraction happens automatically in the service. Each message that
            is saved here is then processed later.

        ???+ example "Examples"
            Store a message.
            ```python
            from langchain_core.messages import HumanMessage
            store.put(("user-1","session-1"), "123", {"message": HumanMessage("My favorite pirate is Blackbeard")})
            ```
        """
        self._validate_namespace(namespace)

        if index is not None and index is not False:
            raise NotImplementedError("Custom indexing is handled by the Bedrock AgentCore service itself.")
        
        if not isinstance(ttl, NotProvided) and ttl is not None:
            raise NotImplementedError("TTL is handled by the Bedrock AgentCore service itself.")

        message = value.get("message")
        if message is None:
            raise ValueError("Value must contain a 'message' key with a BaseMessage object")
        
        if not isinstance(message, BaseMessage):
            raise ValueError("The 'message' value must be a BaseMessage instance")

        op = AgentCoreStoreOp(
            namespace=namespace,
            key=str(key),
            message=message,
            event_timestamp=None
        )
        
        self.batch([op])

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item.

        Args:
            namespace: tuple with (actor_id, session_id)
            key: the event_id of the memory to delete

        """
        op = AgentCoreDeleteOp(memory_record_id=key)
        self.batch([op])

    def list_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List and filter namespaces in the store"""
        raise NotImplementedError("Listing namespaces is not yet implemented for Bedrock AgentCore APIs")

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Asynchronously retrieve a single memory item.

        Args:
            namespace: (actor_id, session_id) indicating where the memory is stored
            key: Unique identifier for the memory event
            refresh_ttl: Not applicable for Bedrock AgentCore Memory

        Returns:
            None - Individual memory retrieval by key is not supported by Bedrock AgentCore.
            Use search() with a specific query instead.

        Note:
            Bedrock AgentCore Memory is designed for semantic search rather than direct
            key-based retrieval. Use the search() method with specific queries to find
            relevant memories.
        """
        raise NotImplementedError("The Bedrock AgentCore Memory client does not yet support async operations")


    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
        # AgentCore-specific parameters
        memory_strategy_id: str | None = None,
        top_k: int | None = None,
    ) -> list[SearchItem]:
        """Asynchronously search for memories within a namespace prefix using Bedrock AgentCore."""
        raise NotImplementedError("The Bedrock AgentCore Memory client does not yet support async operations")

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        """Asynchronously store a message event in Bedrock AgentCore Memory."""
        raise NotImplementedError("The Bedrock AgentCore Memory client does not yet support async operations")

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Asynchronously delete a memory event."""
        raise NotImplementedError("The Bedrock AgentCore Memory client does not yet support async operations")

    async def alist_namespaces(
        self,
        *,
        prefix: NamespacePath | None = None,
        suffix: NamespacePath | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List and filter namespaces in the store asynchronously."""
        raise NotImplementedError("Listing namespaces is not yet implemented for Bedrock AgentCore APIs")
    
    def _retrieve_memories(self, op: AgentCoreRetrieveOp) -> list[SearchItem]:
        """Retrieve memories using semantic search."""
        namespace_str = self._convert_namespace_tuple_to_str(op.namespace)
        
        try:
            retrieve_params = {
                "memory_id": self.memory_id,
                "namespace": namespace_str,
                "query": op.query,
                "top_k": op.top_k
            }
            
            # Add memory_strategy_id if provided
            if op.memory_strategy_id is not None:
                retrieve_params["memory_strategy_id"] = op.memory_strategy_id
            
            memories = self.memory_client.retrieve_memories(**retrieve_params)
            return self._convert_memories_to_search_items(memories, op.namespace)
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def _list_memory_records(self, op: AgentCoreListOp) -> list[SearchItem]:
        """List memory records in a namespace."""
        namespace_str = self._convert_namespace_tuple_to_str(op.namespace)
        
        response = self.memory_client.list_memory_records(
            memoryId=self.memory_id, 
            namespace=namespace_str, 
            maxResults=op.max_results
        )
        memories = response.get("memoryRecordSummaries", [])
        return self._convert_memories_to_search_items(memories, op.namespace)
        
        
    def _get_memory_record(self, op: AgentCoreGetOp) -> Item | None:
        """Get a specific long term memory record by ID."""

        response = self.memory_client.get_memory_record(
            memoryId=self.memory_id,
            memoryRecordId=op.memory_record_id,
        )
        
        record = response.get('memoryRecord')
        if not record:
            return None
            
        text = record.get('content', {}).get('text', '')
        namespaces = record.get('namespaces', [])
        
        # Parse namespace - take first one and split by '/'
        namespace_tuple = tuple(namespaces[0].split('/')) if namespaces else ('', '')
        created_at = record.get('createdAt')

        return Item(
            key=op.memory_record_id,
            namespace=namespace_tuple,
            value={"content": text},
            created_at=created_at,
            updated_at=created_at
        )
    
    def _delete_memory_record(self, op: AgentCoreDeleteOp) -> None:
        """Get a specific long term memory record by ID."""
        self.memory_client.delete_memory_record(
            memoryId=self.memory_id,
            memoryRecordId=op.memory_record_id,
        )

    def _store_message(self, op: AgentCoreStoreOp) -> None:
        """Store a message event."""
        messages_to_store = convert_langchain_messages_to_event_messages([op.message])
        if not messages_to_store:
            logger.warning(f"No valid messages to store for key {op.key}")
            return

        self.memory_client.create_event(
            memory_id=self.memory_id,
            actor_id=op.namespace[0],
            session_id=op.namespace[1],
            messages=messages_to_store,
            event_timestamp=op.event_timestamp
        )
        logger.debug(f"Stored message event with key {op.key}")

    def _validate_namespace(self, namespace: tuple[str, ...]) -> None:
        """Validate namespace format for Bedrock AgentCore."""
        if not isinstance(namespace, tuple) or len(namespace) != 2:
            raise ValueError("Namespace must be a tuple of (actor_id, session_id)")
        if not all(isinstance(part, str) and part.strip() for part in namespace):
            raise ValueError("Namespace parts must be non-empty strings")
        
    def _convert_namespace_tuple_to_str(self, namespace_tuple):
        return "/" + "/".join(namespace_tuple)

    def _convert_memories_to_search_items(self, memories: list, namespace: tuple[str, ...]) -> list[SearchItem]:
        """Convert AgentCore memory records to SearchItem objects."""
        results = []
        
        for item in memories:
            if isinstance(item, dict):
                content = item.get("content", {})
                if isinstance(content, dict):
                    text = content.get("text", "")
                else:
                    text = str(content)
                
                score = item.get("score", 0.0)
                record_id = item.get("memoryRecordId") or item.get("id") or str(len(results))
                
                # Handle datetime parsing
                created_at = item.get("createdAt") or item.get("timestamp")
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        created_at = datetime.now()
                elif created_at is None:
                    created_at = datetime.now()
                
                result = SearchItem(
                    namespace=namespace,
                    key=record_id,
                    value={"content": text, "metadata": item.get("metadata", {})},
                    created_at=created_at,
                    updated_at=created_at,  # memories are not updated
                    score=float(score) if score is not None else None,
                )
                results.append(result)

        return results

def convert_langchain_messages_to_event_messages(
    messages: List[BaseMessage]
) -> List[Dict[str, Any]]:
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
            role = MessageRole.USER.value
        elif msg.type == "ai":
            role = MessageRole.ASSISTANT.value
        elif msg.type == "tool":
            role = MessageRole.TOOL.value
        elif msg.type == "system":
            role = MessageRole.OTHER.value
        else:
            logger.warning(f"Skipping unsupported message type: {msg.type}")
            continue

        converted_messages.append((text, role))

    return converted_messages
