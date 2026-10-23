"""
Unit tests for AgentCore Memory Store.
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchItem,
    SearchOp,
)

from langgraph_checkpoint_aws.checkpoint.agentcore.helpers import (
    convert_langchain_messages_to_event_messages,
)
from langgraph_checkpoint_aws.store.agentcore.store import (
    STORE_KEY_METADATA,
    AgentCoreMemoryStore,
    _store_key_metadata_value,
)


@pytest.mark.filterwarnings("error::langchain_core._api.LangChainDeprecationWarning")
def test_convert_messages_uses_text_property() -> None:
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content=[{"type": "text", "text": "Hi"}]),
        ToolMessage(content="Result", tool_call_id="call-1"),
        SystemMessage(content="Instructions"),
        HumanMessage(content=" "),
        HumanMessage(content="Saved", additional_kwargs={"event_id": "event-1"}),
    ]
    assert convert_langchain_messages_to_event_messages(messages) == [
        ("Hello", "USER"),
        ("Hi", "ASSISTANT"),
        ("Result", "TOOL"),
        ("Instructions", "OTHER"),
    ]


class TestAgentCoreMemoryStore:
    """Test suite for AgentCoreMemoryStore."""

    @pytest.fixture
    def mock_boto_client(self):
        """Mock boto3 client with all required methods."""
        mock_client = Mock()
        mock_client.create_event = MagicMock()
        mock_client.retrieve_memory_records = MagicMock()
        mock_client.get_memory_record = MagicMock()
        mock_client.list_events = MagicMock()
        return mock_client

    @pytest.fixture
    def memory_id(self):
        return "test-memory-id"

    @pytest.fixture
    def store(self, mock_boto_client, memory_id):
        """Create store instance with mocked client."""
        with patch("boto3.client") as mock_boto3_client:
            mock_boto3_client.return_value = mock_boto_client
            return AgentCoreMemoryStore(memory_id=memory_id)

    @pytest.fixture
    def sample_namespace(self):
        return ("test_actor", "test_session")

    @pytest.fixture
    def sample_item_data(self):
        return {
            "message": HumanMessage(content="Hello, world!"),
            "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
        }

    @pytest.fixture
    def sample_memory_record(self):
        return {
            "memoryRecordId": "mem-test-record-1234567890123456789012345678901234",
            "content": {"text": "Hello, world!"},
            "memoryStrategyId": "strategy-123",
            "namespaces": ["test_actor", "test_session"],
            "createdAt": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "score": 0.95,
        }

    def test_init_with_region_name(self, memory_id):
        """Test initialization with region name creates boto3 client."""
        with patch("boto3.client") as mock_boto3_client:
            mock_client = Mock()
            mock_boto3_client.return_value = mock_client

            store = AgentCoreMemoryStore(memory_id=memory_id, region_name="us-west-2")

            assert store.memory_id == memory_id
            mock_boto3_client.assert_called_once_with(
                "bedrock-agentcore", config=ANY, region_name="us-west-2"
            )

    def test_init_missing_parameters(self):
        """Test initialization requires memory_id."""
        with pytest.raises(TypeError):
            AgentCoreMemoryStore()

    def test_init_with_caller_config(self, memory_id):
        """Test successful merge of user Config into final boto client config"""
        from botocore.config import Config

        with patch("boto3.client") as mock_boto3_client:
            AgentCoreMemoryStore(
                memory_id=memory_id,
                region_name="us-west-2",
                config=Config(read_timeout=10),
            )
        merged = mock_boto3_client.call_args[1]["config"]
        assert merged.read_timeout == 10
        assert merged.retries["mode"] == "adaptive"
        assert "langgraph_agentcore_memory_store" in merged.user_agent_extra

    def test_batch_get_op_success(self, store, mock_boto_client):
        """Test successful GetOp operation via list_events."""
        event_timestamp = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "Hello, world!"},
                                "role": "USER",
                            }
                        }
                    ],
                    "eventTimestamp": event_timestamp,
                }
            ]
        }

        op_key = "test-key"
        ops = [GetOp(namespace=("test_actor", "test_session"), key=op_key)]
        results = store.batch(ops)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, Item)
        assert result.namespace == ("test_actor", "test_session")
        assert result.key == op_key
        assert result.value["content"] == "Hello, world!"
        assert result.value["messages"] == [{"text": "Hello, world!", "role": "USER"}]

        # Pin the exact API call shape to protect against regressions in _handle_get
        mock_boto_client.list_events.assert_called_once_with(
            memoryId="test-memory-id",
            actorId="test_actor",
            sessionId="test_session",
            includePayloads=True,
            maxResults=100,
            filter={
                "eventMetadata": [
                    {
                        "left": {"metadataKey": STORE_KEY_METADATA},
                        "operator": "EQUALS_TO",
                        "right": {
                            "metadataValue": {
                                "stringValue": _store_key_metadata_value(op_key)
                            }
                        },
                    }
                ]
            },
        )

    def test_batch_get_op_not_found(self, store, mock_boto_client):
        """Test GetOp when list_events raises ResourceNotFoundException."""
        mock_boto_client.list_events.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFoundException"}},
            operation_name="ListEvents",
        )

        ops = [GetOp(namespace=("test_actor", "test_session"), key="nonexistent")]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] is None

    def test_batch_get_op_other_error(self, store, mock_boto_client):
        """Test GetOp with other boto3 errors on list_events."""
        mock_boto_client.list_events.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDeniedException"}},
            operation_name="ListEvents",
        )

        ops = [GetOp(namespace=("test_actor", "test_session"), key="test-key")]

        with pytest.raises(ClientError):
            store.batch(ops)

    def test_batch_put_op_success(self, store, mock_boto_client, sample_item_data):
        """Test successful PutOp operation."""
        mock_boto_client.create_event.return_value = {
            "event": {"eventId": "event-123", "memoryId": "test-memory-id"}
        }

        ops = [
            PutOp(
                namespace=("test_actor", "test_session"),
                key="test-key",
                value=sample_item_data,
            )
        ]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] is None  # PutOp returns None

        mock_boto_client.create_event.assert_called_once()
        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["memoryId"] == "test-memory-id"
        assert call_args["actorId"] == "test_actor"
        assert call_args["sessionId"] == "test_session"
        assert len(call_args["payload"]) == 1
        assert call_args["payload"][0]["conversational"]["role"] == "USER"

        # Verify metadata carries a hashed form of the user key (hashed so it
        # stays within AgentCore's metadata stringValue constraints)
        assert call_args["metadata"] == {
            STORE_KEY_METADATA: {"stringValue": _store_key_metadata_value("test-key")}
        }

    def test_batch_put_op_delete(self, store, mock_boto_client):
        """Test PutOp with None value (delete operation)."""
        ops = [
            PutOp(namespace=("test_actor", "test_session"), key="test-key", value=None)
        ]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] is None

        # Should not call create_event for delete operations
        mock_boto_client.create_event.assert_not_called()

    def test_batch_put_op_with_ai_message(self, store, mock_boto_client):
        """Test PutOp with AI message."""
        ai_message_data = {"message": AIMessage(content="I can help you with that!")}

        ops = [
            PutOp(
                namespace=("test_actor", "test_session"),
                key="test-key",
                value=ai_message_data,
            )
        ]
        store.batch(ops)

        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["payload"][0]["conversational"]["role"] == "ASSISTANT"
        assert (
            "I can help you with that!"
            in call_args["payload"][0]["conversational"]["content"]["text"]
        )

    def test_batch_put_op_with_tool_message(self, store, mock_boto_client):
        """Test PutOp with tool message."""
        tool_message_data = {
            "message": ToolMessage(content="Tool result", tool_call_id="call-123")
        }

        ops = [
            PutOp(
                namespace=("test_actor", "test_session"),
                key="test-key",
                value=tool_message_data,
            )
        ]
        store.batch(ops)

        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["payload"][0]["conversational"]["role"] == "TOOL"

    def test_batch_put_op_with_system_message(self, store, mock_boto_client):
        """Test PutOp with system message."""
        system_message_data = {
            "message": SystemMessage(content="You are a helpful assistant")
        }

        ops = [
            PutOp(
                namespace=("test_actor", "test_session"),
                key="test-key",
                value=system_message_data,
            )
        ]
        store.batch(ops)

        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["payload"][0]["conversational"]["role"] == "OTHER"

    def test_batch_put_op_with_invalid_value(self, store, mock_boto_client):
        """Test PutOp with value that doesn't contain a message."""
        non_message_data = {"some_key": "some_value", "number": 42}

        ops = [
            PutOp(
                namespace=("test_actor", "test_session"),
                key="test-key",
                value=non_message_data,
            )
        ]

        with pytest.raises(ValueError, match="Value must contain a 'message' key"):
            store.batch(ops)

    def test_batch_search_op_success(
        self, store, mock_boto_client, sample_memory_record
    ):
        """Test successful SearchOp operation."""
        mock_boto_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [sample_memory_record]
        }

        ops = [
            SearchOp(
                namespace_prefix=("test_actor",), query="test query", limit=5, offset=0
            )
        ]
        results = store.batch(ops)

        assert len(results) == 1
        search_results = results[0]
        assert len(search_results) == 1
        assert isinstance(search_results[0], SearchItem)
        assert search_results[0].namespace == ("test_actor",)
        assert search_results[0].score == 0.95

        mock_boto_client.retrieve_memory_records.assert_called_once_with(
            memoryId="test-memory-id",
            namespacePath="/test_actor",
            searchCriteria={"searchQuery": "test query", "topK": 5},
            maxResults=5,
        )
        call_kwargs = mock_boto_client.retrieve_memory_records.call_args.kwargs
        assert "namespace" not in call_kwargs

    def test_batch_search_op_exact_match(
        self, mock_boto_client, memory_id, sample_memory_record
    ):
        """Test SearchOp uses the exact-match namespace when hierarchical_search
        is disabled."""
        with patch("boto3.client") as mock_boto3_client:
            mock_boto3_client.return_value = mock_boto_client
            store = AgentCoreMemoryStore(memory_id=memory_id, hierarchical_search=False)

        mock_boto_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [sample_memory_record]
        }

        ops = [
            SearchOp(
                namespace_prefix=("test_actor",), query="test query", limit=5, offset=0
            )
        ]
        store.batch(ops)

        mock_boto_client.retrieve_memory_records.assert_called_once_with(
            memoryId="test-memory-id",
            namespace="/test_actor",
            searchCriteria={"searchQuery": "test query", "topK": 5},
            maxResults=5,
        )
        call_kwargs = mock_boto_client.retrieve_memory_records.call_args.kwargs
        assert "namespacePath" not in call_kwargs

    def test_batch_search_op_no_query(self, store, mock_boto_client):
        """Test SearchOp without query returns empty results."""
        ops = [SearchOp(namespace_prefix=("test_actor",), limit=5)]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] == []

        # Should not call retrieve_memory_records without query
        mock_boto_client.retrieve_memory_records.assert_not_called()

    def test_batch_search_op_empty_results(self, store, mock_boto_client):
        """Test SearchOp with empty results."""
        mock_boto_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": []
        }

        ops = [
            SearchOp(
                namespace_prefix=("test_actor",), query="nonexistent query", limit=5
            )
        ]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] == []

    def test_batch_list_namespaces_op_success(self, store):
        """Test successful ListNamespacesOp operation."""
        ops = [ListNamespacesOp(limit=10)]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] == []  # Always returns empty list

    def test_batch_list_namespaces_op_with_conditions(self, store):
        """Test ListNamespacesOp with match conditions."""
        ops = [
            ListNamespacesOp(
                match_conditions=(
                    MatchCondition(match_type="prefix", path=("test_actor",)),
                ),
                limit=5,
            )
        ]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] == []  # Always returns empty list

    def test_batch_mixed_operations(
        self, store, mock_boto_client, sample_memory_record, sample_item_data
    ):
        """Test batch with mixed operation types."""
        event_timestamp = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "Hello, world!"},
                                "role": "USER",
                            }
                        }
                    ],
                    "eventTimestamp": event_timestamp,
                }
            ]
        }
        mock_boto_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [sample_memory_record]
        }
        mock_boto_client.create_event.return_value = {"event": {"eventId": "event-123"}}

        ops = [
            GetOp(namespace=("test_actor", "test_session"), key="test-key"),
            PutOp(
                namespace=("test_actor", "test_session"),
                key="new-key",
                value=sample_item_data,
            ),
            SearchOp(namespace_prefix=("test_actor",), query="test query", limit=5),
            ListNamespacesOp(limit=10),
        ]
        results = store.batch(ops)

        assert len(results) == 4
        assert isinstance(results[0], Item)  # GetOp result
        assert results[1] is None  # PutOp result
        assert isinstance(results[2], list)  # SearchOp result
        assert results[3] == []  # ListNamespacesOp result

    def test_batch_unknown_operation(self, store):
        """Test batch with unknown operation type."""

        class UnknownOp:
            pass

        ops = [UnknownOp()]

        with pytest.raises(ValueError, match="Unknown operation type"):
            store.batch(ops)

    async def test_abatch_mixed_operations(
        self, store, mock_boto_client, sample_memory_record, sample_item_data
    ):
        """Test abatch delegates to batch and preserves result order."""
        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "Hello, world!"},
                                "role": "USER",
                            }
                        }
                    ],
                    "eventTimestamp": datetime(
                        2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc
                    ),
                }
            ]
        }
        mock_boto_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [sample_memory_record]
        }
        mock_boto_client.create_event.return_value = {"event": {"eventId": "event-123"}}

        ops = [
            GetOp(namespace=("test_actor", "test_session"), key="test-key"),
            PutOp(
                namespace=("test_actor", "test_session"),
                key="new-key",
                value=sample_item_data,
            ),
            SearchOp(namespace_prefix=("test_actor",), query="test query", limit=5),
            ListNamespacesOp(limit=10),
        ]
        results = await store.abatch(ops)

        assert len(results) == 4
        assert isinstance(results[0], Item)  # GetOp
        assert results[1] is None  # PutOp
        assert isinstance(results[2], list)  # SearchOp
        assert results[3] == []  # ListNamespacesOp
        mock_boto_client.create_event.assert_called_once()

    async def test_abatch_consumes_lazy_iterable(self, store, mock_boto_client):
        """Test abatch materializes generator ops before the executor thread."""
        mock_boto_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": []
        }

        ops = (
            SearchOp(namespace_prefix=("test_actor",), query="test", limit=5)
            for _ in range(2)
        )
        results = await store.abatch(ops)

        assert results == [[], []]

    async def test_async_convenience_methods_route_through_abatch(
        self, store, mock_boto_client, sample_memory_record, sample_item_data
    ):
        """Test BaseStore's aget/aput/asearch work against the sync client."""
        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "Hello, world!"},
                                "role": "USER",
                            }
                        }
                    ],
                    "eventTimestamp": datetime(
                        2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc
                    ),
                }
            ]
        }
        mock_boto_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [sample_memory_record]
        }
        mock_boto_client.create_event.return_value = {"event": {"eventId": "event-123"}}

        await store.aput(("test_actor", "test_session"), "key1", sample_item_data)
        mock_boto_client.create_event.assert_called_once()

        item = await store.aget(("test_actor", "test_session"), "test-key")
        assert isinstance(item, Item)

        found = await store.asearch(("test_actor",), query="test query", limit=5)
        assert len(found) == 1
        assert isinstance(found[0], SearchItem)

    async def test_abatch_does_not_block_event_loop(self, store, mock_boto_client):
        """Test the event loop keeps running while abatch waits on the client."""
        call_duration = 0.2

        def slow_search(**kwargs):
            time.sleep(call_duration)
            return {"memoryRecordSummaries": []}

        mock_boto_client.retrieve_memory_records.side_effect = slow_search
        heartbeats = 0

        async def heartbeat():
            nonlocal heartbeats
            while True:
                await asyncio.sleep(0.01)
                heartbeats += 1

        ticker = asyncio.create_task(heartbeat())
        try:
            await store.abatch(
                [SearchOp(namespace_prefix=("test_actor",), query="q", limit=1)]
            )
        finally:
            ticker.cancel()

        assert heartbeats >= 5

    async def test_abatch_runs_ops_concurrently(self, store, mock_boto_client):
        """Test concurrent abatch calls overlap instead of serializing."""
        call_duration = 0.2
        n_calls = 4

        def slow_search(**kwargs):
            time.sleep(call_duration)
            return {"memoryRecordSummaries": []}

        mock_boto_client.retrieve_memory_records.side_effect = slow_search
        op = SearchOp(namespace_prefix=("test_actor",), query="q", limit=1)

        start = time.perf_counter()
        await asyncio.gather(*(store.abatch([op]) for _ in range(n_calls)))
        elapsed = time.perf_counter() - start

        assert elapsed < (n_calls * call_duration) * 0.75

    async def test_abatch_propagates_validation_errors(self, store):
        """Test ValueErrors raised on the executor thread reach the awaiter."""
        with pytest.raises(ValueError, match="'message' key"):
            await store.aput(
                ("test_actor", "test_session"), "k", {"message": "not a message"}
            )

        with pytest.raises(ValueError, match="tuple of \\(actor_id, session_id\\)"):
            await store.aput(("only_actor",), "k", {"message": HumanMessage("hi")})

    async def test_abatch_propagates_client_errors(self, store, mock_boto_client):
        """Test botocore ClientErrors surface unchanged through abatch."""
        mock_boto_client.list_events.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "ListEvents",
        )

        with pytest.raises(ClientError) as exc_info:
            await store.aget(("test_actor", "test_session"), "test-key")

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"

    def test_convert_memory_records_to_search_items(self, store, sample_memory_record):
        """Test conversion of memory records to SearchItem objects."""
        namespace = ("test_actor",)

        search_items = store._convert_memory_records_to_search_items(
            [sample_memory_record], namespace
        )

        assert len(search_items) == 1
        search_item = search_items[0]
        assert isinstance(search_item, SearchItem)
        assert search_item.namespace == namespace
        assert search_item.key == sample_memory_record["memoryRecordId"]
        assert search_item.value["content"] == "Hello, world!"
        assert search_item.score == 0.95
        assert isinstance(search_item.created_at, datetime)
        assert isinstance(search_item.updated_at, datetime)

    def test_convert_namespace_to_string(self, store):
        """Test namespace tuple to string conversion."""
        # Test empty namespace
        assert store._convert_namespace_to_string(()) == "/"

        # Test single element
        assert store._convert_namespace_to_string(("actor",)) == "/actor"

        # Test multiple elements
        assert (
            store._convert_namespace_to_string(("actor", "session")) == "/actor/session"
        )

    def test_error_handling_preserves_exceptions(self, store, mock_boto_client):
        """Test that boto3 exceptions are properly preserved."""
        mock_boto_client.create_event.side_effect = ClientError(
            error_response={
                "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
            },
            operation_name="CreateEvent",
        )

        ops = [
            PutOp(
                namespace=("test_actor", "test_session"),
                key="test-key",
                value={"message": HumanMessage(content="test")},
            )
        ]

        with pytest.raises(ClientError) as exc_info:
            store.batch(ops)

        assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"

    def test_memory_record_id_format_validation(self, store):
        """Test that _is_memory_record_id matches real memoryRecordIds."""
        from langgraph_checkpoint_aws.store.agentcore.store import (
            _is_memory_record_id,
        )

        test_id = "mem-test-record-1234567890123456789012345"
        assert test_id.startswith("mem-")
        assert len(test_id) >= 40  # Minimum length requirement
        assert all(
            c.isalnum() or c in "-_" for c in test_id[4:]
        )  # Valid characters after prefix
        assert _is_memory_record_id(test_id)
        # A plain put()-style key must not be mistaken for a memoryRecordId
        assert not _is_memory_record_id("test-key")

    def test_namespace_handling_edge_cases(
        self, store, mock_boto_client, sample_memory_record
    ):
        """Test namespace handling with various edge cases."""
        # Test with single element namespace
        ops = [SearchOp(namespace_prefix=("single_actor",), query="test", limit=5)]

        mock_boto_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [sample_memory_record]
        }

        store.batch(ops)

        call_args = mock_boto_client.retrieve_memory_records.call_args[1]
        assert call_args["namespacePath"] == "/single_actor"

    def test_search_with_pagination_parameters(
        self, store, mock_boto_client, sample_memory_record
    ):
        """Test SearchOp with offset parameter."""
        mock_boto_client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [sample_memory_record]
        }

        ops = [
            SearchOp(
                namespace_prefix=("test_actor",),
                query="test query",
                limit=10,
                offset=5,  # This should be ignored in current implementation
            )
        ]
        results = store.batch(ops)

        # Offset is currently ignored, but operation should still work
        assert len(results) == 1
        assert len(results[0]) == 1

    def test_put_op_invalid_namespace(self, store, mock_boto_client):
        """Test PutOp with invalid namespace length."""
        ops = [
            PutOp(
                namespace=("single_element",),  # Should be (actor_id, session_id)
                key="test-key",
                value={"message": HumanMessage(content="test")},
            )
        ]

        with pytest.raises(
            ValueError, match="Namespace must be a tuple of \\(actor_id, session_id\\)"
        ):
            store.batch(ops)

    def test_parse_timestamp_string_with_z(self, store):
        """Test _parse_timestamp with ISO string ending in Z."""
        result = store._parse_timestamp("2024-01-01T12:00:00Z")
        assert result == datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_parse_timestamp_datetime_passthrough(self, store):
        """Test _parse_timestamp returns the same datetime object."""
        dt = datetime(2025, 6, 15, 9, 30, 0, tzinfo=timezone.utc)
        result = store._parse_timestamp(dt)
        assert result is dt  # identity: same object returned, no copy

    def test_parse_invalid_timestamp_returns_now(self, store):
        """Test _parse_timestamp with invalid inputs returns a UTC datetime."""
        result_none = store._parse_timestamp(None)
        result_invalid = store._parse_timestamp("not-a-date")
        assert isinstance(result_none, datetime)
        assert result_none.tzinfo == timezone.utc
        assert isinstance(result_invalid, datetime)
        assert result_invalid.tzinfo == timezone.utc

    def test_put_get_roundtrip(self, store, mock_boto_client):
        """Regression test for #708: put→get round-trip returns stored value."""
        mock_boto_client.create_event.return_value = {
            "event": {"eventId": "event-123", "memoryId": "test-memory-id"}
        }

        put_op = PutOp(
            namespace=("user1", "session1"),
            key="k1",
            value={"message": HumanMessage(content="I love hot chocolate!")},
        )
        store.batch([put_op])

        call_args = mock_boto_client.create_event.call_args[1]
        assert call_args["metadata"] == {
            STORE_KEY_METADATA: {"stringValue": _store_key_metadata_value("k1")}
        }

        event_timestamp = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "I love hot chocolate!"},
                                "role": "USER",
                            }
                        }
                    ],
                    "eventTimestamp": event_timestamp,
                }
            ]
        }

        get_op = GetOp(namespace=("user1", "session1"), key="k1")
        results = store.batch([get_op])

        assert len(results) == 1
        item = results[0]
        assert isinstance(item, Item)
        assert item.key == "k1"
        assert "I love hot chocolate!" in item.value["content"]
        assert item.namespace == ("user1", "session1")

    def test_get_op_with_memory_record_id_key_two_part_namespace(
        self, store, mock_boto_client, sample_memory_record
    ):
        """Regression test: get() on a search()-returned key (a memoryRecordId)
        must look up the memory record directly instead of treating the key as
        a put()-originated key, even for a 2-part namespace."""
        mock_boto_client.get_memory_record.return_value = {
            "memoryRecord": sample_memory_record
        }
        memory_record_id = sample_memory_record["memoryRecordId"]

        ops = [GetOp(namespace=("facts", "actor-1"), key=memory_record_id)]
        results = store.batch(ops)

        assert len(results) == 1
        item = results[0]
        assert isinstance(item, Item)
        assert item.key == memory_record_id
        assert item.namespace == ("facts", "actor-1")
        assert item.value["content"] == "Hello, world!"
        mock_boto_client.get_memory_record.assert_called_once_with(
            memoryId="test-memory-id",
            memoryRecordId=memory_record_id,
        )
        mock_boto_client.list_events.assert_not_called()

    def test_get_op_with_memory_record_id_key_multi_part_namespace(
        self, store, mock_boto_client, sample_memory_record
    ):
        """Regression test: a >2-part namespace must not raise ValueError when
        the key is a memoryRecordId, since that path never unpacks the
        namespace into (actor_id, session_id)."""
        mock_boto_client.get_memory_record.return_value = {
            "memoryRecord": sample_memory_record
        }
        memory_record_id = sample_memory_record["memoryRecordId"]

        ops = [
            GetOp(namespace=("summaries", "actors", "actor-1"), key=memory_record_id)
        ]
        results = store.batch(ops)

        assert len(results) == 1
        item = results[0]
        assert isinstance(item, Item)
        assert item.namespace == ("summaries", "actors", "actor-1")
        mock_boto_client.get_memory_record.assert_called_once_with(
            memoryId="test-memory-id",
            memoryRecordId=memory_record_id,
        )

    def test_get_op_with_memory_record_id_not_found(self, store, mock_boto_client):
        """get_memory_record ResourceNotFoundException is treated as absent."""
        mock_boto_client.get_memory_record.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFoundException"}},
            operation_name="GetMemoryRecord",
        )
        memory_record_id = "mem-missing-record-123456789012345678901234567"

        ops = [GetOp(namespace=("facts", "actor-1"), key=memory_record_id)]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] is None

    def test_get_op_with_memory_record_id_other_error(self, store, mock_boto_client):
        """Only ResourceNotFoundException is treated as a missing key; any
        other get_memory_record error (e.g. access denied, throttling) must
        propagate to the caller instead of being swallowed as None."""
        mock_boto_client.get_memory_record.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDeniedException"}},
            operation_name="GetMemoryRecord",
        )
        memory_record_id = "mem-denied-record-1234567890123456789012345678"

        ops = [GetOp(namespace=("facts", "actor-1"), key=memory_record_id)]
        with pytest.raises(ClientError):
            store.batch(ops)

    def test_batch_get_op_no_events_returns_none(self, store, mock_boto_client):
        """Test GetOp returns None when list_events returns empty events list."""
        mock_boto_client.list_events.return_value = {"events": []}

        ops = [GetOp(namespace=("test_actor", "test_session"), key="absent-key")]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] is None

    def test_batch_get_op_selects_latest_event(self, store, mock_boto_client):
        """Test GetOp selects the most recent event (last-write-wins)."""
        older_timestamp = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        newer_timestamp = datetime(2024, 1, 2, 15, 0, 0, tzinfo=timezone.utc)

        mock_boto_client.list_events.return_value = {
            "events": [
                {
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "old message"},
                                "role": "USER",
                            }
                        }
                    ],
                    "eventTimestamp": newer_timestamp,
                },
                {
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "even older message"},
                                "role": "USER",
                            }
                        }
                    ],
                    "eventTimestamp": older_timestamp,
                },
            ]
        }

        ops = [GetOp(namespace=("test_actor", "test_session"), key="shared-key")]
        results = store.batch(ops)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, Item)
        assert result.value["content"] == "old message"
        assert result.key == "shared-key"

    def test_batch_get_op_selects_latest_event_across_pages(
        self,
        store: AgentCoreMemoryStore,
        mock_boto_client: MagicMock,
    ) -> None:
        """Test GetOp selects the newest event across all response pages."""
        older_timestamp = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        newer_timestamp = datetime(2024, 1, 2, 15, 0, 0, tzinfo=timezone.utc)
        mock_boto_client.list_events.side_effect = [
            {
                "events": [
                    {
                        "payload": [
                            {
                                "conversational": {
                                    "content": {"text": "older page"},
                                    "role": "USER",
                                }
                            }
                        ],
                        "eventTimestamp": older_timestamp,
                    }
                ],
                "nextToken": "page-2",
            },
            {
                "events": [
                    {
                        "payload": [
                            {
                                "conversational": {
                                    "content": {"text": "newer page"},
                                    "role": "USER",
                                }
                            }
                        ],
                        "eventTimestamp": newer_timestamp,
                    }
                ]
            },
        ]

        results = store.batch(
            [GetOp(namespace=("test_actor", "test_session"), key="shared-key")]
        )

        result = results[0]
        assert isinstance(result, Item)
        assert result.value["content"] == "newer page"
        assert mock_boto_client.list_events.call_count == 2
        first_call, second_call = mock_boto_client.list_events.call_args_list
        assert "nextToken" not in first_call.kwargs
        assert second_call.kwargs["nextToken"] == "page-2"

    def test_get_op_invalid_namespace(self, store, mock_boto_client):
        """Test GetOp with invalid namespace raises ValueError."""
        ops = [GetOp(namespace=("single_element",), key="test-key")]

        with pytest.raises(
            ValueError, match="Namespace must be a tuple of \\(actor_id, session_id\\)"
        ):
            store.batch(ops)

    def test_convert_event_to_item_single_payload(self, store):
        """Test _convert_event_to_item with a normal single-entry payload."""
        event = {
            "payload": [
                {
                    "conversational": {
                        "content": {"text": "Hello there"},
                        "role": "USER",
                    }
                }
            ],
            "eventTimestamp": datetime(2024, 3, 10, 8, 0, 0, tzinfo=timezone.utc),
        }

        item = store._convert_event_to_item(event, ("actor1", "sess1"), "my-key")

        assert isinstance(item, Item)
        assert item.key == "my-key"
        assert item.namespace == ("actor1", "sess1")
        assert item.value["content"] == "Hello there"
        assert item.value["messages"] == [{"text": "Hello there", "role": "USER"}]
        assert item.created_at == datetime(2024, 3, 10, 8, 0, 0, tzinfo=timezone.utc)
        assert item.updated_at == datetime(2024, 3, 10, 8, 0, 0, tzinfo=timezone.utc)

    def test_convert_event_to_item_multiple_payloads(self, store):
        """Test _convert_event_to_item joins multiple payload entries."""
        event = {
            "payload": [
                {
                    "conversational": {
                        "content": {"text": "First line"},
                        "role": "USER",
                    }
                },
                {
                    "conversational": {
                        "content": {"text": "Second line"},
                        "role": "ASSISTANT",
                    }
                },
            ],
            "eventTimestamp": datetime(2024, 5, 1, 12, 0, 0, tzinfo=timezone.utc),
        }

        item = store._convert_event_to_item(event, ("a", "b"), "k2")

        assert item.value["content"] == "First line\nSecond line"
        assert item.value["messages"] == [
            {"text": "First line", "role": "USER"},
            {"text": "Second line", "role": "ASSISTANT"},
        ]

    def test_convert_event_to_item_empty_payload(self, store):
        """Test _convert_event_to_item with an empty payload list."""
        event = {
            "payload": [],
            "eventTimestamp": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        }

        item = store._convert_event_to_item(event, ("x", "y"), "k3")

        assert item.value["content"] == ""
        assert item.value["messages"] == []
