"""
Unit tests for AgentCore Memory Store.
"""

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

from langgraph_checkpoint_aws.agentcore.store import AgentCoreMemoryStore


class TestAgentCoreMemoryStore:
    """Test suite for AgentCoreMemoryStore."""

    @pytest.fixture
    def mock_boto_client(self):
        """Mock boto3 client with all required methods."""
        mock_client = Mock()
        mock_client.create_event = MagicMock()
        mock_client.retrieve_memory_records = MagicMock()
        mock_client.get_memory_record = MagicMock()
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
            "memoryRecordId": "mem-test-record-12345678901234567890123456789012345",
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

    def test_batch_get_op_success(self, store, mock_boto_client, sample_memory_record):
        """Test successful GetOp operation."""
        mock_boto_client.get_memory_record.return_value = {
            "memoryRecord": sample_memory_record
        }

        ops = [GetOp(namespace=("test_actor", "test_session"), key="test-key")]
        results = store.batch(ops)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, Item)
        assert result.namespace == ("test_actor", "test_session")
        assert result.key == sample_memory_record["memoryRecordId"]
        assert result.value["content"] == "Hello, world!"

        mock_boto_client.get_memory_record.assert_called_once_with(
            memoryId="test-memory-id", memoryRecordId="test-key"
        )

    def test_batch_get_op_not_found(self, store, mock_boto_client):
        """Test GetOp when record not found."""
        mock_boto_client.get_memory_record.side_effect = ClientError(
            error_response={"Error": {"Code": "ResourceNotFoundException"}},
            operation_name="GetMemoryRecord",
        )

        ops = [GetOp(namespace=("test_actor", "test_session"), key="nonexistent")]
        results = store.batch(ops)

        assert len(results) == 1
        assert results[0] is None

    def test_batch_get_op_other_error(self, store, mock_boto_client):
        """Test GetOp with other boto3 errors."""
        mock_boto_client.get_memory_record.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDeniedException"}},
            operation_name="GetMemoryRecord",
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
            namespace="/test_actor",
            searchCriteria={"searchQuery": "test query", "topK": 5},
            maxResults=5,
        )

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
        mock_boto_client.get_memory_record.return_value = {
            "memoryRecord": sample_memory_record
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

    def test_abatch_not_implemented(self, store):
        """Test that abatch raises NotImplementedError."""
        ops = [SearchOp(namespace_prefix=("test_actor",), query="test", limit=5)]

        # Use asyncio to test async method
        import asyncio

        async def test_async():
            with pytest.raises(NotImplementedError):
                await store.abatch(ops)

        asyncio.run(test_async())

    def test_convert_memory_record_to_item(self, store, sample_memory_record):
        """Test conversion of memory record to Item."""
        namespace = ("test_actor", "test_session")

        item = store._convert_memory_record_to_item(sample_memory_record, namespace)

        assert isinstance(item, Item)
        assert item.namespace == namespace
        assert item.key == sample_memory_record["memoryRecordId"]
        assert item.value["content"] == "Hello, world!"
        assert item.value["memory_strategy_id"] == "strategy-123"
        assert isinstance(item.created_at, datetime)
        assert isinstance(item.updated_at, datetime)

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
        """Test that memory record IDs follow the required format."""
        # This is more of a documentation test to ensure we understand the format
        test_id = "mem-test-record-12345678901234567890123456789012345"
        assert test_id.startswith("mem-")
        assert len(test_id) >= 40  # Minimum length requirement
        assert all(
            c.isalnum() or c in "-_" for c in test_id[4:]
        )  # Valid characters after prefix

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
        assert call_args["namespace"] == "/single_actor"

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
