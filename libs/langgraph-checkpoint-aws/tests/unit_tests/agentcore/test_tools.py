"""
Unit tests for AgentCore Memory Store tools.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.store.base import SearchItem

from langgraph_checkpoint_aws.agentcore.tools import (
    NamespaceTemplate,
    create_search_memory_tool,
    create_store_event_tool,
)


class TestNamespaceTemplate:
    """Test suite for NamespaceTemplate."""

    def test_init_with_string(self):
        """Test initialization with string namespace."""
        template = NamespaceTemplate("test_namespace")
        assert template.namespace_parts == ("test_namespace",)

    def test_init_with_tuple(self):
        """Test initialization with tuple namespace."""
        template = NamespaceTemplate(("part1", "part2", "part3"))
        assert template.namespace_parts == ("part1", "part2", "part3")

    def test_call_with_static_namespace(self):
        """Test calling template with static namespace (no placeholders)."""
        template = NamespaceTemplate(("static", "namespace"))
        # Provide a config to avoid runtime context error
        config = {"configurable": {}}
        result = template(config)
        assert result == ("static", "namespace")

    def test_call_with_placeholders(self):
        """Test calling template with placeholders."""
        template = NamespaceTemplate(("facts", "{actor_id}"))
        config = {"configurable": {"actor_id": "user123"}}
        result = template(config)
        assert result == ("facts", "user123")

    def test_call_with_multiple_placeholders(self):
        """Test calling template with multiple placeholders."""
        template = NamespaceTemplate(("{actor_id}", "{thread_id}", "memories"))
        config = {"configurable": {"actor_id": "user123", "thread_id": "thread456"}}
        result = template(config)
        assert result == ("user123", "thread456", "memories")

    def test_call_missing_placeholder_value(self):
        """Test calling template with missing placeholder value."""
        template = NamespaceTemplate(("facts", "{actor_id}"))
        config = {"configurable": {}}  # Missing actor_id

        with pytest.raises(ValueError, match="Missing required configurable key"):
            template(config)

    @patch("langgraph_checkpoint_aws.agentcore.tools.get_config")
    def test_call_without_config_uses_get_config(self, mock_get_config):
        """Test calling template without config uses get_config."""
        # Test successful get_config call
        mock_get_config.return_value = {"configurable": {"actor_id": "user123"}}
        template = NamespaceTemplate(("facts", "{actor_id}"))
        result = template()
        assert result == ("facts", "user123")
        mock_get_config.assert_called_once()

        # Test RuntimeError handling - returns template as-is
        mock_get_config.reset_mock()
        mock_get_config.side_effect = RuntimeError("Not in runnable context")
        template = NamespaceTemplate(("facts", "{actor_id}"))
        result = template()
        assert result == ("facts", "{actor_id}")  # Returns template unchanged
        mock_get_config.assert_called_once()


class TestCreateSearchMemoryTool:
    """Test suite for create_search_memory_tool."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock store with search capabilities."""
        store = Mock()
        store.search = Mock()
        store.asearch = AsyncMock()
        return store

    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results."""
        return [
            SearchItem(
                namespace=("facts", "user123"),
                key="mem-123",
                value={"content": "User likes coffee"},
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
                score=0.95,
            ),
            SearchItem(
                namespace=("facts", "user123"),
                key="mem-456",
                value={"content": "User is allergic to peanuts"},
                created_at="2024-01-02T00:00:00Z",
                updated_at="2024-01-02T00:00:00Z",
                score=0.87,
            ),
        ]

    def test_create_search_tool_basic(self, mock_store):
        """Test creating a basic search tool."""
        tool = create_search_memory_tool(
            namespace=("facts", "{actor_id}"),
            store=mock_store,
        )

        assert isinstance(tool, StructuredTool)
        assert tool.name == "search_memory"
        assert "Search AgentCore Memory" in tool.description

    def test_search_tool_sync_invocation(self, mock_store, sample_search_results):
        """Test synchronous search tool invocation."""
        mock_store.search.return_value = sample_search_results

        tool = create_search_memory_tool(
            namespace=("facts", "user123"),
            store=mock_store,
        )

        result = tool.invoke({"query": "user preferences"})

        assert "User likes coffee" in result
        assert "User is allergic to peanuts" in result
        assert "relevance: 0.95" in result
        assert "relevance: 0.87" in result

        mock_store.search.assert_called_once_with(
            ("facts", "user123"),
            query="user preferences",
            filter=None,
            limit=10,
            offset=0,
        )

    @pytest.mark.asyncio
    async def test_search_tool_async_invocation(
        self, mock_store, sample_search_results
    ):
        """Test asynchronous search tool invocation."""
        mock_store.asearch.return_value = sample_search_results

        tool = create_search_memory_tool(
            namespace=("facts", "user123"),
            store=mock_store,
        )

        result = await tool.ainvoke({"query": "user preferences", "limit": 5})

        assert "User likes coffee" in result
        mock_store.asearch.assert_called_once_with(
            ("facts", "user123"),
            query="user preferences",
            filter=None,
            limit=5,
            offset=0,
        )

    def test_search_tool_with_content_and_artifact(
        self, mock_store, sample_search_results
    ):
        """Test search tool with content_and_artifact response format."""
        mock_store.search.return_value = sample_search_results

        tool = create_search_memory_tool(
            namespace=("facts", "user123"),
            store=mock_store,
            response_format="content_and_artifact",
        )

        # When response_format is "content_and_artifact", the tool returns a tuple
        result = tool.invoke({"query": "test"})

        # The result should be a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        content, artifacts = result
        assert isinstance(content, str)
        assert artifacts == sample_search_results

    def test_search_tool_empty_results(self, mock_store):
        """Test search tool with empty results."""
        mock_store.search.return_value = []

        tool = create_search_memory_tool(
            namespace=("facts", "user123"),
            store=mock_store,
        )

        result = tool.invoke({"query": "nonexistent"})
        assert result == "No memories found."

    def test_search_tool_with_filter(self, mock_store, sample_search_results):
        """Test search tool with filter parameter."""
        mock_store.search.return_value = sample_search_results

        tool = create_search_memory_tool(
            namespace=("facts", "user123"),
            store=mock_store,
        )

        filter_dict = {"category": "preferences"}
        tool.invoke({"query": "test", "filter": filter_dict})

        mock_store.search.assert_called_once_with(
            ("facts", "user123"),
            query="test",
            filter=filter_dict,
            limit=10,
            offset=0,
        )

    def test_search_tool_with_runtime_namespace(self, mock_store):
        """Test search tool with runtime namespace resolution."""
        mock_store.search.return_value = []

        tool = create_search_memory_tool(
            namespace=("facts", "{actor_id}"),
            store=mock_store,
        )

        with patch(
            "langgraph_checkpoint_aws.agentcore.tools.get_config"
        ) as mock_get_config:
            mock_get_config.return_value = {"configurable": {"actor_id": "user456"}}

            tool.invoke({"query": "test"})

            call_args = mock_store.search.call_args
            assert call_args[0][0] == ("facts", "user456")


class TestCreateStoreEventTool:
    """Test suite for create_store_event_tool."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock store."""
        store = Mock()
        store.put = Mock()
        store.aput = AsyncMock()
        return store

    def test_create_store_event_tool(self, mock_store):
        """Test creating a store event tool."""
        tool = create_store_event_tool(store=mock_store)

        assert isinstance(tool, StructuredTool)
        assert tool.name == "store_conversation_event"
        assert "Store a conversation event" in tool.description

    def test_store_event_sync_invocation(self, mock_store):
        """Test synchronous store event invocation."""
        tool = create_store_event_tool(store=mock_store)

        message = HumanMessage(content="Test message")
        result = tool.invoke(
            {"message": message, "actor_id": "user123", "session_id": "session456"}
        )

        assert "Stored conversation event" in result
        mock_store.put.assert_called_once()
        call_args = mock_store.put.call_args
        assert call_args[0][0] == ("user123", "session456")  # namespace
        assert call_args[0][2]["message"] == message  # value

    @pytest.mark.asyncio
    async def test_store_event_async_invocation(self, mock_store):
        """Test asynchronous store event invocation."""
        tool = create_store_event_tool(store=mock_store)

        message = AIMessage(content="AI response")
        result = await tool.ainvoke(
            {"message": message, "actor_id": "user123", "session_id": "session456"}
        )

        assert "Stored conversation event" in result
        mock_store.aput.assert_called_once()

    def test_store_event_with_custom_name(self, mock_store):
        """Test store event tool with custom name."""
        tool = create_store_event_tool(store=mock_store, name="custom_store_tool")

        assert tool.name == "custom_store_tool"


class TestGetStore:
    """Test suite for _get_store helper function."""

    def test_get_store_with_provided_store(self):
        """Test _get_store returns provided store."""
        from langgraph_checkpoint_aws.agentcore.tools import _get_store

        mock_store = Mock()
        result = _get_store(mock_store)
        assert result == mock_store

    @patch("langgraph_checkpoint_aws.agentcore.tools.get_store")
    def test_get_store_without_provided_store(self, mock_get_store):
        """Test _get_store uses get_store when no store provided."""
        from langgraph_checkpoint_aws.agentcore.tools import _get_store

        mock_store = Mock()
        mock_get_store.return_value = mock_store

        result = _get_store(None)
        assert result == mock_store
        mock_get_store.assert_called_once()

    @patch("langgraph_checkpoint_aws.agentcore.tools.get_store")
    def test_get_store_runtime_error(self, mock_get_store):
        """Test _get_store handles RuntimeError."""
        from langgraph_checkpoint_aws.agentcore.tools import _get_store

        mock_get_store.side_effect = RuntimeError("No store configured")

        with pytest.raises(RuntimeError, match="Could not get store"):
            _get_store(None)


class TestFormatSearchResults:
    """Test suite for _format_search_results helper function."""

    def test_format_empty_results(self):
        """Test formatting empty search results."""
        from langgraph_checkpoint_aws.agentcore.tools import _format_search_results

        result = _format_search_results([])
        assert result == "No memories found."

    def test_format_single_result(self):
        """Test formatting single search result."""
        from langgraph_checkpoint_aws.agentcore.tools import _format_search_results

        memory = SearchItem(
            namespace=("facts", "user123"),
            key="mem-123",
            value={"content": "Test content"},
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            score=0.95,
        )

        result = _format_search_results([memory])
        assert "1. Test content" in result
        assert "relevance: 0.95" in result
        assert "id: mem-123" in result

    def test_format_multiple_results(self):
        """Test formatting multiple search results."""
        from langgraph_checkpoint_aws.agentcore.tools import _format_search_results

        memories = [
            SearchItem(
                namespace=("facts", "user123"),
                key="mem-123",
                value={"content": "First memory"},
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
                score=0.95,
            ),
            SearchItem(
                namespace=("facts", "user123"),
                key="mem-456",
                value={"content": "Second memory"},
                created_at="2024-01-02T00:00:00Z",
                updated_at="2024-01-02T00:00:00Z",
                score=None,  # No score
            ),
        ]

        result = _format_search_results(memories)
        assert "1. First memory" in result
        assert "2. Second memory" in result
        assert "relevance: 0.95" in result
        assert "relevance:" not in result.split("\n")[1]  # No score for second item
