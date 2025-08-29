from unittest.mock import Mock
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from langchain_aws.memory.bedrock_agentcore import (
    store_agentcore_memory_events,
    list_agentcore_memory_events,
    retrieve_agentcore_memories,
    create_store_memory_events_tool,
    create_list_memory_events_tool,
    create_retrieve_memory_tool,
)


@pytest.fixture
def mock_memory_client():
    client = Mock()
    # Set up realistic mock responses
    client.create_event.return_value = {"eventId": "test-event-123"}
    client.list_events.return_value = [
        {
            "eventId": "event-1",
            "payload": [
                {"conversational": {"role": "USER", "content": {"text": "Hello"}}}
            ],
        }
    ]
    client.retrieve_memories.return_value = [
        {
            "content": {"text": "User likes coffee"},
            "score": 0.95,
            "metadata": {"category": "preferences"},
        }
    ]
    return client


@pytest.mark.compile
def test_agentcore_memory_integration_workflow(mock_memory_client):
    """Test the complete workflow of storing, listing, and retrieving memories."""
    # Test storing messages
    messages = [HumanMessage("I love coffee"), AIMessage("Great! I'll remember that.")]

    event_id = store_agentcore_memory_events(
        mock_memory_client,
        messages=messages,
        memory_id="test-memory",
        actor_id="user-1",
        session_id="session-1",
    )

    assert event_id == "test-event-123"
    mock_memory_client.create_event.assert_called_once()

    # Test listing messages
    retrieved_messages = list_agentcore_memory_events(
        mock_memory_client,
        memory_id="test-memory",
        actor_id="user-1",
        session_id="session-1",
    )

    assert len(retrieved_messages) == 1
    assert isinstance(retrieved_messages[0], HumanMessage)

    # Test memory search
    memories = retrieve_agentcore_memories(
        mock_memory_client,
        memory_id="test-memory",
        namespace_str="/preferences/user-1",
        query="coffee preferences",
    )

    assert len(memories) == 1
    assert memories[0]["content"] == "User likes coffee"


@pytest.mark.compile
def test_tool_creation_integration(mock_memory_client):
    """Test that the tool factory functions create working tools."""
    store_tool = create_store_memory_events_tool(
        mock_memory_client, "test-memory", "user-1", "session-1"
    )

    list_tool = create_list_memory_events_tool(
        mock_memory_client, "test-memory", "user-1", "session-1"
    )

    search_tool = create_retrieve_memory_tool(
        mock_memory_client, "test-memory", "/preferences"
    )

    assert store_tool.name == "store_memory_events"
    assert list_tool.name == "list_memory_events"
    assert search_tool.name == "retrieve_memory"

    messages = [HumanMessage("Test message")]
    result = store_tool.invoke({"messages": messages})
    assert result == "test-event-123"
