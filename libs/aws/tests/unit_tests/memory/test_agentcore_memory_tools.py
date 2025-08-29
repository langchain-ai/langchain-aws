from unittest.mock import MagicMock
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_aws.memory.bedrock_agentcore import (
    create_store_memory_events_tool,
    create_list_memory_events_tool,
    create_retrieve_memory_tool,
)


@pytest.fixture
def mock_memory_client():
    client = MagicMock()
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


def test_create_store_memory_events_tool(mock_memory_client):
    tool = create_store_memory_events_tool(
        mock_memory_client, "test-memory", "user-1", "session-1"
    )

    assert tool.name == "store_memory_events"
    assert "Store conversation messages" in tool.description

    messages = [HumanMessage("Test message")]
    result = tool.invoke({"messages": messages})

    assert result == "test-event-123"
    mock_memory_client.create_event.assert_called_once_with(
        memory_id="test-memory",
        actor_id="user-1",
        session_id="session-1",
        messages=[("Test message", "USER")],
    )


def test_create_list_memory_events_tool(mock_memory_client):
    tool = create_list_memory_events_tool(
        mock_memory_client, "test-memory", "user-1", "session-1"
    )

    assert tool.name == "list_memory_events"
    assert "Retrieve recent conversation messages" in tool.description

    result = tool.invoke({"max_results": 50})

    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello"

    mock_memory_client.list_events.assert_called_once_with(
        memory_id="test-memory",
        actor_id="user-1",
        session_id="session-1",
        max_results=50,
        include_payload=True,
    )


def test_create_list_memory_events_tool_default_max_results(mock_memory_client):
    tool = create_list_memory_events_tool(
        mock_memory_client, "test-memory", "user-1", "session-1"
    )

    tool.invoke({})

    mock_memory_client.list_events.assert_called_once_with(
        memory_id="test-memory",
        actor_id="user-1",
        session_id="session-1",
        max_results=100,
        include_payload=True,
    )


def test_create_retrieve_memory_tool_default_params(mock_memory_client):
    tool = create_retrieve_memory_tool(
        mock_memory_client, "test-memory", "/summaries/actor-1/session-1"
    )

    assert tool.name == "retrieve_memory"
    assert "Search for relevant memories" in tool.description

    result = tool.invoke({"query": "coffee preferences", "limit": 5})

    assert len(result) == 1
    assert result[0]["content"] == "User likes coffee"
    assert result[0]["score"] == 0.95

    mock_memory_client.retrieve_memories.assert_called_once_with(
        memory_id="test-memory",
        namespace="/summaries/actor-1/session-1",
        query="coffee preferences",
        top_k=5,
    )


def test_create_retrieve_memory_tool_custom_params(mock_memory_client):
    tool = create_retrieve_memory_tool(
        mock_memory_client,
        "test-memory",
        "/summaries/actor-1/session-1",
        tool_name="search_user_preferences",
        tool_description="Search for user preferences",
    )

    assert tool.name == "search_user_preferences"
    assert tool.description == "Search for user preferences"

    tool.invoke({"query": "food preferences", "limit": 3})

    mock_memory_client.retrieve_memories.assert_called_once_with(
        memory_id="test-memory",
        namespace="/summaries/actor-1/session-1",
        query="food preferences",
        top_k=3,
    )


def test_create_retrieve_memory_tool_default_limit(mock_memory_client):
    tool = create_retrieve_memory_tool(
        mock_memory_client, "test-memory", "/summaries/actor-1/session-1"
    )

    tool.invoke({"query": "test query"})

    mock_memory_client.retrieve_memories.assert_called_once_with(
        memory_id="test-memory",
        namespace="/summaries/actor-1/session-1",
        query="test query",
        top_k=3,
    )


def test_store_tool_with_multiple_messages(mock_memory_client):
    tool = create_store_memory_events_tool(
        mock_memory_client, "test-memory", "user-1", "session-1"
    )

    messages = [
        HumanMessage("Hello"),
        AIMessage("Hi there"),
        ToolMessage("Tool result", tool_call_id="123"),
    ]

    result = tool.invoke({"messages": messages})

    assert result == "test-event-123"
    mock_memory_client.create_event.assert_called_once_with(
        memory_id="test-memory",
        actor_id="user-1",
        session_id="session-1",
        messages=[
            ("Hello", "USER"),
            ("Hi there", "ASSISTANT"),
            ("Tool result", "TOOL"),
        ],
    )


def test_tools_handle_client_errors(mock_memory_client):
    mock_memory_client.create_event.side_effect = Exception("AWS Error")
    mock_memory_client.list_events.side_effect = Exception("AWS Error")
    mock_memory_client.retrieve_memories.side_effect = Exception("AWS Error")

    store_tool = create_store_memory_events_tool(
        mock_memory_client, "test-memory", "user-1", "session-1"
    )
    list_tool = create_list_memory_events_tool(
        mock_memory_client, "test-memory", "user-1", "session-1"
    )
    retrieve_tool = create_retrieve_memory_tool(
        mock_memory_client, "test-memory", "/summaries/actor-1/session-1"
    )

    with pytest.raises(Exception, match="AWS Error"):
        store_tool.invoke({"messages": [HumanMessage("test")]})

    with pytest.raises(Exception, match="AWS Error"):
        list_tool.invoke({})

    with pytest.raises(Exception, match="AWS Error"):
        retrieve_tool.invoke({"query": "test", "limit": 3})
