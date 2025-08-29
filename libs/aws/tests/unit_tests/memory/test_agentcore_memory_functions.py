from unittest.mock import MagicMock

import pytest

from langchain_aws.memory.bedrock_agentcore import (
    convert_langchain_messages_to_events,
    convert_events_to_langchain_messages,
    store_agentcore_memory_events,
    list_agentcore_memory_events,
    retrieve_agentcore_memories,
)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_agentcore_memory_client() -> MagicMock:
    memory_client = MagicMock()
    memory_client.create_event.return_value = {"eventId": "12345"}

    return memory_client


def test_store_messages(mock_agentcore_memory_client) -> None:
    messages = [
        SystemMessage("You are a friendly chatbot"),
        HumanMessage("Hello, world!"),
        AIMessage("Hello there! What can I help you with today"),
        HumanMessage("Tell me a joke"),
        ToolMessage("Joke of the day retrieved.", tool_call_id="test_tool_call"),
    ]

    result = store_agentcore_memory_events(
        mock_agentcore_memory_client,
        messages=messages,
        memory_id="1234",
        actor_id="5678",
        session_id="9101",
    )
    mock_agentcore_memory_client.create_event.assert_called_once_with(
        memory_id="1234",
        actor_id="5678",
        session_id="9101",
        messages=[
            ("Hello, world!", "USER"),
            ("Hello there! What can I help you with today", "ASSISTANT"),
            ("Tell me a joke", "USER"),
            ("Joke of the day retrieved.", "TOOL"),
        ],
    )

    assert result == "12345"


def test_store_messages_with_system_messages(mock_agentcore_memory_client) -> None:
    messages = [
        SystemMessage("You are a friendly chatbot"),
        HumanMessage("Hello, world!"),
        AIMessage("Hello there! What can I help you with today"),
        HumanMessage("Tell me a joke"),
        ToolMessage("Joke of the day retrieved.", tool_call_id="test_tool_call"),
    ]

    result = store_agentcore_memory_events(
        mock_agentcore_memory_client,
        messages=messages,
        memory_id="1234",
        actor_id="5678",
        session_id="9101",
        include_system_messages=True,
    )
    mock_agentcore_memory_client.create_event.assert_called_once_with(
        memory_id="1234",
        actor_id="5678",
        session_id="9101",
        messages=[
            ("You are a friendly chatbot", "OTHER"),
            ("Hello, world!", "USER"),
            ("Hello there! What can I help you with today", "ASSISTANT"),
            ("Tell me a joke", "USER"),
            ("Joke of the day retrieved.", "TOOL"),
        ],
    )

    assert result == "12345"


def test_store_messages_empty_list_raises_error(mock_agentcore_memory_client):
    with pytest.raises(ValueError, match="The messages field cannot be empty."):
        store_agentcore_memory_events(
            mock_agentcore_memory_client,
            messages=[],
            memory_id="1234",
            actor_id="5678",
            session_id="9101",
        )


def test_list_memory_events_success(mock_agentcore_memory_client):
    # Mock the response from list_events
    mock_agentcore_memory_client.list_events.return_value = [
        {
            "eventId": "event-1",
            "payload": [
                {
                    "conversational": {
                        "role": "USER",
                        "content": {"text": "Hello, world!"},
                    }
                }
            ],
        },
        {
            "eventId": "event-2",
            "payload": [
                {
                    "conversational": {
                        "role": "ASSISTANT",
                        "content": {"text": "Hi there!"},
                    }
                }
            ],
        },
    ]

    result = list_agentcore_memory_events(
        mock_agentcore_memory_client,
        memory_id="test-memory",
        actor_id="test-actor",
        session_id="test-session",
        max_results=50,
    )

    # Assert the client was called correctly
    mock_agentcore_memory_client.list_events.assert_called_once_with(
        memory_id="test-memory",
        actor_id="test-actor",
        session_id="test-session",
        max_results=50,
        include_payload=True,
    )

    # Assert the return value is correct
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Hello, world!"
    assert result[0].additional_kwargs["event_id"] == "event-1"
    assert isinstance(result[1], AIMessage)
    assert result[1].content == "Hi there!"


def test_list_memory_events_tool_message(mock_agentcore_memory_client):
    mock_agentcore_memory_client.list_events.return_value = [
        {
            "eventId": "event-1",
            "payload": [
                {"conversational": {"role": "TOOL", "content": {"text": "Tool result"}}}
            ],
        }
    ]

    result = list_agentcore_memory_events(
        mock_agentcore_memory_client,
        memory_id="test-memory",
        actor_id="test-actor",
        session_id="test-session",
    )

    assert len(result) == 1
    assert isinstance(result[0], ToolMessage)
    assert result[0].content == "Tool result"
    assert result[0].tool_call_id == "unknown"  # Default value


def test_retrieve_agentcore_memories_success(mock_agentcore_memory_client):
    # Mock the response from retrieve_memories
    mock_agentcore_memory_client.retrieve_memories.return_value = [
        {
            "content": {"text": "User prefers coffee over tea"},
            "score": 0.95,
            "metadata": {"category": "preferences", "timestamp": "2024-01-01"},
        },
        {
            "content": {"text": "User lives in San Francisco"},
            "score": 0.87,
            "metadata": {"category": "location"},
        },
    ]

    result = retrieve_agentcore_memories(
        mock_agentcore_memory_client,
        memory_id="test-memory",
        namespace_str="/userPreferences/actor-1/session-1",
        query="coffee preferences",
        limit=5,
    )

    # Assert the client was called correctly
    mock_agentcore_memory_client.retrieve_memories.assert_called_once_with(
        memory_id="test-memory",
        namespace="/userPreferences/actor-1/session-1",
        query="coffee preferences",
        top_k=5,
    )

    # Assert the return value structure
    assert len(result) == 2
    assert result[0]["content"] == "User prefers coffee over tea"
    assert result[0]["score"] == 0.95
    assert result[0]["metadata"] == {
        "category": "preferences",
        "timestamp": "2024-01-01",
    }
    assert result[1]["content"] == "User lives in San Francisco"
    assert result[1]["score"] == 0.87


def test_convert_langchain_messages_to_events_basic():
    """Test basic message conversion to events."""
    messages = [
        HumanMessage("Hello"),
        AIMessage("Hi there"),
        ToolMessage("Tool result", tool_call_id="123"),
        SystemMessage("System prompt"),
    ]

    # Without system messages
    events = convert_langchain_messages_to_events(
        messages, include_system_messages=False
    )
    expected = [("Hello", "USER"), ("Hi there", "ASSISTANT"), ("Tool result", "TOOL")]
    assert events == expected

    # With system messages
    events = convert_langchain_messages_to_events(
        messages, include_system_messages=True
    )
    expected = [
        ("Hello", "USER"),
        ("Hi there", "ASSISTANT"),
        ("Tool result", "TOOL"),
        ("System prompt", "OTHER"),
    ]
    assert events == expected


def test_convert_langchain_messages_skips_existing_event_ids():
    """Test that messages with event_id are skipped."""
    msg_with_id = HumanMessage("Already saved")
    msg_with_id.additional_kwargs["event_id"] = "existing-123"

    messages = [msg_with_id, HumanMessage("New message")]

    events = convert_langchain_messages_to_events(messages)
    assert events == [("New message", "USER")]


def test_convert_langchain_messages_filters_empty_content():
    """Test that empty/whitespace messages are filtered out."""
    messages = [
        HumanMessage(""),
        HumanMessage("   "),
        HumanMessage("Valid message"),
        AIMessage("\n\t"),
    ]

    events = convert_langchain_messages_to_events(messages)
    assert events == [("Valid message", "USER")]


def test_convert_events_to_langchain_messages_basic():
    """Test basic event conversion to LangChain messages."""
    events = [
        {
            "eventId": "event-1",
            "payload": [
                {"conversational": {"role": "USER", "content": {"text": "Hello"}}}
            ],
        },
        {
            "eventId": "event-2",
            "payload": [
                {
                    "conversational": {
                        "role": "ASSISTANT",
                        "content": {"text": "Hi there"},
                    }
                }
            ],
        },
    ]

    messages = convert_events_to_langchain_messages(events)

    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello"
    assert messages[0].additional_kwargs["event_id"] == "event-1"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Hi there"


def test_convert_events_handles_malformed_data():
    """Test handling of malformed event data."""
    events = [
        {"eventId": "event-1"},  # Missing payload
        {"eventId": "event-2", "payload": [{}]},  # Missing conversational
        {
            "eventId": "event-3",
            "payload": [
                {
                    "conversational": {
                        "role": "USER"
                        # Missing content
                    }
                }
            ],
        },
        {
            "eventId": "event-4",
            "payload": [
                {
                    "conversational": {
                        "role": "UNKNOWN_ROLE",
                        "content": {"text": "Should be skipped"},
                    }
                }
            ],
        },
    ]

    messages = convert_events_to_langchain_messages(events)
    assert len(messages) == 0  # All should be filtered out


# TODO: Delete test once AgentCore adds support for memory metadata
def test_convert_events_tool_message_hack():
    """Test the tool_call_id='unknown' hack."""
    events = [
        {
            "eventId": "event-1",
            "payload": [
                {"conversational": {"role": "TOOL", "content": {"text": "Tool result"}}}
            ],
        }
    ]

    messages = convert_events_to_langchain_messages(events)

    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].content == "Tool result"
    assert messages[0].tool_call_id == "unknown"  # The hack


def test_roundtrip_conversion():
    """Test that converting messages to events and back preserves data."""
    original_messages = [
        HumanMessage("Hello world"),
        AIMessage("Hi there!"),
        ToolMessage("Tool executed", tool_call_id="tool-123"),
    ]

    # Convert to events
    events_data = convert_langchain_messages_to_events(original_messages)

    # Simulate AgentCore storage format
    mock_events = []
    for i, (text, role) in enumerate(events_data):
        mock_events.append(
            {
                "eventId": f"event-{i}",
                "payload": [
                    {"conversational": {"role": role, "content": {"text": text}}}
                ],
            }
        )

    # Convert back to messages
    recovered_messages = convert_events_to_langchain_messages(mock_events)

    # Verify content is preserved (note: tool_call_id will be "unknown")
    assert len(recovered_messages) == 3
    assert recovered_messages[0].content == "Hello world"
    assert recovered_messages[1].content == "Hi there!"
    assert recovered_messages[2].content == "Tool executed"
    assert isinstance(recovered_messages[0], HumanMessage)
    assert isinstance(recovered_messages[1], AIMessage)
    assert isinstance(recovered_messages[2], ToolMessage)
