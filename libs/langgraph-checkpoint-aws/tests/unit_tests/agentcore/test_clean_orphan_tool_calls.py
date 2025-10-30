"""
Unit tests for clean_orphan_tool_calls function.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_checkpoint_aws.agentcore.helpers import clean_orphan_tool_calls


class TestCleanOrphanToolCalls:
    """Test suite for clean_orphan_tool_calls function."""

    def test_empty_messages_list(self):
        """Test with empty messages list."""
        result = clean_orphan_tool_calls([])
        assert result == []

    def test_messages_without_tool_calls(self):
        """Test with messages that don't have tool calls."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        result = clean_orphan_tool_calls(messages)
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there!"

    def test_complete_tool_call_pair_preserved(self):
        """Test that complete AIMessage + ToolMessage pairs are preserved."""
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_123",
                        "name": "get_weather",
                        "args": {"city": "SF"},
                    }
                ],
            ),
            ToolMessage(content="Sunny, 75F", tool_call_id="tool_123"),
            AIMessage(content="The weather in SF is sunny and 75F"),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 4
        # Verify the AIMessage still has its tool_calls
        assert isinstance(result[1], AIMessage)
        assert len(result[1].tool_calls) == 1
        assert result[1].tool_calls[0]["id"] == "tool_123"

    def test_orphan_tool_call_removed(self):
        """Test that orphan tool_calls are removed from AIMessage."""
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "orphan_tool_123",
                        "name": "get_weather",
                        "args": {"city": "SF"},
                    }
                ],
            ),
            # No ToolMessage here - this is an orphan tool_call
            AIMessage(content="Sorry, I couldn't get the weather"),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 3
        # Verify the AIMessage has its tool_calls removed
        assert isinstance(result[1], AIMessage)
        assert len(result[1].tool_calls) == 0

    def test_mixed_complete_and_orphan_tool_calls(self):
        """Test with both complete and orphan tool calls."""
        messages = [
            HumanMessage(content="Get weather and time"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_weather",
                        "name": "get_weather",
                        "args": {"city": "SF"},
                    },
                    {
                        "id": "tool_time",
                        "name": "get_time",
                        "args": {"timezone": "PST"},
                    },
                ],
            ),
            # Only weather tool has a response
            ToolMessage(content="Sunny, 75F", tool_call_id="tool_weather"),
            # No ToolMessage for tool_time - it's orphan
            AIMessage(content="The weather is sunny"),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 4
        # Verify only the complete tool_call is kept
        assert isinstance(result[1], AIMessage)
        assert len(result[1].tool_calls) == 1
        assert result[1].tool_calls[0]["id"] == "tool_weather"

    def test_multiple_tool_calls_all_complete(self):
        """Test with multiple tool calls that all have responses."""
        messages = [
            HumanMessage(content="Get weather and time"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tool_weather",
                        "name": "get_weather",
                        "args": {"city": "SF"},
                    },
                    {
                        "id": "tool_time",
                        "name": "get_time",
                        "args": {"timezone": "PST"},
                    },
                ],
            ),
            ToolMessage(content="Sunny, 75F", tool_call_id="tool_weather"),
            ToolMessage(content="10:30 AM", tool_call_id="tool_time"),
            AIMessage(content="Weather is sunny and time is 10:30 AM"),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 5
        # Verify both tool_calls are preserved
        assert isinstance(result[1], AIMessage)
        assert len(result[1].tool_calls) == 2

    def test_multiple_tool_calls_all_orphan(self):
        """Test with multiple tool calls that are all orphans."""
        messages = [
            HumanMessage(content="Get weather and time"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "orphan_weather",
                        "name": "get_weather",
                        "args": {"city": "SF"},
                    },
                    {
                        "id": "orphan_time",
                        "name": "get_time",
                        "args": {"timezone": "PST"},
                    },
                ],
            ),
            # No ToolMessages at all
            AIMessage(content="Sorry, couldn't get the info"),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 3
        # Verify all tool_calls are removed
        assert isinstance(result[1], AIMessage)
        assert len(result[1].tool_calls) == 0

    def test_ai_message_without_tool_calls_preserved(self):
        """Test that AIMessages without tool_calls are unchanged."""
        messages = [
            AIMessage(content="Hello"),
            AIMessage(content="How can I help?", tool_calls=[]),
            HumanMessage(content="Hi"),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 3
        assert result[0].content == "Hello"
        assert result[1].content == "How can I help?"
        assert result[2].content == "Hi"

    def test_tool_message_without_ai_message(self):
        """Test handling of ToolMessage without corresponding AIMessage."""
        # This is an edge case that shouldn't normally happen
        messages = [
            HumanMessage(content="Hello"),
            ToolMessage(content="Tool result", tool_call_id="random_id"),
            AIMessage(content="Here's the result"),
        ]

        result = clean_orphan_tool_calls(messages)

        # Should preserve all messages
        assert len(result) == 3
        assert isinstance(result[1], ToolMessage)

    def test_preserves_message_order(self):
        """Test that message order is preserved."""
        messages = [
            HumanMessage(content="First"),
            AIMessage(
                content="",
                tool_calls=[{"id": "orphan", "name": "tool", "args": {}}],
            ),
            HumanMessage(content="Second"),
            AIMessage(content="Third"),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 4
        assert result[0].content == "First"
        assert isinstance(result[1], AIMessage)
        assert result[2].content == "Second"
        assert result[3].content == "Third"

    def test_none_messages_list(self):
        """Test with None instead of list."""
        result = clean_orphan_tool_calls(None)
        assert result is None
