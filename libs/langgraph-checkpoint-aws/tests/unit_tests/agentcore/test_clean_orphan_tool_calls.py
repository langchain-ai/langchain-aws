"""
Unit tests for clean_orphan_tool_calls function.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_checkpoint_aws.agentcore.helpers import clean_orphan_tool_calls


class TestCleanOrphanToolCalls:
    """Test suite for clean_orphan_tool_calls function."""

    def test_empty_messages_list(self):
        assert clean_orphan_tool_calls([]) == []

    def test_none_messages_list(self):
        assert clean_orphan_tool_calls(None) is None

    def test_messages_without_tool_calls(self):
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        result = clean_orphan_tool_calls(messages)
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi there!"

    def test_patch_mode_complete_tool_call_only(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"id": "tool_123", "name": "get_weather", "args": {}}],
            ),
            ToolMessage(content="Sunny, 75F", tool_call_id="tool_123"),
        ]
        result = clean_orphan_tool_calls(messages)
        assert len(result) == 2
        assert len(result[0].tool_calls) == 1

    def test_patch_mode_orphan_tool_call_only(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "orphan_123", "name": "get_weather", "args": {}},
                ],
            ),
            AIMessage(content="Sorry, I couldn't get the weather"),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 3
        assert len(result[0].tool_calls) == 1
        assert isinstance(result[1], ToolMessage)
        assert result[1].tool_call_id == "orphan_123"
        assert "interrupted" in result[1].content
        assert "get_weather" in result[1].content
        assert result[1].status == "error"

    def test_patch_mode_mixed_complete_and_orphan_tool_calls(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "tool_weather", "name": "get_weather", "args": {}},
                    {"id": "tool_time", "name": "get_time", "args": {}},
                ],
            ),
            ToolMessage(content="Sunny", tool_call_id="tool_weather"),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 3
        assert len(result[0].tool_calls) == 2
        assert isinstance(result[1], ToolMessage)
        assert result[1].tool_call_id == "tool_time"
        assert result[2].tool_call_id == "tool_weather"

    def test_patch_mode_multiple_orphan_tool_calls(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "orphan_1", "name": "tool_a", "args": {}},
                    {"id": "orphan_2", "name": "tool_b", "args": {}},
                ],
            ),
        ]

        result = clean_orphan_tool_calls(messages)

        assert len(result) == 3
        assert result[1].tool_call_id == "orphan_1"
        assert result[1].status == "error"
        assert result[2].tool_call_id == "orphan_2"
        assert result[2].status == "error"

    def test_removal_mode_complete_tool_call_only(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"id": "tool_123", "name": "get_weather", "args": {}}],
            ),
            ToolMessage(content="Sunny, 75F", tool_call_id="tool_123"),
        ]
        result = clean_orphan_tool_calls(messages, remove_dangling_tool_calls=True)
        assert len(result) == 2
        assert len(result[0].tool_calls) == 1

    def test_removal_mode_orphan_tool_call_only(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "orphan_123", "name": "get_weather", "args": {}},
                ],
            ),
        ]

        result = clean_orphan_tool_calls(messages, remove_dangling_tool_calls=True)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert len(result[0].tool_calls) == 0

    def test_removal_mode_mixed_complete_and_orphan_tool_calls(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "valid", "name": "get_weather", "args": {}},
                    {"id": "orphan", "name": "get_time", "args": {}},
                ],
            ),
            ToolMessage(content="Sunny", tool_call_id="valid"),
        ]

        result = clean_orphan_tool_calls(messages, remove_dangling_tool_calls=True)

        assert len(result) == 2
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0]["id"] == "valid"

    def test_removal_mode_multiple_orphan_tool_calls(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "orphan_1", "name": "tool_a", "args": {}},
                    {"id": "orphan_2", "name": "tool_b", "args": {}},
                ],
            ),
        ]

        result = clean_orphan_tool_calls(messages, remove_dangling_tool_calls=True)

        assert len(result) == 1
        assert len(result[0].tool_calls) == 0
