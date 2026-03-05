"""Tests for checkpoint sanitization utilities."""

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_checkpoint_aws.sanitization import (
    _parse_tool_args,
    _sanitize_content_blocks,
    _sanitize_tool_calls,
    sanitize_checkpoint,
    sanitize_message,
)


def _make_buggy_ai_message(**kwargs):
    """Create an AIMessage with malformed tool_call args.

    In the real bug, malformed data comes from checkpoint deserialization
    which bypasses Pydantic validation. We use model_construct() to
    simulate this.
    """
    defaults = {
        "additional_kwargs": {},
        "response_metadata": {},
        "id": None,
        "type": "ai",
    }
    defaults.update(kwargs)
    return AIMessage.model_construct(**defaults)


class TestParseToolArgs:
    """Test _parse_tool_args function."""

    def test_dict_passthrough(self):
        args = {"name": "test", "price": 50}
        assert _parse_tool_args(args) == args

    def test_string_to_dict(self):
        args = '{"name": "test", "price": 50}'
        result = _parse_tool_args(args)
        assert result == {"name": "test", "price": 50}
        assert isinstance(result, dict)

    def test_none_to_empty_dict(self):
        assert _parse_tool_args(None) == {}

    def test_invalid_json_string(self):
        assert _parse_tool_args("not json") == "not json"

    def test_nested_string_in_dict(self):
        args = {"name": "test", "nested": '{"key": "value"}'}
        result = _parse_tool_args(args)
        assert result["nested"] == {"key": "value"}


class TestSanitizeToolCalls:
    """Test _sanitize_tool_calls function."""

    def test_empty_list(self):
        assert _sanitize_tool_calls([]) == []

    def test_dict_args_unchanged(self):
        tool_calls = [
            {
                "name": "create_product",
                "id": "call_123",
                "args": {"name": "X", "price": 50},
            }
        ]
        result = _sanitize_tool_calls(tool_calls)
        assert result == tool_calls

    def test_string_args_converted(self):
        tool_calls = [
            {
                "name": "create_product",
                "id": "call_123",
                "args": '{"name": "X", "price": 50}',
            }
        ]
        result = _sanitize_tool_calls(tool_calls)
        assert result[0]["args"] == {"name": "X", "price": 50}
        assert isinstance(result[0]["args"], dict)

    def test_multiple_tool_calls(self):
        tool_calls = [
            {"name": "tool1", "id": "1", "args": '{"a": 1}'},
            {"name": "tool2", "id": "2", "args": {"b": 2}},
            {"name": "tool3", "id": "3", "args": '{"c": 3}'},
        ]
        result = _sanitize_tool_calls(tool_calls)
        assert result[0]["args"] == {"a": 1}
        assert result[1]["args"] == {"b": 2}
        assert result[2]["args"] == {"c": 3}


class TestSanitizeContentBlocks:
    """Test _sanitize_content_blocks function."""

    def test_string_content_unchanged(self):
        assert _sanitize_content_blocks("Hello!") == "Hello!"

    def test_text_block_unchanged(self):
        content = [{"type": "text", "text": "Hello"}]
        assert _sanitize_content_blocks(content) == content

    def test_tool_use_dict_input_unchanged(self):
        content = [
            {
                "type": "tool_use",
                "name": "create_product",
                "id": "call_123",
                "input": {"name": "X", "price": 50},
            }
        ]
        assert _sanitize_content_blocks(content) == content

    def test_tool_use_string_input_converted(self):
        content = [
            {
                "type": "tool_use",
                "name": "create_product",
                "id": "call_123",
                "input": '{"name": "X", "price": 50}',
            }
        ]
        result = _sanitize_content_blocks(content)
        assert result[0]["input"] == {"name": "X", "price": 50}
        assert isinstance(result[0]["input"], dict)

    def test_mixed_blocks(self):
        content = [
            {"type": "text", "text": "Creating product."},
            {
                "type": "tool_use",
                "name": "create_product",
                "id": "call_123",
                "input": '{"name": "X"}',
            },
        ]
        result = _sanitize_content_blocks(content)
        assert result[0]["type"] == "text"
        assert result[1]["input"] == {"name": "X"}


class TestSanitizeMessage:
    """Test sanitize_message function."""

    def test_human_message_unchanged(self):
        msg = HumanMessage(content="Hello")
        assert sanitize_message(msg) is msg

    def test_ai_message_no_tools_unchanged(self):
        msg = AIMessage(content="Hello back!")
        result = sanitize_message(msg)
        assert result.content == "Hello back!"

    def test_ai_message_tool_calls_sanitized(self):
        msg = _make_buggy_ai_message(
            content="I'll create that.",
            tool_calls=[
                {
                    "name": "create_product",
                    "id": "call_123",
                    "args": '{"name": "X", "price": 50}',
                }
            ],
        )
        result = sanitize_message(msg)
        assert result.tool_calls[0]["args"] == {
            "name": "X",
            "price": 50,
        }

    def test_ai_message_content_blocks_sanitized(self):
        msg = AIMessage(
            content=[
                {"type": "text", "text": "Creating..."},
                {
                    "type": "tool_use",
                    "name": "create_product",
                    "id": "call_123",
                    "input": '{"name": "X"}',
                },
            ]
        )
        result = sanitize_message(msg)
        assert result.content[1]["input"] == {"name": "X"}


class TestSanitizeCheckpoint:
    """Test sanitize_checkpoint function."""

    def test_none_checkpoint(self):
        assert sanitize_checkpoint(None) is None

    def test_empty_checkpoint(self):
        assert sanitize_checkpoint({}) == {}

    def test_no_channel_values(self):
        checkpoint = {"v": 1, "id": "123"}
        assert sanitize_checkpoint(checkpoint) == checkpoint

    def test_no_messages(self):
        checkpoint = {"channel_values": {"other": "data"}}
        assert sanitize_checkpoint(checkpoint) == checkpoint

    def test_messages_sanitized(self):
        buggy_ai_msg = _make_buggy_ai_message(
            content="Creating...",
            tool_calls=[
                {
                    "name": "create_product",
                    "id": "call_123",
                    "args": '{"name": "X", "price": 50}',
                }
            ],
        )
        checkpoint = {
            "v": 1,
            "id": "123",
            "channel_values": {
                "messages": [
                    HumanMessage(content="Create product X"),
                    buggy_ai_msg,
                ]
            },
        }
        result = sanitize_checkpoint(checkpoint)

        # Original should be unchanged
        orig_args = checkpoint["channel_values"]["messages"][1].tool_calls[0]["args"]
        assert orig_args == '{"name": "X", "price": 50}'

        # Result should be sanitized
        fixed_args = result["channel_values"]["messages"][1].tool_calls[0]["args"]
        assert fixed_args == {"name": "X", "price": 50}

    def test_preserves_other_fields(self):
        checkpoint = {
            "v": 1,
            "id": "checkpoint_123",
            "ts": "2026-01-08T10:00:00Z",
            "channel_values": {
                "messages": [HumanMessage(content="Hello")],
                "other_channel": {"data": "value"},
            },
            "channel_versions": {"messages": 1},
            "versions_seen": {},
        }
        result = sanitize_checkpoint(checkpoint)

        assert result["v"] == 1
        assert result["id"] == "checkpoint_123"
        assert result["ts"] == "2026-01-08T10:00:00Z"
        assert result["channel_values"]["other_channel"] == {
            "data": "value",
        }
        assert result["channel_versions"] == {"messages": 1}


class TestIntegration:
    """Integration tests for real-world Bedrock HITL scenarios."""

    def test_bedrock_hitl_checkpoint_roundtrip(self):
        """Simulate HITL checkpoint flow with serialization bug."""
        buggy_ai_msg = _make_buggy_ai_message(
            content=[
                {
                    "type": "text",
                    "text": "I'll create that product.",
                },
                {
                    "type": "tool_use",
                    "name": "Foodics___create_product",
                    "id": "toolu_abc",
                    "input": '{"name": "Latte", "price": 25}',
                },
            ],
            tool_calls=[
                {
                    "name": "Foodics___create_product",
                    "id": "toolu_abc",
                    "args": '{"name": "Latte", "price": 25}',
                }
            ],
        )
        buggy_checkpoint = {
            "v": 1,
            "id": "cp_123",
            "channel_values": {
                "messages": [
                    HumanMessage(content="Create a Latte"),
                    buggy_ai_msg,
                ]
            },
        }

        fixed = sanitize_checkpoint(buggy_checkpoint)
        ai_msg = fixed["channel_values"]["messages"][1]

        assert isinstance(ai_msg.content[1]["input"], dict)
        assert ai_msg.content[1]["input"]["name"] == "Latte"
        assert isinstance(ai_msg.tool_calls[0]["args"], dict)
        assert ai_msg.tool_calls[0]["args"]["name"] == "Latte"

    def test_already_correct_checkpoint_unchanged(self):
        """Correct checkpoints should not be modified."""
        correct_checkpoint = {
            "v": 1,
            "id": "cp_123",
            "channel_values": {
                "messages": [
                    HumanMessage(content="Create a product"),
                    AIMessage(
                        content=[
                            {"type": "text", "text": "Creating..."},
                            {
                                "type": "tool_use",
                                "name": "create_product",
                                "id": "toolu_abc",
                                "input": {"name": "X"},
                            },
                        ],
                        tool_calls=[
                            {
                                "name": "create_product",
                                "id": "toolu_abc",
                                "args": {"name": "X"},
                            }
                        ],
                    ),
                ]
            },
        }

        result = sanitize_checkpoint(correct_checkpoint)
        assert result is correct_checkpoint
