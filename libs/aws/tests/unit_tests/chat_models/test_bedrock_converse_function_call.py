"""
Unit tests for _lc_content_to_bedrock handling of OpenAI Responses API
function_call content blocks.

Regression for:
  https://github.com/langchain-ai/langchain/issues/36531
"""

import pytest
from langchain_aws.chat_models.bedrock_converse import _lc_content_to_bedrock


class TestFunctionCallToToolUse:

    def test_function_call_with_string_arguments(self):
        content = [
            {
                "type": "function_call",
                "id": "call_abc123",
                "name": "get_weather",
                "arguments": '{"location": "London", "unit": "celsius"}',
            }
        ]
        result = _lc_content_to_bedrock(content)

        assert len(result) == 1
        tool_use = result[0]["toolUse"]
        assert tool_use["name"] == "get_weather"
        assert tool_use["toolUseId"] == "call_abc123"
        assert tool_use["input"] == {"location": "London", "unit": "celsius"}

    def test_function_call_with_call_id_instead_of_id(self):
        content = [
            {
                "type": "function_call",
                "call_id": "call_xyz789",
                "name": "search_web",
                "arguments": '{"query": "langchain bedrock"}',
            }
        ]
        result = _lc_content_to_bedrock(content)

        assert len(result) == 1
        tool_use = result[0]["toolUse"]
        assert tool_use["toolUseId"] == "call_xyz789"
        assert tool_use["name"] == "search_web"

    def test_function_call_with_empty_arguments(self):
        content = [
            {
                "type": "function_call",
                "id": "call_empty",
                "name": "get_current_time",
                "arguments": "",
            }
        ]
        result = _lc_content_to_bedrock(content)

        assert len(result) == 1
        assert result[0]["toolUse"]["input"] == {}

    def test_function_call_with_missing_arguments(self):
        content = [
            {
                "type": "function_call",
                "id": "call_noargs",
                "name": "get_current_time",
            }
        ]
        result = _lc_content_to_bedrock(content)

        assert len(result) == 1
        assert result[0]["toolUse"]["input"] == {}

    def test_function_call_with_dict_arguments(self):
        content = [
            {
                "type": "function_call",
                "id": "call_dict",
                "name": "calculate",
                "arguments": {"expression": "2 + 2"},
            }
        ]
        result = _lc_content_to_bedrock(content)

        assert len(result) == 1
        assert result[0]["toolUse"]["input"] == {"expression": "2 + 2"}

    def test_mixed_text_and_function_call_blocks(self):
        content = [
            {"type": "text", "text": "I will call the weather tool now."},
            {
                "type": "function_call",
                "id": "call_mixed",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        ]
        result = _lc_content_to_bedrock(content)

        assert len(result) == 2
        assert result[0] == {"text": "I will call the weather tool now."}
        tool_use = result[1]["toolUse"]
        assert tool_use["name"] == "get_weather"
        assert tool_use["input"] == {"location": "Paris"}

    def test_multiple_function_call_blocks(self):
        content = [
            {
                "type": "function_call",
                "id": "call_1",
                "name": "tool_one",
                "arguments": '{"a": 1}',
            },
            {
                "type": "function_call",
                "id": "call_2",
                "name": "tool_two",
                "arguments": '{"b": 2}',
            },
        ]
        result = _lc_content_to_bedrock(content)

        assert len(result) == 2
        assert result[0]["toolUse"]["name"] == "tool_one"
        assert result[1]["toolUse"]["name"] == "tool_two"

    def test_function_call_id_takes_priority_over_call_id(self):
        content = [
            {
                "type": "function_call",
                "id": "primary_id",
                "call_id": "secondary_id",
                "name": "some_tool",
                "arguments": "{}",
            }
        ]
        result = _lc_content_to_bedrock(content)

        assert result[0]["toolUse"]["toolUseId"] == "primary_id"

    def test_function_call_parity_with_tool_use(self):
        function_call_content = [
            {
                "type": "function_call",
                "id": "call_parity",
                "name": "my_tool",
                "arguments": '{"key": "value"}',
            }
        ]
        tool_use_content = [
            {
                "type": "tool_use",
                "id": "call_parity",
                "name": "my_tool",
                "input": {"key": "value"},
            }
        ]

        function_call_result = _lc_content_to_bedrock(function_call_content)
        tool_use_result = _lc_content_to_bedrock(tool_use_content)

        assert function_call_result == tool_use_result