# type:ignore
"""Test chat model integration."""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_aws.chat_models.sagemaker_endpoint import (
    ChatModelContentHandler,
    ChatSagemakerEndpoint,
    OpenAICompatibleChatModelContentHandler,
    _messages_to_sagemaker,
)


class GetWeather(BaseModel):
    """Get the current weather in a given location."""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: Optional[str] = Field(
        default="fahrenheit", description="Temperature unit: 'celsius' or 'fahrenheit'"
    )


class GetTime(BaseModel):
    """Get current time in a timezone."""

    timezone: str = Field(..., description="The timezone, e.g. America/New_York")


class DefaultHandler(ChatModelContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs: Dict) -> bytes:
        return json.dumps(prompt).encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.decode())
        return AIMessage(content=response_json[0]["generated_text"])


def test_format_messages_request() -> None:
    client = Mock()
    messages = [
        SystemMessage("Output everything you have."),  # type: ignore[misc]
        HumanMessage("What is an llm?"),  # type: ignore[misc]
    ]
    kwargs = {}

    llm = ChatSagemakerEndpoint(
        endpoint_name="my-endpoint",
        region_name="us-west-2",
        content_handler=DefaultHandler(),
        model_kwargs={
            "parameters": {
                "max_new_tokens": 50,
            }
        },
        client=client,
    )
    invocation_params = llm._format_messages_request(messages=messages, **kwargs)

    expected_invocation_params = {
        "EndpointName": "my-endpoint",
        "Body": (
            b'[{"role": "system", "content": "Output everything you have."}, '
            b'{"role": "user", "content": "What is an llm?"}]'
        ),
        "ContentType": "application/json",
        "Accept": "application/json",
    }
    assert invocation_params == expected_invocation_params


def test__messages_to_sagemaker() -> None:
    messages = [
        SystemMessage("foo"),  # type: ignore[misc]
        HumanMessage("bar"),  # type: ignore[misc]
        AIMessage("some answer"),
        HumanMessage("follow-up question"),  # type: ignore[misc]
    ]
    expected = [
        {"role": "system", "content": "foo"},
        {"role": "user", "content": "bar"},
        {"role": "assistant", "content": "some answer"},
        {"role": "user", "content": "follow-up question"},
    ]
    actual = _messages_to_sagemaker(messages)
    assert expected == actual


class TestOpenAICompatibleChatModelContentHandler:
    """Tests for OpenAICompatibleChatModelContentHandler
    - critical for payload transformation."""

    @pytest.fixture
    def handler(self) -> OpenAICompatibleChatModelContentHandler:
        """Create a handler instance."""
        return OpenAICompatibleChatModelContentHandler()

    def test_transform_input_basic(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify basic message transformation to JSON bytes."""
        messages = [{"role": "user", "content": "Hello"}]
        model_kwargs = {"temperature": 0.7}

        result = handler.transform_input(messages, model_kwargs)
        parsed = json.loads(result.decode("utf-8"))

        assert parsed["messages"] == messages
        assert parsed["temperature"] == 0.7

    def test_transform_input_with_multiple_messages(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify multi-turn conversation is properly transformed."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        model_kwargs = {}

        result = handler.transform_input(messages, model_kwargs)
        parsed = json.loads(result.decode("utf-8"))

        assert len(parsed["messages"]) == 4
        assert parsed["messages"][0]["role"] == "system"
        assert parsed["messages"][-1]["role"] == "user"

    def test_transform_input_with_all_model_kwargs(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify all common model kwargs are included in payload."""
        messages = [{"role": "user", "content": "Test"}]
        model_kwargs = {
            "temperature": 0.5,
            "max_tokens": 1000,
            "top_p": 0.9,
            "stop": ["\n", "END"],
            "stream": True,
        }

        result = handler.transform_input(messages, model_kwargs)
        parsed = json.loads(result.decode("utf-8"))

        assert parsed["temperature"] == 0.5
        assert parsed["max_tokens"] == 1000
        assert parsed["top_p"] == 0.9
        assert parsed["stop"] == ["\n", "END"]
        assert parsed["stream"] is True

    def test_transform_input_with_tools(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify tools are included in the payload."""
        messages = [{"role": "user", "content": "What's the weather?"}]
        model_kwargs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        }

        result = handler.transform_input(messages, model_kwargs)
        parsed = json.loads(result.decode("utf-8"))

        assert "tools" in parsed
        assert len(parsed["tools"]) == 1
        assert parsed["tools"][0]["function"]["name"] == "get_weather"

    def test_transform_input_returns_bytes(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify transform_input returns bytes, not string."""
        result = handler.transform_input([{"role": "user", "content": "test"}], {})
        assert isinstance(result, bytes)

    def test_transform_input_empty_model_kwargs(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify empty model_kwargs doesn't cause issues."""
        messages = [{"role": "user", "content": "Hello"}]
        result = handler.transform_input(messages, {})
        parsed = json.loads(result.decode("utf-8"))

        assert parsed["messages"] == messages
        # Only messages key should be present
        assert set(parsed.keys()) == {"messages"}

    def test_transform_output_non_streaming(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify non-streaming response parsing."""
        response = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help?",
                        }
                    }
                ]
            }
        ).encode("utf-8")

        result = handler.transform_output(response)

        assert isinstance(result, AIMessage)
        assert result.content == "Hello! How can I help?"

    def test_transform_output_non_streaming_with_tool_calls(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify tool calls are properly parsed from non-streaming response."""
        response = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Paris"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        ).encode("utf-8")

        result = handler.transform_output(response)

        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_123"
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"] == {"location": "Paris"}

    def test_transform_output_non_streaming_multiple_tool_calls(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify multiple tool calls are properly parsed."""
        response = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll check both locations.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "Paris"}',
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "London"}',
                                    },
                                },
                            ],
                        }
                    }
                ]
            }
        ).encode("utf-8")

        result = handler.transform_output(response)

        assert isinstance(result, AIMessage)
        assert result.content == "I'll check both locations."
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["args"]["location"] == "Paris"
        assert result.tool_calls[1]["args"]["location"] == "London"

    def test_transform_output_non_streaming_empty_content(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify empty content is handled correctly."""
        response = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                        }
                    }
                ]
            }
        ).encode("utf-8")

        result = handler.transform_output(response)

        assert isinstance(result, AIMessage)
        assert result.content == ""

    def test_transform_output_non_streaming_null_content(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify null content is handled correctly."""
        response = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                        }
                    }
                ]
            }
        ).encode("utf-8")

        result = handler.transform_output(response)

        assert isinstance(result, AIMessage)
        assert result.content == ""

    def test_transform_output_streaming_chunk(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify streaming chunk parsing."""
        response = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "content": "Hello",
                        }
                    }
                ]
            }
        ).encode("utf-8")

        result = handler.transform_output(response)

        assert isinstance(result, AIMessageChunk)
        assert result.content == "Hello"

    def test_transform_output_streaming_with_tool_call_chunks(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify streaming tool call chunks are properly parsed."""
        response = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_123",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location":',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")

        result = handler.transform_output(response)

        assert isinstance(result, AIMessageChunk)
        assert len(result.tool_call_chunks) == 1
        assert result.tool_call_chunks[0]["index"] == 0
        assert result.tool_call_chunks[0]["id"] == "call_123"
        assert result.tool_call_chunks[0]["name"] == "get_weather"

    def test_transform_output_streaming_empty_delta(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify empty delta is handled (common at stream end)."""
        response = json.dumps({"choices": [{"delta": {}}]}).encode("utf-8")

        result = handler.transform_output(response)

        assert isinstance(result, AIMessageChunk)
        assert result.content == ""

    def test_transform_output_streaming_partial_tool_args(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify partial tool arguments are preserved as strings for accumulation."""
        # First chunk - name and start of args
        response1 = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"loc',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")

        result1 = handler.transform_output(response1)
        assert result1.tool_call_chunks[0]["args"] == '{"loc'

        # Second chunk - continuation of args
        response2 = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'ation": "Paris"}'},
                                }
                            ]
                        }
                    }
                ]
            }
        ).encode("utf-8")

        result2 = handler.transform_output(response2)
        assert result2.tool_call_chunks[0]["args"] == 'ation": "Paris"}'

    def test_transform_output_unsupported_format(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify unsupported format raises NotImplementedError."""
        response = json.dumps(
            {
                "generated_text": "Some text"  # TGI format, not OpenAI
            }
        ).encode("utf-8")

        with pytest.raises(NotImplementedError):
            handler.transform_output(response)

    def test_transform_output_empty_choices(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify empty choices list returns empty AIMessage."""
        response = json.dumps({"choices": []}).encode("utf-8")

        result = handler.transform_output(response)

        assert isinstance(result, AIMessage)
        assert result.content == ""

    def test_transform_output_invalid_json(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify invalid JSON raises appropriate error."""
        response = b"not valid json"

        with pytest.raises(json.JSONDecodeError):
            handler.transform_output(response)

    def test_transform_output_handles_bytes_and_string(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify both bytes and string inputs are handled."""
        response_dict = {
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}]
        }

        # Test with bytes
        result_bytes = handler.transform_output(
            json.dumps(response_dict).encode("utf-8")
        )
        assert result_bytes.content == "Hi"

        # Test with string (some endpoints might return string)
        result_str = handler.transform_output(json.dumps(response_dict))
        assert result_str.content == "Hi"

    def test_parse_openai_style_tool_calls_non_streaming(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify non-streaming tool call parsing returns ToolCall objects."""
        tool_calls_data = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ]

        result = handler._parse_openai_style_tool_calls(tool_calls_data)

        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["name"] == "get_weather"
        assert result[0]["args"] == {"location": "Paris"}

    def test_parse_openai_style_tool_calls_chunks_streaming(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify streaming tool call parsing returns ToolCallChunk objects."""
        tool_calls_data = [
            {
                "index": 0,
                "id": "call_123",
                "function": {"name": "get_weather", "arguments": '{"loc'},
            }
        ]

        result = handler._parse_openai_style_tool_calls_chunks(tool_calls_data)

        assert len(result) == 1
        assert result[0]["index"] == 0
        assert result[0]["id"] == "call_123"
        assert result[0]["name"] == "get_weather"
        assert result[0]["args"] == '{"loc'  # Kept as string for accumulation

    def test_parse_openai_style_tool_calls_empty(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify empty tool calls return empty list."""
        result = handler._parse_openai_style_tool_calls(None)
        assert result == []

        result = handler._parse_openai_style_tool_calls([])
        assert result == []

        result = handler._parse_openai_style_tool_calls_chunks(None)
        assert result == []

        result = handler._parse_openai_style_tool_calls_chunks([])
        assert result == []

    def test_parse_openai_style_tool_calls_complex_args(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify complex nested arguments are properly parsed."""
        tool_calls_data = [
            {
                "id": "call_456",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps(
                        {
                            "query": "test",
                            "filters": {"category": "books", "price_range": [10, 50]},
                            "options": {"limit": 10, "sort": "relevance"},
                        }
                    ),
                },
            }
        ]

        result = handler._parse_openai_style_tool_calls(tool_calls_data)

        assert result[0]["args"]["query"] == "test"
        assert result[0]["args"]["filters"]["category"] == "books"
        assert result[0]["args"]["options"]["limit"] == 10

    def test_parse_openai_style_tool_calls_empty_arguments(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify empty arguments are handled correctly."""
        tool_calls_data = [
            {
                "id": "call_789",
                "type": "function",
                "function": {"name": "get_current_time", "arguments": "{}"},
            }
        ]

        result = handler._parse_openai_style_tool_calls(tool_calls_data)

        assert result[0]["args"] == {}

    def test_parse_openai_style_tool_calls_missing_function(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify missing function key is handled gracefully."""
        tool_calls_data = [{"id": "call_abc"}]

        result = handler._parse_openai_style_tool_calls(tool_calls_data)

        assert len(result) == 1
        assert result[0]["id"] == "call_abc"
        assert result[0]["name"] == ""
        assert result[0]["args"] == {}

    def test_parse_openai_style_response_non_streaming_basic(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify non-streaming response with message key returns AIMessage."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello, how can I help you?",
                    }
                }
            ]
        }

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessage)
        assert not isinstance(result, AIMessageChunk)
        assert result.content == "Hello, how can I help you?"
        assert result.tool_calls == []

    def test_parse_openai_style_response_streaming_basic(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify streaming response with delta key returns AIMessageChunk."""
        response = {
            "choices": [
                {
                    "delta": {
                        "content": "Hello",
                    }
                }
            ]
        }

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessageChunk)
        assert result.content == "Hello"
        assert result.tool_call_chunks == []

    def test_parse_openai_style_response_non_streaming_with_tool_calls(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify non-streaming response with tool calls
        returns AIMessage with tool_calls."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check the weather.",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco", "unit": "celsius"}',  # noqa: E501
                                },
                            }
                        ],
                    }
                }
            ]
        }

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessage)
        assert result.content == "Let me check the weather."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_abc123"
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[0]["args"] == {
            "location": "San Francisco",
            "unit": "celsius",
        }

    def test_parse_openai_style_response_streaming_with_tool_call_chunks(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify streaming response with tool calls returns
        AIMessageChunk with tool_call_chunks."""
        response = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_xyz789",
                                "function": {
                                    "name": "search_database",
                                    "arguments": '{"query": "python"',
                                },
                            }
                        ]
                    }
                }
            ]
        }

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessageChunk)
        assert result.content == ""
        assert len(result.tool_call_chunks) == 1
        assert result.tool_call_chunks[0]["index"] == 0
        assert result.tool_call_chunks[0]["id"] == "call_xyz789"
        assert result.tool_call_chunks[0]["name"] == "search_database"
        assert result.tool_call_chunks[0]["args"] == '{"query": "python"'

    def test_parse_openai_style_response_empty_choices(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify empty choices returns empty AIMessage."""
        response = {"choices": []}

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessage)
        assert result.content == ""

    def test_parse_openai_style_response_missing_choices(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify missing choices key returns empty AIMessage."""
        response = {"id": "chatcmpl-123", "object": "chat.completion"}

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessage)
        assert result.content == ""

    def test_parse_openai_style_response_non_streaming_empty_content(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify non-streaming response with empty content."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                    }
                }
            ]
        }

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessage)
        assert result.content == ""

    def test_parse_openai_style_response_non_streaming_null_content(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify non-streaming response with null content."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                    }
                }
            ]
        }

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessage)
        assert result.content == ""

    def test_parse_openai_style_response_streaming_empty_delta(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify streaming response with empty delta (common at stream end)."""
        response = {"choices": [{"delta": {}}]}

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessageChunk)
        assert result.content == ""
        assert result.tool_call_chunks == []

    def test_parse_openai_style_response_streaming_null_delta(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify streaming response with null delta."""
        response = {"choices": [{"delta": None}]}

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessageChunk)
        assert result.content == ""

    def test_parse_openai_style_response_non_streaming_multiple_tool_calls(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify non-streaming response with multiple tool calls."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll check both.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_time",
                                    "arguments": '{"timezone": "EST"}',
                                },
                            },
                        ],
                    }
                }
            ]
        }

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessage)
        assert result.content == "I'll check both."
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["name"] == "get_weather"
        assert result.tool_calls[1]["name"] == "get_time"

    def test_parse_openai_style_response_streaming_multiple_tool_call_chunks(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify streaming response with multiple tool call chunks."""
        response = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "func1", "arguments": '{"a":'},
                            },
                            {
                                "index": 1,
                                "id": "call_2",
                                "function": {"name": "func2", "arguments": '{"b":'},
                            },
                        ]
                    }
                }
            ]
        }

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessageChunk)
        assert len(result.tool_call_chunks) == 2
        assert result.tool_call_chunks[0]["index"] == 0
        assert result.tool_call_chunks[1]["index"] == 1

    def test_parse_openai_style_response_non_streaming_null_message(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify non-streaming response with null message."""
        response = {"choices": [{"message": None}]}

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessage)
        assert result.content == ""

    def test_parse_openai_style_response_non_streaming_tool_calls_only(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify non-streaming response with tool calls but no content."""
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        result = handler._parse_openai_style_response(response)

        assert isinstance(result, AIMessage)
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "get_weather"

    def test_parse_openai_style_response_detects_streaming_by_delta_key(
        self, handler: OpenAICompatibleChatModelContentHandler
    ) -> None:
        """Verify streaming detection is based on presence of 'delta' key."""
        # Even with empty delta, should return AIMessageChunk
        streaming_response = {"choices": [{"delta": {"content": ""}}]}
        non_streaming_response = {"choices": [{"message": {"content": ""}}]}

        streaming_result = handler._parse_openai_style_response(streaming_response)
        non_streaming_result = handler._parse_openai_style_response(
            non_streaming_response
        )

        assert isinstance(streaming_result, AIMessageChunk)
        assert isinstance(non_streaming_result, AIMessage)
        assert not isinstance(non_streaming_result, AIMessageChunk)


class TestChatSagemakerEndpoint:
    """End-to-end tests for ChatSagemakerEndpoint verifying client calls."""

    @pytest.fixture
    def mock_client(self) -> Mock:
        """Create a mock SageMaker runtime client."""
        return Mock()

    @pytest.fixture
    def llm(self, mock_client: Mock) -> ChatSagemakerEndpoint:
        """Create a ChatSagemakerEndpoint instance with mock client."""
        return ChatSagemakerEndpoint(
            endpoint_name="my-endpoint",
            region_name="us-west-2",
            client=mock_client,
        )

    def _create_openai_response(
        self,
        content: str = "Hello!",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> bytes:
        """Helper to create OpenAI-style response bytes."""
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return json.dumps({"choices": [{"message": message}]}).encode("utf-8")

    def test_bind_tools_returns_runnable_binding(
        self, llm: ChatSagemakerEndpoint
    ) -> None:
        """Verify bind_tools returns proper RunnableBinding type."""
        llm_with_tools = llm.bind_tools([GetWeather])
        assert isinstance(llm_with_tools, RunnableBinding)

    def test_bind_tools_includes_tools_in_kwargs(
        self, llm: ChatSagemakerEndpoint
    ) -> None:
        """Verify tools are properly included in bound kwargs."""
        llm_with_tools = llm.bind_tools([GetWeather])

        assert "tools" in llm_with_tools.kwargs
        tools = llm_with_tools.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "GetWeather"

    def test_bind_tools_with_multiple_tools(self, llm: ChatSagemakerEndpoint) -> None:
        """Verify multiple tools can be bound together."""
        llm_with_tools = llm.bind_tools([GetWeather, GetTime])

        tools = llm_with_tools.kwargs["tools"]
        assert len(tools) == 2
        tool_names = {t["function"]["name"] for t in tools}
        assert tool_names == {"GetWeather", "GetTime"}

    def test_bind_tools_preserves_tool_schema(self, llm: ChatSagemakerEndpoint) -> None:
        """Verify tool schema including descriptions and parameters is preserved."""
        llm_with_tools = llm.bind_tools([GetWeather])

        tool = llm_with_tools.kwargs["tools"][0]
        func = tool["function"]

        assert func["name"] == "GetWeather"
        assert "description" in func
        assert "Get the current weather" in func["description"]
        assert "parameters" in func
        assert func["parameters"]["type"] == "object"
        assert "location" in func["parameters"]["properties"]
        assert "unit" in func["parameters"]["properties"]
        assert "location" in func["parameters"]["required"]

    def test_bind_tools_with_empty_list(self, llm: ChatSagemakerEndpoint) -> None:
        """Verify bind_tools handles empty tool list gracefully."""
        llm_with_tools = llm.bind_tools([])
        assert llm_with_tools.kwargs["tools"] == []

    def test_bind_tools_with_langchain_tool_decorator(
        self, llm: ChatSagemakerEndpoint
    ) -> None:
        """Verify bind_tools works with @tool decorated functions."""

        @tool
        def search_database(query: str) -> str:
            """Search the database for relevant information."""
            return f"Results for: {query}"

        llm_with_tools = llm.bind_tools([search_database])

        tools = llm_with_tools.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search_database"

    def test_bind_tools_with_openai_dict_format(
        self, llm: ChatSagemakerEndpoint
    ) -> None:
        """Verify bind_tools accepts OpenAI-style dict format."""
        openai_tool = {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
        llm_with_tools = llm.bind_tools([openai_tool])

        tools = llm_with_tools.kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "calculate"

    def test_bind_tools_preserves_additional_kwargs(
        self, llm: ChatSagemakerEndpoint
    ) -> None:
        """Verify additional kwargs passed to bind_tools are preserved."""
        llm_with_tools = llm.bind_tools(
            [GetWeather], tool_choice="auto", custom_param="value"
        )

        assert llm_with_tools.kwargs["tool_choice"] == "auto"
        assert llm_with_tools.kwargs["custom_param"] == "value"

    def test_bind_tools_chaining(self, llm: ChatSagemakerEndpoint) -> None:
        """Verify bind_tools can be chained (rebinding tools)."""
        llm_with_weather = llm.bind_tools([GetWeather])
        llm_with_both = llm_with_weather.bind(tools=[GetWeather, GetTime])

        # Note: this creates a new binding, not merging
        assert isinstance(llm_with_both, RunnableBinding)


class TestChatSagemakerEndpointEndToEnd:
    """End-to-end tests for ChatSagemakerEndpoint tool calling flows."""

    @pytest.fixture
    def mock_client(self) -> Mock:
        """Create a mock SageMaker runtime client."""
        return Mock()

    def test_tool_calling_invoke(self, mock_client: Mock) -> None:
        """
        End-to-end test for tool calling with invoke (non-streaming).

        Tests the complete flow:
        1. User asks a question requiring a tool
        2. LLM responds with a tool call
        3. Tool result is provided
        4. LLM provides final response

        Verifies:
        - Request payload structure (endpoint, headers, body with tools)
        - Message formatting for each role (user, assistant with tool_calls, tool)
        - Response parsing for both tool calls and final text response
        """
        # === PHASE 1: Initial request - LLM returns tool call ===
        tool_call_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_weather_123",
                                "type": "function",
                                "function": {
                                    "name": "GetWeather",
                                    "arguments": '{"location": "Paris", "unit": "celsius"}',  # noqa: E501
                                },
                            }
                        ],
                    }
                }
            ]
        }
        mock_client.invoke_endpoint.return_value = {
            "Body": json.dumps(tool_call_response).encode("utf-8")
        }

        llm = ChatSagemakerEndpoint(
            endpoint_name="test-endpoint",
            region_name="us-west-2",
            client=mock_client,
            model_kwargs={"temperature": 0.7, "max_tokens": 1000},
        )
        llm_with_tools = llm.bind_tools([GetWeather, GetTime])

        messages = [
            SystemMessage(content="You are a helpful weather assistant."),
            HumanMessage(content="What's the weather in Paris?"),
        ]
        result = llm_with_tools.invoke(messages)

        call_kwargs = mock_client.invoke_endpoint.call_args[1]
        assert call_kwargs["EndpointName"] == "test-endpoint"
        assert call_kwargs["ContentType"] == "application/json"
        assert call_kwargs["Accept"] == "application/json"

        body = json.loads(call_kwargs["Body"].decode("utf-8"))

        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are a helpful weather assistant."
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "What's the weather in Paris?"

        assert "tools" in body
        assert len(body["tools"]) == 2
        tool_names = {t["function"]["name"] for t in body["tools"]}
        assert tool_names == {"GetWeather", "GetTime"}

        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 1000

        assert isinstance(result, AIMessage)
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_weather_123"
        assert result.tool_calls[0]["name"] == "GetWeather"
        assert result.tool_calls[0]["args"] == {"location": "Paris", "unit": "celsius"}

        final_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The weather in Paris is currently 22°C and sunny.",
                    }
                }
            ]
        }
        mock_client.invoke_endpoint.return_value = {
            "Body": json.dumps(final_response).encode("utf-8")
        }

        messages_with_tool_result = [
            SystemMessage(content="You are a helpful weather assistant."),
            HumanMessage(content="What's the weather in Paris?"),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_weather_123",
                        name="GetWeather",
                        args={"location": "Paris", "unit": "celsius"},
                    )
                ],
            ),
            ToolMessage(
                content='{"temperature": 22, "condition": "sunny", "unit": "celsius"}',
                tool_call_id="call_weather_123",
            ),
        ]
        final_result = llm_with_tools.invoke(messages_with_tool_result)

        call_kwargs = mock_client.invoke_endpoint.call_args[1]
        body = json.loads(call_kwargs["Body"].decode("utf-8"))

        assert len(body["messages"]) == 4

        ai_msg = body["messages"][2]
        assert ai_msg["role"] == "assistant"
        assert ai_msg["content"] == ""
        assert "tool_calls" in ai_msg
        assert ai_msg["tool_calls"][0]["id"] == "call_weather_123"
        assert ai_msg["tool_calls"][0]["function"]["name"] == "GetWeather"
        assert (
            ai_msg["tool_calls"][0]["function"]["arguments"]
            == '{"location": "Paris", "unit": "celsius"}'
        )

        tool_msg = body["messages"][3]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_weather_123"
        assert (
            tool_msg["content"]
            == '{"temperature": 22, "condition": "sunny", "unit": "celsius"}'
        )

        assert isinstance(final_result, AIMessage)
        assert (
            final_result.content == "The weather in Paris is currently 22°C and sunny."
        )
        assert final_result.tool_calls == []

    def test_tool_calling_streaming(self, mock_client: Mock) -> None:
        """
        End-to-end test for tool calling with streaming.

        Tests the complete streaming flow:
        1. User asks a question requiring a tool
        2. LLM streams back a tool call in chunks
        3. Tool result is provided
        4. LLM streams back final response

        Verifies:
        - Request uses invoke_endpoint_with_response_stream
        - Streaming chunks are properly parsed (delta format)
        - Tool call chunks are accumulated correctly
        - Final response chunks are handled
        """
        streaming_chunks = [
            {
                "PayloadPart": {
                    "Bytes": json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_stream_123",
                                                "function": {
                                                    "name": "GetWeather",
                                                    "arguments": "",
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    ).encode("utf-8")
                    + b"\n"
                }
            },
            {
                "PayloadPart": {
                    "Bytes": json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {
                                                    "arguments": '{"location":'
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    ).encode("utf-8")
                    + b"\n"
                }
            },
            {
                "PayloadPart": {
                    "Bytes": json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {"arguments": ' "Paris"}'},
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    ).encode("utf-8")
                    + b"\n"
                }
            },
        ]

        mock_client.invoke_endpoint_with_response_stream.return_value = {
            "Body": iter(streaming_chunks)
        }

        llm = ChatSagemakerEndpoint(
            endpoint_name="test-endpoint",
            region_name="us-west-2",
            client=mock_client,
            streaming=True,
        )
        llm_with_tools = llm.bind_tools([GetWeather])

        messages = [HumanMessage(content="What's the weather in Paris?")]

        chunks = list(llm_with_tools.stream(messages))

        mock_client.invoke_endpoint_with_response_stream.assert_called_once()

        call_kwargs = mock_client.invoke_endpoint_with_response_stream.call_args[1]
        assert call_kwargs["EndpointName"] == "test-endpoint"
        body = json.loads(call_kwargs["Body"].decode("utf-8"))
        assert "tools" in body
        assert body["tools"][0]["function"]["name"] == "GetWeather"

        assert len(chunks) >= 3

        tool_chunks = [c for c in chunks if c.tool_call_chunks]
        assert len(tool_chunks) == 3

        assert tool_chunks[0].tool_call_chunks[0]["id"] == "call_stream_123"
        assert tool_chunks[0].tool_call_chunks[0]["name"] == "GetWeather"

        assert tool_chunks[1].tool_call_chunks[0]["args"] == '{"location":'
        assert tool_chunks[2].tool_call_chunks[0]["args"] == ' "Paris"}'

        text_streaming_chunks = [
            {
                "PayloadPart": {
                    "Bytes": json.dumps(
                        {"choices": [{"delta": {"content": "The weather "}}]}
                    ).encode("utf-8")
                    + b"\n"
                }
            },
            {
                "PayloadPart": {
                    "Bytes": json.dumps(
                        {"choices": [{"delta": {"content": "in Paris is "}}]}
                    ).encode("utf-8")
                    + b"\n"
                }
            },
            {
                "PayloadPart": {
                    "Bytes": json.dumps(
                        {"choices": [{"delta": {"content": "sunny and 22°C."}}]}
                    ).encode("utf-8")
                    + b"\n"
                }
            },
        ]

        mock_client.invoke_endpoint_with_response_stream.return_value = {
            "Body": iter(text_streaming_chunks)
        }

        messages_with_tool = [
            HumanMessage(content="What's the weather in Paris?"),
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_stream_123",
                        name="GetWeather",
                        args={"location": "Paris"},
                    )
                ],
            ),
            ToolMessage(
                content='{"temp": 22, "condition": "sunny"}',
                tool_call_id="call_stream_123",
            ),
        ]

        final_chunks = list(llm_with_tools.stream(messages_with_tool))

        content_chunks = [c for c in final_chunks if c.content]

        # Verify text content is streamed
        assert len(content_chunks) == 3
        assert content_chunks[0].content == "The weather "
        assert content_chunks[1].content == "in Paris is "
        assert content_chunks[2].content == "sunny and 22°C."

        full_content = "".join(chunk.content for chunk in content_chunks)
        assert full_content == "The weather in Paris is sunny and 22°C."


class TestMessagesToSagemaker:
    """Tests for _messages_to_sagemaker conversion function."""

    def test_basic_conversation(self) -> None:
        """Verify basic message types are converted correctly."""
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        result = _messages_to_sagemaker(messages)

        assert result == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    def test_complete_conversation(self) -> None:
        """Verify complete conversation is converted correctly."""
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Check the weather in NYC."),
            AIMessage(
                content="Let me check.",
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", args={"location": "NYC"})
                ],
            ),
            ToolMessage(content="The weather in NYC is sunny.", tool_call_id="call_1"),
            AIMessage(content="The weather in NYC is sunny."),
            HumanMessage(content="Thank you!"),
            AIMessage(content="You're welcome!"),
        ]

        result = _messages_to_sagemaker(messages)

        assert len(result) == 7
        assert result[0] == {"role": "system", "content": "You are helpful."}
        assert result[1] == {"role": "user", "content": "Check the weather in NYC."}
        # AI message with tool calls - verify structure
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "Let me check."
        assert "tool_calls" in result[2]
        assert result[2]["tool_calls"][0]["id"] == "call_1"
        assert result[2]["tool_calls"][0]["function"]["name"] == "get_weather"
        # Tool message
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "call_1"
        assert result[3]["content"] == "The weather in NYC is sunny."
        # Remaining messages
        assert result[4] == {
            "role": "assistant",
            "content": "The weather in NYC is sunny.",
        }  # noqa: E501
        assert result[5] == {"role": "user", "content": "Thank you!"}
        assert result[6] == {"role": "assistant", "content": "You're welcome!"}

    def test_ai_message_with_tool_calls(self) -> None:
        """Verify AI messages with tool calls include tool_calls in output."""
        messages = [
            HumanMessage(content="What's the weather?"),
            AIMessage(
                content="Let me check.",
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", args={"location": "NYC"})
                ],
            ),
        ]

        result = _messages_to_sagemaker(messages)

        assert len(result) == 2
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[1]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_message_conversion(self) -> None:
        """Verify tool messages are properly converted."""
        messages = [
            ToolMessage(
                content='{"temp": 72}',
                tool_call_id="call_123",
            ),
        ]

        result = _messages_to_sagemaker(messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["content"] == '{"temp": 72}'

    def test_user_message_merging(self) -> None:
        """Verify consecutive same-role messages are merged."""
        messages = [
            HumanMessage(content="First part."),
            HumanMessage(content="Second part."),
        ]

        result = _messages_to_sagemaker(messages)

        # Messages should be merged (with newline separator)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First part.\nSecond part."

    def test_system_message_merging(self) -> None:
        """Verify consecutive same-role messages are merged."""
        messages = [
            SystemMessage(content="First part."),
            SystemMessage(content="Second part."),
        ]

        result = _messages_to_sagemaker(messages)

        # Messages should be merged (with newline separator)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "First part.\nSecond part."

    def test_unsupported_message_type_raises(self) -> None:
        """Verify unsupported message types raise ValueError."""
        from langchain_core.messages import ChatMessage

        messages = [ChatMessage(content="test", role="custom")]

        with pytest.raises(ValueError, match="Unsupported message type"):
            _messages_to_sagemaker(messages)
