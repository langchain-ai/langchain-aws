"""Test chat model integration."""

import base64
import os
from typing import Any, Dict, List, Tuple, Type, Union, cast
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableBinding
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel, Field

from langchain_aws import ChatBedrockConverse
from langchain_aws.chat_models.bedrock_converse import (
    _bedrock_to_lc,
    _camel_to_snake,
    _camel_to_snake_keys,
    _extract_response_metadata,
    _lc_content_to_bedrock,
    _messages_to_bedrock,
    _snake_to_camel,
    _snake_to_camel_keys,
)


class TestBedrockStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @property
    def chat_model_params(self) -> dict:
        return {
            "client": None,
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region_name": "us-west-1",
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {
            "temperature": 0,
            "max_tokens": 100,
            "stop": [],
        }

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Return env vars, init args, and expected instance attrs for initializing
        from env vars."""
        return (
            {
                "AWS_ACCESS_KEY_ID": "key_id",
                "AWS_SECRET_ACCESS_KEY": "secret_key",
                "AWS_SESSION_TOKEN": "token",
                "AWS_REGION": "region",
            },
            {
                "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            },
            {
                "aws_access_key_id": "key_id",
                "aws_secret_access_key": "secret_key",
                "aws_session_token": "token",
            },
        )

    @pytest.mark.xfail(reason="Doesn't support streaming init param.")
    def test_init_streaming(self) -> None:
        super().test_init_streaming()


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def test_anthropic_bind_tools_tool_choice() -> None:
    chat_model = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-west-2"
    )  # type: ignore[call-arg]
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], tool_choice={"tool": {"name": "GetWeather"}}
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "tool": {"name": "GetWeather"}
    }
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], tool_choice="GetWeather"
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "tool": {"name": "GetWeather"}
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "auto": {}
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "any": {}
    }


def test_amazon_bind_tools_tool_choice() -> None:
    chat_model = ChatBedrockConverse(
        model="us.amazon.nova-lite-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], tool_choice="GetWeather"
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "tool": {"name": "GetWeather"}
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "auto": {}
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "any": {}
    }


def test__messages_to_bedrock() -> None:
    messages = [
        SystemMessage(content="sys1"),
        SystemMessage(content=["sys2"]),
        SystemMessage(
            content=[
                {"text": "sys3"},
                {"type": "text", "text": "sys4"},
                {"guardContent": {"text": {"text": "sys5"}}},
                {"type": "guard_content", "text": "sys6"},
            ]
        ),
        HumanMessage(content="hu1"),
        HumanMessage(content=["hu2"]),
        AIMessage(content="ai1"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="tool1",
                    args={"arg1": "arg1"},
                    id="tool_call1",
                    type="tool_call",
                )
            ],
        ),
        AIMessage(
            content=[
                {
                    "toolUse": {
                        "toolUseId": "tool_call2",
                        "input": {"arg2": 2},
                        "name": "tool2",
                    }
                },
                {
                    "type": "tool_use",
                    "id": "tool_call3",
                    "input": {"arg3": ["a", "b"]},
                    "name": "tool3",
                },
            ]
        ),
        HumanMessage(
            content=[
                {"text": "hu3"},
                {
                    "toolResult": {
                        "toolUseId": "tool_call1",
                        "content": [{"text": "tool_res1"}],
                        "status": "success",
                    }
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_call2",
                    "is_error": True,
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "data",
                            },
                        },
                    ],
                },
            ]
        ),
        ToolMessage(content="tool_res3", tool_call_id="tool_call3"),
        HumanMessage(
            content=[
                {"guardContent": {"text": {"text": "hu5"}}},
                {"type": "guard_content", "text": "hu6"},
            ]
        ),
        HumanMessage(
            content=[
                {
                    "type": "document",
                    "document": {
                        "format": "pdf",
                        "name": "doc1",
                        "source": {"bytes": b"doc1_data"},
                    },
                }
            ]
        ),
        HumanMessage(
            content=[
                {
                    "type": "image",
                    "image": {"format": "jpeg", "source": {"bytes": b"image_data"}},
                }
            ]
        ),
        HumanMessage(
            content=[
                {
                    "type": "video",
                    "video": {"format": "mp4", "source": {"bytes": b"video_data"}},
                }
            ]
        ),
        HumanMessage(
            content=[
                {
                    "type": "video",
                    "video": {
                        "format": "mp4",
                        "source": {"s3Location": {"uri": "s3_url"}},
                    },
                }
            ]
        ),
    ]
    expected_messages = [
        {"role": "user", "content": [{"text": "hu1"}, {"text": "hu2"}]},
        {
            "role": "assistant",
            "content": [
                {"text": "ai1"},
                {
                    "toolUse": {
                        "toolUseId": "tool_call2",
                        "input": {"arg2": 2},
                        "name": "tool2",
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "tool_call3",
                        "input": {"arg3": ["a", "b"]},
                        "name": "tool3",
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "tool_call1",
                        "input": {"arg1": "arg1"},
                        "name": "tool1",
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"text": "hu3"},
                {
                    "toolResult": {
                        "toolUseId": "tool_call1",
                        "content": [{"text": "tool_res1"}],
                        "status": "success",
                    }
                },
                {
                    "toolResult": {
                        "toolUseId": "tool_call2",
                        "content": [
                            {
                                "image": {
                                    "format": "jpeg",
                                    "source": {
                                        "bytes": base64.b64decode(
                                            "data".encode("utf-8")
                                        )
                                    },
                                }
                            }
                        ],
                        "status": "error",
                    }
                },
                {
                    "toolResult": {
                        "toolUseId": "tool_call3",
                        "content": [{"text": "tool_res3"}],
                        "status": "success",
                    }
                },
                {"guardContent": {"text": {"text": "hu5"}}},
                {"guardContent": {"text": {"text": "hu6"}}},
                {
                    "document": {
                        "format": "pdf",
                        "name": "doc1",
                        "source": {"bytes": b"doc1_data"},
                    }
                },
                {"image": {"format": "jpeg", "source": {"bytes": b"image_data"}}},
                {"video": {"format": "mp4", "source": {"bytes": b"video_data"}}},
                {
                    "video": {
                        "format": "mp4",
                        "source": {"s3Location": {"uri": "s3_url"}},
                    }
                },
            ],
        },
    ]
    expected_system = [
        {"text": "sys1"},
        {"text": "sys2"},
        {"text": "sys3"},
        {"text": "sys4"},
        {"guardContent": {"text": {"text": "sys5"}}},
        {"guardContent": {"text": {"text": "sys6"}}},
    ]
    actual_messages, actual_system = _messages_to_bedrock(messages)
    assert expected_messages == actual_messages
    assert expected_system == actual_system


def test__bedrock_to_lc() -> None:
    bedrock: List[Dict] = [
        {"text": "text1"},
        {
            "toolUse": {
                "toolUseId": "tool_call1",
                "input": {"arg1": "val1"},
                "name": "tool1",
            }
        },
        {"image": {"format": "jpeg", "source": {"bytes": b"data"}}},
        {
            "toolResult": {
                "toolUseId": "tool_call1",
                "content": [
                    {"text": "tool_text1"},
                    {"image": {"format": "png", "source": {"bytes": b"tool_img"}}},
                    {"json": {"output": [1, 2, 3]}},
                ],
            }
        },
        {"video": {"format": "mp4", "source": {"bytes": b"video_data"}}},
        {"video": {"format": "mp4", "source": {"s3Location": {"uri": "video_data"}}}},
    ]
    expected = [
        {"type": "text", "text": "text1"},
        {
            "type": "tool_use",
            "id": "tool_call1",
            "input": {"arg1": "val1"},
            "name": "tool1",
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(b"data").decode("utf-8"),
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "tool_call1",
            "content": [
                {"type": "text", "text": "tool_text1"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(b"tool_img").decode("utf-8"),
                    },
                },
                {"type": "json", "json": {"output": [1, 2, 3]}},
            ],
            "is_error": False,
        },
        {
            "type": "video",
            "source": {
                "type": "base64",
                "media_type": "video/mp4",
                "data": base64.b64encode(b"video_data").decode("utf-8"),
            },
        },
        {
            "type": "video",
            "source": {
                "type": "s3Location",
                "media_type": "video/mp4",
                "data": {"uri": "video_data"},
            },
        },
    ]
    actual = _bedrock_to_lc(bedrock)
    assert expected == actual


_CAMEL_SNAKE = [
    ("text", "text"),
    ("toolUseId", "tool_use_id"),
]


@pytest.mark.parametrize(["camel", "snake"], _CAMEL_SNAKE)
def test__snake_to_camel(camel: str, snake: str) -> None:
    assert _snake_to_camel(snake) == camel


@pytest.mark.parametrize(["camel", "snake"], _CAMEL_SNAKE)
def test__camel_to_snake(camel: str, snake: str) -> None:
    assert _camel_to_snake(camel) == snake


_CAMEL_DICT = {"toolUse": {"toolUseId": "fooBar"}}
_SNAKE_DICT = {"tool_use": {"tool_use_id": "fooBar"}}


def test__camel_to_snake_keys() -> None:
    assert _camel_to_snake_keys(_CAMEL_DICT) == _SNAKE_DICT


def test__snake_to_camel_keys() -> None:
    assert _snake_to_camel_keys(_SNAKE_DICT) == _CAMEL_DICT


def test__format_openai_image_url() -> None:
    ...


def test_standard_tracing_params() -> None:
    llm = ChatBedrockConverse(
        model="foo", temperature=0.1, max_tokens=10, region_name="us-west-2"
    )
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "amazon_bedrock",
        "ls_model_type": "chat",
        "ls_model_name": "foo",
        "ls_temperature": 0.1,
        "ls_max_tokens": 10,
    }


@pytest.mark.parametrize(
    "model_id, disable_streaming",
    [
        ("us.anthropic.claude-3-7-sonnet-20250219-v1:0", False),
        ("anthropic.claude-3-5-sonnet-20240620-v1:0", False),
        ("us.anthropic.claude-3-haiku-20240307-v1:0", False),
        ("cohere.command-r-v1:0", False),
        ("meta.llama3-1-405b-instruct-v1:0", "tool_calling"),
        ("us.meta.llama3-3-70b-instruct-v1:0", "tool_calling"),
        ("us.amazon.nova-lite-v1:0", False),
        ("us.amazon.nonstreaming-model-v1:0", True),
        ("us.deepseek.r1-v1:0", "tool_calling"),
    ],
)
def test_set_disable_streaming(
    model_id: str, disable_streaming: Union[bool, str]
) -> None:
    llm = ChatBedrockConverse(model=model_id, region_name="us-west-2")
    assert llm.disable_streaming == disable_streaming


def test__extract_response_metadata() -> None:
    response = {
        "ResponseMetadata": {
            "RequestId": "xxxxxx",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {
                "date": "Wed, 06 Nov 2024 23:28:31 GMT",
                "content-type": "application/json",
                "content-length": "212",
                "connection": "keep-alive",
                "x-amzn-requestid": "xxxxx",
            },
            "RetryAttempts": 0,
        },
        "stopReason": "end_turn",
        "metrics": {"latencyMs": 191},
    }
    response_metadata = _extract_response_metadata(response)
    assert response_metadata["metrics"]["latencyMs"] == [191]


@mock.patch.dict(os.environ, {"AWS_REGION": "us-west-1"})
def test_chat_bedrock_converse_different_regions() -> None:
    region = "ap-south-2"
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0", region_name=region
    )
    assert llm.region_name == region


def test__bedrock_to_lc_anthropic_reasoning() -> None:
    bedrock_content: List[Dict[str, Any]] = [
        # Expected LC format for non-reasoning block
        {"text": "Thought text"},
        # Invoke format with reasoning_text
        {
            "reasoning_content": {
                "reasoning_text": {"text": "Thought text", "signature": "sig"}
            }
        },
        # Streaming format with text only
        {"reasoning_content": {"text": "Thought text"}},
        # Streaming format with signature only
        {"reasoning_content": {"signature": "sig"}},
        # Expected LC format for reasoning with no text
        {"reasoning_content": {"reasoning_text": {"text": "", "signature": "sig"}}},
        # Expected LC format for reasoning with no signature
        {
            "reasoning_content": {
                "reasoning_text": {"text": "Another reasoning block", "signature": ""}
            }
        },
    ]

    expected_lc = [
        # Expected LC format for non-reasoning block
        {"type": "text", "text": "Thought text"},
        # Expected LC format for invoke reasoning_text
        {
            "type": "reasoning_content",
            "reasoning_content": {
                "type": "text",
                "text": "Thought text",
                "signature": "sig",
            },
        },
        # Expected LC format for streaming text
        {
            "type": "reasoning_content",
            "reasoning_content": {"type": "text", "text": "Thought text"},
        },
        # Expected LC format for streaming signature
        {
            "type": "reasoning_content",
            "reasoning_content": {"type": "signature", "signature": "sig"},
        },
        # Expected LC format for reasoning with no text
        {
            "type": "reasoning_content",
            "reasoning_content": {"type": "text", "text": "", "signature": "sig"},
        },
        # Expected LC format for reasoning with no signature
        {
            "type": "reasoning_content",
            "reasoning_content": {
                "type": "text",
                "text": "Another reasoning block",
                "signature": "",
            },
        },
    ]

    actual = _bedrock_to_lc(bedrock_content)
    assert expected_lc == actual


def test__lc_content_to_bedrock_anthropic_reasoning() -> None:
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        # Anthropic "thinking" type
        HumanMessage(content="Solve this problem step by step: what is 27 * 14?"),
        AIMessage(
            content=[
                {
                    "type": "thinking",
                    "thinking": "To solve 27 * 14, I'll break it down into steps...",
                    "signature": "sig-123",
                },
                {"type": "text", "text": "The answer is 378."},
            ]
        ),
        # Bedrock "reasoning_content" type
        HumanMessage(content="Can you re-check your last answer?"),
        AIMessage(
            content=[
                {
                    "type": "reasoning_content",
                    "reasoning_content": {
                        "text": (
                            "To solve 27 * 14, I'll break it down:\n1. "
                            "First multiply 7 × 14 = 98\n2. Then multiply "
                            "20 × 14 = 280\n3. Add the results: 98 + 280 = 378"
                        ),
                        "signature": "math-sig-456",
                    },
                },
                {
                    "type": "text",
                    "text": "I've double-checked and confirm that 27 * 14 = 378.",
                },
            ]
        ),
    ]

    expected_messages = [
        {
            "role": "user",
            "content": [{"text": "Solve this problem step by step: what is 27 * 14?"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": (
                                "To solve 27 * 14, I'll break it down into steps..."
                            ),
                            "signature": "sig-123",
                        }
                    }
                },
                {"text": "The answer is 378."},
            ],
        },
        {"role": "user", "content": [{"text": "Can you re-check your last answer?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": (
                                "To solve 27 * 14, I'll break it down:\n1. First "
                                "multiply 7 × 14 = 98\n2. Then multiply 20 × 14 = 280"
                                "\n3. Add the results: 98 + 280 = 378"
                            ),
                            "signature": "math-sig-456",
                        }
                    }
                },
                {"text": "I've double-checked and confirm that 27 * 14 = 378."},
            ],
        },
    ]

    expected_system = [{"text": "You are a helpful assistant."}]

    actual_messages, actual_system = _messages_to_bedrock(messages)

    assert expected_messages == actual_messages
    assert expected_system == actual_system


def test__lc_content_to_bedrock_empty_signature() -> None:
    """Test that thinking and reasoning blocks without signatures are omitted."""
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        # Human message with a question
        HumanMessage(content="Tell me about machine learning."),
        # AI message with thinking block that has NO signature (should be omitted)
        AIMessage(
            content=[
                {
                    "type": "thinking",
                    "thinking": "I'll explain machine learning concepts...",
                    "signature": "",  # Empty signature
                },
                {
                    "type": "text",
                    "text": (
                        "Machine learning is a field of AI that enables systems to "
                        "learn from data."
                    ),
                },
            ]
        ),
        # Human follow-up
        HumanMessage(content="What about deep learning?"),
        # AI message with reasoning_content block
        # that has NO signature (should be omitted)
        AIMessage(
            content=[
                {
                    "type": "reasoning_content",
                    "reasoningContent": {
                        "text": (
                            "Deep learning is a subset of machine learning using "
                            "neural networks."
                        ),
                        "signature": "",  # Empty signature
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Deep learning is a subfield of machine learning that uses "
                        "neural networks with many layers."
                    ),
                },
            ]
        ),
    ]

    expected_messages = [
        {"role": "user", "content": [{"text": "Tell me about machine learning."}]},
        {
            "role": "assistant",
            "content": [
                # No reasoning block because signature is empty
                {
                    "text": (
                        "Machine learning is a field of AI that enables systems to "
                        "learn from data."
                    )
                }
            ],
        },
        {"role": "user", "content": [{"text": "What about deep learning?"}]},
        {
            "role": "assistant",
            "content": [
                # No reasoning block because signature is empty
                {
                    "text": (
                        "Deep learning is a subfield of machine learning that uses "
                        "neural networks with many layers."
                    )
                }
            ],
        },
    ]

    expected_system = [{"text": "You are a helpful assistant."}]

    actual_messages, actual_system = _messages_to_bedrock(messages)

    assert expected_messages == actual_messages
    assert expected_system == actual_system


def test__lc_content_to_bedrock_mixed_signatures() -> None:
    """Test a mix of thinking/reasoning blocks with and without signatures."""
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        # Human message
        HumanMessage(content="Explain AI and machine learning."),
        # AI message with mixed thinking/reasoning blocks (with/without signatures)
        AIMessage(
            content=[
                {
                    "type": "thinking",
                    "thinking": "I should start with AI definitions...",
                    "signature": "",  # Empty signature - should be omitted
                },
                {"type": "text", "text": "AI is a broad field of computer science."},
                {
                    "type": "thinking",
                    "thinking": "Now I'll explain machine learning...",
                    "signature": "signature-xyz",  # Has signature - should be included
                },
                {"type": "text", "text": "Machine learning is a subset of AI."},
            ]
        ),
        # Human follow-up
        HumanMessage(content="What about neural networks?"),
        # AI response with mixed reasoning_content blocks
        AIMessage(
            content=[
                {
                    "type": "reasoning_content",
                    "reasoningContent": {
                        "text": "Neural networks are inspired by biological neurons.",
                        "signature": (
                            "nn-signature"
                        ),  # Has signature - should be included
                    },
                },
                {
                    "type": "reasoning_content",
                    "reasoningContent": {
                        "text": "They consist of interconnected layers of nodes.",
                        "signature": "",  # Empty signature - should be omitted
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Neural networks are computing systems inspired by biological "
                        "neural networks."
                    ),
                },
            ]
        ),
    ]

    expected_messages = [
        {"role": "user", "content": [{"text": "Explain AI and machine learning."}]},
        {
            "role": "assistant",
            "content": [
                # First thinking block omitted (empty signature)
                {"text": "AI is a broad field of computer science."},
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "Now I'll explain machine learning...",
                            "signature": "signature-xyz",
                        }
                    }
                },
                {"text": "Machine learning is a subset of AI."},
            ],
        },
        {"role": "user", "content": [{"text": "What about neural networks?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": (
                                "Neural networks are inspired by biological neurons."
                            ),
                            "signature": "nn-signature",
                        }
                    }
                },
                # Second reasoning block omitted (empty signature)
                {
                    "text": (
                        "Neural networks are computing systems inspired by biological "
                        "neural networks."
                    )
                },
            ],
        },
    ]

    expected_system = [{"text": "You are a helpful assistant."}]

    actual_messages, actual_system = _messages_to_bedrock(messages)

    assert expected_messages == actual_messages
    assert expected_system == actual_system
    """Test that reasoning_content blocks without signatures are omitted."""
    # Test with empty signature
    content: List[Union[str, Dict[str, Any]]] = [
        {"type": "text", "text": "Some text"},
        {
            "type": "reasoning_content",
            "reasoningContent": {"text": "This is reasoning", "signature": ""},
        },
    ]

    bedrock_content = _lc_content_to_bedrock(content)

    # Verify reasoning_content block was omitted because it has an empty signature
    assert len(bedrock_content) == 1
    assert bedrock_content[0] == {"text": "Some text"}

    # Test with signature present
    content = [
        {"type": "text", "text": "Some text"},
        {
            "type": "reasoning_content",
            "reasoningContent": {
                "text": "This is reasoning",
                "signature": "some-signature",
            },
        },
    ]

    bedrock_content = _lc_content_to_bedrock(content)

    # Verify that the reasoning_content block is included when it has a signature
    assert len(bedrock_content) == 2
    assert bedrock_content[0] == {"text": "Some text"}
    assert bedrock_content[1] == {
        "reasoningContent": {
            "reasoningText": {
                "text": "This is reasoning",
                "signature": "some-signature",
            }
        }
    }


def test__lc_content_to_bedrock_reasoning_content_signature() -> None:
    """
    Test that reasoning_content blocks with and without signatures
    are handled correctly.
    """
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        # Human message with a question
        HumanMessage(content="Explain quantum computing."),
        # AI message with reasoning_content blocks with and without signatures
        AIMessage(
            content=[
                {
                    "type": "reasoning_content",
                    "reasoningContent": {
                        "text": "Quantum computing uses quantum bits or qubits...",
                        "signature": (
                            "qc-signature"
                        ),  # Has signature - should be included
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Quantum computing is a type of computation that harnesses "
                        "quantum mechanics."
                    ),
                },
            ]
        ),
        # Human follow-up
        HumanMessage(content="How does it differ from classical computing?"),
        # AI response with reasoning_content block without signature
        AIMessage(
            content=[
                {
                    "type": "reasoning_content",
                    "reasoningContent": {
                        "text": (
                            "Unlike classical bits that are either 0 or 1, qubits can "
                            "be in superposition."
                        ),
                        "signature": "",  # Empty signature - should be omitted
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Classical computers use bits as the smallest unit of data, "
                        "while quantum computers use qubits which can exist in "
                        "multiple states simultaneously."
                    ),
                },
            ]
        ),
        # Another follow-up
        HumanMessage(content="What are the practical applications?"),
        # AI response with multiple reasoning_content blocks with mixed signatures
        AIMessage(
            content=[
                {
                    "type": "reasoning_content",
                    "reasoningContent": {
                        "text": (
                            "Quantum computing excels at certain types of problems..."
                        ),
                        "signature": (
                            "apps-signature"
                        ),  # Has signature - should be included
                    },
                },
                {
                    "type": "reasoning_content",
                    "reasoningContent": {
                        "text": (
                            "Examples include cryptography, optimization, and "
                            "simulation."
                        ),
                        "signature": "",  # Empty signature - should be omitted
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Practical applications include cryptography, drug discovery, "
                        "materials science, and complex optimization problems."
                    ),
                },
            ]
        ),
    ]

    expected_messages = [
        {"role": "user", "content": [{"text": "Explain quantum computing."}]},
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": "Quantum computing uses quantum bits or qubits...",
                            "signature": "qc-signature",
                        }
                    }
                },
                {
                    "text": (
                        "Quantum computing is a type of computation that harnesses "
                        "quantum mechanics."
                    )
                },
            ],
        },
        {
            "role": "user",
            "content": [{"text": "How does it differ from classical computing?"}],
        },
        {
            "role": "assistant",
            "content": [
                # No reasoning block because signature is empty
                {
                    "text": (
                        "Classical computers use bits as the smallest unit of data, "
                        "while quantum computers use qubits which can exist in "
                        "multiple states simultaneously."
                    )
                }
            ],
        },
        {"role": "user", "content": [{"text": "What are the practical applications?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": (
                                "Quantum computing excels at certain types of "
                                "problems..."
                            ),
                            "signature": "apps-signature",
                        }
                    }
                },
                # Second reasoning block omitted (empty signature)
                {
                    "text": (
                        "Practical applications include cryptography, drug discovery, "
                        "materials science, and complex optimization problems."
                    )
                },
            ],
        },
    ]

    expected_system = [{"text": "You are a helpful assistant."}]

    actual_messages, actual_system = _messages_to_bedrock(messages)

    assert expected_messages == actual_messages
    assert expected_system == actual_system
