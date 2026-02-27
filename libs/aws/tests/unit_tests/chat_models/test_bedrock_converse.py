"""Test chat model integration."""

import base64
import os
from typing import Any, Dict, Iterator, List, Literal, Tuple, Type, Union, cast
from unittest import mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableBinding
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel, Field
from syrupy import SnapshotAssertion

from langchain_aws import ChatBedrockConverse
from langchain_aws.chat_models.bedrock_converse import (
    _bedrock_to_lc,
    _camel_to_snake,
    _camel_to_snake_keys,
    _convert_tool_blocks_to_text,
    _extract_response_metadata,
    _format_data_content_block,
    _format_tools,
    _has_tool_use_or_result_blocks,
    _lc_content_to_bedrock,
    _messages_to_bedrock,
    _parse_stream_event,
    _snake_to_camel,
    _snake_to_camel_keys,
)
from langchain_aws.function_calling import convert_to_anthropic_tool


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

    @pytest.mark.xfail(reason="Pending mapping + init validator in core")
    def test_serdes(self, model: BaseChatModel, snapshot: SnapshotAssertion) -> None:
        super().test_serdes(model, snapshot)


def test_profile() -> None:
    model = ChatBedrockConverse(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-west-2",
    )
    assert model.profile
    assert not model.profile["reasoning_output"]

    model = ChatBedrockConverse(
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        region_name="us-west-2",
    )
    assert model.profile
    assert model.profile["reasoning_output"]

    model = ChatBedrockConverse(model="foo")
    assert model.profile == {}


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


@pytest.mark.parametrize(
    "thinking_model",
    [
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "anthropic.claude-opus-4-20250514-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
)
def test_anthropic_thinking_bind_tools_tool_choice(thinking_model: str) -> None:
    chat_model = ChatBedrockConverse(
        model=thinking_model,
        region_name="us-west-2",
        additional_model_request_fields={
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    )
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "auto": {}
    }
    with pytest.raises(ValueError):
        chat_model.bind_tools([GetWeather], tool_choice="any")
    with pytest.raises(ValueError):
        chat_model.bind_tools([GetWeather], tool_choice="GetWeather")
    with pytest.raises(ValueError):
        chat_model.bind_tools(
            [GetWeather], tool_choice={"tool": {"name": "GetWeather"}}
        )


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


@pytest.mark.parametrize(
    "model, should_support_auto",
    [
        ("us.meta.llama4-maverick-17b-instruct-v1:0", True),
        ("us.meta.llama3-3-70b-instruct-v1:0", True),
        ("us.meta.llama3-2-90b-instruct-v1:0", True),
        ("us.meta.llama3-2-1b-instruct-v1:0", False),
        ("us.meta.llama3-1-405b-instruct-v1:0", True),
        ("meta.llama3-70b-instruct-v1:0", False),
    ],
)
def test_llama_bind_tools_tool_choice_variants(
    model: str, should_support_auto: bool
) -> None:
    chat_model = ChatBedrockConverse(model=model, region_name="us-east-1")  # type: ignore[call-arg]

    if should_support_auto:
        chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="auto")
        assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
            "auto": {}
        }
        with pytest.raises(ValueError):
            chat_model.bind_tools([GetWeather], tool_choice="any")
        with pytest.raises(ValueError):
            chat_model.bind_tools([GetWeather], tool_choice="GetWeather")
    else:
        with pytest.raises(ValueError):
            chat_model.bind_tools([GetWeather], tool_choice="auto")
        with pytest.raises(ValueError):
            chat_model.bind_tools([GetWeather], tool_choice="any")
        with pytest.raises(ValueError):
            chat_model.bind_tools([GetWeather], tool_choice="GetWeather")


@pytest.mark.parametrize(
    "model,expected_values",
    [
        ("us.deepseek.r1-v1:0", ()),
        ("deepseek.v3-v1:0", ("any",)),
        (
            "deepseek.v3-x:0",
            (
                "any",
                "tool",
            ),
        ),
    ],
)
def test_deepseek_supports_tool_choice_values(
    model: str, expected_values: tuple[Literal["auto", "any", "tool"], ...]
) -> None:
    chat_model = ChatBedrockConverse(model=model, region_name="us-east-1")
    assert chat_model.supports_tool_choice_values == expected_values


def test_deepseek_r1_no_tool_choice_support() -> None:
    chat_model = ChatBedrockConverse(model="deepseek.r1-v1:0", region_name="us-east-1")  # type: ignore[call-arg]

    assert chat_model.supports_tool_choice_values == ()

    with pytest.raises(ValueError):
        chat_model.bind_tools([GetWeather], tool_choice="auto")

    with pytest.raises(ValueError):
        chat_model.bind_tools([GetWeather], tool_choice="any")

    with pytest.raises(ValueError):
        chat_model.bind_tools(
            [GetWeather], tool_choice={"tool": {"name": "GetWeather"}}
        )

    with pytest.raises(ValueError):
        chat_model.bind_tools([GetWeather], tool_choice="GetWeather")


def test_deepseek_v3_bind_tools_tool_choice_variants() -> None:
    chat_model = ChatBedrockConverse(model="deepseek.v3-v1:0", region_name="us-east-1")  # type: ignore[call-arg]

    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "any": {}
    }

    with pytest.raises(ValueError):
        chat_model.bind_tools([GetWeather], tool_choice="auto")

    with pytest.raises(ValueError):
        chat_model.bind_tools([GetWeather], tool_choice="GetWeather")


def test_deepseek_v3_bind_tools_default_tool_choice() -> None:
    chat_model = ChatBedrockConverse(model="deepseek.v3-v1:0", region_name="us-east-1")  # type: ignore[call-arg]

    chat_model_with_tools = chat_model.bind_tools([GetWeather])
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


def test_messages_to_bedrock_with_cache_point() -> None:
    messages = [
        HumanMessage(content=["Hello!", {"cachePoint": {"type": "default"}}]),
        ToolMessage(
            content=[
                {"type": "text", "text": "Tool response"},
                {"cachePoint": {"type": "default"}},
            ],
            tool_call_id="tool-123",
            status="success",
        ),
    ]

    actual_messages, actual_system = _messages_to_bedrock(messages)
    expected_messages = [
        {
            "role": "user",
            "content": [
                {"text": "Hello!"},
                {"cachePoint": {"type": "default"}},
                {
                    "toolResult": {
                        "content": [{"text": "Tool response"}],
                        "status": "success",
                        "toolUseId": "tool-123",
                    }
                },
                {"cachePoint": {"type": "default"}},
            ],
        }
    ]
    assert expected_messages == actual_messages
    assert [] == actual_system


def test__messages_to_bedrock_empty_list() -> None:
    messages: List[BaseMessage] = []
    actual_messages, actual_system = _messages_to_bedrock(messages)

    expected_messages: List[Dict] = [{"role": "user", "content": [{"text": "."}]}]
    expected_system: List[Dict] = []

    assert expected_messages == actual_messages
    assert expected_system == actual_system


def test__messages_to_bedrock_system_only() -> None:
    messages: List[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant.")
    ]
    actual_messages, actual_system = _messages_to_bedrock(messages)

    expected_messages: List[Dict] = [{"role": "user", "content": [{"text": "."}]}]
    expected_system: List[Dict] = [{"text": "You are a helpful assistant."}]

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


def test__format_openai_image_url() -> None: ...


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
        "ls_invocation_params": {
            "provider": "foo",
            "region_name": "us-west-2",
        },
    }


@pytest.mark.parametrize(
    "model_id, disable_streaming",
    [
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", False),
        ("us.anthropic.claude-sonnet-4-20250514-v1:0", False),
        ("us.anthropic.claude-opus-4-20250514-v1:0", False),
        ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", False),
        ("us.anthropic.claude-3-haiku-20240307-v1:0", False),
        ("cohere.command-r-v1:0", False),
        ("meta.llama3-1-405b-instruct-v1:0", "tool_calling"),
        ("us.meta.llama3-3-70b-instruct-v1:0", "tool_calling"),
        ("us.amazon.nova-lite-v1:0", False),
        ("us.amazon.nova-2-lite-v1:0", False),
        ("us.amazon.nonstreaming-model-v1:0", True),
        ("us.deepseek.r1-v1:0", "tool_calling"),
        ("deepseek.v3-v1:0", False),
        ("openai.gpt-oss-120b-1:0", False),
        ("openai.gpt-oss-20b-1:0", False),
        ("qwen.qwen3-32b-v1:0", False),
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
                "text": "Thought text",
                "signature": "sig",
            },
        },
        # Expected LC format for streaming text
        {
            "type": "reasoning_content",
            "reasoning_content": {"text": "Thought text"},
        },
        # Expected LC format for streaming signature
        {
            "type": "reasoning_content",
            "reasoning_content": {"signature": "sig"},
        },
        # Expected LC format for reasoning with no text
        {
            "type": "reasoning_content",
            "reasoning_content": {"text": "", "signature": "sig"},
        },
        # Expected LC format for reasoning with no signature
        {
            "type": "reasoning_content",
            "reasoning_content": {
                "text": "Another reasoning block",
                "signature": "",
            },
        },
    ]

    actual = _bedrock_to_lc(bedrock_content)
    assert expected_lc == actual


def test__bedrock_to_lc_nova_reasoning_content() -> None:
    """Test parsing Nova's reasoningContent format."""
    bedrock_content: List[Dict[str, Any]] = [
        # Nova's reasoningContent format
        {
            "reasoningContent": {
                "reasoningText": {
                    "text": "I need to search for information about the Nobel Prize..."
                }
            }
        },
        # System tool use block
        {
            "toolUse": {
                "toolUseId": "tool_123",
                "name": "nova_grounding",
                "input": {"query": "2024 Nobel Prize Physics"},
            }
        },
        # System tool result block
        {
            "toolResult": {
                "toolUseId": "tool_123",
                "content": [
                    {"text": "The 2024 Nobel Prize in Physics was awarded to..."}
                ],
                "status": "success",
            }
        },
        # Final text response
        {
            "text": (
                "Based on my search, the 2024 Nobel Prize in Physics was awarded to..."
            )
        },
    ]

    expected_lc = [
        # Reasoning content should be parsed
        {
            "type": "reasoning_content",
            "reasoning_content": {
                "text": "I need to search for information about the Nobel Prize...",
                "signature": "",
            },
        },
        # System tool use
        {
            "type": "tool_use",
            "id": "tool_123",
            "name": "nova_grounding",
            "input": {"query": "2024 Nobel Prize Physics"},
        },
        # System tool result
        {
            "type": "tool_result",
            "tool_use_id": "tool_123",
            "content": [
                {
                    "type": "text",
                    "text": "The 2024 Nobel Prize in Physics was awarded to...",
                }
            ],
            "is_error": False,
        },
        # Text response
        {
            "type": "text",
            "text": (
                "Based on my search, the 2024 Nobel Prize in Physics was awarded to..."
            ),
        },
    ]

    actual = _bedrock_to_lc(bedrock_content)
    assert expected_lc == actual


def test__bedrock_to_lc_nova_code_interpreter() -> None:
    """Test parsing Nova code interpreter tool blocks."""
    bedrock_content: List[Dict[str, Any]] = [
        # Code interpreter tool use
        {
            "toolUse": {
                "toolUseId": "code_123",
                "name": "nova_code_interpreter",
                "input": {"code": "import math\nmath.sqrt(475878756857)"},
            }
        },
        # Code interpreter result
        {
            "toolResult": {
                "toolUseId": "code_123",
                "content": [{"text": "689690.0"}],
                "status": "success",
            }
        },
    ]

    expected_lc = [
        {
            "type": "tool_use",
            "id": "code_123",
            "name": "nova_code_interpreter",
            "input": {"code": "import math\nmath.sqrt(475878756857)"},
        },
        {
            "type": "tool_result",
            "tool_use_id": "code_123",
            "content": [{"type": "text", "text": "689690.0"}],
            "is_error": False,
        },
    ]

    actual = _bedrock_to_lc(bedrock_content)
    assert expected_lc == actual


def test__bedrock_to_lc_redacted_tool_result() -> None:
    """Test handling redacted/empty tool results."""
    bedrock_content: List[Dict[str, Any]] = [
        {
            "toolUse": {
                "toolUseId": "tool_456",
                "name": "nova_grounding",
                "input": {"query": "test query"},
            }
        },
        # Redacted result with empty content
        {
            "toolResult": {
                "toolUseId": "tool_456",
                "content": [],  # Empty/redacted
                "status": "success",
            }
        },
    ]

    expected_lc = [
        {
            "type": "tool_use",
            "id": "tool_456",
            "name": "nova_grounding",
            "input": {"query": "test query"},
        },
        {
            "type": "tool_result",
            "tool_use_id": "tool_456",
            "content": [],  # Empty content preserved
            "is_error": False,
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


def test__lc_content_to_bedrock_reasoning_content_snake_case() -> None:
    """Test that reasoning_content blocks with snake case are
    handled correctly.
    """
    persisted_ai_content: List[Union[str, Dict[str, Any]]] = [
        {
            "type": "reasoning_content",
            "reasoning_content": {
                "text": "Let me think about this step by step...",
                "signature": "abc123signature",
            },
        },
        {"type": "text", "text": "Here is my response."},
    ]

    bedrock_content = _lc_content_to_bedrock(persisted_ai_content)

    assert len(bedrock_content) == 2
    assert "reasoningContent" in bedrock_content[0]
    assert "reasoningText" in bedrock_content[0]["reasoningContent"]
    assert (
        bedrock_content[0]["reasoningContent"]["reasoningText"]["text"]
        == "Let me think about this step by step..."
    )
    assert (
        bedrock_content[0]["reasoningContent"]["reasoningText"]["signature"]
        == "abc123signature"
    )

    assert bedrock_content[1] == {"text": "Here is my response."}


def test__lc_content_to_bedrock_reasoning_content_camel_case() -> None:
    """Test that reasoning_content blocks with camelcase keys are
    handled correctly.
    """
    content_camelcase: List[Union[str, Dict[str, Any]]] = [
        {
            "type": "reasoning_content",
            "reasoningContent": {
                "text": "Thinking in camelCase...",
                "signature": "a_signature",
            },
        },
        {"type": "text", "text": "Response text."},
    ]

    bedrock_content = _lc_content_to_bedrock(content_camelcase)

    assert len(bedrock_content) == 2
    assert "reasoningContent" in bedrock_content[0]
    assert (
        bedrock_content[0]["reasoningContent"]["reasoningText"]["text"]
        == "Thinking in camelCase..."
    )
    assert (
        bedrock_content[0]["reasoningContent"]["reasoningText"]["signature"]
        == "a_signature"
    )


def test__lc_content_to_bedrock_mime_types() -> None:
    video_data = base64.b64encode(b"video_test_data").decode("utf-8")
    image_data = base64.b64encode(b"image_test_data").decode("utf-8")
    file_data = base64.b64encode(b"file_test_data").decode("utf-8")

    # Create content with one of each type
    content: List[Union[str, Dict[str, Any]]] = [
        {
            "type": "video",
            "source": {
                "type": "base64",
                "mediaType": "video/mp4",
                "data": video_data,
            },
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "mediaType": "image/jpeg",
                "data": image_data,
            },
        },
        {
            "type": "file",
            "sourceType": "base64",
            "mimeType": "application/pdf",
            "data": file_data,
            "name": "test_document.pdf",
        },
    ]

    expected_content = [
        {
            "video": {
                "format": "mp4",
                "source": {"bytes": base64.b64decode(video_data.encode("utf-8"))},
            }
        },
        {
            "image": {
                "format": "jpeg",
                "source": {"bytes": base64.b64decode(image_data.encode("utf-8"))},
            }
        },
        {
            "document": {
                "format": "pdf",
                "name": "test_document.pdf",
                "source": {"bytes": base64.b64decode(file_data.encode("utf-8"))},
            }
        },
    ]

    bedrock_content = _lc_content_to_bedrock(content)
    assert bedrock_content == expected_content


def test__lc_content_to_bedrock_mime_types_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid MIME type format"):
        _lc_content_to_bedrock(
            [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "mediaType": "invalidmimetype",
                        "data": base64.b64encode(b"test_data").decode("utf-8"),
                    },
                }
            ]
        )

    with pytest.raises(ValueError, match="Unsupported MIME type"):
        _lc_content_to_bedrock(
            [
                {
                    "type": "file",
                    "sourceType": "base64",
                    "mimeType": "application/unknown-format",
                    "data": base64.b64encode(b"test_data").decode("utf-8"),
                    "name": "test_document.xyz",
                }
            ]
        )


def test__lc_content_to_bedrock_empty_content() -> None:
    content: List[Union[str, Dict[str, Any]]] = []

    bedrock_content = _lc_content_to_bedrock(content)

    assert len(bedrock_content) > 0
    assert bedrock_content[0]["text"] == "."


def test__lc_content_to_bedrock_whitespace_only_content() -> None:
    content = "   \n  \t  "

    bedrock_content = _lc_content_to_bedrock(content)

    assert len(bedrock_content) > 0
    assert bedrock_content[0]["text"] == "."


def test__lc_content_to_bedrock_empty_string_content() -> None:
    content = ""

    bedrock_content = _lc_content_to_bedrock(content)

    assert len(bedrock_content) > 0
    assert bedrock_content[0]["text"] == "."


def test__lc_content_to_bedrock_mixed_empty_content() -> None:
    content: List[Union[str, Dict[str, Any]]] = [
        {"type": "text", "text": ""},
        {"type": "text", "text": "   "},
        {"type": "text", "text": ""},
    ]

    bedrock_content = _lc_content_to_bedrock(content)

    assert len(bedrock_content) > 0
    assert bedrock_content[0]["text"] == "."


def test__lc_content_to_bedrock_empty_text_block() -> None:
    content: List[Union[str, Dict[str, Any]]] = [{"type": "text", "text": ""}]

    bedrock_content = _lc_content_to_bedrock(content)

    assert len(bedrock_content) > 0
    assert bedrock_content[0]["text"] == "."


def test__lc_content_to_bedrock_whitespace_text_block() -> None:
    content: List[Union[str, Dict[str, Any]]] = [{"type": "text", "text": "  \n  "}]

    bedrock_content = _lc_content_to_bedrock(content)

    assert len(bedrock_content) > 0
    assert bedrock_content[0]["text"] == "."


def test__lc_content_to_bedrock_mixed_valid_and_empty_content() -> None:
    content: List[Union[str, Dict[str, Any]]] = [
        {"type": "text", "text": "Valid text"},
        {"type": "text", "text": ""},
        {"type": "text", "text": "   "},
    ]

    bedrock_content = _lc_content_to_bedrock(content)

    assert len(bedrock_content) == 3
    assert bedrock_content[0]["text"] == "Valid text"
    assert bedrock_content[1]["text"] == "."
    assert bedrock_content[2]["text"] == "."


def test__lc_content_to_bedrock_mixed_types_with_empty_content() -> None:
    content: List[Union[str, Dict[str, Any]]] = [
        {"type": "text", "text": "Valid text"},
        {
            "type": "tool_use",
            "id": "tool_call1",
            "input": {"arg1": "val1"},
            "name": "tool1",
        },
        {"type": "text", "text": "   "},
    ]

    expected = [
        {"text": "Valid text"},
        {
            "toolUse": {
                "toolUseId": "tool_call1",
                "input": {"arg1": "val1"},
                "name": "tool1",
            }
        },
        {"text": "."},
    ]

    bedrock_content = _lc_content_to_bedrock(content)

    assert len(bedrock_content) == 3
    assert bedrock_content == expected


def test__lc_content_to_bedrock_non_standard_unwrap() -> None:
    """Test that `non_standard` blocks are unwrapped to their original value.

    When content passes through `langchain-core`'s `content_blocks`
    normalization, blocks like `cachePoint` get wrapped as `non_standard`.
    `_lc_content_to_bedrock` should unwrap them back to the original block.
    """
    content: List[Union[str, Dict[str, Any]]] = [
        {"type": "text", "text": "System prompt"},
        {"type": "non_standard", "value": {"cachePoint": {"type": "default"}}},
    ]

    bedrock_content = _lc_content_to_bedrock(content)

    assert bedrock_content == [
        {"text": "System prompt"},
        {"cachePoint": {"type": "default"}},
    ]


def test__lc_content_to_bedrock_non_standard_unwrap_guard_content() -> None:
    """Test that `non_standard` blocks with `guardContent` are unwrapped."""
    content: List[Union[str, Dict[str, Any]]] = [
        {
            "type": "non_standard",
            "value": {"guardContent": {"text": {"text": "guard"}}},
        },
    ]

    bedrock_content = _lc_content_to_bedrock(content)

    assert bedrock_content == [
        {"guardContent": {"text": {"text": "guard"}}},
    ]


def test__get_provider() -> None:
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-west-2"
    )

    assert llm.provider == "anthropic"

    llm = ChatBedrockConverse(
        model="arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        provider="anthropic",
        region_name="us-west-2",
    )

    assert llm.provider == "anthropic"

    with pytest.raises(
        ValueError,
        match="Model provider should be supplied when passing a model ARN as model_id.",
    ):
        ChatBedrockConverse(
            model="arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-west-2",
        )


def test__get_base_model() -> None:
    llm_model_only = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-west-2"
    )

    assert llm_model_only._get_base_model() == "anthropic.claude-3-sonnet-20240229-v1:0"

    llm_with_regional_model = ChatBedrockConverse(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0", region_name="us-west-2"
    )

    assert (
        llm_with_regional_model._get_base_model()
        == "anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    llm_with_base_model = ChatBedrockConverse(
        model="arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        base_model="anthropic.claude-3-sonnet-20240229-v1:0",
        provider="anthropic",
        region_name="us-west-2",
    )

    assert (
        llm_with_base_model._get_base_model()
        == "anthropic.claude-3-sonnet-20240229-v1:0"
    )


@pytest.mark.parametrize(
    "arn_model_id, base_model_id, provider, expected_disable_streaming",
    [
        (
            "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic",
            False,
        ),
        (
            "arn:aws:bedrock:us-west-2::custom-model/anthropic.claude-v2:1/MyModel",
            "anthropic.claude-v2:1",
            "anthropic",
            "tool_calling",
        ),
        (
            "arn:aws:bedrock:us-west-2::custom-model/meta.llama3-8b-instruct-v1:0/MyModel",
            "meta.llama3-8b-instruct-v1:0",
            "meta",
            "tool_calling",
        ),
        (
            "arn:aws:bedrock:us-west-2::custom-model/meta.llama2-70b-chat-v1/MyModel",
            "meta.llama2-70b-chat-v1",
            "meta",
            "tool_calling",
        ),
        (
            "arn:aws:bedrock:us-west-2::custom-model/mistral.mistral-large-2402-v1:0/MyModel",
            "mistral.mistral-large-2402-v1:0",
            "mistral",
            "tool_calling",
        ),
        (
            "arn:aws:bedrock:us-west-2::custom-model/cohere.command-r-v1:0/MyModel",
            "cohere.command-r-v1:0",
            "cohere",
            False,
        ),
        (
            "arn:aws:bedrock:us-west-2::custom-model/amazon.nova-8b/MyModel",
            "amazon.nova-8b",
            "amazon",
            True,
        ),
        (
            "arn:aws:bedrock:us-west-2::custom-model/amazon.titan-text-express-v1/MyModel",
            "amazon.titan-text-express-v1",
            "amazon",
            "tool_calling",
        ),
        (
            "arn:aws:sagemaker:us-west-2::endpoint/endpoint-quick-start-xxxxx",
            "deepseek.r1-v1:0",
            "deepseek",
            "tool_calling",
        ),
    ],
)
def test_disable_streaming_with_arn(
    arn_model_id: str,
    base_model_id: str,
    provider: str,
    expected_disable_streaming: Union[bool, str],
) -> None:
    """Test that disable_streaming is properly set when base_model is provided."""
    llm = ChatBedrockConverse(
        model=arn_model_id,
        base_model=base_model_id,
        provider=provider,
        region_name="us-west-2",
    )
    assert llm.disable_streaming == expected_disable_streaming


def test_create_cache_point() -> None:
    """Test creating a cache point configuration"""
    cache_point = ChatBedrockConverse.create_cache_point()
    assert cache_point["cachePoint"]["type"] == "default"


def test_anthropic_tool_with_cache_point() -> None:
    """Test convert_to_anthropic_tool with cache point"""
    # Test with cache point
    cache_point = {"cachePoint": {"type": "default"}}

    # Test with other tool types
    tool_dict = {
        "name": "calculator",
        "description": "A tool that performs calculations",
        "input_schema": {"properties": {}},
    }
    result = convert_to_anthropic_tool(tool_dict)
    assert result["name"] == "calculator"
    assert result["description"] == "A tool that performs calculations"

    # Test bind_tools with cache point
    chat_model = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", region_name="us-east-1"
    )
    chat_model_with_tools = chat_model.bind_tools([tool_dict, cache_point])

    # Verify that both the tool_dict and cache_point are in the tools list
    runnable_binding = cast(RunnableBinding, chat_model_with_tools)
    tools = runnable_binding.kwargs.get("tools", [])

    # Assert that we have two tools
    assert len(tools) == 2

    # Check that the cache_point was passed through unchanged
    cache_points = [t for t in tools if "cachePoint" in t]
    assert len(cache_points) == 1


def test_model_kwargs() -> None:
    """Test we can transfer unknown params to model_kwargs."""
    llm = ChatBedrockConverse(
        model="my-model",
        region_name="us-west-2",
        system=["System message"],
        additional_model_request_fields={"foo": "bar"},
    )
    assert llm.model_id == "my-model"
    assert llm.region_name == "us-west-2"
    assert llm.system == ["System message"]
    assert llm.additional_model_request_fields == {"foo": "bar"}

    with pytest.warns(
        UserWarning,
        match="uses 'additional_model_request_fields' instead of 'model_kwargs'",
    ):
        llm = ChatBedrockConverse(
            model="my-model",
            region_name="us-west-2",
            model_kwargs={"foo": "bar"},  # type: ignore[call-arg]
        )
    assert llm.model_id == "my-model"
    assert llm.region_name == "us-west-2"
    assert llm.additional_model_request_fields == {"foo": "bar"}

    with pytest.warns(match="transferred to model_kwargs"):
        llm = ChatBedrockConverse(  # type: ignore[call-arg]
            model="my-model",
            region_name="us-west-2",
            foo="bar",  # type: ignore[call-arg]
        )
    assert llm.model_id == "my-model"
    assert llm.region_name == "us-west-2"
    assert llm.additional_model_request_fields == {"foo": "bar"}

    with pytest.warns(match="transferred to model_kwargs"):
        llm = ChatBedrockConverse(  # type: ignore[call-arg]
            model="my-model",
            region_name="us-west-2",
            foo="bar",  # type: ignore[call-arg]
            additional_model_request_fields={"baz": "qux"},
        )
    assert llm.model_id == "my-model"
    assert llm.region_name == "us-west-2"
    assert llm.additional_model_request_fields == {"foo": "bar", "baz": "qux"}

    # For backward compatibility, test that we don't transfer known parameters out
    # of model_kwargs
    llm = ChatBedrockConverse(
        model="my-model",
        region_name="us-west-2",
        additional_model_request_fields={"temperature": 0.2},
    )
    assert llm.additional_model_request_fields == {"temperature": 0.2}
    assert llm.temperature is None


def _create_mock_llm_guard_last_turn_only() -> Tuple[
    ChatBedrockConverse, mock.MagicMock
]:
    """Utility to create an LLM with guard_last_turn_only=True and a mocked client."""
    mocked_client = mock.MagicMock()
    llm = ChatBedrockConverse(
        client=mocked_client,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
        guard_last_turn_only=True,
        guardrails={"guardrailId": "dummy-guardrail", "guardrailVersion": "1"},
    )
    return llm, mocked_client


def test_guard_last_turn_only_no_guardrail_config() -> None:
    """Test that an error is raised if guard_last_turn_only is True but no
    guardrail_config is provided."""
    with pytest.raises(ValueError):
        ChatBedrockConverse(
            client=mock.MagicMock(),
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-west-2",
            guard_last_turn_only=True,
        )


def test_generate_guard_last_turn_only() -> None:
    """Test that _generate() wraps ONLY the final user turn with guardContent."""
    llm, mocked_client = _create_mock_llm_guard_last_turn_only()

    mocked_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "ok"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }

    messages = [
        HumanMessage(content="First user message"),
        AIMessage(content="Assistant reply"),
        HumanMessage(content="Second user message"),
    ]

    llm.invoke(messages)
    _, kwargs = mocked_client.converse.call_args
    bedrock_msgs = kwargs["messages"]

    assert bedrock_msgs[0]["content"][0] == {"text": "First user message"}
    # Last user turn is wrapped in guardContent
    assert bedrock_msgs[-1]["content"][0] == {
        "guardContent": {"text": {"text": "Second user message"}}
    }


def test_stream_guard_last_turn_only() -> None:
    """Test that stream() applies guardContent to final user turn."""
    llm, mocked_client = _create_mock_llm_guard_last_turn_only()

    mocked_client.converse_stream.return_value = {
        "stream": [{"messageStart": {"role": "assistant"}}]
    }

    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi!"),
        HumanMessage(content="How are you?"),
    ]
    list(llm.stream(messages))

    _, kwargs = mocked_client.converse_stream.call_args
    bedrock_msgs = kwargs["messages"]

    assert bedrock_msgs[0]["content"][0] == {"text": "Hello"}
    assert bedrock_msgs[-1]["content"][0] == {
        "guardContent": {"text": {"text": "How are you?"}}
    }


@mock.patch("langchain_aws.chat_models.bedrock_converse.create_aws_client")
def test_bedrock_client_creation(mock_create_client: mock.Mock) -> None:
    """Test that bedrock_client is created during validation."""
    mock_bedrock_client = mock.Mock()
    mock_runtime_client = mock.Mock()

    def side_effect(service_name: str, **kwargs: Any) -> mock.Mock:
        if service_name == "bedrock":
            return mock_bedrock_client
        elif service_name == "bedrock-runtime":
            return mock_runtime_client
        return mock.Mock()

    mock_create_client.side_effect = side_effect

    chat_model = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0", region_name="us-west-2"
    )

    assert chat_model.bedrock_client == mock_bedrock_client
    assert chat_model.client == mock_runtime_client
    assert mock_create_client.call_count == 2


@mock.patch("langchain_aws.chat_models.bedrock_converse.create_aws_client")
def test_get_base_model_with_application_inference_profile(
    mock_create_client: mock.Mock,
) -> None:
    """Test _get_base_model method with application inference profile."""
    mock_bedrock_client = mock.Mock()
    mock_runtime_client = mock.Mock()

    # Mock the get_inference_profile response
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [
            {
                "modelArn": (
                    "arn:aws:bedrock:us-east-1::foundation-model/"
                    "anthropic.claude-3-sonnet-20240229-v1:0"
                )
            }
        ]
    }

    def side_effect(service_name: str, **kwargs: Any) -> mock.Mock:
        if service_name == "bedrock":
            return mock_bedrock_client
        elif service_name == "bedrock-runtime":
            return mock_runtime_client
        return mock.Mock()

    mock_create_client.side_effect = side_effect

    chat_model = ChatBedrockConverse(
        model="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test-profile",
        region_name="us-west-2",
        provider="anthropic",
    )

    base_model = chat_model._get_base_model()
    assert base_model == "anthropic.claude-3-sonnet-20240229-v1:0"
    mock_bedrock_client.get_inference_profile.assert_called_once_with(
        inferenceProfileIdentifier="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test-profile"
    )


@mock.patch("langchain_aws.chat_models.bedrock_converse.create_aws_client")
def test_get_base_model_without_application_inference_profile(
    mock_create_client: mock.Mock,
) -> None:
    """Test _get_base_model method without application inference profile."""
    mock_bedrock_client = mock.Mock()
    mock_runtime_client = mock.Mock()

    def side_effect(service_name: str, **kwargs: Any) -> mock.Mock:
        if service_name == "bedrock":
            return mock_bedrock_client
        elif service_name == "bedrock-runtime":
            return mock_runtime_client
        return mock.Mock()

    mock_create_client.side_effect = side_effect

    chat_model = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
        provider="anthropic",
    )

    base_model = chat_model._get_base_model()
    assert base_model == "anthropic.claude-3-sonnet-20240229-v1:0"
    mock_bedrock_client.get_inference_profile.assert_not_called()


@mock.patch("langchain_aws.chat_models.bedrock_converse.create_aws_client")
def test_configure_streaming_for_resolved_model(mock_create_client: mock.Mock) -> None:
    """Test _configure_streaming_for_resolved_model method."""
    mock_bedrock_client = mock.Mock()
    mock_runtime_client = mock.Mock()

    # Mock the get_inference_profile response for a model with full streaming support
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [
            {
                "modelArn": (
                    "arn:aws:bedrock:us-east-1::foundation-model/"
                    "anthropic.claude-3-sonnet-20240229-v1:0"
                )
            }
        ]
    }

    def side_effect(service_name: str, **kwargs: Any) -> mock.Mock:
        if service_name == "bedrock":
            return mock_bedrock_client
        elif service_name == "bedrock-runtime":
            return mock_runtime_client
        return mock.Mock()

    mock_create_client.side_effect = side_effect

    chat_model = ChatBedrockConverse(
        model="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test-profile",
        region_name="us-west-2",
        provider="anthropic",
    )

    # The streaming should be configured based on the resolved model
    assert chat_model.disable_streaming is False


@mock.patch("langchain_aws.chat_models.bedrock_converse.create_aws_client")
def test_configure_streaming_for_resolved_model_no_tools(
    mock_create_client: mock.Mock,
) -> None:
    """Test _configure_streaming_for_resolved_model method with no-tools streaming."""
    mock_bedrock_client = mock.Mock()
    mock_runtime_client = mock.Mock()

    # Mock the get_inference_profile response for a model with no-tools streaming
    # support
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [
            {
                "modelArn": (
                    "arn:aws:bedrock:us-east-1::foundation-model/"
                    "amazon.titan-text-express-v1"
                )
            }
        ]
    }

    def side_effect(service_name: str, **kwargs: Any) -> mock.Mock:
        if service_name == "bedrock":
            return mock_bedrock_client
        elif service_name == "bedrock-runtime":
            return mock_runtime_client
        return mock.Mock()

    mock_create_client.side_effect = side_effect

    chat_model = ChatBedrockConverse(
        model="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test-profile",
        region_name="us-west-2",
        provider="amazon",
    )

    # The streaming should be configured as "tool_calling" for no-tools models
    assert chat_model.disable_streaming == "tool_calling"


@mock.patch("langchain_aws.chat_models.bedrock_converse.create_aws_client")
def test_configure_streaming_for_resolved_model_no_streaming(
    mock_create_client: mock.Mock,
) -> None:
    """Test _configure_streaming_for_resolved_model method with no streaming support."""
    mock_bedrock_client = mock.Mock()
    mock_runtime_client = mock.Mock()

    # Mock the get_inference_profile response for a model with no streaming support
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [
            {
                "modelArn": (
                    "arn:aws:bedrock:us-east-1::foundation-model/"
                    "stability.stable-image-core-v1:0"
                )
            }
        ]
    }

    def side_effect(service_name: str, **kwargs: Any) -> mock.Mock:
        if service_name == "bedrock":
            return mock_bedrock_client
        elif service_name == "bedrock-runtime":
            return mock_runtime_client
        return mock.Mock()

    mock_create_client.side_effect = side_effect

    chat_model = ChatBedrockConverse(
        model="arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/test-profile",
        region_name="us-west-2",
        provider="stability",
    )

    # The streaming should be disabled for models with no streaming support
    assert chat_model.disable_streaming is True


def test_nova_provider_extraction() -> None:
    """Test that provider is correctly extracted from Nova model ID when not
    provided."""
    model = ChatBedrockConverse(
        client=mock.MagicMock(),
        model="us.amazon.nova-pro-v1:0",
        region_name="us-west-2",
    )
    assert model.provider == "amazon"


def test__messages_to_bedrock_strips_trailing_whitespace_string() -> None:
    """
    Test that _messages_to_bedrock strips trailing whitespace from string
    AIMessage content.
    """
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
        AIMessage(content="AI message with trailing whitespace    \n  \t  "),
    ]

    bedrock_messages, _ = _messages_to_bedrock(messages)

    assert (
        bedrock_messages[1]["content"][0]["text"]
        == "AI message with trailing whitespace"
    )


def test__messages_to_bedrock_strips_trailing_whitespace_blocks() -> None:
    """
    Test that _messages_to_bedrock strips trailing whitespace from block
    AIMessage content.
    """
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="Human message"),
        AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "AI message with trailing whitespace    \n  \t  ",
                },
                {"type": "text", "text": "Another text block with whitespace  \n "},
            ]
        ),
    ]

    bedrock_messages, _ = _messages_to_bedrock(messages)

    assert (
        bedrock_messages[1]["content"][0]["text"]
        == "AI message with trailing whitespace"
    )
    assert (
        bedrock_messages[1]["content"][1]["text"]
        == "Another text block with whitespace"
    )


def test__messages_to_bedrock_preserves_whitespace_non_last_aimessage_string() -> None:
    """
    Test that _messages_to_bedrock preserves trailing whitespace in non-last AIMessages.
    """
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="First human message"),
        AIMessage(content="AI message with trailing whitespace    \n  \t  "),
        HumanMessage(content="Second human message"),
    ]

    bedrock_messages, _ = _messages_to_bedrock(messages)

    assert (
        bedrock_messages[1]["content"][0]["text"]
        == "AI message with trailing whitespace    \n  \t  "
    )


def test__messages_to_bedrock_preserves_whitespace_non_last_aimessage_blocks() -> None:
    """
    Test that _messages_to_bedrock preserves trailing whitespace in non-last AIMessages.
    """
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="First human message"),
        AIMessage(
            content=[
                {
                    "type": "text",
                    "text": "AI message with trailing whitespace    \n  \t  ",
                },
            ]
        ),
        HumanMessage(content="Second human message"),
    ]

    bedrock_messages, _ = _messages_to_bedrock(messages)

    assert (
        bedrock_messages[1]["content"][0]["text"]
        == "AI message with trailing whitespace    \n  \t  "
    )


@pytest.mark.parametrize(
    "system_prompt_parameter, expected_system",
    [
        # No system parameter → use only the system message from messages
        (None, [{"text": "System message"}]),
        # Simple string input → converted into a dict with text
        (
            ["System message from param"],
            [
                {"text": "System message from param"},
                {"text": "System message"},
            ],
        ),
        # Dict input → passed through as-is
        (
            [
                {
                    "text": "Structured system message",
                    "guardContent": {"text": {"text": "guarded"}},
                }
            ],
            [
                {
                    "text": "Structured system message",
                    "guardContent": {"text": {"text": "guarded"}},
                },
                {"text": "System message"},
            ],
        ),
        # Mixed string and dict → both should be handled correctly
        (
            [
                "Simple system prompt",
                {"text": "Advanced system prompt", "cachePoint": {"type": "default"}},
            ],
            [
                {"text": "Simple system prompt"},
                {"text": "Advanced system prompt", "cachePoint": {"type": "default"}},
                {"text": "System message"},
            ],
        ),
    ],
)
def test__messages_to_bedrock_appends_system_prompt_from_parameter(
    system_prompt_parameter: List[str | Dict[str, Any]] | None,
    expected_system: List[Dict[str, Any]],
) -> None:
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content="First human message"),
    ]

    _, actual_system = _messages_to_bedrock(messages, system_prompt_parameter)

    assert actual_system == expected_system


@mock.patch("langchain_aws.chat_models.bedrock_converse.create_aws_client")
def test_bedrock_client_inherits_from_runtime_client(
    mock_create_client: mock.Mock,
) -> None:
    """Test that bedrock_client inherits region and config from runtime client."""
    mock_runtime_client = mock.Mock()
    mock_bedrock_client = mock.Mock()

    mock_runtime_client.meta.region_name = "us-west-2"
    mock.Mock()

    def side_effect(service_name: str, **kwargs: Any) -> mock.Mock:
        if service_name == "bedrock":
            return mock_bedrock_client
        elif service_name == "bedrock-runtime":
            return mock_runtime_client
        return mock.Mock()

    mock_create_client.side_effect = side_effect

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        ChatBedrockConverse(
            model="us.meta.llama3-3-70b-instruct-v1:0", client=mock_runtime_client
        )

        mock_create_client.assert_called_with(
            region_name="us-west-2",
            credentials_profile_name=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            endpoint_url=None,
            config=None,
            service_name="bedrock",
            api_key=None,
        )


@mock.patch("langchain_aws.chat_models.bedrock_converse.create_aws_client")
def test_bedrock_client_uses_explicit_values_over_runtime_client(
    mock_create_client: mock.Mock,
) -> None:
    """Test that explicitly provided values override those from runtime client."""
    mock_runtime_client = mock.Mock()
    mock_bedrock_client = mock.Mock()

    mock_runtime_client.meta.region_name = "us-west-2"
    mock.Mock()
    mock.Mock()

    def side_effect(service_name: str, **kwargs: Any) -> mock.Mock:
        if service_name == "bedrock":
            return mock_bedrock_client
        elif service_name == "bedrock-runtime":
            return mock_runtime_client
        return mock.Mock()

    mock_create_client.side_effect = side_effect

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
        ChatBedrockConverse(
            model="us.meta.llama3-3-70b-instruct-v1:0",
            client=mock_runtime_client,
            region_name="us-east-1",
        )

        mock_create_client.assert_called_with(
            region_name="us-east-1",
            credentials_profile_name=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            endpoint_url=None,
            config=None,
            service_name="bedrock",
            api_key=None,
        )


def test__has_tool_use_or_result_blocks() -> None:
    """Test detection of toolUse and toolResult blocks in messages."""
    # No tool blocks
    messages_no_tools = [{"role": "user", "content": [{"text": "Hello"}]}]
    assert not _has_tool_use_or_result_blocks(messages_no_tools)

    # With toolUse blocks
    messages_with_tools = [
        {"role": "assistant", "content": [{"toolUse": {"name": "calc"}}]}
    ]
    assert _has_tool_use_or_result_blocks(messages_with_tools)


def test__convert_tool_blocks_to_text() -> None:
    """Test conversion of toolUse and toolResult blocks to text format."""
    input_messages: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {"text": "Calculating..."},
                {"toolUse": {"toolUseId": "1", "name": "calc", "input": {"a": 5}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "1", "content": [{"text": "10"}]}}
            ],
        },
    ]

    result = _convert_tool_blocks_to_text(input_messages)

    # Check toolUse converted to text
    assert '[Called calc with parameters: {"a": 5}]' in result[0]["content"][1]["text"]

    # Check toolResult converted to text
    assert "[Tool output: 10]" in result[1]["content"][0]["text"]


def test_tool_conversion_warning_integration() -> None:
    """Test that tool blocks without toolConfig trigger conversion and warning."""
    mocked_client = mock.MagicMock()
    mocked_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Done"}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
    )

    messages = [
        AIMessage(
            content="",
            tool_calls=[ToolCall(name="calc", args={"x": 1}, id="1", type="tool_call")],
        )
    ]

    with pytest.warns(
        RuntimeWarning, match="Tool messages were passed without toolConfig"
    ):
        llm.invoke(messages)

    # Verify conversion happened
    call_args = mocked_client.converse.call_args[1]["messages"]
    assert any("Called calc" in str(block) for block in call_args[0]["content"])


def test_get_num_tokens_from_messages_supported_model() -> None:
    """Test get_num_tokens_from_messages for models that support count_tokens API."""
    mocked_client = mock.MagicMock()
    mocked_client.count_tokens.return_value = {"inputTokens": 42}

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1",
    )

    messages: List[BaseMessage] = [
        HumanMessage(content="What is the capital of France?")
    ]
    token_count = llm.get_num_tokens_from_messages(messages)

    assert token_count == 42
    mocked_client.count_tokens.assert_called_once()

    # Verify API call format
    call_args = mocked_client.count_tokens.call_args
    assert call_args[1]["modelId"] == "anthropic.claude-haiku-4-5-20251001-v1:0"
    assert "converse" in call_args[1]["input"]


def test_get_num_tokens_from_messages_unsupported_model_fallback() -> None:
    """Test fallback behavior for unsupported models."""
    mocked_client = mock.MagicMock()

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="amazon.titan-text-express-v1",
        region_name="us-west-2",
    )

    messages: List[BaseMessage] = [HumanMessage(content="Hello")]

    with mock.patch.object(
        BaseChatModel, "get_num_tokens_from_messages", return_value=10
    ) as mock_base:
        token_count = llm.get_num_tokens_from_messages(messages)
        assert token_count == 10
        mock_base.assert_called_once()

    mocked_client.count_tokens.assert_not_called()


def test_get_num_tokens_from_messages_api_error_fallback() -> None:
    """Test fallback when count_tokens API fails."""
    mocked_client = mock.MagicMock()
    mocked_client.count_tokens.side_effect = Exception("API Error")

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-west-2",
    )

    messages: List[BaseMessage] = [HumanMessage(content="Hello")]

    with mock.patch.object(
        BaseChatModel, "get_num_tokens_from_messages", return_value=5
    ) as mock_base:
        token_count = llm.get_num_tokens_from_messages(messages)
        assert token_count == 5
        mock_base.assert_called_once()


def test_bind_tools_with_nova_system_tool_instances() -> None:
    """Test bind_tools with NovaSystemTool instances."""
    from langchain_aws.tools import NovaCodeInterpreterTool, NovaGroundingTool

    chat_model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]

    # Test with NovaGroundingTool
    chat_model_with_grounding = chat_model.bind_tools([NovaGroundingTool()])
    tools = cast(RunnableBinding, chat_model_with_grounding).kwargs["toolConfig"][
        "tools"
    ]
    assert len(tools) == 1
    assert tools[0] == {"systemTool": {"name": "nova_grounding"}}

    # Test with NovaCodeInterpreterTool
    chat_model_with_code = chat_model.bind_tools([NovaCodeInterpreterTool()])
    tools = cast(RunnableBinding, chat_model_with_code).kwargs["toolConfig"]["tools"]
    assert len(tools) == 1
    assert tools[0] == {"systemTool": {"name": "nova_code_interpreter"}}

    # Test with both system tools
    chat_model_with_both = chat_model.bind_tools(
        [NovaGroundingTool(), NovaCodeInterpreterTool()]
    )
    tools = cast(RunnableBinding, chat_model_with_both).kwargs["toolConfig"]["tools"]
    assert len(tools) == 2
    assert tools[0] == {"systemTool": {"name": "nova_grounding"}}
    assert tools[1] == {"systemTool": {"name": "nova_code_interpreter"}}


def test_bind_tools_with_system_tool_name_strings() -> None:
    """Test bind_tools with system tool name strings."""
    chat_model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]

    # Test with direct string
    chat_model_with_grounding = chat_model.bind_tools(["nova_grounding"])
    tools = cast(RunnableBinding, chat_model_with_grounding).kwargs["toolConfig"][
        "tools"
    ]
    assert len(tools) == 1
    assert tools[0] == {"systemTool": {"name": "nova_grounding"}}

    # Test with direct string
    chat_model_with_code = chat_model.bind_tools(["nova_code_interpreter"])
    tools = cast(RunnableBinding, chat_model_with_code).kwargs["toolConfig"]["tools"]
    assert len(tools) == 1
    assert tools[0] == {"systemTool": {"name": "nova_code_interpreter"}}

    # Test with both as strings
    chat_model_with_both = chat_model.bind_tools(
        ["nova_grounding", "nova_code_interpreter"]
    )
    tools = cast(RunnableBinding, chat_model_with_both).kwargs["toolConfig"]["tools"]
    assert len(tools) == 2
    assert tools[0] == {"systemTool": {"name": "nova_grounding"}}
    assert tools[1] == {"systemTool": {"name": "nova_code_interpreter"}}


def test_bind_tools_with_mixed_system_and_custom_tools() -> None:
    """Test bind_tools with mixed system and custom tools."""
    from langchain_aws.tools import NovaGroundingTool

    chat_model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]

    # Mix NovaGroundingTool with custom GetWeather tool
    chat_model_with_mixed = chat_model.bind_tools([GetWeather, NovaGroundingTool()])
    tools = cast(RunnableBinding, chat_model_with_mixed).kwargs["toolConfig"]["tools"]
    assert len(tools) == 2

    # First tool should be the custom tool (GetWeather)
    assert "function" in tools[0]
    assert tools[0]["function"]["name"] == "GetWeather"

    # Second tool should be the system tool
    assert tools[1] == {"systemTool": {"name": "nova_grounding"}}


def test_bind_tools_system_tools_with_tool_choice() -> None:
    """Test that tool_choice works with system tools."""
    from langchain_aws.tools import NovaGroundingTool

    chat_model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]

    # Test with auto tool choice
    chat_model_with_tools = chat_model.bind_tools(
        [NovaGroundingTool()], tool_choice="auto"
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["toolConfig"][
        "toolChoice"
    ] == {"auto": {}}

    # Test with any tool choice
    chat_model_with_tools = chat_model.bind_tools(
        [NovaGroundingTool()], tool_choice="any"
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["toolConfig"][
        "toolChoice"
    ] == {"any": {}}


def test_bind_tools_toolconfig_structure_with_system_tools() -> None:
    """Test that toolConfig structure is correct with system tools."""
    from langchain_aws.tools import NovaCodeInterpreterTool, NovaGroundingTool

    chat_model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]

    # Bind tools with both system and custom tools
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather, NovaGroundingTool(), NovaCodeInterpreterTool()],
        tool_choice="auto",
    )

    bound_kwargs = cast(RunnableBinding, chat_model_with_tools).kwargs

    # Verify toolConfig is present
    assert "toolConfig" in bound_kwargs
    tool_config = bound_kwargs["toolConfig"]

    # Verify tools are present in toolConfig
    assert "tools" in tool_config
    tools = tool_config["tools"]
    assert len(tools) == 3

    # Verify custom tool format
    assert "function" in tools[0]
    assert tools[0]["function"]["name"] == "GetWeather"

    # Verify system tools format
    assert tools[1] == {"systemTool": {"name": "nova_grounding"}}
    assert tools[2] == {"systemTool": {"name": "nova_code_interpreter"}}

    # Verify toolChoice is present in toolConfig
    assert "toolChoice" in tool_config
    assert tool_config["toolChoice"] == {"auto": {}}


def test_bind_tools_formats_custom_tools_to_dicts() -> None:
    """Test that custom tools are dict formatted with no Nova tools present."""
    from langchain_core.tools import tool

    @tool
    def my_custom_tool(query: str) -> str:
        """A simple test tool.

        Args:
            query: The query string.
        """
        return f"Result for: {query}"

    chat_model = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]

    chat_model_with_tools = chat_model.bind_tools([my_custom_tool])

    bound_kwargs = cast(RunnableBinding, chat_model_with_tools).kwargs

    assert "tools" in bound_kwargs
    assert "toolConfig" not in bound_kwargs

    tools = bound_kwargs["tools"]
    assert isinstance(tools, list)
    assert len(tools) == 1

    tool_def = tools[0]
    assert isinstance(tool_def, dict), f"Expected dict, got {type(tool_def)}"

    assert tool_def.get("type") == "function"
    assert "function" in tool_def
    assert tool_def["function"].get("name") == "my_custom_tool"


def test_bind_tools_strict_true() -> None:
    """Test that strict=True sets the flag and applies schema transforms."""
    chat_model = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]
    chat_model_with_tools = chat_model.bind_tools([GetWeather], strict=True)

    bound_kwargs = cast(RunnableBinding, chat_model_with_tools).kwargs
    tools = bound_kwargs["tools"]
    assert len(tools) == 1

    func = tools[0]["function"]
    assert func["strict"] is True
    assert "location" in func["parameters"]["required"]
    assert func["parameters"]["additionalProperties"] is False


def test_bind_tools_strict_none_default() -> None:
    """Test that default strict=None does not include strict key in tool definition."""
    chat_model = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]
    chat_model_with_tools = chat_model.bind_tools([GetWeather])

    bound_kwargs = cast(RunnableBinding, chat_model_with_tools).kwargs
    func = bound_kwargs["tools"][0]["function"]
    assert "strict" not in func


def test_bind_tools_strict_false() -> None:
    """Test that strict=False explicitly sets strict to False."""
    chat_model = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]
    chat_model_with_tools = chat_model.bind_tools([GetWeather], strict=False)

    bound_kwargs = cast(RunnableBinding, chat_model_with_tools).kwargs
    func = bound_kwargs["tools"][0]["function"]
    assert func["strict"] is False


def test_format_tools_preserves_strict() -> None:
    """Test that _format_tools preserves strict when converting to Bedrock toolSpec."""
    openai_tool = {
        "type": "function",
        "function": {
            "name": "GetWeather",
            "description": "Get the current weather in a given location",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }
    result = _format_tools([openai_tool])
    assert len(result) == 1
    tool_spec = result[0]["toolSpec"]
    assert tool_spec["strict"] is True
    assert tool_spec["name"] == "GetWeather"
    assert "inputSchema" in tool_spec


def test_bind_tools_strict_with_tool_choice() -> None:
    """Test that strict and tool_choice work together without interference."""
    chat_model = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], strict=True, tool_choice="auto"
    )

    bound_kwargs = cast(RunnableBinding, chat_model_with_tools).kwargs
    assert bound_kwargs["tool_choice"] == {"auto": {}}
    func = bound_kwargs["tools"][0]["function"]
    assert func["strict"] is True


def test_with_structured_output_passes_strict_to_bind_tools() -> None:
    """Test that with_structured_output passes strict through to bind_tools."""
    chat_model = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0", region_name="us-east-1"
    )  # type: ignore[call-arg]
    structured = chat_model.with_structured_output(GetWeather, strict=True)
    llm = structured.first  # type: ignore[attr-defined]
    bound_kwargs = cast(RunnableBinding, llm).kwargs
    func = bound_kwargs["tools"][0]["function"]
    assert func["strict"] is True


def test_reasoning_config_validation_accepts_strings() -> None:
    """Test that reasoning config validation accepts string values."""
    # Should not raise an error with string values
    for effort in ["low", "medium", "high"]:
        llm = ChatBedrockConverse(
            model="amazon.nova-2-lite-v1:0",
            region_name="us-east-1",
            additional_model_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}
            },
        )  # type: ignore[call-arg]
        assert llm is not None


def test_reasoning_config_validation_rejects_invalid_effort() -> None:
    """Test that reasoning config validation rejects invalid effort levels."""
    with pytest.raises(
        ValueError,
        match=(
            "Invalid maxReasoningEffort: 'invalid'.*"
            "Must be one of: \\['low', 'medium', 'high'\\]"
        ),
    ):
        ChatBedrockConverse(
            model="amazon.nova-2-lite-v1:0",
            region_name="us-east-1",
            additional_model_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "invalid"}
            },
        )  # type: ignore[call-arg]


def test_reasoning_config_validation_requires_max_effort_when_enabled() -> None:
    """Test that reasoning config validation requires maxReasoningEffort."""
    with pytest.raises(
        ValueError,
        match=(
            "When reasoningConfig type is 'enabled', "
            "'maxReasoningEffort' must be specified"
        ),
    ):
        ChatBedrockConverse(
            model="amazon.nova-2-lite-v1:0",
            region_name="us-east-1",
            additional_model_request_fields={"reasoningConfig": {"type": "enabled"}},
        )  # type: ignore[call-arg]


def test_reasoning_config_validation_disabled_no_effort_required() -> None:
    """Test that maxReasoningEffort is not required when reasoning is disabled."""
    # Should not raise an error when type is disabled
    llm = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
        additional_model_request_fields={"reasoningConfig": {"type": "disabled"}},
    )  # type: ignore[call-arg]
    assert llm is not None


def test_reasoning_config_validation_no_config() -> None:
    """Test that validation passes when no reasoning config is provided."""
    # Should not raise an error when no reasoningConfig is provided
    llm = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
    )  # type: ignore[call-arg]
    assert llm is not None


def test_reasoning_config_validation_rejects_invalid_type() -> None:
    """Test that reasoning config validation rejects invalid effort types."""
    with pytest.raises(
        ValueError,
        match=(
            "Invalid maxReasoningEffort type: int.*"
            "Must be a string, one of: \\['low', 'medium', 'high'\\]"
        ),
    ):
        ChatBedrockConverse(
            model="amazon.nova-2-lite-v1:0",
            region_name="us-east-1",
            additional_model_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": 123}
            },
        )  # type: ignore[call-arg]


def test_iam_permission_error_detection() -> None:
    """Test that IAM permission errors for InvokeTool are detected and enhanced."""
    from botocore.exceptions import ClientError

    from langchain_aws.chat_models.bedrock_converse import _handle_bedrock_error

    # Create a mock AccessDeniedException for InvokeTool
    error_response: Any = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": (
                "User: arn:aws:iam::834212829299:user/demo is not authorized to "
                "perform: bedrock:InvokeTool on resource: "
                "arn:aws:bedrock::834212829299:system-tool/amazon.nova_grounding "
                "with an explicit deny in an identity-based policy"
            ),
        },
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }
    error = ClientError(error_response, "Converse")  # type: ignore[arg-type]

    # Should raise ValueError with enhanced message
    with pytest.raises(ValueError) as exc_info:
        _handle_bedrock_error(error)

    error_message = str(exc_info.value)
    assert "bedrock:InvokeTool" in error_message
    assert "IAM permission" in error_message
    assert "Example policy statement" in error_message
    assert "Original error" in error_message


def test_iam_permission_error_other_access_denied() -> None:
    """Test that other AccessDeniedException errors are not modified."""
    from botocore.exceptions import ClientError

    from langchain_aws.chat_models.bedrock_converse import _handle_bedrock_error

    # Create a mock AccessDeniedException without InvokeTool
    error_response: Any = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": "User is not authorized to perform: bedrock:InvokeModel",
        },
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }
    error = ClientError(error_response, "Converse")  # type: ignore[arg-type]

    # Should re-raise the original error
    with pytest.raises(ClientError) as exc_info:
        _handle_bedrock_error(error)

    assert exc_info.value == error


def test_iam_permission_error_non_access_denied() -> None:
    """Test that non-AccessDeniedException errors are not modified."""
    from botocore.exceptions import ClientError

    from langchain_aws.chat_models.bedrock_converse import _handle_bedrock_error

    # Create a different error type
    error_response: Any = {
        "Error": {
            "Code": "ValidationException",
            "Message": "Invalid input",
        },
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }
    error = ClientError(error_response, "Converse")  # type: ignore[arg-type]

    # Should re-raise the original error
    with pytest.raises(ClientError) as exc_info:
        _handle_bedrock_error(error)

    assert exc_info.value == error


def test_generate_with_iam_permission_error() -> None:
    """Test that _generate wraps IAM permission errors."""
    from botocore.exceptions import ClientError

    llm = ChatBedrockConverse(model="amazon.nova-2-lite-v1:0", region_name="us-east-1")

    # Mock the client to raise AccessDeniedException
    error_response: Any = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": (
                "User is not authorized to perform: bedrock:InvokeTool on resource: "
                "arn:aws:bedrock::123456789:system-tool/amazon.nova_grounding"
            ),
        },
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }
    llm.client.converse = mock.Mock(side_effect=ClientError(error_response, "Converse"))

    # Should raise ValueError with enhanced message
    with pytest.raises(ValueError) as exc_info:
        llm.invoke("What's the weather?")

    error_message = str(exc_info.value)
    assert "bedrock:InvokeTool" in error_message
    assert "IAM permission" in error_message
    assert "Example policy statement" in error_message


def test_stream_with_iam_permission_error() -> None:
    """Test that _stream wraps IAM permission errors."""
    from botocore.exceptions import ClientError
    from langchain_core.messages import HumanMessage

    llm = ChatBedrockConverse(model="amazon.nova-2-lite-v1:0", region_name="us-east-1")

    # Mock the client to raise AccessDeniedException when converse_stream is called
    error_response: Any = {
        "Error": {
            "Code": "AccessDeniedException",
            "Message": (
                "User is not authorized to perform: bedrock:InvokeTool on resource: "
                "arn:aws:bedrock::123456789:system-tool/amazon.nova_grounding"
            ),
        },
        "ResponseMetadata": {"RequestId": "test-request-id"},
    }

    # Create a mock that raises the exception
    mock_client = mock.Mock()
    mock_client.converse_stream.side_effect = ClientError(
        error_response,  # type: ignore[arg-type]
        "Converse",
    )
    llm.client = mock_client

    # Should raise ValueError with enhanced message
    # Call _stream directly to avoid invoke() being called
    with pytest.raises(ValueError) as exc_info:
        # Consume the generator to trigger the error
        for _ in llm._stream([HumanMessage(content="What's the weather?")]):
            pass

    error_message = str(exc_info.value)
    assert "bedrock:InvokeTool" in error_message
    assert "IAM permission" in error_message
    assert "Example policy statement" in error_message


def test_reasoning_config_validation_only_applies_to_nova_2() -> None:
    """Test that reasoning config validation only applies to Nova 2 models."""
    # Should not raise an error for non-Nova-2 models even with invalid config
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
        additional_model_request_fields={
            "reasoningConfig": {"type": "enabled"}  # Missing maxReasoningEffort
        },
    )  # type: ignore[call-arg]
    assert llm is not None

    # Should not raise an error for Nova 1 models
    llm = ChatBedrockConverse(
        model="amazon.nova-lite-v1:0",
        region_name="us-east-1",
        additional_model_request_fields={
            "reasoningConfig": {"type": "enabled"}  # Missing maxReasoningEffort
        },
    )  # type: ignore[call-arg]
    assert llm is not None

    # Should raise an error for Nova 2 models with invalid config
    with pytest.raises(
        ValueError,
        match="When reasoningConfig type is 'enabled', "
        "'maxReasoningEffort' must be specified",
    ):
        ChatBedrockConverse(
            model="amazon.nova-2-lite-v1:0",
            region_name="us-east-1",
            additional_model_request_fields={
                "reasoningConfig": {"type": "enabled"}  # Missing maxReasoningEffort
            },
        )  # type: ignore[call-arg]


def test_service_tier_parameter() -> None:
    """Test that service_tier parameter is correctly set and passed."""
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
        service_tier="priority",
    )
    assert llm.service_tier == "priority"


@pytest.mark.parametrize(
    "service_tier",
    ["priority", "default", "flex", "reserved"],
)
def test_service_tier_valid_values(service_tier: str) -> None:
    """Test that all valid service_tier values are accepted."""
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
        service_tier=service_tier,  # type: ignore[arg-type]
    )
    assert llm.service_tier == service_tier


def test_service_tier_in_converse_params() -> None:
    """Test that service_tier is included in _converse_params with correct format."""
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
        service_tier="priority",
    )

    params = llm._converse_params()
    assert params["serviceTier"] == {"type": "priority"}


def test_service_tier_none_not_in_params() -> None:
    """Test that service_tier is not in params when None."""
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
    )

    params = llm._converse_params()
    assert "serviceTier" not in params


def test_service_tier_passed_to_converse() -> None:
    """Test that service_tier is passed to the converse API call."""
    mocked_client = mock.MagicMock()
    mocked_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
        service_tier="priority",
    )

    messages = [HumanMessage(content="Hi")]
    llm.invoke(messages)

    # Verify converse was called with serviceTier in the correct format
    call_kwargs = mocked_client.converse.call_args[1]
    assert call_kwargs["serviceTier"] == {"type": "priority"}


def test_service_tier_passed_to_converse_stream() -> None:
    """Test that service_tier is passed to the converse_stream API call."""
    mocked_client = mock.MagicMock()
    mocked_client.converse_stream.return_value = {
        "stream": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
                }
            },
        ]
    }

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
        service_tier="flex",
    )

    messages = [HumanMessage(content="Hi")]
    list(llm.stream(messages))

    # Verify converse_stream was called with serviceTier in the correct format
    call_kwargs = mocked_client.converse_stream.call_args[1]
    assert call_kwargs["serviceTier"] == {"type": "flex"}


def test_additional_model_request_fields_merge_no_duplicate_keys() -> None:
    """Test that additional_model_request_fields from constructor and invoke are merged
    correctly when both use the same key.

    This test ensures that when additional_model_request_fields is provided both
    at initialization and at invocation, the final request contains only one
    field and the invoke value wins.
    """
    mocked_client = mock.MagicMock()
    mocked_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    # Create model with additional_model_request_fields in constructor (snake_case)
    llm = ChatBedrockConverse(
        client=mocked_client,
        model="openai.gpt-oss-120b-1:0",
        region_name="us-west-2",
        additional_model_request_fields={"reasoning_effort": "low"},
    )

    messages = [HumanMessage(content="Hi")]
    # Call invoke with additional_model_request_fields in kwargs (snake_case)
    llm.invoke(
        messages,
        additional_model_request_fields={"reasoning_effort": "medium"},
    )

    # Verify converse was called
    assert mocked_client.converse.called
    call_kwargs = mocked_client.converse.call_args[1]

    # Verify additionalModelRequestFields exists and is a dict
    assert "additionalModelRequestFields" in call_kwargs
    additional_fields = call_kwargs["additionalModelRequestFields"]
    assert isinstance(additional_fields, dict)

    # Verify the key is passed through exactly as provided
    assert "reasoning_effort" in additional_fields
    assert "reasoningEffort" not in additional_fields

    # Verify the value is from invoke (invoke overrides constructor)
    assert additional_fields["reasoning_effort"] == "medium"

    # Verify no duplicate keys exist
    keys = list(additional_fields.keys())
    assert len(keys) == 1
    assert keys[0] == "reasoning_effort"


def test_additional_model_request_fields_merge_constructor_only() -> None:
    """Test correctness of additional_model_request_fields from constructor only."""
    mocked_client = mock.MagicMock()
    mocked_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="openai.gpt-oss-120b-1:0",
        region_name="us-west-2",
        additional_model_request_fields={"reasoning_effort": "low"},
    )

    messages = [HumanMessage(content="Hi")]
    llm.invoke(messages)

    call_kwargs = mocked_client.converse.call_args[1]
    assert "additionalModelRequestFields" in call_kwargs
    additional_fields = call_kwargs["additionalModelRequestFields"]
    assert "reasoning_effort" in additional_fields
    assert additional_fields["reasoning_effort"] == "low"


def test_additional_model_request_fields_merge_invoke_only() -> None:
    """Test that additional_model_request_fields from invoke only works correctly."""
    mocked_client = mock.MagicMock()
    mocked_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="openai.gpt-oss-120b-1:0",
        region_name="us-west-2",
    )

    messages = [HumanMessage(content="Hi")]
    llm.invoke(
        messages,
        additional_model_request_fields={"reasoning_effort": "high"},
    )

    call_kwargs = mocked_client.converse.call_args[1]
    assert "additionalModelRequestFields" in call_kwargs
    additional_fields = call_kwargs["additionalModelRequestFields"]
    assert "reasoning_effort" in additional_fields
    assert additional_fields["reasoning_effort"] == "high"


def test_additional_model_request_fields_invoke_overrides_constructor() -> None:
    """Test that invoke values take priority over constructor values for same key.

    This test explicitly verifies that when the same key is provided in both
    constructor and invoke, the invoke value wins.
    """
    mocked_client = mock.MagicMock()
    mocked_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    # Constructor sets reasoning_effort to "constructor_value"
    llm = ChatBedrockConverse(
        client=mocked_client,
        model="openai.gpt-oss-120b-1:0",
        region_name="us-west-2",
        additional_model_request_fields={"reasoning_effort": "constructor_value"},
    )

    # Invoke sets reasoning_effort to "invoke_value"
    llm.invoke(
        [HumanMessage(content="Hi")],
        additional_model_request_fields={"reasoning_effort": "invoke_value"},
    )

    call_kwargs = mocked_client.converse.call_args[1]
    additional_fields = call_kwargs["additionalModelRequestFields"]

    # Verify invoke value takes priority over constructor value
    assert additional_fields["reasoning_effort"] == "invoke_value"
    assert additional_fields["reasoning_effort"] != "constructor_value"


def test_additional_model_request_fields_camel_keys_passthrough() -> None:
    """Test additional_model_request_fields keys are not transformed."""
    mocked_client = mock.MagicMock()
    mocked_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="amazon.nova-pro-v1:0",
        region_name="us-west-2",
    )

    llm.invoke(
        [HumanMessage(content="Hi")],
        additional_model_request_fields={
            "inferenceConfig": {"topK": 50},
        },
    )

    call_kwargs = mocked_client.converse.call_args[1]
    additional_fields = call_kwargs["additionalModelRequestFields"]

    assert "inferenceConfig" in additional_fields
    assert "inference_config" not in additional_fields
    assert additional_fields["inferenceConfig"] == {"topK": 50}


def test_additional_model_request_fields_keys_passthrough_stream() -> None:
    """Test that additional_model_request_fields keys are not transformed
    when streaming."""
    mocked_client = mock.MagicMock()
    mock_stream = mock.MagicMock()
    mock_stream.__iter__ = mock.Mock(
        return_value=iter(
            [
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockDelta": {
                        "delta": {"text": "Hi"},
                        "contentBlockIndex": 0,
                    }
                },
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
    )
    mocked_client.converse_stream.return_value = {"stream": mock_stream}

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="amazon.nova-pro-v1:0",
        region_name="us-west-2",
    )

    list(
        llm.stream(
            [HumanMessage(content="Hi")],
            additional_model_request_fields={
                "inferenceConfig": {"topK": 50},
            },
        )
    )

    call_kwargs = mocked_client.converse_stream.call_args[1]
    additional_fields = call_kwargs["additionalModelRequestFields"]

    assert "inferenceConfig" in additional_fields
    assert "inference_config" not in additional_fields
    assert additional_fields["inferenceConfig"] == {"topK": 50}


def test_additional_model_request_fields_mixed_formats() -> None:
    """Test that both camelCase and snake_case keys coexist without transformation."""
    mocked_client = mock.MagicMock()
    mocked_client.converse.return_value = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="amazon.nova-pro-v1:0",
        region_name="us-west-2",
    )

    llm.invoke(
        [HumanMessage(content="Hi")],
        additional_model_request_fields={
            "inferenceConfig": {"topK": 50},
            "custom_param": "value",
        },
    )

    call_kwargs = mocked_client.converse.call_args[1]
    additional_fields = call_kwargs["additionalModelRequestFields"]

    assert additional_fields["inferenceConfig"] == {"topK": 50}
    assert additional_fields["custom_param"] == "value"
    assert "inference_config" not in additional_fields


def test_stream_closes_event_stream() -> None:
    """Test that stream() explicitly closes the EventStream after iteration."""
    mocked_client = mock.MagicMock()
    mock_stream = mock.MagicMock()
    mock_stream.__iter__ = mock.Mock(
        return_value=iter(
            [
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockDelta": {
                        "delta": {"text": "Hi"},
                        "contentBlockIndex": 0,
                    }
                },
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
    )
    mocked_client.converse_stream.return_value = {"stream": mock_stream}

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
    )

    list(llm.stream([HumanMessage(content="Hi")]))

    mock_stream.close.assert_called_once()


def test_stream_closes_event_stream_on_exception() -> None:
    """Test that stream() closes the EventStream even when iteration raises."""
    mocked_client = mock.MagicMock()
    mock_stream = mock.MagicMock()

    def raise_error() -> Iterator[Dict[str, Any]]:
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": "Hi"}, "contentBlockIndex": 0}}
        raise RuntimeError("Test error")

    mock_stream.__iter__ = mock.Mock(return_value=raise_error())
    mocked_client.converse_stream.return_value = {"stream": mock_stream}

    llm = ChatBedrockConverse(
        client=mocked_client,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
    )

    with pytest.raises(RuntimeError, match="Test error"):
        list(llm.stream([HumanMessage(content="Hi")]))

    mock_stream.close.assert_called_once()


def test_guardrail_config_snake_to_camel_conversion() -> None:
    """Test that guardrail_config is properly converted
    from snake_case to camelCase."""
    mocked_client = mock.MagicMock()
    mocked_bedrock_client = mock.MagicMock()

    llm = ChatBedrockConverse(
        client=mocked_client,
        bedrock_client=mocked_bedrock_client,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2",
        guardrails={
            "guardrail_identifier": "test-id",
            "guardrail_version": "1",
            "trace": "enabled",
        },
    )

    params = llm._converse_params()

    # Check that guardrailConfig has camelCase keys
    assert "guardrailConfig" in params
    guardrail_config = params["guardrailConfig"]
    assert "guardrailIdentifier" in guardrail_config
    assert "guardrailVersion" in guardrail_config
    assert "trace" in guardrail_config
    assert guardrail_config["guardrailIdentifier"] == "test-id"
    assert guardrail_config["guardrailVersion"] == "1"
    assert guardrail_config["trace"] == "enabled"


def test_ls_invocation_params_includes_provider_and_region() -> None:
    """Test that _get_ls_params includes provider and
    region_name in ls_invocation_params."""
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        provider="anthropic",
        region_name="us-west-2",
    )

    ls_params = llm._get_ls_params()

    assert "ls_invocation_params" in ls_params
    invocation_params = ls_params["ls_invocation_params"]  # type: ignore[typeddict-item]
    assert invocation_params["provider"] == "anthropic"
    assert invocation_params["region_name"] == "us-west-2"


def test_ls_invocation_params_infers_region_from_client(
    mock_boto3_client: mock.MagicMock,
) -> None:
    """Test that _get_ls_params infers region from client
    when not explicitly provided."""
    # Configure the mock client to return a region
    mock_boto3_client.meta.region_name = "eu-west-1"

    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        provider="anthropic",
    )

    ls_params = llm._get_ls_params()

    assert "ls_invocation_params" in ls_params
    invocation_params = ls_params["ls_invocation_params"]  # type: ignore[typeddict-item]
    assert invocation_params["provider"] == "anthropic"
    assert invocation_params["region_name"] == "eu-west-1"


def test_ls_invocation_params_prefers_explicit_region_over_inferred(
    mock_boto3_client: mock.MagicMock,
) -> None:
    """Test that explicit region_name takes precedence over inferred region."""
    # Configure the mock client to return a different region
    mock_boto3_client.meta.region_name = "eu-west-1"

    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        provider="anthropic",
        region_name="us-west-2",  # Explicit region should take precedence
    )

    ls_params = llm._get_ls_params()

    assert "ls_invocation_params" in ls_params
    invocation_params = ls_params["ls_invocation_params"]  # type: ignore[typeddict-item]
    # Explicit region_name should be used, not the one from client
    assert invocation_params["region_name"] == "us-west-2"


def test__lc_content_to_bedrock_tool_use_dict_input_unchanged() -> None:
    """Dict input should pass through unchanged."""
    content: List[Union[str, Dict[str, Any]]] = [
        {
            "type": "tool_use",
            "id": "tool_1",
            "name": "get_weather",
            "input": {"city": "Paris"},
        }
    ]
    result = _lc_content_to_bedrock(content)
    assert result == [
        {
            "toolUse": {
                "toolUseId": "tool_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        }
    ]


def test__lc_content_to_bedrock_tool_use_string_input_parsed_to_dict() -> None:
    """String input (from streaming accumulation) should be parsed to dict."""
    content: List[Union[str, Dict[str, Any]]] = [
        {
            "type": "tool_use",
            "id": "tool_1",
            "name": "get_weather",
            "input": '{"city": "Paris"}',
        }
    ]
    result = _lc_content_to_bedrock(content)
    assert result == [
        {
            "toolUse": {
                "toolUseId": "tool_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        }
    ]


def test__lc_content_to_bedrock_tool_use_empty_string_input() -> None:
    """Empty string input should become empty dict."""
    content: List[Union[str, Dict[str, Any]]] = [
        {
            "type": "tool_use",
            "id": "tool_1",
            "name": "no_args_tool",
            "input": "",
        }
    ]
    result = _lc_content_to_bedrock(content)
    assert result == [
        {
            "toolUse": {
                "toolUseId": "tool_1",
                "name": "no_args_tool",
                "input": {},
            }
        }
    ]


def test__lc_content_to_bedrock_server_tool_use_string_input_parsed() -> None:
    """String input on server_tool_use should also be parsed to dict."""
    content: List[Union[str, Dict[str, Any]]] = [
        {
            "type": "server_tool_use",
            "id": "tool_1",
            "name": "grounding",
            "input": '{"query": "latest news"}',
        }
    ]
    result = _lc_content_to_bedrock(content)
    assert result == [
        {
            "toolUse": {
                "toolUseId": "tool_1",
                "name": "grounding",
                "input": {"query": "latest news"},
            }
        }
    ]


def test_content_block_start_tool_call_chunk_args_type() -> None:
    """contentBlockStart should produce tool_call_chunk with string/None args."""
    event = {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {
                "toolUse": {
                    "toolUseId": "tool_1",
                    "name": "get_weather",
                }
            },
        }
    }
    chunk = _parse_stream_event(event)
    assert isinstance(chunk, AIMessageChunk)
    assert len(chunk.tool_call_chunks) == 1
    args = chunk.tool_call_chunks[0]["args"]
    assert args is None or isinstance(args, str)


def test_content_block_delta_tool_call_chunk_args_type() -> None:
    """contentBlockDelta should produce tool_call_chunk with string args."""
    event = {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {
                "toolUse": {
                    "input": '{"city": "Paris"}',
                }
            },
        }
    }
    chunk = _parse_stream_event(event)
    assert isinstance(chunk, AIMessageChunk)
    assert len(chunk.tool_call_chunks) == 1
    args = chunk.tool_call_chunks[0]["args"]
    assert isinstance(args, str)
    assert args == '{"city": "Paris"}'


def test_streaming_tool_use_round_trip() -> None:
    """Simulate streaming tool call, accumulate chunks, convert back to Bedrock."""
    events = [
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {
                    "toolUse": {
                        "toolUseId": "tool_abc",
                        "name": "get_weather",
                    }
                },
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "toolUse": {
                        "input": '{"city":',
                    }
                },
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {
                    "toolUse": {
                        "input": ' "Paris"}',
                    }
                },
            }
        },
    ]

    full = None
    for event in events:
        chunk = _parse_stream_event(event)
        if chunk is not None:
            full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    tool_block = next(
        b for b in full.content if isinstance(b, dict) and b.get("type") == "tool_use"
    )
    assert isinstance(tool_block["input"], str)
    assert len(full.tool_call_chunks) > 0

    bedrock_content = _lc_content_to_bedrock(
        cast(List[Union[str, Dict[str, Any]]], full.content)
    )
    tool_use_block = bedrock_content[0]["toolUse"]
    assert isinstance(tool_use_block["input"], dict)
    assert tool_use_block["input"] == {"city": "Paris"}


def test__format_data_content_block_video_base64() -> None:
    """Test that _format_data_content_block handles video blocks with base64 data."""
    video_data = base64.b64encode(b"video_test_data").decode("utf-8")
    block = {
        "type": "video",
        "base64": video_data,
        "mimeType": "video/mp4",
    }
    result = _format_data_content_block(block)
    assert result == {
        "video": {
            "format": "mp4",
            "source": {"bytes": base64.b64decode(video_data.encode("utf-8"))},
        }
    }


def test__format_data_content_block_video_source_type() -> None:
    """Test that _format_data_content_block handles video blocks with sourceType."""
    video_data = base64.b64encode(b"video_test_data").decode("utf-8")
    block = {
        "type": "video",
        "sourceType": "base64",
        "mimeType": "video/mp4",
        "data": video_data,
    }
    result = _format_data_content_block(block)
    assert result == {
        "video": {
            "format": "mp4",
            "source": {"bytes": base64.b64decode(video_data.encode("utf-8"))},
        }
    }


def test__format_data_content_block_video_missing_mime_type() -> None:
    """Test _format_data_content_block raises for video without mimeType."""
    block = {
        "type": "video",
        "base64": base64.b64encode(b"video_test_data").decode("utf-8"),
    }
    with pytest.raises(ValueError, match="mime_type key is required"):
        _format_data_content_block(block)


def test__format_data_content_block_video_no_base64() -> None:
    """Test _format_data_content_block raises for video without base64."""
    block = {
        "type": "video",
        "mimeType": "video/mp4",
    }
    with pytest.raises(
        ValueError, match="Video data only supported through in-line base64 format"
    ):
        _format_data_content_block(block)


def test__format_data_content_block_unsupported_type() -> None:
    """Test that _format_data_content_block raises ValueError for unsupported types."""
    block = {
        "type": "audio",
        "base64": base64.b64encode(b"audio_data").decode("utf-8"),
        "mimeType": "audio/mp3",
    }
    with pytest.raises(ValueError, match="Unsupported data content block type"):
        _format_data_content_block(block)
