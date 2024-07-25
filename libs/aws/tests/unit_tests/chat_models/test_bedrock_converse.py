"""Test chat model integration."""

import base64
from typing import Dict, List, Type, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBinding
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_aws import ChatBedrockConverse
from langchain_aws.chat_models.bedrock_converse import (
    _bedrock_to_anthropic,
    _camel_to_snake,
    _camel_to_snake_keys,
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

    @pytest.mark.xfail()
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
    ]
    expected_messages = [
        {"role": "user", "content": [{"text": "hu1"}]},
        {"role": "user", "content": [{"text": "hu2"}]},
        {"role": "assistant", "content": [{"text": "ai1"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tool_call1",
                        "input": {"arg1": "arg1"},
                        "name": "tool1",
                    }
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
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
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "text": "hu3",
                },
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
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"guardContent": {"text": {"text": "hu5"}}},
                {"guardContent": {"text": {"text": "hu6"}}},
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


def test__bedrock_to_anthropic() -> None:
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
    ]
    actual = _bedrock_to_anthropic(bedrock)
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
