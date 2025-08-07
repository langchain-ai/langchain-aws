# type:ignore

"""Test chat model integration."""

import os
from contextlib import nullcontext
from typing import Any, Callable, Dict, Literal, Type, cast
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_aws import ChatBedrock
from langchain_aws.chat_models.bedrock import (
    ChatPromptAdapter,
    _format_anthropic_messages,
    _merge_messages,
    convert_messages_to_prompt_anthropic,
)
from langchain_aws.function_calling import convert_to_anthropic_tool


def test__merge_messages() -> None:
    messages = [
        SystemMessage("foo"),  # type: ignore[misc]
        SystemMessage("barfoo"),  # type: ignore[misc]
        HumanMessage("bar"),  # type: ignore[misc]
        AIMessage(  # type: ignore[misc]
            [
                {"text": "baz", "type": "text"},
                {
                    "tool_input": {"a": "b"},
                    "type": "tool_use",
                    "id": "1",
                    "text": None,
                    "name": "buz",
                },
                {"text": "baz", "type": "text"},
                {
                    "tool_input": {"a": "c"},
                    "type": "tool_use",
                    "id": "2",
                    "text": None,
                    "name": "blah",
                },
            ]
        ),
        ToolMessage("buz output", tool_call_id="1"),  # type: ignore[misc]
        ToolMessage("blah output", tool_call_id="2"),  # type: ignore[misc]
        HumanMessage("next thing"),  # type: ignore[misc]
    ]
    expected = [
        SystemMessage(
            [
                {'type': 'text', 'text': 'foo'},
                {'type': 'text', 'text': 'barfoo'}
            ]
        ),  # type: ignore[misc]
        HumanMessage("bar"),  # type: ignore[misc]
        AIMessage(  # type: ignore[misc]
            [
                {"text": "baz", "type": "text"},
                {
                    "tool_input": {"a": "b"},
                    "type": "tool_use",
                    "id": "1",
                    "text": None,
                    "name": "buz",
                },
                {"text": "baz", "type": "text"},
                {
                    "tool_input": {"a": "c"},
                    "type": "tool_use",
                    "id": "2",
                    "text": None,
                    "name": "blah",
                },
            ]
        ),
        HumanMessage(  # type: ignore[misc]
            [
                {"type": "tool_result", "content": "buz output", "tool_use_id": "1"},
                {"type": "tool_result", "content": "blah output", "tool_use_id": "2"},
                {"type": "text", "text": "next thing"},
            ]
        ),
    ]
    actual = _merge_messages(messages)
    assert expected == actual


def test__merge_messages_mutation() -> None:
    original_messages = [
        HumanMessage([{"type": "text", "text": "bar"}]),  # type: ignore[misc]
        HumanMessage("next thing"),  # type: ignore[misc]
    ]
    messages = [
        HumanMessage([{"type": "text", "text": "bar"}]),  # type: ignore[misc]
        HumanMessage("next thing"),  # type: ignore[misc]
    ]
    expected = [
        HumanMessage(  # type: ignore[misc]
            [{"type": "text", "text": "bar"}, {"type": "text", "text": "next thing"}]
        ),
    ]
    actual = _merge_messages(messages)
    assert expected == actual
    assert messages == original_messages


def test__format_anthropic_messages_with_tool_calls() -> None:
    system = SystemMessage("fuzz")  # type: ignore[misc]
    human = HumanMessage("foo")  # type: ignore[misc]
    ai = AIMessage(  # type: ignore[misc]
        "",
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "buzz"}}],
    )
    tool = ToolMessage(  # type: ignore[misc]
        "blurb",
        tool_call_id="1",
    )
    human_2 = HumanMessage("try again.")
    messages = [system, human, ai, tool, human_2]
    expected = (
        "fuzz",
        [
            {"role": "user", "content": "foo"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "bar",
                        "id": "1",
                        "input": {"baz": "buzz"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "blurb", "tool_use_id": "1"},
                    {"type": "text", "text": "try again."},
                ],
            },
        ],
    )
    actual = _format_anthropic_messages(messages)
    assert expected == actual


def test__format_anthropic_messages_with_str_content_and_tool_calls() -> None:
    system = SystemMessage("fuzz")  # type: ignore[misc]
    human = HumanMessage("foo")  # type: ignore[misc]
    # If content and tool_calls are specified and content is a string, then both are
    # included with content first.
    ai = AIMessage(  # type: ignore[misc]
        "thought",
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "buzz"}}],
    )
    tool = ToolMessage("blurb", tool_call_id="1")  # type: ignore[misc]
    messages = [system, human, ai, tool]
    expected = (
        "fuzz",
        [
            {"role": "user", "content": "foo"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thought"},
                    {
                        "type": "tool_use",
                        "name": "bar",
                        "id": "1",
                        "input": {"baz": "buzz"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "blurb", "tool_use_id": "1"}
                ],
            },
        ],
    )
    actual = _format_anthropic_messages(messages)
    assert expected == actual


def test__format_anthropic_messages_with_list_content_and_tool_calls() -> None:
    system = SystemMessage("fuzz")  # type: ignore[misc]
    human = HumanMessage("foo")  # type: ignore[misc]
    # If content and tool_calls are specified and content is a list, both are
    # included
    ai = AIMessage(  # type: ignore[misc]
        [{"type": "text", "text": "thought"}],
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "buzz"}}],
    )
    tool = ToolMessage(  # type: ignore[misc]
        "blurb",
        tool_call_id="1",
    )
    messages = [system, human, ai, tool]
    expected = (
        "fuzz",
        [
            {"role": "user", "content": "foo"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thought"},
                    {
                        "type": "tool_use",
                        "name": "bar",
                        "id": "1",
                        "input": {"baz": "buzz"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "blurb", "tool_use_id": "1"}
                ],
            },
        ],
    )
    actual = _format_anthropic_messages(messages)
    assert expected == actual


def test__format_anthropic_messages_with_tool_use_blocks_and_tool_calls() -> None:
    """Show that tool_calls are preferred to tool_use blocks when both have same id."""
    system = SystemMessage("fuzz")  # type: ignore[misc]
    human = HumanMessage("foo")  # type: ignore[misc]
    # NOTE: tool_use block in contents and tool_calls have different arguments.
    ai = AIMessage(  # type: ignore[misc]
        [
            {"type": "text", "text": "thought"},
            {
                "type": "tool_use",
                "name": "bar",
                "id": "1",
                "input": {"baz": "NOT_BUZZ"},
            },
        ],
        tool_calls=[{"name": "bar", "id": "1", "args": {"baz": "BUZZ"}}],
    )
    tool = ToolMessage("blurb", tool_call_id="1")  # type: ignore[misc]
    messages = [system, human, ai, tool]
    expected = (
        "fuzz",
        [
            {"role": "user", "content": "foo"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thought"},
                    {
                        "type": "tool_use",
                        "name": "bar",
                        "id": "1",
                        "input": {"baz": "BUZZ"},  # tool_calls value preferred.
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "blurb", "tool_use_id": "1"}
                ],
            },
        ],
    )
    actual = _format_anthropic_messages(messages)
    assert expected == actual


def test__format_anthropic_messages_with_cache_control() -> None:
    system = SystemMessage(
        [
            {
                "type": "text",
                "text": "fuzz",
                "cache_control": {"type": "ephemeral"},
            },
            "bar",
        ],
    )
    human = HumanMessage(
        [
            {
                "type": "text",
                "text": "foo",
                "cache_control": {"type": "ephemeral"},
            },
        ],
    )
    messages = [system, human]
    expected = (
        [
            {
                "type": "text",
                "text": "fuzz",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "bar"},
        ],
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "foo",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ],
    )

    actual = _format_anthropic_messages(messages)
    assert expected == actual


def test__format_anthropic_messages_system_message_list_content() -> None:
    """Test that system messages with list content return as list of content blocks"""
    system = SystemMessage(
        [
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": "Additional instructions here."},
        ],
    )
    human = HumanMessage("Hello!")
    messages = [system, human]
    expected = (
        [
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": "Additional instructions here."},
        ],
        [
            {"role": "user", "content": "Hello!"},
        ],
    )

    actual = _format_anthropic_messages(messages)
    assert expected == actual

def test__format_anthropic_multiple_system_messages() -> None:
    """Test that multiple system messages can be passed, and that none of them are required to be at position 0."""
    system1 = SystemMessage("foo")  # type: ignore[misc]
    system2 = SystemMessage("bar")  # type: ignore[misc]
    human = HumanMessage("Hello!")
    messages = [human, system1, system2]
    expected_system = [
        {'text': 'foo', 'type': 'text'},
        {'text': 'bar', 'type': 'text'}
    ]
    expected_messages = [
        {"role": "user", "content": "Hello!"}
    ]

    actual_system, actual_messages = _format_anthropic_messages(messages)
    assert expected_system == actual_system
    assert expected_messages == actual_messages


@pytest.fixture()
def pydantic() -> Type[BaseModel]:
    class dummy_function(BaseModel):
        """dummy function"""

        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    return dummy_function


@pytest.fixture()
def function() -> Callable:
    def dummy_function(arg1: int, arg2: Literal["bar", "baz"]) -> None:
        """dummy function

        Args:
            arg1: foo
            arg2: one of 'bar', 'baz'
        """
        pass

    return dummy_function


@pytest.fixture()
def dummy_tool() -> BaseTool:
    class Schema(BaseModel):
        arg1: int = Field(..., description="foo")
        arg2: Literal["bar", "baz"] = Field(..., description="one of 'bar', 'baz'")

    class DummyFunction(BaseTool):
        args_schema: Type[BaseModel] = Schema
        name: str = "dummy_function"
        description: str = "dummy function"

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            pass

    return DummyFunction()


@pytest.fixture()
def json_schema() -> Dict:
    return {
        "title": "dummy_function",
        "description": "dummy function",
        "type": "object",
        "properties": {
            "arg1": {"description": "foo", "type": "integer"},
            "arg2": {
                "description": "one of 'bar', 'baz'",
                "enum": ["bar", "baz"],
                "type": "string",
            },
        },
        "required": ["arg1", "arg2"],
    }


@pytest.fixture()
def openai_function() -> Dict:
    return {
        "name": "dummy_function",
        "description": "dummy function",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg1", "arg2"],
        },
    }


@pytest.fixture()
def tool_with_empty_description() -> Dict:
    return {
        "name": "dummy_function",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg1", "arg2"],
        },
    }


def test_convert_to_anthropic_tool(
    pydantic: Type[BaseModel],
    function: Callable,
    dummy_tool: BaseTool,
    json_schema: Dict,
    openai_function: Dict,
    tool_with_empty_description: Dict,
) -> None:
    expected = {
        "name": "dummy_function",
        "description": "dummy function",
        "input_schema": {
            "type": "object",
            "properties": {
                "arg1": {"description": "foo", "type": "integer"},
                "arg2": {
                    "description": "one of 'bar', 'baz'",
                    "enum": ["bar", "baz"],
                    "type": "string",
                },
            },
            "required": ["arg1", "arg2"],
        },
    }

    for fn in (pydantic, function, dummy_tool, json_schema, expected, openai_function):
        actual = convert_to_anthropic_tool(fn)  # type: ignore[arg-type]
        assert actual == expected

    expected["description"] = expected["name"]
    actual = convert_to_anthropic_tool(tool_with_empty_description)
    assert actual == expected


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def test_anthropic_bind_tools_tool_choice() -> None:
    chat_model = ChatBedrock(
        model_id="anthropic.claude-3-opus-20240229", region_name="us-west-2"
    )  # type: ignore[call-arg]
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], tool_choice={"type": "tool", "name": "GetWeather"}
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "tool",
        "name": "GetWeather",
    }
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], tool_choice="GetWeather"
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "tool",
        "name": "GetWeather",
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "auto"
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "type": "any"
    }


def test_standard_tracing_params() -> None:
    llm = ChatBedrock(model_id="foo", region_name="us-west-2")  # type: ignore[call-arg]
    expected = {
        "ls_provider": "amazon_bedrock",
        "ls_model_type": "chat",
        "ls_model_name": "foo",
    }
    assert llm._get_ls_params() == expected

    # Test initialization with `model` alias
    llm = ChatBedrock(model="foo", region_name="us-west-2")
    assert llm._get_ls_params() == expected

    llm = ChatBedrock(
        model_id="foo", model_kwargs={"temperature": 0.1}, region_name="us-west-2"
    )  # type: ignore[call-arg]
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "amazon_bedrock",
        "ls_model_type": "chat",
        "ls_model_name": "foo",
        "ls_temperature": 0.1,
    }


def test_beta_use_converse_api() -> None:
    llm = ChatBedrock(model_id="amazon.nova.foo", region_name="us-west-2")  # type: ignore[call-arg]
    assert llm.beta_use_converse_api

    llm = ChatBedrock(
        model="foobar",
        base_model="amazon.nova.foo",
        region_name="us-west-2")  # type: ignore[call-arg]
    assert llm.beta_use_converse_api

    llm = ChatBedrock(
        model="arn:aws:bedrock:::application-inference-profile/my-profile",
        base_model="claude.foo",
        region_name="us-west-2")  # type: ignore[call-arg]
    assert not llm.beta_use_converse_api

    llm = ChatBedrock(
        model="nova.foo", region_name="us-west-2", beta_use_converse_api=False
    )
    assert not llm.beta_use_converse_api

    llm = ChatBedrock(
        model="foobar",
        base_model="nova.foo",
        region_name="us-west-2",
        beta_use_converse_api=False
    )
    assert not llm.beta_use_converse_api

    llm = ChatBedrock(model="foo", region_name="us-west-2", beta_use_converse_api=True)
    assert llm.beta_use_converse_api

    llm = ChatBedrock(model="foo", region_name="us-west-2", beta_use_converse_api=False)
    assert not llm.beta_use_converse_api


@mock.patch("langchain_aws.chat_models.bedrock.create_aws_client")
def test_beta_use_converse_api_with_inference_profile(mock_create_aws_client):
    mock_bedrock_client = mock.MagicMock()
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [
            {
                "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
            }
        ]
    }
    mock_create_aws_client.return_value = mock_bedrock_client

    aip_model_id = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/my-profile"
    chat = ChatBedrock(
        model_id=aip_model_id, 
        region_name="us-west-2",
        bedrock_client=mock_bedrock_client
    ) # type: ignore[call-arg]

    mock_bedrock_client.get_inference_profile.assert_called_with(
        inferenceProfileIdentifier=aip_model_id
    )

    assert chat.beta_use_converse_api is False


@mock.patch("langchain_aws.chat_models.bedrock.create_aws_client")
def test_beta_use_converse_api_with_inference_profile_as_nova_model(mock_create_aws_client):
    mock_bedrock_client = mock.MagicMock()
    mock_bedrock_client.get_inference_profile.return_value = {
        "models": [
            {
                "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-micro-v1:0"
            }
        ]
    }
    mock_create_aws_client.return_value = mock_bedrock_client

    aip_model_id = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/my-profile"
    chat = ChatBedrock(
        model_id=aip_model_id, 
        region_name="us-west-2",
        bedrock_client=mock_bedrock_client
    ) # type: ignore[call-arg]

    mock_bedrock_client.get_inference_profile.assert_called_with(
        inferenceProfileIdentifier=aip_model_id
    )

    assert chat.beta_use_converse_api is True


@pytest.mark.parametrize(
    "model_id, provider, expected_provider, expectation, region_name",
    [
        (
            "amer.amazon.nova-pro-v1:0",
            None,
            "amazon",
            nullcontext(),
            "us-west-2",
        ),
        (
            "global.anthropic.claude-sonnet-4-20250514-v1:0",
            None,
            "anthropic",
            nullcontext(),
            "us-west-2",
        ),
        (
            "eu.anthropic.claude-3-haiku-20240307-v1:0",
            None,
            "anthropic",
            nullcontext(),
            "eu-west-1",
        ),
        (
            "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
            None,
            "anthropic",
            nullcontext(),
            "ap-northeast-1",
        ),
        (
            "us-gov.anthropic.claude-3-5-sonnet-20240620-v1:0",
            None,
            "anthropic",
            nullcontext(),
            "us-gov-west-1",
        ),
        (
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            None,
            "anthropic",
            nullcontext(),
            "us-west-2",
        ),
        ("meta.llama3-1-405b-instruct-v1:0", None, "meta", nullcontext(), "us-west-2"),
        (
            "arn:aws:bedrock:us-east-1::custom-model/cohere.command-r-v1:0/MyCustomModel2",
            "cohere",
            "cohere",
            nullcontext(),
            "us-east-1",
        ),
        (
            "arn:aws:bedrock:us-east-1::custom-model/cohere.command-r-v1:0/MyCustomModel2",
            None,
            "cohere",
            pytest.raises(ValueError),
            "us-east-1",
        ),
    ],
)
def test__get_provider(model_id, provider, expected_provider, expectation, region_name) -> None:
    llm = ChatBedrock(model_id=model_id, provider=provider, region_name=region_name)
    with expectation:
        assert llm._get_provider() == expected_provider


@mock.patch.dict(os.environ, {"AWS_REGION": "us-west-1"})
def test_chat_bedrock_different_regions() -> None:
    region = "ap-south-2"
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", region_name=region
    )
    assert llm.region_name == region


def test__format_anthropic_messages_with_thinking_blocks() -> None:
    """Test that thinking blocks are correctly formatted and preserved in messages."""
    system = SystemMessage("System instruction")  # type: ignore[misc]
    human = HumanMessage("What is the weather in NYC?")  # type: ignore[misc]
    ai = AIMessage(  # type: ignore[misc]
        "",
        additional_kwargs={
            "thinking": {
                "text": "I need to check the weather in NYC.",
                "signature": "SIG123",
            }
        },
        tool_calls=[{"name": "get_weather", "id": "1", "args": {"city": "nyc"}}],
    )
    tool = ToolMessage(  # type: ignore[misc]
        "It might be cloudy in nyc",
        tool_call_id="1",
    )

    messages = [system, human, ai, tool]
    expected_system, expected_messages = (
        "System instruction",
        [
            {"role": "user", "content": "What is the weather in NYC?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I need to check the weather in NYC.",
                        "signature": "SIG123",
                    },
                    {
                        "type": "tool_use",
                        "name": "get_weather",
                        "id": "1",
                        "input": {"city": "nyc"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "It might be cloudy in nyc",
                        "tool_use_id": "1",
                    }
                ],
            },
        ],
    )

    actual_system, actual_messages = _format_anthropic_messages(messages)
    assert expected_system == actual_system
    assert expected_messages == actual_messages


def test__format_anthropic_messages_with_image_conversion_in_tool() -> None:
    """Test that ToolMessage with OpenAI-style image content is correctly converted to Anthropic format."""
    # Create a dummy base64 image string
    dummy_base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    messages = [
        ToolMessage(  # type: ignore[misc]
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{dummy_base64_image}"
                    }
                }
            ],
            tool_call_id="test_tool_call_123"
        ),
        HumanMessage("What do you see in the image?"),  # type: ignore[misc]
    ]
    
    expected = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "test_tool_call_123",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": dummy_base64_image
                            }
                        }
                    ]
                },
                {"type": "text", "text": "What do you see in the image?"}
            ]
        }
    ]
    
    _ , actual = _format_anthropic_messages(messages)
    assert expected == actual


def test__convert_messages_to_prompt_anthropic_message_is_none() -> None:
    messages = None
    assert convert_messages_to_prompt_anthropic(messages) == ""


def test__convert_messages_to_prompt_anthropic_message_is_empty() -> None:
    messages = []
    assert convert_messages_to_prompt_anthropic(messages) == ""


def test__format_anthropic_messages_with_thinking_in_content_blocks() -> None:
    """Test that thinking blocks in content are correctly ordered (first) in messages."""
    system = SystemMessage("System instruction")  # type: ignore[misc]
    human = HumanMessage("What is the weather in NYC?")  # type: ignore[misc]

    # Create AIMessage with content list that has thinking block not at the start
    ai = AIMessage(  # type: ignore[misc]
        [
            {"type": "text", "text": "Let me check the weather."},
            {
                "type": "thinking",
                "thinking": "I should use the get_weather tool.",
                "signature": "SIG456",
            },
            {
                "type": "tool_use",
                "id": "tool1",
                "name": "get_weather",
                "input": {"city": "nyc"},
            },
        ],
    )
    tool = ToolMessage("It might be cloudy in nyc", tool_call_id="tool1")  # type: ignore[misc]

    messages = [system, human, ai, tool]
    expected_system, expected_messages = (
        "System instruction",
        [
            {"role": "user", "content": "What is the weather in NYC?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I should use the get_weather tool.",
                        "signature": "SIG456",
                    },
                    {"type": "text", "text": "Let me check the weather."},
                    {
                        "type": "tool_use",
                        "id": "tool1",
                        "name": "get_weather",
                        "input": {"city": "nyc"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "It might be cloudy in nyc",
                        "tool_use_id": "tool1",
                    }
                ],
            },
        ],
    )

    actual_system, actual_messages = _format_anthropic_messages(messages)
    assert expected_system == actual_system

    # Verify thinking blocks are placed first in the content array
    assert actual_messages[1]["content"][0]["type"] == "thinking"
    assert expected_messages == actual_messages


def test__format_anthropic_messages_after_tool_use_no_thinking() -> None:
    """Test message formatting for assistant responses after tool use (which shouldn't have thinking)."""
    system = SystemMessage("System instruction")  # type: ignore[misc]
    human = HumanMessage("What is the weather in NYC?")  # type: ignore[misc]

    # First assistant turn with thinking and tool use
    assistant1 = AIMessage(  # type: ignore[misc]
        "",
        additional_kwargs={
            "thinking": {
                "text": "I need to check the weather in NYC.",
                "signature": "SIG123",
            }
        },
        tool_calls=[{"name": "get_weather", "id": "1", "args": {"city": "nyc"}}],
    )

    # Tool result from user
    tool_result = ToolMessage("It might be cloudy in nyc", tool_call_id="1")  # type: ignore[misc]

    # Final assistant response without thinking
    assistant2 = AIMessage("Based on the data, it's cloudy in NYC.")  # type: ignore[misc]

    messages = [system, human, assistant1, tool_result, assistant2]
    _, actual_messages = _format_anthropic_messages(messages)

    # Check that the final assistant message has no thinking blocks
    assert len(actual_messages) == 4  # system isn't included in the array
    assert actual_messages[3]["role"] == "assistant"

    # The content should be a list with a single text block
    assert isinstance(actual_messages[3]["content"], list)
    assert len(actual_messages[3]["content"]) == 1
    assert actual_messages[3]["content"][0]["type"] == "text"
    assert (
        actual_messages[3]["content"][0]["text"]
        == "Based on the data, it's cloudy in NYC."
    )

    # Verify no thinking blocks in the final message
    assert not any(
        block.get("type") in ["thinking", "redacted_thinking"]
        for block in actual_messages[3]["content"]
    )


def test__format_anthropic_messages_tool_result_ordering() -> None:
    """Test that tool type content blocks in UserMessage are moved to the beginning."""
    human_message = HumanMessage(  # type: ignore[misc]
        [
            {"type": "text", "text": "I have a question about the data."},
            {
                "type": "tool_result",
                "content": "Data analysis result",
                "tool_use_id": "tool1"
            },
            {"type": "text", "text": "Can you explain this result?"},
        ]
    )

    system = SystemMessage("You are a helpful assistant")  # type: ignore[misc]
    ai = AIMessage("Let me think about that")  # type: ignore[misc]

    messages = [system, human_message, ai]

    _, formatted_messages = _format_anthropic_messages(messages)

    user_content = formatted_messages[0]["content"]

    assert user_content[0]["type"] == "tool_result"
    assert user_content[0]["content"] == "Data analysis result"
    assert user_content[1]["type"] == "text"
    assert user_content[1]["text"] == "I have a question about the data."
    assert user_content[2]["type"] == "text"
    assert user_content[2]["text"] == "Can you explain this result?"


def test__format_anthropic_messages_tool_use_ordering() -> None:
    """Test that tool type content blocks in AssistantMessage are always moved to the end."""
    ai_message = AIMessage(  # type: ignore[misc]
        [
            {"type": "text", "text": "Let me analyze this for you."},
            {
                "type": "tool_use",
                "name": "data_analyzer",
                "id": "tool1",
                "input": {"data": "sample_data"}
            },
            {"type": "text", "text": "This will help us understand the pattern."},
        ]
    )

    system = SystemMessage("You are a helpful assistant")  # type: ignore[misc]
    human = HumanMessage("Can you analyze this data?")  # type: ignore[misc]

    messages = [system, human, ai_message]

    _, formatted_messages = _format_anthropic_messages(messages)

    assistant_content = formatted_messages[1]["content"]

    assert assistant_content[0]["type"] == "text"
    assert assistant_content[0]["text"] == "Let me analyze this for you."
    assert assistant_content[1]["type"] == "text"
    assert assistant_content[1]["text"] == "This will help us understand the pattern."
    assert assistant_content[2]["type"] == "tool_use"
    assert assistant_content[2]["name"] == "data_analyzer"


def test__format_anthropic_messages_preserves_content_order() -> None:
    """Test that _format_anthropic_messages preserves the original order of mixed text and image content."""
    content = [
        {"type": "text", "text": "Some text..."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,1337C0DE"},
        },
        {"type": "text", "text": "Caption for 1st image..."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,FACADE42"},
        },
        {"type": "text", "text": "Another caption for 2nd image..."},
        {"type": "text", "text": "Now, analyze the following image..."},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,C0FFEE42"},
        },
    ]

    messages = [HumanMessage(content=content)]

    _, formatted_messages = _format_anthropic_messages(messages)

    assert len(formatted_messages) == 1
    assert formatted_messages[0]["role"] == "user"

    user_content = formatted_messages[0]["content"]

    # The expected order should preserve the original interleaved sequence:
    # text -> image -> text -> image -> text -> text -> image
    expected_sequence = [
        ("text", "Some text..."),
        ("image", "1337C0DE"),
        ("text", "Caption for 1st image..."),
        ("image", "FACADE42"),
        ("text", "Another caption for 2nd image..."),
        ("text", "Now, analyze the following image..."),
        ("image", "C0FFEE42"),
    ]

    actual_sequence = []
    for item in user_content:
        if item["type"] == "text":
            actual_sequence.append(("text", item["text"]))
        elif item["type"] == "image":
            actual_sequence.append(("image", item["source"]["data"]))

    assert actual_sequence == expected_sequence, (
        f"Content order was not preserved. Expected: {expected_sequence}, "
        f"but got: {actual_sequence}"
    )


@pytest.mark.parametrize(
    "model_id, base_model_id, provider, expected_format_marker",
    [
        (
            "arn:aws:bedrock:us-west-2::custom-model/meta.llama3-8b-instruct-v1:0/MyModel",
            "meta.llama3-8b-instruct-v1:0",
            "meta",
            "<|begin_of_text|>"
        ),
        (
            "arn:aws:bedrock:us-west-2::custom-model/meta.llama2-70b-chat-v1/MyModel",
            "meta.llama2-70b-chat-v1",
            "meta",
            "[INST]"
        ),
        (
            "meta.llama2-70b-chat-v1",
            "meta.llama3-8b-instruct-v1:0",
            "meta",
            "<|begin_of_text|>"
        ),
        (
            "arn:aws:sagemaker:us-west-2::endpoint/endpoint-quick-start-xxxxx",
            "deepseek.r1-v1:0",
            "deepseek",
            "<|begin_of_sentence|>"
        ),
    ]
)
def test_chat_prompt_adapter_with_model_detection(model_id, base_model_id, provider, expected_format_marker):
    """Test that ChatPromptAdapter correctly formats prompts when base_model is provided."""
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Hello")
    ]

    chat = ChatBedrock(
        model_id=model_id,
        base_model_id=base_model_id,
        provider=provider,
        region_name="us-west-2"
    )

    model_name = chat._get_base_model()
    provider_name = chat._get_provider()

    prompt = ChatPromptAdapter.convert_messages_to_prompt(
        provider=provider_name,
        messages=messages,
        model=model_name
    )

    assert expected_format_marker in prompt


def test__format_anthropic_messages_empty_content_fix() -> None:
    """
    Test that empty content arrays are handled correctly to prevent ValidationException.
    """
    messages = [
        HumanMessage("What is the capital of India?"),  # type: ignore[misc]
        AIMessage([{"type": "text", "text": ""}])  # type: ignore[misc]
    ]
    
    system, formatted_messages = _format_anthropic_messages(messages)
    
    assert len(formatted_messages) == 2
    ai_content = formatted_messages[1]["content"]
    assert isinstance(ai_content, list)
    assert len(ai_content) > 0  # Should not be empty
    assert ai_content[0]["type"] == "text"
    assert ai_content[0]["text"] == "."


def test__format_anthropic_messages_whitespace_only_content() -> None:
    """Test that whitespace-only content is handled correctly."""
    messages = [
        HumanMessage("What is the capital of India?"),  # type: ignore[misc]
        AIMessage([{"type": "text", "text": "   \n  \t  "}])  # type: ignore[misc]
    ]
    
    system, formatted_messages = _format_anthropic_messages(messages)
    
    assert len(formatted_messages) == 2
    ai_content = formatted_messages[1]["content"]
    assert isinstance(ai_content, list)
    assert len(ai_content) > 0
    assert ai_content[0]["type"] == "text"
    assert ai_content[0]["text"] == "."


def test__format_anthropic_messages_empty_string_content() -> None:
    """Test that empty string content is handled correctly."""
    messages = [
        HumanMessage("What is the capital of India?"),  # type: ignore[misc]
        AIMessage("")  # type: ignore[misc]
    ]
    
    system, formatted_messages = _format_anthropic_messages(messages)
    
    assert len(formatted_messages) == 2
    ai_content = formatted_messages[1]["content"]
    assert isinstance(ai_content, list)
    assert len(ai_content) > 0
    assert ai_content[0]["type"] == "text"
    assert ai_content[0]["text"] == "."


def test__format_anthropic_messages_mixed_empty_content() -> None:
    """Test that mixed content with some empty blocks is handled correctly."""
    messages = [
        HumanMessage("What is the capital of India?"),  # type: ignore[misc]
        AIMessage([  # type: ignore[misc]
            {"type": "text", "text": ""},
            {"type": "text", "text": "   "},
            {"type": "text", "text": ""}
        ])
    ]
    
    system, formatted_messages = _format_anthropic_messages(messages)
    
    # Verify that the content is not empty even when all text blocks are filtered out
    assert len(formatted_messages) == 2
    ai_content = formatted_messages[1]["content"]
    assert isinstance(ai_content, list)
    assert len(ai_content) > 0
    assert ai_content[0]["type"] == "text"
    assert ai_content[0]["text"] == "."


def test__format_anthropic_messages_mixed_type_blocks_and_empty_content() -> None:
    """Test that empty blocks mixed with non-text type blocks is handled correctly."""
    messages = [
        AIMessage([  # type: ignore[misc]
            {"type": "text", "text": "\n\t"},
            {
                "type": "tool_use",
                "id": "tool_call1",
                "input": {"arg1": "val1"},
                "name": "tool1",
            },
        ])
    ]

    expected_content = [
        {
            'role': 'assistant',
            'content': [
                {
                    'type': 'tool_use',
                    'id': 'tool_call1',
                    'input': {'arg1': 'val1'},
                    'name': 'tool1'
                }
            ]
        }
    ]

    system, formatted_messages = _format_anthropic_messages(messages)

    assert len(formatted_messages) == 1
    assert formatted_messages == expected_content


def test_model_kwargs() -> None:
    """Test we can transfer unknown params to model_kwargs."""
    llm = ChatBedrock(
        model_id="my-model",
        region_name="us-west-2",
        model_kwargs={"foo": "bar"},
    )
    assert llm.model_id == "my-model"
    assert llm.region_name == "us-west-2"
    assert llm.model_kwargs == {"foo": "bar"}

    with pytest.warns(match="transferred to model_kwargs"):
        llm = ChatBedrock(
            model_id="my-model",
            region_name="us-west-2",
            foo="bar",
        )
    assert llm.model_id == "my-model"
    assert llm.region_name == "us-west-2"
    assert llm.model_kwargs == {"foo": "bar"}

    with pytest.warns(match="transferred to model_kwargs"):
        llm = ChatBedrock(
            model_id="my-model",
            region_name="us-west-2",
            foo="bar",
            model_kwargs={"baz": "qux"},
        )
    assert llm.model_id == "my-model"
    assert llm.region_name == "us-west-2"
    assert llm.model_kwargs == {"foo": "bar", "baz": "qux"}

    # For backward compatibility, test that we don't transfer known parameters out
    # of model_kwargs
    llm = ChatBedrock(
        model_id="my-model",
        region_name="us-west-2",
        model_kwargs={"stop_sequences": ["test"]},
    )
    assert llm.model_kwargs == {"stop_sequences": ["test"]}
    assert llm.stop_sequences is None
