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
    convert_messages_to_prompt_anthropic
)
from langchain_aws.function_calling import convert_to_anthropic_tool


def test__merge_messages() -> None:
    messages = [
        SystemMessage("foo"),  # type: ignore[misc]
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
        SystemMessage("foo"),  # type: ignore[misc]
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
    # If content and tool_calls are specified and content is a list, then content is
    # preferred.
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
                "content": [{"type": "text", "text": "thought"}],
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
        "fuzzbar",
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
    llm = ChatBedrock(model_id="nova.foo", region_name="us-west-2")  # type: ignore[call-arg]
    assert llm.beta_use_converse_api

    llm = ChatBedrock(
        model="nova.foo", region_name="us-west-2", beta_use_converse_api=False
    )
    assert not llm.beta_use_converse_api

    llm = ChatBedrock(model="foo", region_name="us-west-2", beta_use_converse_api=True)
    assert llm.beta_use_converse_api

    llm = ChatBedrock(model="foo", region_name="us-west-2", beta_use_converse_api=False)
    assert not llm.beta_use_converse_api


@pytest.mark.parametrize(
    "model_id, provider, expected_provider, expectation, region_name",
    [
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
