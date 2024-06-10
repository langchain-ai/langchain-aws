"""Test Bedrock chat model."""

from typing import Any, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langchain_aws.chat_models.bedrock import ChatBedrock
from tests.callbacks import FakeCallbackHandler, FakeCallbackHandlerWithTokenCounts


@pytest.fixture
def chat() -> ChatBedrock:
    return ChatBedrock(model_id="anthropic.claude-v2", model_kwargs={"temperature": 0})  # type: ignore[call-arg]


@pytest.mark.scheduled
def test_chat_bedrock(chat: ChatBedrock) -> None:
    """Test BedrockChat wrapper."""
    system = SystemMessage(content="You are a helpful assistant.")
    human = HumanMessage(content="Hello")
    response = chat([system, human])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_bedrock_generate(chat: ChatBedrock) -> None:
    """Test BedrockChat wrapper with generate."""
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.scheduled
def test_chat_bedrock_generate_with_token_usage(chat: ChatBedrock) -> None:
    """Test BedrockChat wrapper with generate."""
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert isinstance(response.llm_output, dict)

    usage = response.llm_output["usage"]
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] > 0


@pytest.mark.scheduled
def test_chat_bedrock_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    chat = ChatBedrock(  # type: ignore[call-arg]
        model_id="anthropic.claude-v2",
        streaming=True,
        callbacks=[callback_handler],
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_chat_bedrock_streaming_llama3() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    chat = ChatBedrock(  # type: ignore[call-arg]
        model_id="meta.llama3-8b-instruct-v1:0",
        streaming=True,
        callbacks=[callback_handler],
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_chat_bedrock_streaming_generation_info() -> None:
    """Test that generation info is preserved when streaming."""

    class _FakeCallback(FakeCallbackHandler):
        saved_things: dict = {}

        def on_llm_end(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Save the generation
            self.saved_things["generation"] = args[0]

    callback = _FakeCallback()
    chat = ChatBedrock(  # type: ignore[call-arg]
        model_id="anthropic.claude-v2",
        callbacks=[callback],
        model_kwargs={"temperature": 0},
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello!"


@pytest.mark.scheduled
def test_bedrock_streaming(chat: ChatBedrock) -> None:
    """Test streaming tokens from OpenAI."""

    full = None
    for token in chat.stream("I'm Pickle Rick"):
        full = token if full is None else full + token  # type: ignore[operator]
        assert isinstance(token.content, str)
    assert isinstance(cast(AIMessageChunk, full).content, str)


@pytest.mark.scheduled
async def test_bedrock_astream(chat: ChatBedrock) -> None:
    """Test streaming tokens from OpenAI."""

    async for token in chat.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_bedrock_abatch(chat: ChatBedrock) -> None:
    """Test streaming tokens from BedrockChat."""
    result = await chat.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_bedrock_abatch_tags(chat: ChatBedrock) -> None:
    """Test batch tokens from BedrockChat."""
    result = await chat.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_bedrock_batch(chat: ChatBedrock) -> None:
    """Test batch tokens from BedrockChat."""
    result = chat.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_bedrock_ainvoke(chat: ChatBedrock) -> None:
    """Test invoke tokens from BedrockChat."""
    result = await chat.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_bedrock_invoke(chat: ChatBedrock) -> None:
    """Test invoke tokens from BedrockChat."""
    result = chat.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
    assert "usage" in result.additional_kwargs
    assert result.additional_kwargs["usage"]["prompt_tokens"] == 13


@pytest.mark.scheduled
def test_function_call_invoke_with_system(chat: ChatBedrock) -> None:
    class GetWeather(BaseModel):
        location: str = Field(..., description="The city and state")

    llm_with_tools = chat.bind_tools([GetWeather])

    messages = [
        SystemMessage(content="answer only in french"),
        HumanMessage(content="what is the weather like in San Francisco"),
    ]

    response = llm_with_tools.invoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_tool_use_call_invoke() -> None:
    chat = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0.001},
    )

    class GetWeather(BaseModel):
        location: str = Field(..., description="The city and state")

    @tool(args_schema=GetWeather)
    def get_weather(location: str) -> str:
        """Useful for getting the weather in a location."""
        return f"The weather in {location} is nice."

    llm_with_tools = chat.bind_tools([get_weather])

    messages = [HumanMessage(content="what is the weather like in San Francisco CA")]

    response = llm_with_tools.invoke(messages)
    assert isinstance(response, AIMessage)
    assert "tool_calls" in response.__dict__.keys()
    tool_calls = response.tool_calls
    assert isinstance(tool_calls, list)
    tool_name = tool_calls[0]["name"]
    assert tool_name == "get_weather"


@pytest.mark.scheduled
@pytest.mark.parametrize("streaming", [True, False])
def test_chat_bedrock_token_callbacks(streaming: bool) -> None:
    """
    Test that streaming correctly invokes on_llm_end
    and stores token counts and stop reason.
    """
    callback_handler = FakeCallbackHandlerWithTokenCounts()
    chat = ChatBedrock(  # type: ignore[call-arg]
        model_id="anthropic.claude-v2",
        streaming=streaming,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat.invoke([message], RunnableConfig(callbacks=[callback_handler]))
    assert callback_handler.input_token_count > 0
    assert callback_handler.output_token_count > 0
    assert callback_handler.stop_reason is not None
    assert isinstance(response, BaseMessage)


@pytest.mark.scheduled
def test_function_call_invoke_without_system(chat: ChatBedrock) -> None:
    class GetWeather(BaseModel):
        location: str = Field(..., description="The city and state")

    llm_with_tools = chat.bind_tools([GetWeather])

    messages = [HumanMessage(content="what is the weather like in San Francisco")]

    response = llm_with_tools.invoke(messages)
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
async def test_function_call_invoke_with_system_astream(chat: ChatBedrock) -> None:
    class GetWeather(BaseModel):
        location: str = Field(..., description="The city and state")

    llm_with_tools = chat.bind_tools([GetWeather])

    messages = [
        SystemMessage(content="answer only in french"),
        HumanMessage(content="what is the weather like in San Francisco"),
    ]

    for chunk in llm_with_tools.stream(messages):
        assert isinstance(chunk.content, str)


@pytest.mark.scheduled
async def test_function_call_invoke_without_system_astream(chat: ChatBedrock) -> None:
    class GetWeather(BaseModel):
        location: str = Field(..., description="The city and state")

    llm_with_tools = chat.bind_tools([GetWeather])

    messages = [HumanMessage(content="what is the weather like in San Francisco")]

    for chunk in llm_with_tools.stream(messages):
        assert isinstance(chunk.content, str)
