"""Test Bedrock chat model."""

import json
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from langchain_aws.chat_models.bedrock import ChatBedrock
from tests.callbacks import FakeCallbackHandler, FakeCallbackHandlerWithTokenCounts


@pytest.fixture
def chat() -> ChatBedrock:
    return ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0},
    )  # type: ignore[call-arg]


@pytest.mark.scheduled
def test_chat_bedrock(chat: ChatBedrock) -> None:
    """Test ChatBedrock wrapper."""
    system = SystemMessage(content="You are a helpful assistant.")
    human = HumanMessage(content="Hello")
    response = chat.invoke([system, human])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.scheduled
def test_chat_bedrock_generate(chat: ChatBedrock) -> None:
    """Test ChatBedrock wrapper with generate."""
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
    """Test ChatBedrock wrapper with generate."""
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert isinstance(response.llm_output, dict)

    usage = response.llm_output["usage"]
    assert usage["prompt_tokens"] == 16
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] > 0


@pytest.mark.scheduled
def test_chat_bedrock_streaming() -> None:
    """Test that streaming correctly streams chunks."""
    chat = ChatBedrock(  # type: ignore[call-arg]
        model_id="anthropic.claude-v2"
    )
    message = HumanMessage(content="Hello")
    stream = chat.stream([message])

    full = next(stream)
    for chunk in stream:
        full += chunk  # type: ignore[assignment]

    assert full.content
    assert full.response_metadata
    assert full.usage_metadata  # type: ignore[attr-defined]


@pytest.mark.scheduled
def test_chat_bedrock_token_counts() -> None:
    chat = ChatBedrock(  # type: ignore[call-arg]
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0},
    )
    invoke_response = chat.invoke("hi", max_tokens=6)
    assert isinstance(invoke_response, AIMessage)
    assert invoke_response.usage_metadata is not None
    assert invoke_response.usage_metadata["output_tokens"] <= 6

    stream = chat.stream("hi", max_tokens=6)
    stream_response = next(stream)
    for chunk in stream:
        stream_response += chunk
    assert isinstance(stream_response, AIMessage)
    assert stream_response.usage_metadata is not None
    assert stream_response.usage_metadata["output_tokens"] <= 6


@pytest.mark.scheduled
def test_chat_bedrock_streaming_llama3() -> None:
    """Test that streaming correctly streams message chunks"""
    chat = ChatBedrock(  # type: ignore[call-arg]
        model_id="meta.llama3-8b-instruct-v1:0"
    )
    message = HumanMessage(content="Hello")

    response = AIMessageChunk(content="")
    for chunk in chat.stream([message]):
        response += chunk  # type: ignore[assignment]

    assert response.content
    assert response.response_metadata
    assert response.usage_metadata


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
@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
    ],
)
def test_bedrock_streaming(model_id: str) -> None:
    chat = ChatBedrock(
        model_id=model_id,
        model_kwargs={"temperature": 0},
    )  # type: ignore[call-arg]
    full = None
    for token in chat.stream("I'm Pickle Rick"):
        full = token if full is None else full + token  # type: ignore[operator]
        assert isinstance(token.content, str)
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, str)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert full.usage_metadata["total_tokens"] > 0


@pytest.mark.scheduled
@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
    ],
)
async def test_bedrock_astream(model_id: str) -> None:
    """Test streaming tokens from OpenAI."""
    chat = ChatBedrock(
        model_id=model_id,
        model_kwargs={"temperature": 0},
    )  # type: ignore[call-arg]
    full = None
    async for token in chat.astream("I'm Pickle Rick"):
        full = token if full is None else full + token  # type: ignore[operator]
        assert isinstance(token.content, str)
    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, str)
    assert full.usage_metadata is not None
    assert full.usage_metadata["input_tokens"] > 0
    assert full.usage_metadata["output_tokens"] > 0
    assert full.usage_metadata["total_tokens"] > 0


@pytest.mark.scheduled
async def test_bedrock_abatch(chat: ChatBedrock) -> None:
    """Test streaming tokens from ChatBedrock."""
    result = await chat.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_bedrock_abatch_tags(chat: ChatBedrock) -> None:
    """Test batch tokens from ChatBedrock."""
    result = await chat.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_bedrock_batch(chat: ChatBedrock) -> None:
    """Test batch tokens from ChatBedrock."""
    result = chat.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
async def test_bedrock_ainvoke(chat: ChatBedrock) -> None:
    """Test invoke tokens from ChatBedrock."""
    result = await chat.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_bedrock_invoke(chat: ChatBedrock) -> None:
    """Test invoke tokens from ChatBedrock."""
    result = chat.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
    assert "usage" in result.additional_kwargs
    assert result.additional_kwargs["usage"]["prompt_tokens"] == 12


class GetWeather(BaseModel):
    """Useful for getting the weather in a location."""

    location: str = Field(..., description="The city and state")


class AnswerWithJustification(BaseModel):
    """An answer to the user question along with justification for the answer."""

    answer: str
    justification: str


@pytest.mark.scheduled
def test_structured_output() -> None:
    chat = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0.001},
    )  # type: ignore[call-arg]
    structured_llm = chat.with_structured_output(AnswerWithJustification)

    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )

    assert isinstance(response, AnswerWithJustification)


@pytest.mark.scheduled
def test_tool_use_call_invoke() -> None:
    chat = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0.001},
    )  # type: ignore[call-arg]

    llm_with_tools = chat.bind_tools([GetWeather])

    messages = [HumanMessage(content="what is the weather like in San Francisco CA")]

    response = llm_with_tools.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.tool_calls, list)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "GetWeather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]

    # Test streaming
    first = True
    for chunk in llm_with_tools.stream("what's the weather in san francisco, ca"):
        if first:
            gathered = chunk
            first = False
        else:
            gathered = gathered + chunk  # type: ignore
    assert isinstance(gathered, AIMessageChunk)
    assert isinstance(gathered.tool_call_chunks, list)
    assert len(gathered.tool_call_chunks) == 1
    tool_call_chunk = gathered.tool_call_chunks[0]
    assert tool_call_chunk["name"] == "GetWeather"
    assert isinstance(tool_call_chunk["args"], str)
    assert "location" in json.loads(tool_call_chunk["args"])


@pytest.mark.parametrize("tool_choice", ["GetWeather", "auto", "any"])
def test_anthropic_bind_tools_tool_choice(tool_choice: str) -> None:
    chat = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0.001},
    )  # type: ignore[call-arg]
    chat_model_with_tools = chat.bind_tools([GetWeather], tool_choice=tool_choice)
    response = chat_model_with_tools.invoke("what's the weather in ny and la")
    assert isinstance(response, AIMessage)
    assert response.tool_calls
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "GetWeather"
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]


@pytest.mark.scheduled
def test_chat_bedrock_token_callbacks() -> None:
    """
    Test that streaming correctly invokes on_llm_end
    and stores token counts and stop reason.
    """
    callback_handler = FakeCallbackHandlerWithTokenCounts()
    chat = ChatBedrock(  # type: ignore[call-arg]
        model_id="anthropic.claude-v2", streaming=False, verbose=True
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
async def test_function_call_invoke_with_system(chat: ChatBedrock) -> None:
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
async def test_function_call_invoke_without_system_astream(chat: ChatBedrock) -> None:
    class GetWeather(BaseModel):
        location: str = Field(..., description="The city and state")

    llm_with_tools = chat.bind_tools([GetWeather])

    messages = [HumanMessage(content="what is the weather like in San Francisco")]

    astream = llm_with_tools.astream(messages)
    full = await astream.__anext__()
    async for chunk in astream:
        full += chunk  # type: ignore[assignment]

    assert full.tool_calls  # type: ignore[attr-defined]


@pytest.mark.skip(reason="Needs guardrails setup to run.")
def test_guardrails() -> None:
    params = {
        "region_name": "us-west-2",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "guardrails": {
            "guardrailIdentifier": "e7esbceow153",
            "guardrailVersion": "1",
            "trace": "enabled",
        },
        "beta_use_converse_api": True,
    }
    chat_model = ChatBedrock(**params)  # type: ignore[arg-type]
    messages = [
        HumanMessage(
            content=[
                "Create a playlist of 2 heavy metal songs.",
                {
                    "guardContent": {
                        "text": {"text": "Only answer with a list of songs."}
                    }
                },
            ]
        )
    ]
    response = chat_model.invoke(messages)

    assert (
        response.content == "Sorry, I can't answer questions about heavy metal music."
    )
    assert response.response_metadata["stopReason"] == "guardrail_intervened"
    assert response.response_metadata["trace"] is not None

    stream = chat_model.stream(messages)
    response = next(stream)
    for chunk in stream:
        response += chunk

    assert (
        response.content[0]["text"]  # type: ignore[index]
        == "Sorry, I can't answer questions about heavy metal music."
    )
    assert response.response_metadata["stopReason"] == "guardrail_intervened"
    assert response.response_metadata["trace"] is not None
