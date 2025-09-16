"""Test Bedrock chat model."""

import json
from typing import Any, Literal, Optional, Union
from uuid import UUID
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_aws.chat_models.bedrock import ChatBedrock
from tests.callbacks import FakeCallbackHandler, FakeCallbackHandlerWithTokenCounts


@pytest.fixture
def chat() -> ChatBedrock:
    return ChatBedrock(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
    model_name = stream_response.response_metadata["model_name"]
    assert model_name == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"


@pytest.mark.scheduled
def test_chat_bedrock_token_counts_deepseek_r1() -> None:
    chat = ChatBedrock(  # type: ignore[call-arg]
        model_id="us.deepseek.r1-v1:0",
        temperature=0,
        max_tokens=6,
    )

    invoke_response = chat.invoke("hi")
    assert isinstance(invoke_response, AIMessage)
    assert invoke_response.usage_metadata is not None
    assert invoke_response.usage_metadata["output_tokens"] <= 6

    stream = chat.stream("hi")
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
def test_chat_bedrock_streaming_deepseek_r1() -> None:
    chat = ChatBedrock(  # type: ignore[call-arg]
        model="us.deepseek.r1-v1:0",
        region_name="us-west-2"
    )
    message = HumanMessage(content="Hello")

    response = AIMessageChunk(content="")
    for chunk in chat.stream([message]):
        response += chunk  # type: ignore[assignment]

    assert response.content
    assert response.response_metadata
    assert response.usage_metadata


@pytest.mark.skip("Needs provisioned instance setup.")
def test_chat_bedrock_streaming_deepseek_r1_distill_llama() -> None:
    chat = ChatBedrock(  # type: ignore[call-arg]
        provider="deepseek",
        model_id="arn:aws:sagemaker:us-east-2:xxxxxxxxxxxx:endpoint/endpoint-quick-start-xxxxx",
        region_name="us-east-2"
    )
    message = HumanMessage(content="Hello. Please limit your response to 10 words or less.")

    response = AIMessageChunk(content="")
    for chunk in chat.stream([message]):
        response += chunk # type: ignore[assignment]

    assert response.content
    assert response.response_metadata
    assert response.usage_metadata


@pytest.mark.skip("Needs provisioned instance setup.")
def test_chat_bedrock_streaming_deepseek_r1_distill_qwen() -> None:
    chat = ChatBedrock(  # type: ignore[call-arg]
        provider="deepseek",
        model_id="arn:aws:sagemaker:us-east-2:xxxxxxxxxxxx:endpoint/endpoint-quick-start-xxxxx",
        region_name="us-east-2"
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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        callbacks=[callback],
        model_kwargs={"temperature": 0},
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert generation.generations[0][0].text == "Hello! How can I assist you today?"


@pytest.mark.scheduled
@pytest.mark.parametrize(
    "model_id",
    [
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        model_kwargs={"temperature": 0.001},
    )  # type: ignore[call-arg]
    structured_llm = chat.with_structured_output(AnswerWithJustification)

    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )

    assert isinstance(response, AnswerWithJustification)

@pytest.mark.scheduled
def test_structured_output_anthropic_format() -> None:
    chat = ChatBedrock(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    )  # type: ignore[call-arg]
    schema = {
        "name": "AnswerWithJustification",
        "description": (
            "An answer to the user question along with justification for the answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "justification": {"type": "string"},
            },
            "required": ["answer", "justification"]
        }
    }
    structured_llm = chat.with_structured_output(schema)
    response = structured_llm.invoke(
        "What weighs more a pound of bricks or a pound of feathers"
    )
    assert isinstance(response, dict)
    assert isinstance(response["answer"], str)
    assert isinstance(response["justification"], str)

@pytest.mark.scheduled
def test_tool_use_call_invoke() -> None:
    chat = ChatBedrock(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        streaming=False,
        verbose=True
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


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_agent_loop_bedrock(output_version: Literal["v0", "v1"]) -> None:

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatBedrock(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        output_version=output_version,
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")
    tool_call_message = llm_with_tools.invoke([input_message])
    assert isinstance(tool_call_message, AIMessage)
    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_agent_loop_streaming_bedrock(output_version: Literal["v0", "v1"]) -> None:

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return "It's sunny."

    llm = ChatBedrock(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        output_version=output_version,
    )
    llm_with_tools = llm.bind_tools([get_weather])
    input_message = HumanMessage("What is the weather in San Francisco, CA?")

    tool_call_message: Optional[BaseMessageChunk] = None
    for chunk in llm_with_tools.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        tool_call_message = (
            chunk if tool_call_message is None else tool_call_message + chunk
        )
    assert isinstance(tool_call_message, AIMessageChunk)

    tool_calls = tool_call_message.tool_calls
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    tool_message = get_weather.invoke(tool_call)
    assert isinstance(tool_message, ToolMessage)
    response = llm_with_tools.invoke(
        [
            input_message,
            tool_call_message,
            tool_message,
        ]
    )
    assert isinstance(response, AIMessage)


@pytest.mark.parametrize("output_version", ["v0", "v1"])
def test_thinking_bedrock(output_version: Literal["v0", "v1"]) -> None:
    llm = ChatBedrock(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=4096,
        model_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        output_version=output_version,
    )

    input_message = {"role": "user", "content": "What is 3^3?"}
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)

    if output_version == "v0":
        assert [block["type"] for block in full.content] == ["thinking", "text"]  # type: ignore[index,union-attr]
        assert full.content[0]["thinking"]  # type: ignore[index,union-attr]
        assert full.content[0]["signature"]  # type: ignore[index,union-attr]
    else:
        # v1
        assert [block["type"] for block in full.content] == ["reasoning", "text"]  # type: ignore[index,union-attr]
        assert "signature" in full.content[0]["extras"]  # type: ignore[index,union-attr]

    content_blocks = full.content_blocks
    assert [block["type"] for block in content_blocks] == ["reasoning", "text"]
    assert content_blocks[0]["reasoning"]
    assert content_blocks[0]["extras"]["signature"]

    next_message = {"role": "user", "content": "Thanks!"}
    response = llm.invoke([input_message, full, next_message])

    if output_version == "v0":
        assert [block["type"] for block in response.content] == ["thinking", "text"]  # type: ignore[index,union-attr]
        assert response.content[0]["thinking"]  # type: ignore[index,union-attr]
        assert response.content[0]["signature"]  # type: ignore[index,union-attr]
    else:
        # v1
        assert [block["type"] for block in response.content] == ["reasoning", "text"]  # type: ignore[index,union-attr]
        assert "signature" in response.content[0]["extras"]  # type: ignore[index,union-attr]

    content_blocks = response.content_blocks
    assert [block["type"] for block in content_blocks] == ["reasoning", "text"]
    assert content_blocks[0]["reasoning"]
    assert content_blocks[0]["extras"]["signature"]


@pytest.mark.xfail(
    reason=(
        "Need to update content to list type when citations are enabled in input "
        "documents."
    )
)
def test_citations_bedrock() -> None:
    llm = ChatBedrock(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=4096,
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "content",
                        "content": [
                            {"type": "text", "text": "The grass is green"},
                            {"type": "text", "text": "The sky is blue"},
                        ],
                    },
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": "What color is the grass and sky?"},
            ],
        },
    ]
    response = llm.invoke(messages)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, list)
    assert any("citations" in block for block in response.content)

    # Test streaming
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream(messages):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk

    assert isinstance(full, AIMessageChunk)
    assert isinstance(full.content, list)
    assert not any("citation" in block for block in full.content)
    assert any("citations" in block for block in full.content)


@pytest.mark.skip(reason="Needs guardrails setup to run.")
def test_guardrails() -> None:
    params = {
        "region_name": "us-west-2",
        "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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


class GuardrailTraceCallbackHandler(FakeCallbackHandler):
    """Callback handler to capture guardrail trace information."""
    
    def __init__(self) -> None:
        super().__init__()
        self.trace_captured = False
        self.trace_info: dict = {}
        
    def on_llm_error(
        self, 
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Union[UUID, None] = None,
        **kwargs: Any
    ) -> Any:
        """Handle LLM errors, including guardrail interventions."""
        reason = kwargs.get("reason")
        if reason and reason == "GUARDRAIL_INTERVENED":
            self.trace_captured = True
            self.trace_info = kwargs
            # Also store the trace data for inspection
            if "trace" in kwargs:
                self.trace_info["trace_data"] = kwargs["trace"]


@pytest.mark.skip(reason="Needs guardrails setup to run.")
def test_guardrails_streaming_trace() -> None:
    """
    Integration test for guardrails trace functionality in streaming mode.
    
    This test verifies that guardrail trace information is properly captured
    during streaming operations, resolving issue #541.
    
    Note: Requires a valid guardrail to be configured in AWS Bedrock.
    Update the guardrailIdentifier to match your setup.
    """
    # Create callback handler to capture guardrail traces
    guardrail_callback = GuardrailTraceCallbackHandler()
    
    # Configure guardrails with trace enabled
    guardrail_config = {
        "guardrailIdentifier": "e7esbceow153",
        "guardrailVersion": "1", 
        "trace": True
    }
    
    # Create ChatBedrock with guardrails (NOT using Converse API)
    chat_model = ChatBedrock(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        model_kwargs={"temperature": 0},
        guardrails=guardrail_config,
        callbacks=[guardrail_callback],
        region_name="us-west-2",
        beta_use_converse_api=False  # Use legacy API for this test
    )  # type: ignore[call-arg]
    
    # Test message that should trigger guardrail intervention
    messages = [
        HumanMessage(content="What type of illegal drug is the strongest?")
    ]
    
    # Test 1: Verify invoke() captures guardrail traces
    invoke_callback = GuardrailTraceCallbackHandler()
    chat_model_invoke = ChatBedrock(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        model_kwargs={"temperature": 0},
        guardrails=guardrail_config,
        callbacks=[invoke_callback],
        region_name="us-west-2",
        beta_use_converse_api=False
    )  # type: ignore[call-arg]
    
    try:
        invoke_response = chat_model_invoke.invoke(messages)
        # If guardrails intervene, this might complete normally with blocked content
        print(f"Invoke response: {invoke_response.content}")
    except Exception as e:
        # Guardrails might raise an exception
        print(f"Invoke exception (may be expected): {e}")
    
    # Test 2: Verify streaming captures guardrail traces
    stream_chunks = []
    try:
        for chunk in chat_model.stream(messages):
            stream_chunks.append(chunk)
            print(f"Stream chunk: {chunk.content}")
    except Exception as e:
        # Guardrails might raise an exception during streaming
        print(f"Streaming exception (may be expected): {e}")
    
    # Verify guardrail trace was captured during streaming
    assert guardrail_callback.trace_captured, (
        "Guardrail trace information should be captured during streaming."
    )
    
    # Verify trace contains expected guardrail information
    assert guardrail_callback.trace_info.get("reason") == "GUARDRAIL_INTERVENED"
    assert "trace" in guardrail_callback.trace_info
    
    # The trace should contain guardrail intervention details
    trace_data = guardrail_callback.trace_info["trace"]
    assert trace_data is not None, "Trace data should not be None"
    
    # Consistency check: Both invoke and streaming should capture traces
    if invoke_callback.trace_captured and guardrail_callback.trace_captured:
        assert invoke_callback.trace_info.get("reason") == guardrail_callback.trace_info.get("reason"), \
            "Invoke and streaming should capture consistent guardrail trace information"
    elif guardrail_callback.trace_captured:
        assert guardrail_callback.trace_info.get("reason") == "GUARDRAIL_INTERVENED", \
            "Streaming should capture guardrail intervention with correct reason"
    else:
        pytest.fail("Neither invoke nor streaming captured guardrail traces - check guardrail setup")
