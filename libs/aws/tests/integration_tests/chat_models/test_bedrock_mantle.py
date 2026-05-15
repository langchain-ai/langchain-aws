"""Integration tests for ChatBedrockMantle.

These tests require:
  - pip install "langchain-aws[mantle]"
  - Valid AWS credentials with Bedrock access
  - Access to Mantle models in the configured region

Run with:
  AWS_REGION=us-east-1 \
    uv run --group test --group test_integration \
    pytest tests/integration_tests/chat_models/test_bedrock_mantle.py -v
"""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

try:
    from langchain_aws import ChatBedrockMantle

    MANTLE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    MANTLE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MANTLE_AVAILABLE,
    reason='Mantle deps not installed. Run: pip install "langchain-aws[mantle]"',
)

REGION = os.environ.get("AWS_REGION", "us-east-1")

MODELS = [
    "openai.gpt-oss-20b",
]

# Models that only support Chat Completions (not Responses API)
CHAT_COMPLETIONS_MODELS = [
    "qwen.qwen3-32b",
    "deepseek.v3.2",
]


@pytest.fixture(params=MODELS)
def llm(request: pytest.FixtureRequest) -> "ChatBedrockMantle":
    return ChatBedrockMantle(model=request.param, region_name=REGION)


@pytest.fixture
def llm_default() -> "ChatBedrockMantle":
    return ChatBedrockMantle(model=MODELS[0], region_name=REGION)


# ── Basic invoke ────────────────────────────────────────────────────


class TestBasicInvoke:
    def test_invoke(self, llm: "ChatBedrockMantle") -> None:
        response = llm.invoke("What is 2+2? Answer with just the number.")
        assert isinstance(response, AIMessage)
        assert response.content
        assert "4" in response.content

    def test_invoke_with_system(self, llm: "ChatBedrockMantle") -> None:
        messages = [
            SystemMessage(content="You are a helpful math tutor."),
            HumanMessage(content="What is 3*3? Answer with just the number."),
        ]
        response = llm.invoke(messages)
        assert isinstance(response, AIMessage)
        assert "9" in response.content

    def test_response_has_id(self, llm_default: "ChatBedrockMantle") -> None:
        """AIMessage.id should be set from the response ID."""
        response = llm_default.invoke("Say hi")
        assert isinstance(response, AIMessage)
        assert response.id is not None
        assert response.id.startswith("resp_")
        assert response.response_metadata["id"] == response.id


# ── Streaming ───────────────────────────────────────────────────────


class TestStreaming:
    def test_stream(self, llm: "ChatBedrockMantle") -> None:
        """Default streaming uses the Responses API."""
        chunks = list(llm.stream("Say hello in one word"))
        assert len(chunks) > 0
        full_content = "".join(
            c.content for c in chunks if isinstance(c.content, str) and c.content
        )
        assert full_content

    def test_stream_chat_completions(self) -> None:
        """Streaming falls back to Chat Completions with response_format."""
        llm = ChatBedrockMantle(
            model=MODELS[0],
            region_name=REGION,
            use_responses_api=False,
        )
        chunks = list(llm.stream("Say hello in one word"))
        assert len(chunks) > 0
        full_content = "".join(
            c.content for c in chunks if isinstance(c.content, str) and c.content
        )
        assert full_content


# ── Async ───────────────────────────────────────────────────────────


class TestAsync:
    @pytest.mark.asyncio
    async def test_ainvoke(self, llm: "ChatBedrockMantle") -> None:
        response = await llm.ainvoke("What is 2+2? Answer with just the number.")
        assert isinstance(response, AIMessage)
        assert response.content
        assert "4" in response.content

    @pytest.mark.asyncio
    async def test_astream(self, llm: "ChatBedrockMantle") -> None:
        chunks = []
        async for chunk in llm.astream("Say hello in one word"):
            chunks.append(chunk)
        assert len(chunks) > 0
        full_content = "".join(
            c.content for c in chunks if isinstance(c.content, str) and c.content
        )
        assert full_content


# ── Tool calling ────────────────────────────────────────────────────


class TestToolCalling:
    def test_bind_tools(self, llm: "ChatBedrockMantle") -> None:
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get the current weather in a given location."""

            location: str = Field(description="City and state")

        llm_with_tools = llm.bind_tools([GetWeather])
        try:
            response = llm_with_tools.invoke("What's the weather in Seattle?")
        except Exception as e:
            if "not supported" in str(e).lower() or "tool" in str(e).lower():
                pytest.skip(f"Model does not support tool calling: {e}")
            raise
        assert isinstance(response, AIMessage)
        assert response.content or response.tool_calls

    def test_streaming_tool_calls(self, llm_default: "ChatBedrockMantle") -> None:
        """Tool calls should stream via Responses API."""
        from pydantic import BaseModel, Field

        class GetWeather(BaseModel):
            """Get the current weather in a given location."""

            location: str = Field(description="City and state")

        llm_with_tools = llm_default.bind_tools([GetWeather])
        chunks = list(llm_with_tools.stream("What's the weather in Seattle?"))
        assert len(chunks) > 0
        # At least one chunk should have tool call info or text
        has_content = any(isinstance(c.content, str) and c.content for c in chunks)
        has_tool_chunks = any(c.tool_call_chunks for c in chunks)
        assert has_content or has_tool_chunks


# ── Model discovery ─────────────────────────────────────────────────


class TestModelDiscovery:
    def test_list_models(self, llm_default: "ChatBedrockMantle") -> None:
        models = llm_default.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "id" in models[0]


# ── Usage metadata ──────────────────────────────────────────────────


class TestUsageMetadata:
    def test_usage_in_invoke(self, llm_default: "ChatBedrockMantle") -> None:
        response = llm_default.invoke("Hi")
        assert isinstance(response, AIMessage)
        if response.usage_metadata:
            assert response.usage_metadata["input_tokens"] > 0
            assert response.usage_metadata["output_tokens"] > 0

    def test_usage_in_stream(self, llm_default: "ChatBedrockMantle") -> None:
        """Last streaming chunk should carry usage metadata."""
        chunks = list(llm_default.stream("Hi"))
        last = chunks[-1]
        if last.usage_metadata:
            assert last.usage_metadata["total_tokens"] > 0


# ── Conversation state (previous_response_id) ──────────────────────


class TestConversationState:
    def test_previous_response_id(self, llm_default: "ChatBedrockMantle") -> None:
        first = llm_default.invoke("My name is Alice.")
        assert isinstance(first, AIMessage)
        response_id = first.response_metadata.get("id")
        assert response_id

        followup = llm_default.invoke(
            "What is my name?", previous_response_id=response_id
        )
        assert isinstance(followup, AIMessage)
        assert followup.content

    def test_use_previous_response_id_auto(self) -> None:
        """use_previous_response_id extracts ID from message history."""
        llm = ChatBedrockMantle(
            model=MODELS[0],
            region_name=REGION,
            use_previous_response_id=True,
        )
        first = llm.invoke("My name is Bob.")
        assert isinstance(first, AIMessage)
        assert first.id and first.id.startswith("resp_")

        # Pass full history — llm should extract previous_response_id
        messages = [
            HumanMessage(content="My name is Bob."),
            first,
            HumanMessage(content="What is my name?"),
        ]
        followup = llm.invoke(messages)
        assert isinstance(followup, AIMessage)
        assert followup.content


# ── Chat Completions fallback ───────────────────────────────────────


class TestChatCompletionsFallback:
    def test_response_format_routes_to_chat_completions(self) -> None:
        llm = ChatBedrockMantle(model=MODELS[0], region_name=REGION)
        response = llm.invoke(
            "Return a JSON object with fields name and age for Alice who is 30. "
            "Output only valid JSON, no other text.",
            response_format={"type": "json_object"},
        )
        assert isinstance(response, AIMessage)
        assert response.content

    def test_explicit_chat_completions(self) -> None:
        llm = ChatBedrockMantle(
            model=MODELS[0],
            region_name=REGION,
            use_responses_api=False,
        )
        response = llm.invoke("Say hi")
        assert isinstance(response, AIMessage)
        assert response.content

    def test_chat_completions_only_model(self) -> None:
        """Models that don't support Responses API work via Chat Completions."""
        for model in CHAT_COMPLETIONS_MODELS:
            llm = ChatBedrockMantle(
                model=model,
                region_name=REGION,
                use_responses_api=False,
            )
            response = llm.invoke("Say hi in one word")
            assert isinstance(response, AIMessage)
            assert response.content

    def test_chat_completions_only_model_stream(self) -> None:
        """Streaming works for Chat Completions-only models."""
        llm = ChatBedrockMantle(
            model=CHAT_COMPLETIONS_MODELS[0],
            region_name=REGION,
            use_responses_api=False,
        )
        chunks = list(llm.stream("Say hi"))
        assert len(chunks) > 0
        full = "".join(
            c.content for c in chunks if isinstance(c.content, str) and c.content
        )
        assert full


# ── Structured output ────────────────────────────────────────────────


class TestStructuredOutput:
    def test_with_structured_output_json_mode(self) -> None:
        """json_mode routes through Chat Completions response_format
        and returns valid JSON."""
        llm = ChatBedrockMantle(model=MODELS[0], region_name=REGION)
        structured = llm.with_structured_output(
            {"type": "object", "properties": {"answer": {"type": "string"}}},
            method="json_mode",
        )
        result = structured.invoke(
            "Return a JSON object with a single key 'answer' "
            "whose value is 'hello'. Output only valid JSON."
        )
        assert isinstance(result, dict)


# ── Explicit use_responses_api ──────────────────────────────────────


class TestExplicitResponsesApi:
    def test_use_responses_api_true(self) -> None:
        llm = ChatBedrockMantle(
            model=MODELS[0],
            region_name=REGION,
            use_responses_api=True,
        )
        response = llm.invoke("Say hi")
        assert isinstance(response, AIMessage)
        assert response.content
        assert response.id and response.id.startswith("resp_")


# ── Retry and timeout ───────────────────────────────────────────────


class TestRetryAndTimeout:
    def test_custom_timeout(self) -> None:
        llm = ChatBedrockMantle(model=MODELS[0], region_name=REGION, timeout=60.0)
        response = llm.invoke("Say hi")
        assert isinstance(response, AIMessage)
        assert response.content

    def test_custom_max_retries(self) -> None:
        llm = ChatBedrockMantle(model=MODELS[0], region_name=REGION, max_retries=3)
        response = llm.invoke("Say hi")
        assert isinstance(response, AIMessage)
        assert response.content
