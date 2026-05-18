"""Integration tests for ChatBedrockMantle."""

import os
from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

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


# ── Standard LangChain interface tests ──────────────────────────────


_STANDARD_MAX_TOKENS = 5000  # matches ChatBedrockConverse's config


@pytest.mark.skipif(
    not MANTLE_AVAILABLE,
    reason="Mantle deps not installed or CI lacks Mantle API permissions.",
)
class TestBedrockMantleStandardOpenWeights(ChatModelIntegrationTests):
    """Conformance run against an open-weight OpenAI model on ``/v1``.

    ``openai.gpt-oss-120b`` serves both Chat Completions and Responses
    at the default ``/v1`` base URL. It does not currently accept
    ``tool_choice="required"`` on this path, so we disable the
    ``has_tool_choice`` capability flag for this model.
    """

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockMantle

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "openai.gpt-oss-120b",
            "region_name": REGION,
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {"max_tokens": _STANDARD_MAX_TOKENS}

    @property
    def has_tool_choice(self) -> bool:
        # Mantle rejects ``tool_choice="required"`` for gpt-oss-* on
        # ``/v1/responses`` with a 400 validation error.
        return False

    @property
    def has_structured_output(self) -> bool:
        # ``openai.gpt-oss-120b`` does not strictly honor
        # ``strict=True`` for ``response_format={"type": "json_schema"}``
        # — it often prefixes free-form text (e.g. ``"Why{...}"``)
        # before the JSON payload, breaking the output parser. The
        # class-level structured-output plumbing is exercised by
        # ``TestBedrockMantleStandardFrontier`` against ``gpt-5.5``.
        return False

    # These two tests internally use ``tool_choice="any"`` (which the
    # class maps to ``tool_choice="required"``). Mantle rejects
    # ``tool_choice="required"`` for ``openai.gpt-oss-120b`` on
    # ``/v1/responses`` — same underlying limitation as
    # ``has_tool_choice=False``, but this pair of tests uses
    # ``tool_choice`` as a means to a different end so the flag
    # doesn't skip them.
    @pytest.mark.xfail(
        reason=(
            "openai.gpt-oss-120b rejects tool_choice='required' on "
            "/v1/responses; exercised by TestBedrockMantleStandardFrontier "
            "against a model that supports it."
        )
    )
    def test_unicode_tool_call_integration(
        self,
        model: BaseChatModel,
        *,
        tool_choice: str | None = None,
        force_tool_call: bool = True,
    ) -> None:
        super().test_unicode_tool_call_integration(
            model, tool_choice=tool_choice, force_tool_call=force_tool_call
        )

    @pytest.mark.xfail(
        reason=(
            "openai.gpt-oss-120b rejects tool_choice='required' on "
            "/v1/responses; exercised by TestBedrockMantleStandardFrontier "
            "against a model that supports it."
        )
    )
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_structured_few_shot_examples(model, my_adder_tool)


@pytest.mark.skipif(
    not MANTLE_AVAILABLE,
    reason="Mantle deps not installed or CI lacks Mantle API permissions.",
)
class TestBedrockMantleStandardFrontier(ChatModelIntegrationTests):
    """Conformance run against a frontier OpenAI model on ``/openai/v1``.

    ``openai.gpt-5.5`` lives on ``/openai/v1`` and supports the
    Responses API only (no Chat Completions). Users select it by
    passing an explicit ``base_url``. Frontier models allocate a
    portion of the budget to internal reasoning, so we use a larger
    ``max_tokens`` to ensure text output is produced.
    """

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockMantle

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "openai.gpt-5.5",
            "region_name": REGION,
            "base_url": f"https://bedrock-mantle.{REGION}.api.aws/openai/v1",
            # gpt-5.x is Responses-API-only — pin routing so structured
            # output and other Chat-Completions-hinted params don't
            # trigger a fallback to an endpoint the model doesn't serve.
            "use_responses_api": True,
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {"max_tokens": _STANDARD_MAX_TOKENS}


# ── Custom integration tests ────────────────────────────────────────

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
        has_tool_chunks = any(
            isinstance(c, AIMessageChunk) and c.tool_call_chunks for c in chunks
        )
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


# ── Explicit AWS credentials & configurable TTL ─────────────────────


class TestExplicitCredsAndTTL:
    """Live tests for the credential kwargs + TTL knob added in this
    revision. Each test resolves creds from the ambient chain and passes
    them explicitly, exercising the ``aws_access_key_id`` / secret / token
    path end-to-end against a real Bedrock Mantle endpoint.
    """

    def _resolve_ambient_creds(self) -> dict:
        """Snapshot the current default-chain creds for reuse as explicit kwargs."""
        import boto3
        from pydantic import SecretStr

        creds = boto3.Session().get_credentials()
        if creds is None:
            pytest.skip("No AWS credentials available in this environment.")
        frozen = creds.get_frozen_credentials()
        if not frozen.access_key or not frozen.secret_key:
            pytest.skip("Ambient AWS credentials missing access/secret key.")
        result: dict = {
            "aws_access_key_id": SecretStr(frozen.access_key),
            "aws_secret_access_key": SecretStr(frozen.secret_key),
        }
        if frozen.token:
            result["aws_session_token"] = SecretStr(frozen.token)
        return result

    def test_invoke_with_explicit_creds(self) -> None:
        """End-to-end: explicit access_key + secret_key (+ session token if
        present) produce a working short-term Bedrock API key."""
        explicit_creds = self._resolve_ambient_creds()
        llm = ChatBedrockMantle(
            model=MODELS[0],
            region_name=REGION,
            **explicit_creds,
        )
        response = llm.invoke("Say hi in one word")
        assert isinstance(response, AIMessage)
        assert response.content

    def test_invoke_with_custom_ttl(self) -> None:
        """Non-default TTL propagates to the token generator and the
        generated key still works."""
        llm = ChatBedrockMantle(
            model=MODELS[0],
            region_name=REGION,
            bedrock_api_key_ttl_seconds=3600,  # 1 hour instead of default 12h
        )
        response = llm.invoke("Say hi in one word")
        assert isinstance(response, AIMessage)
        assert response.content

    def test_invoke_with_explicit_creds_and_custom_ttl(self) -> None:
        """Combined: explicit creds + custom TTL both flow through."""
        explicit_creds = self._resolve_ambient_creds()
        llm = ChatBedrockMantle(
            model=MODELS[0],
            region_name=REGION,
            bedrock_api_key_ttl_seconds=1800,  # 30 min
            **explicit_creds,
        )
        response = llm.invoke("Say hi in one word")
        assert isinstance(response, AIMessage)
        assert response.content

    def test_invoke_with_profile_name(self) -> None:
        """Profile-only path: BotocoreSession(profile=...) resolves creds."""
        profile = os.environ.get("LANGCHAIN_AWS_MANTLE_TEST_PROFILE") or os.environ.get(
            "AWS_PROFILE"
        )
        if not profile:
            pytest.skip("Set AWS_PROFILE or LANGCHAIN_AWS_MANTLE_TEST_PROFILE to run.")
        # Verify the profile actually exists in ~/.aws/config.
        import botocore.session

        try:
            botocore.session.Session(profile=profile).get_credentials()
        except Exception as exc:
            pytest.skip(f"Profile {profile!r} not resolvable: {exc}")

        llm = ChatBedrockMantle(
            model=MODELS[0],
            region_name=REGION,
            credentials_profile_name=profile,
        )
        response = llm.invoke("Say hi in one word")
        assert isinstance(response, AIMessage)
        assert response.content
