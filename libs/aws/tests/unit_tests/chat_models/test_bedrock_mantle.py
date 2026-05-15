"""Unit tests for ChatBedrockMantle."""

from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from pydantic import SecretStr

# We need to mock openai and aws_bedrock_token_generator before importing
# ChatBedrockMantle since they are optional deps.


@pytest.fixture(autouse=True)
def _mock_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the optional dependencies so tests run without installing them."""
    mock_openai_mod = MagicMock()
    mock_token_mod = MagicMock()
    mock_token_mod.provide_token.return_value = "bedrock-api-key-dGVzdA==&Version=1"
    monkeypatch.setitem(__import__("sys").modules, "openai", mock_openai_mod)
    monkeypatch.setitem(
        __import__("sys").modules,
        "aws_bedrock_token_generator",
        mock_token_mod,
    )


from langchain_aws.chat_models.bedrock_mantle import (  # noqa: E402
    ChatBedrockMantle,
    _handle_openai_error,
    _openai_response_to_ai_message,
    _parse_responses_tool_args,
)
from langchain_aws.utils import _BedrockApiKeyProvider  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────


def _make_llm(**overrides: object) -> ChatBedrockMantle:
    defaults: dict = {
        "model": "test",
        "region_name": "us-east-1",
        "api_key": SecretStr("k"),
    }
    defaults.update(overrides)
    return ChatBedrockMantle(**defaults)


def _mock_responses_response(
    text: str = "hello",
    resp_id: str = "resp_123",
    model: str = "test",
) -> MagicMock:
    """Build a mock matching the Responses API shape."""
    mock_text_block = MagicMock()
    mock_text_block.text = text

    mock_output_item = MagicMock()
    mock_output_item.content = [mock_text_block]
    del mock_output_item.name  # so hasattr(item, "name") is False

    mock_response = MagicMock()
    mock_response.id = resp_id
    mock_response.model = model
    mock_response.status = "completed"
    mock_response.output = [mock_output_item]
    mock_response.usage = MagicMock()
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 5
    mock_response.usage.total_tokens = 15
    return mock_response


# ── OpenAI response conversion ──────────────────────────────────────


class TestOpenAIResponseConversion:
    """Tests for _openai_response_to_ai_message."""

    def test_plain_text_response(self) -> None:
        choice = MagicMock()
        choice.message.content = "Hello there"
        choice.message.tool_calls = None
        choice.finish_reason = "stop"

        msg = _openai_response_to_ai_message(choice)
        assert isinstance(msg, AIMessage)
        assert msg.content == "Hello there"
        assert msg.tool_calls == []
        assert msg.response_metadata["finish_reason"] == "stop"

    def test_empty_content_response(self) -> None:
        choice = MagicMock()
        choice.message.content = None
        choice.message.tool_calls = None
        choice.finish_reason = "stop"

        msg = _openai_response_to_ai_message(choice)
        assert msg.content == ""

    def test_response_with_tool_calls(self) -> None:
        tc = MagicMock()
        tc.function.name = "get_weather"
        tc.function.arguments = '{"location": "Seattle"}'
        tc.id = "call_abc"

        choice = MagicMock()
        choice.message.content = ""
        choice.message.tool_calls = [tc]
        choice.finish_reason = "tool_calls"

        msg = _openai_response_to_ai_message(choice)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["args"] == {"location": "Seattle"}
        assert msg.tool_calls[0]["id"] == "call_abc"

    def test_response_with_malformed_tool_args(self) -> None:
        tc = MagicMock()
        tc.function.name = "my_tool"
        tc.function.arguments = "not valid json"
        tc.id = "call_xyz"

        choice = MagicMock()
        choice.message.content = ""
        choice.message.tool_calls = [tc]
        choice.finish_reason = "tool_calls"

        msg = _openai_response_to_ai_message(choice)
        assert msg.tool_calls[0]["args"] == {"raw": "not valid json"}


# ── Responses API tool-arg parsing ──────────────────────────────────


class TestParseResponsesToolArgs:
    def test_valid_json_dict(self) -> None:
        assert _parse_responses_tool_args('{"a": 1}') == {"a": 1}

    def test_json_list_wrapped(self) -> None:
        assert _parse_responses_tool_args("[1, 2]") == {"raw": [1, 2]}

    def test_invalid_json_wrapped(self) -> None:
        assert _parse_responses_tool_args("not json") == {"raw": "not json"}

    def test_dict_passthrough(self) -> None:
        assert _parse_responses_tool_args({"x": 1}) == {"x": 1}

    def test_none_wrapped(self) -> None:
        assert _parse_responses_tool_args(None) == {"raw": None}


# ── Message conversion ──────────────────────────────────────────────


class TestMessageConversion:
    def test_human_message(self) -> None:
        msgs = convert_to_openai_messages([HumanMessage(content="hello")])
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_system_message(self) -> None:
        msgs = convert_to_openai_messages([SystemMessage(content="you are helpful")])
        assert msgs == [{"role": "system", "content": "you are helpful"}]

    def test_ai_message_plain(self) -> None:
        msgs = convert_to_openai_messages([AIMessage(content="hi there")])
        assert msgs == [{"role": "assistant", "content": "hi there"}]

    def test_tool_message(self) -> None:
        msgs = convert_to_openai_messages(
            [ToolMessage(content="72F", tool_call_id="call_123")]
        )
        assert msgs == [{"role": "tool", "tool_call_id": "call_123", "content": "72F"}]

    def test_multi_turn(self) -> None:
        msgs = convert_to_openai_messages(
            [
                SystemMessage(content="be helpful"),
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
                HumanMessage(content="bye"),
            ]
        )
        assert len(msgs) == 4
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]


# ── Token provider ──────────────────────────────────────────────────


class TestBedrockApiKeyProvider:
    def test_caches_token(self) -> None:
        provider = _BedrockApiKeyProvider("us-east-1")
        with patch(
            "langchain_aws.utils._BedrockApiKeyProvider._do_refresh"
        ) as mock_refresh:

            def _set_token() -> None:
                provider._token = "token-1"
                provider._expires_at = __import__("time").monotonic() + 50000

            mock_refresh.side_effect = _set_token
            provider()
            assert mock_refresh.call_count == 1

            t2 = provider()
            assert mock_refresh.call_count == 1
            assert t2 == "token-1"

    def test_advisory_refresh_succeeds(self) -> None:
        """Between advisory (15 min) and mandatory (10 min) thresholds,
        refresh is attempted and succeeds."""
        import time

        provider = _BedrockApiKeyProvider("us-east-1")
        provider._token = "old-token"
        # Set expiry so we're in the advisory window (12 min remaining)
        provider._expires_at = time.monotonic() + 720  # 12 min

        with patch(
            "langchain_aws.utils._BedrockApiKeyProvider._do_refresh"
        ) as mock_refresh:

            def _set_new_token() -> None:
                provider._token = "new-token"
                provider._expires_at = time.monotonic() + 43200

            mock_refresh.side_effect = _set_new_token
            result = provider()
            assert result == "new-token"
            mock_refresh.assert_called_once()

    def test_advisory_refresh_fails_returns_existing_token(self) -> None:
        """If advisory refresh fails but token is still valid (>10 min),
        return the existing token instead of raising."""
        import time

        provider = _BedrockApiKeyProvider("us-east-1")
        provider._token = "still-valid-token"
        # Set expiry so we're in advisory window but not mandatory (12 min)
        provider._expires_at = time.monotonic() + 720

        with patch(
            "langchain_aws.utils._BedrockApiKeyProvider._do_refresh"
        ) as mock_refresh:
            mock_refresh.side_effect = RuntimeError("network error")
            result = provider()
            assert result == "still-valid-token"

    def test_mandatory_refresh_fails_raises(self) -> None:
        """If mandatory refresh fails (<10 min remaining), must raise."""
        import time

        provider = _BedrockApiKeyProvider("us-east-1")
        provider._token = "expiring-token"
        # Set expiry so we're in the mandatory window (5 min remaining)
        provider._expires_at = time.monotonic() + 300

        with patch(
            "langchain_aws.utils._BedrockApiKeyProvider._do_refresh"
        ) as mock_refresh:
            mock_refresh.side_effect = RuntimeError("network error")
            with pytest.raises(RuntimeError, match="network error"):
                provider()

    def test_no_token_refresh_fails_raises(self) -> None:
        """First call with no cached token — refresh failure must raise."""
        provider = _BedrockApiKeyProvider("us-east-1")
        assert provider._token is None

        with patch(
            "langchain_aws.utils._BedrockApiKeyProvider._do_refresh"
        ) as mock_refresh:
            mock_refresh.side_effect = RuntimeError("no creds")
            with pytest.raises(RuntimeError, match="no creds"):
                provider()


# ── Init ────────────────────────────────────────────────────────────


class TestChatBedrockMantleInit:
    def test_missing_region_raises(self) -> None:
        with pytest.raises(ValueError, match="region_name"):
            with mock.patch.dict("os.environ", {}, clear=True):
                ChatBedrockMantle(model="test-model")

    def test_with_api_key(self) -> None:
        llm = _make_llm(model="deepseek.v3.2", api_key=SecretStr("my-static-key"))
        assert llm.model == "deepseek.v3.2"
        assert llm._resolved_region == "us-east-1"

    def test_with_region_constructs_url(self) -> None:
        llm = _make_llm(region_name="us-west-2")
        assert llm._resolved_region == "us-west-2"

    def test_with_base_url_override(self) -> None:
        llm = _make_llm(base_url="https://custom-endpoint.example.com/v1")
        assert llm.base_url == "https://custom-endpoint.example.com/v1"

    def test_llm_type(self) -> None:
        assert _make_llm()._llm_type == "bedrock-mantle"

    def test_max_retries_default(self) -> None:
        assert _make_llm().max_retries == 2

    def test_max_retries_custom(self) -> None:
        assert _make_llm(max_retries=5).max_retries == 5

    def test_timeout_custom(self) -> None:
        assert _make_llm(timeout=30.0).timeout == 30.0


# ── _build_params ───────────────────────────────────────────────────


class TestBuildParams:
    def test_basic_params(self) -> None:
        llm = _make_llm(model="deepseek.v3.2", temperature=0.5, max_tokens=100)
        params = llm._build_params(stop=["END"], stream=False)
        assert params["model"] == "deepseek.v3.2"
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 100
        assert params["stop"] == ["END"]
        assert params["stream"] is False

    def test_stream_includes_usage(self) -> None:
        params = _make_llm()._build_params(stream=True)
        assert params["stream_options"] == {"include_usage": True}

    def test_previous_response_id_passthrough(self) -> None:
        params = _make_llm()._build_params(
            stream=False, previous_response_id="resp_abc123"
        )
        assert params["previous_response_id"] == "resp_abc123"


# ── _use_responses_api routing ──────────────────────────────────────


class TestUseResponsesApiRouting:
    """Responses API is the default; Chat Completions is the fallback."""

    def test_default_is_responses_api(self) -> None:
        llm = _make_llm()
        params = {"model": "test", "stream": False}
        assert llm._use_responses_api(params) is True

    def test_response_format_triggers_chat_completions(self) -> None:
        llm = _make_llm()
        params = {"model": "test", "response_format": {"type": "json_object"}}
        assert llm._use_responses_api(params) is False

    def test_previous_response_id_stays_on_responses(self) -> None:
        llm = _make_llm()
        params = {"model": "test", "previous_response_id": "resp_abc"}
        assert llm._use_responses_api(params) is True

    def test_explicit_true_always_responses(self) -> None:
        llm = _make_llm(use_responses_api=True)
        # Even with response_format, explicit True wins
        params = {"model": "test", "response_format": {"type": "json_object"}}
        assert llm._use_responses_api(params) is True

    def test_explicit_false_always_chat_completions(self) -> None:
        llm = _make_llm(use_responses_api=False)
        params = {"model": "test", "stream": False}
        assert llm._use_responses_api(params) is False

    def test_generate_routes_to_responses_by_default(self) -> None:
        llm = _make_llm()
        mock_resp = _mock_responses_response(text="from responses api")
        llm._sync_client.responses.create = MagicMock(return_value=mock_resp)
        llm._sync_client.chat.completions.create = MagicMock(
            side_effect=AssertionError("should not call chat completions")
        )

        result = llm._generate([HumanMessage(content="hi")])
        assert result.generations[0].message.content == "from responses api"
        llm._sync_client.responses.create.assert_called_once()

    def test_generate_falls_back_to_chat_completions(self) -> None:
        llm = _make_llm()

        mock_choice = MagicMock()
        mock_choice.message.content = "from chat completions"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = None
        mock_resp.model = "test"
        mock_resp.id = "chatcmpl_123"

        llm._sync_client.chat.completions.create = MagicMock(return_value=mock_resp)
        llm._sync_client.responses.create = MagicMock(
            side_effect=AssertionError("should not call responses")
        )

        result = llm._generate(
            [HumanMessage(content="hi")],
            response_format={"type": "json_object"},
        )
        assert result.generations[0].message.content == "from chat completions"
        llm._sync_client.chat.completions.create.assert_called_once()


# ── Error handling ──────────────────────────────────────────────────


class TestErrorHandling:
    def _make_mock_openai(self) -> MagicMock:
        mock_mod = MagicMock()
        for name in (
            "AuthenticationError",
            "RateLimitError",
            "NotFoundError",
            "BadRequestError",
        ):
            cls = type(name, (Exception,), {"__module__": "openai"})
            setattr(mock_mod, name, cls)
        return mock_mod

    def test_authentication_error(self) -> None:
        mock_mod = self._make_mock_openai()
        error = mock_mod.AuthenticationError("invalid key")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(ValueError, match="authentication failed"):
                _handle_openai_error(error)

    def test_rate_limit_error(self) -> None:
        mock_mod = self._make_mock_openai()
        error = mock_mod.RateLimitError("rate limited")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(ValueError, match="rate limit"):
                _handle_openai_error(error)

    def test_not_found_error(self) -> None:
        mock_mod = self._make_mock_openai()
        error = mock_mod.NotFoundError("model not found")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(ValueError, match="model not found"):
                _handle_openai_error(error)

    def test_bad_request_tool_choice(self) -> None:
        mock_mod = self._make_mock_openai()
        error = mock_mod.BadRequestError("Invalid 'tool_choice': value did not match")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(ValueError, match="rejected the tool_choice"):
                _handle_openai_error(error)

    def test_bad_request_not_supported(self) -> None:
        mock_mod = self._make_mock_openai()
        error = mock_mod.BadRequestError("model does not support /v1/responses")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(ValueError, match="not supported"):
                _handle_openai_error(error)

    def test_unknown_error_reraised(self) -> None:
        mock_mod = self._make_mock_openai()
        error = RuntimeError("something else")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(RuntimeError, match="something else"):
                _handle_openai_error(error)

    def test_bad_request_generic(self) -> None:
        mock_mod = self._make_mock_openai()
        error = mock_mod.BadRequestError("some other bad request")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(ValueError, match="bad request"):
                _handle_openai_error(error)


# ── Refusal propagation ────────────────────────────────────────────


class TestRefusalPropagation:
    def test_refusal_in_additional_kwargs(self) -> None:
        choice = MagicMock()
        choice.message.content = ""
        choice.message.tool_calls = None
        choice.message.refusal = "I cannot help with that."
        choice.finish_reason = "stop"

        msg = _openai_response_to_ai_message(choice)
        assert msg.additional_kwargs["refusal"] == "I cannot help with that."

    def test_no_refusal_omitted(self) -> None:
        choice = MagicMock()
        choice.message.content = "Hello"
        choice.message.tool_calls = None
        choice.message.refusal = None
        choice.finish_reason = "stop"

        msg = _openai_response_to_ai_message(choice)
        assert "refusal" not in msg.additional_kwargs


# ── Async: Responses API uses async client ──────────────────────────


class TestAGenerateResponsesApi:
    def test_agenerate_uses_async_client(self) -> None:
        """Regression: _agenerate must use _async_client, not _sync_client."""
        import asyncio

        llm = _make_llm()
        mock_resp = _mock_responses_response(text="async response", resp_id="resp_a1")

        async def _mock_create(**kwargs):  # type: ignore[no-untyped-def]
            return mock_resp

        llm._async_client.responses.create = _mock_create
        llm._sync_client.responses.create = MagicMock(
            side_effect=AssertionError("sync client should not be called")
        )

        result = asyncio.run(llm._agenerate([HumanMessage(content="hi")]))
        assert result.generations[0].message.content == "async response"
        assert result.generations[0].message.response_metadata["id"] == "resp_a1"
        llm._sync_client.responses.create.assert_not_called()


# ── Responses API streaming ─────────────────────────────────────────


class TestResponsesApiStreaming:
    """Verify streaming works for the Responses API (default path)."""

    def _make_text_delta_event(self, text: str) -> MagicMock:
        event = MagicMock()
        event.type = "response.output_text.delta"
        event.delta = text
        return event

    def _make_completed_event(
        self,
        resp_id: str = "resp_s1",
        model: str = "test",
    ) -> MagicMock:
        event = MagicMock()
        event.type = "response.completed"
        event.response.id = resp_id
        event.response.model = model
        event.response.usage.input_tokens = 5
        event.response.usage.output_tokens = 3
        event.response.usage.total_tokens = 8
        return event

    def test_stream_yields_text_chunks(self) -> None:
        llm = _make_llm()
        events = [
            self._make_text_delta_event("Hel"),
            self._make_text_delta_event("lo"),
            self._make_completed_event(),
        ]
        llm._sync_client.responses.create = MagicMock(return_value=iter(events))

        chunks = list(llm._stream([HumanMessage(content="hi")]))
        text = "".join(
            c.message.content
            for c in chunks
            if isinstance(c.message.content, str) and c.message.content
        )
        assert text == "Hello"

    def test_stream_final_chunk_has_usage(self) -> None:
        llm = _make_llm()
        events = [
            self._make_text_delta_event("ok"),
            self._make_completed_event(),
        ]
        llm._sync_client.responses.create = MagicMock(return_value=iter(events))

        chunks = list(llm._stream([HumanMessage(content="hi")]))
        last = chunks[-1]
        assert isinstance(last.message, AIMessageChunk)
        assert last.message.usage_metadata is not None
        assert last.message.usage_metadata["input_tokens"] == 5
        assert last.message.usage_metadata["output_tokens"] == 3

    def test_stream_skips_unknown_events(self) -> None:
        llm = _make_llm()
        unknown = MagicMock()
        unknown.type = "response.reasoning_text.delta"
        events = [
            unknown,
            self._make_text_delta_event("hi"),
            self._make_completed_event(),
        ]
        llm._sync_client.responses.create = MagicMock(return_value=iter(events))

        chunks = list(llm._stream([HumanMessage(content="hi")]))
        text = "".join(
            c.message.content
            for c in chunks
            if isinstance(c.message.content, str) and c.message.content
        )
        assert text == "hi"

    def test_astream_yields_text_chunks(self) -> None:
        import asyncio

        llm = _make_llm()
        events = [
            self._make_text_delta_event("async "),
            self._make_text_delta_event("stream"),
            self._make_completed_event(),
        ]

        async def _async_iter():  # type: ignore[no-untyped-def]
            for e in events:
                yield e

        async def _mock_create(**kwargs):  # type: ignore[no-untyped-def]
            return _async_iter()

        llm._async_client.responses.create = _mock_create

        async def _collect() -> list:
            return [chunk async for chunk in llm._astream([HumanMessage(content="hi")])]

        chunks = asyncio.run(_collect())
        text = "".join(
            c.message.content
            for c in chunks
            if isinstance(c.message.content, str) and c.message.content
        )
        assert text == "async stream"


# ── Tool format conversion ──────────────────────────────────────────


class TestToolFormatConversion:
    """_prepare_responses_params must convert Chat Completions tool format
    to Responses API format."""

    def test_converts_chat_completions_tool_format(self) -> None:
        llm = _make_llm()
        params = llm._build_params(
            stream=False,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
        )
        resp_params = llm._prepare_responses_params(params)
        tool = resp_params["tools"][0]
        # Responses API format: flat, no nested "function" key
        assert tool["type"] == "function"
        assert tool["name"] == "get_weather"
        assert "function" not in tool
        assert "parameters" in tool

    def test_passes_through_responses_format_tools(self) -> None:
        llm = _make_llm()
        params = llm._build_params(
            stream=False,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
        )
        resp_params = llm._prepare_responses_params(params)
        tool = resp_params["tools"][0]
        assert tool["name"] == "get_weather"
        assert "function" not in tool


# ── AIMessage.id set from response ID ───────────────────────────────


class TestResponseIdOnMessage:
    """response.id must be set on AIMessage.id for LangGraph compat."""

    def test_invoke_sets_message_id(self) -> None:
        llm = _make_llm()
        mock_resp = _mock_responses_response(resp_id="resp_abc123")
        llm._sync_client.responses.create = MagicMock(return_value=mock_resp)

        result = llm._generate([HumanMessage(content="hi")])
        msg = result.generations[0].message
        assert msg.id == "resp_abc123"
        assert msg.response_metadata["id"] == "resp_abc123"

    def test_stream_completed_sets_message_id(self) -> None:
        llm = _make_llm()
        completed = MagicMock()
        completed.type = "response.completed"
        completed.response.id = "resp_stream_456"
        completed.response.model = "test"
        completed.response.usage.input_tokens = 5
        completed.response.usage.output_tokens = 3
        completed.response.usage.total_tokens = 8

        llm._sync_client.responses.create = MagicMock(return_value=iter([completed]))

        chunks = list(llm._stream([HumanMessage(content="hi")]))
        assert chunks[-1].message.id == "resp_stream_456"


# ── Streaming tool-call events ──────────────────────────────────────


class TestStreamingToolCalls:
    """Responses API streaming must emit tool-call chunks."""

    def test_function_call_added_emits_tool_chunk(self) -> None:
        llm = _make_llm()

        added = MagicMock()
        added.type = "response.output_item.added"
        added.item.type = "function_call"
        added.item.name = "get_weather"
        added.item.call_id = "call_abc"
        added.output_index = 0

        delta = MagicMock()
        delta.type = "response.function_call_arguments.delta"
        delta.delta = '{"location":'
        delta.output_index = 0

        delta2 = MagicMock()
        delta2.type = "response.function_call_arguments.delta"
        delta2.delta = '"Seattle"}'
        delta2.output_index = 0

        completed = MagicMock()
        completed.type = "response.completed"
        completed.response.id = "resp_tc"
        completed.response.model = "test"
        completed.response.usage.input_tokens = 10
        completed.response.usage.output_tokens = 5
        completed.response.usage.total_tokens = 15

        llm._sync_client.responses.create = MagicMock(
            return_value=iter([added, delta, delta2, completed])
        )

        chunks = list(llm._stream([HumanMessage(content="weather?")]))

        # First chunk should have tool name and id
        assert isinstance(chunks[0].message, AIMessageChunk)
        tc_chunks = chunks[0].message.tool_call_chunks
        assert len(tc_chunks) == 1
        assert tc_chunks[0]["name"] == "get_weather"
        assert tc_chunks[0]["id"] == "call_abc"

        # Subsequent chunks carry argument fragments
        all_args = ""
        for c in chunks[1:-1]:  # skip first (name) and last (completed)
            assert isinstance(c.message, AIMessageChunk)
            if c.message.tool_call_chunks:
                args = c.message.tool_call_chunks[0]["args"]
                if args:
                    all_args += args
        assert all_args == '{"location":"Seattle"}'


# ── stream_usage toggle ─────────────────────────────────────────────


class TestStreamUsage:
    def test_default_includes_usage(self) -> None:
        params = _make_llm()._build_params(stream=True)
        assert params["stream_options"] == {"include_usage": True}

    def test_explicit_true(self) -> None:
        params = _make_llm(stream_usage=True)._build_params(stream=True)
        assert params["stream_options"] == {"include_usage": True}

    def test_explicit_false_omits_stream_options(self) -> None:
        params = _make_llm(stream_usage=False)._build_params(stream=True)
        assert "stream_options" not in params


# ── use_previous_response_id ────────────────────────────────────────


class TestUsePreviousResponseId:
    def test_extracts_id_from_message_history(self) -> None:
        llm = _make_llm(use_previous_response_id=True)
        mock_resp = _mock_responses_response(text="followup", resp_id="resp_f1")
        llm._sync_client.responses.create = MagicMock(return_value=mock_resp)

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!", id="resp_first_123"),
            HumanMessage(content="How are you?"),
        ]
        result = llm._generate(messages)
        assert result.generations[0].message.content == "followup"

        # Verify previous_response_id was passed to the API
        call_kwargs = llm._sync_client.responses.create.call_args[1]
        assert call_kwargs["previous_response_id"] == "resp_first_123"

    def test_no_response_id_sends_all_messages(self) -> None:
        llm = _make_llm(use_previous_response_id=True)
        mock_resp = _mock_responses_response()
        llm._sync_client.responses.create = MagicMock(return_value=mock_resp)

        messages: list[BaseMessage] = [
            HumanMessage(content="Hello"),
            HumanMessage(content="How are you?"),
        ]
        llm._generate(messages)

        call_kwargs = llm._sync_client.responses.create.call_args[1]
        assert call_kwargs.get("previous_response_id") is None

    def test_disabled_by_default(self) -> None:
        llm = _make_llm()
        assert llm.use_previous_response_id is False


# ── tool_choice validation ──────────────────────────────────────────


class TestToolChoiceValidation:
    def test_auto_passes_without_warning(self) -> None:
        llm = _make_llm()
        params = llm._build_params(stream=False, tool_choice="auto")
        with patch("langchain_aws.chat_models.bedrock_mantle.logger") as mock_log:
            llm._prepare_responses_params(params)
            mock_log.warning.assert_not_called()

    def test_none_passes_without_warning(self) -> None:
        llm = _make_llm()
        params = llm._build_params(stream=False, tool_choice="none")
        with patch("langchain_aws.chat_models.bedrock_mantle.logger") as mock_log:
            llm._prepare_responses_params(params)
            mock_log.warning.assert_not_called()

    def test_required_logs_warning(self) -> None:
        llm = _make_llm()
        params = llm._build_params(stream=False, tool_choice="required")
        with patch("langchain_aws.chat_models.bedrock_mantle.logger") as mock_log:
            llm._prepare_responses_params(params)
            mock_log.warning.assert_called_once()
            assert "tool_choice" in mock_log.warning.call_args[0][0]


# ── _use_responses_api routing with text param ──────────────────────


class TestRoutingWithTextParam:
    def test_text_param_forces_responses_api(self) -> None:
        llm = _make_llm()
        params = {"model": "test", "text": {"format": {"type": "json_object"}}}
        assert llm._use_responses_api(params) is True

    def test_text_overrides_response_format(self) -> None:
        """If both text and response_format are present, text wins."""
        llm = _make_llm()
        params = {
            "model": "test",
            "text": {"format": {"type": "json_object"}},
            "response_format": {"type": "json_object"},
        }
        assert llm._use_responses_api(params) is True


# ── with_structured_output json_schema ──────────────────────────────


class TestWithStructuredOutputJsonSchema:
    """with_structured_output(method='json_schema') must bind the text param
    so the Responses API receives the json_schema format."""

    def test_json_schema_binds_text_param(self) -> None:
        llm = _make_llm()
        mock_resp = _mock_responses_response(
            text='{"name": "Alice", "age": 30}', resp_id="resp_js1"
        )
        llm._sync_client.responses.create = MagicMock(return_value=mock_resp)

        schema = {
            "type": "object",
            "title": "Person",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        structured = llm.with_structured_output(schema, method="json_schema")

        # Invoke to trigger the bound params
        structured.invoke("Give me a person")

        # Verify the text param was passed to the Responses API
        call_kwargs = llm._sync_client.responses.create.call_args[1]
        assert "text" in call_kwargs
        text_format = call_kwargs["text"]["format"]
        assert text_format["type"] == "json_schema"
        assert text_format["name"] == "Person"
        assert text_format["schema"] == schema
        assert text_format["strict"] is True
