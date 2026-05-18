"""Unit tests for ChatBedrockMantle.

The standard LangChain interface tests (``ChatModelUnitTests``) for this
model live in ``tests/unit_tests/test_standard.py`` alongside the other
Bedrock chat model conformance suites.
"""

import sys
from typing import AsyncGenerator, Generator, Optional, cast
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
from langchain_core.outputs import ChatGenerationChunk
from pydantic import SecretStr

# We need to mock openai and aws_bedrock_token_generator before importing
# ChatBedrockMantle since they are optional deps.


def _make_mock_module(name: str) -> MagicMock:
    """Build a MagicMock that satisfies ``importlib.util.find_spec``.

    ``find_spec`` reads ``module.__spec__``; a bare ``MagicMock`` auto-
    creates one but it isn't a valid ``ModuleSpec``, which raises when
    the caller code uses ``find_spec`` to feature-detect the module.
    """
    from importlib.machinery import ModuleSpec

    mod = MagicMock()
    mod.__spec__ = ModuleSpec(name, loader=None)
    mod.__name__ = name
    return mod


_mock_openai_mod = _make_mock_module("openai")
_mock_token_gen_mod = _make_mock_module("aws_bedrock_token_generator")
_mock_token_gen_mod.provide_token.return_value = "bedrock-api-key-dGVzdA==&Version=1"
sys.modules.setdefault("openai", _mock_openai_mod)
sys.modules.setdefault("aws_bedrock_token_generator", _mock_token_gen_mod)


@pytest.fixture(autouse=True)
def _mock_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the optional dependencies so tests run without installing them."""
    mock_openai_mod = _make_mock_module("openai")
    mock_token_mod = _make_mock_module("aws_bedrock_token_generator")
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


class _MockSyncStream:
    """Mimic an ``openai.Stream``: both an iterator and a context manager."""

    def __init__(self, events: list) -> None:
        self._events = list(events)
        self._i = 0

    def __enter__(self) -> "_MockSyncStream":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def __iter__(self) -> "_MockSyncStream":
        return self

    def __next__(self) -> object:
        if self._i >= len(self._events):
            raise StopIteration
        event = self._events[self._i]
        self._i += 1
        return event


class _MockAsyncStream:
    """Mimic an ``openai.AsyncStream``: supports ``async with`` and ``async for``."""

    def __init__(self, events: list) -> None:
        self._events = list(events)
        self._i = 0

    async def __aenter__(self) -> "_MockAsyncStream":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def __aiter__(self) -> "_MockAsyncStream":
        return self

    async def __anext__(self) -> object:
        if self._i >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._i]
        self._i += 1
        return event


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


# ── Responses API output extraction ─────────────────────────────────


class TestExtractResponsesOutput:
    """Tests for ``_extract_responses_output``.

    Guards against known heterogeneous output items in the Responses API
    (text, tool call, reasoning). Reasoning items expose a ``content``
    attribute that is ``None`` on gpt-5.x; iterating it must not raise.
    """

    @staticmethod
    def _extract(response: object) -> tuple:
        from langchain_aws.chat_models.bedrock_mantle import (
            _extract_responses_output,
        )

        return _extract_responses_output(response)

    def _text_item(self, text: str) -> MagicMock:
        block = MagicMock()
        block.text = text
        item = MagicMock()
        item.content = [block]
        # Ensure hasattr(item, "name") is False so it's treated as a text item.
        del item.name
        return item

    def _tool_call_item(
        self, name: str, args: str, call_id: str = "call_1"
    ) -> MagicMock:
        item = MagicMock()
        item.content = None  # tool calls do not carry text content
        item.name = name
        item.arguments = args
        item.call_id = call_id
        return item

    def _reasoning_item(self) -> MagicMock:
        """Reasoning items on gpt-5.x expose ``content=None``."""
        item = MagicMock()
        item.content = None
        # No `name` attribute → not a tool call either.
        del item.name
        return item

    def _response(self, output_items: list) -> MagicMock:
        r = MagicMock()
        r.output = output_items
        return r

    def test_text_only(self) -> None:
        response = self._response([self._text_item("hello")])
        content, tool_calls = self._extract(response)
        assert content == "hello"
        assert tool_calls == []

    def test_reasoning_item_with_none_content_skipped(self) -> None:
        response = self._response([self._reasoning_item(), self._text_item("hi")])
        content, tool_calls = self._extract(response)
        assert content == "hi"
        assert tool_calls == []

    def test_reasoning_and_tool_call_no_text(self) -> None:
        response = self._response(
            [
                self._reasoning_item(),
                self._tool_call_item("get_weather", '{"city": "SEA"}'),
            ]
        )
        content, tool_calls = self._extract(response)
        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["args"] == {"city": "SEA"}
        assert tool_calls[0]["id"] == "call_1"

    def test_empty_output(self) -> None:
        response = self._response([])
        content, tool_calls = self._extract(response)
        assert content == ""
        assert tool_calls == []

    def test_none_output(self) -> None:
        r = MagicMock()
        r.output = None
        content, tool_calls = self._extract(r)
        assert content == ""
        assert tool_calls == []


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
        assert not provider._token

        with patch(
            "langchain_aws.utils._BedrockApiKeyProvider._do_refresh"
        ) as mock_refresh:
            mock_refresh.side_effect = RuntimeError("no creds")
            with pytest.raises(RuntimeError, match="no creds"):
                provider()

    def test_async_call_returns_same_token(self) -> None:
        """``async_call`` delegates to ``__call__`` via ``asyncio.to_thread``
        and returns the same cached token as the sync path.
        """
        import asyncio
        import time as _time

        provider = _BedrockApiKeyProvider("us-east-1")

        def _set_token() -> None:
            provider._token = "async-test-token"
            provider._expires_at = _time.monotonic() + 43200

        with patch.object(provider, "_do_refresh", side_effect=_set_token):
            sync_result = provider()
            async_result = asyncio.run(provider.async_call())

        assert sync_result == "async-test-token"
        assert async_result == "async-test-token"


# ── Token provider _do_refresh credential-expiry awareness ──────────


class TestBedrockApiKeyProviderDoRefresh:
    """Tests for _do_refresh respecting underlying credential expiry."""

    def test_static_credentials_use_full_ttl(self) -> None:
        """Static (non-refreshable) creds → use full 12h TTL."""
        import time

        provider = _BedrockApiKeyProvider("us-east-1")

        mock_static_creds = MagicMock()
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_static_creds

        with (
            patch("botocore.session.Session", return_value=mock_session),
            patch(
                "aws_bedrock_token_generator.provide_token",
                return_value="bedrock-api-key-abc123",
            ) as mock_provide,
        ):
            provider._do_refresh()

        assert provider._token == "bedrock-api-key-abc123"
        # Should use full 12h TTL since creds are not RefreshableCredentials
        expected_expiry = time.monotonic() + 43200
        assert abs(provider._expires_at - expected_expiry) < 2
        mock_provide.assert_called_once()
        call_kwargs = mock_provide.call_args
        assert call_kwargs[1]["expiry"].total_seconds() == 43200

    def test_refreshable_credentials_uses_remaining_time(self) -> None:
        """RefreshableCredentials with 2h remaining → TTL capped at 2h."""
        import time

        from botocore.credentials import RefreshableCredentials

        provider = _BedrockApiKeyProvider("us-east-1")

        mock_creds = MagicMock(spec=RefreshableCredentials)
        mock_creds._seconds_remaining.return_value = 7200.0  # 2 hours

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        with (
            patch("botocore.session.Session", return_value=mock_session),
            patch(
                "aws_bedrock_token_generator.provide_token",
                return_value="bedrock-api-key-short",
            ) as mock_provide,
        ):
            provider._do_refresh()

        assert provider._token == "bedrock-api-key-short"
        expected_expiry = time.monotonic() + 7200
        assert abs(provider._expires_at - expected_expiry) < 2
        call_kwargs = mock_provide.call_args
        assert call_kwargs[1]["expiry"].total_seconds() == 7200

    def test_refreshable_credentials_longer_than_12h_capped(self) -> None:
        """RefreshableCredentials with 24h remaining → still capped at 12h."""
        import time

        from botocore.credentials import RefreshableCredentials

        provider = _BedrockApiKeyProvider("us-east-1")

        mock_creds = MagicMock(spec=RefreshableCredentials)
        mock_creds._seconds_remaining.return_value = 86400.0  # 24 hours

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        with (
            patch("botocore.session.Session", return_value=mock_session),
            patch(
                "aws_bedrock_token_generator.provide_token",
                return_value="bedrock-api-key-long",
            ) as mock_provide,
        ):
            provider._do_refresh()

        assert provider._token == "bedrock-api-key-long"
        expected_expiry = time.monotonic() + 43200
        assert abs(provider._expires_at - expected_expiry) < 2
        call_kwargs = mock_provide.call_args
        assert call_kwargs[1]["expiry"].total_seconds() == 43200

    def test_session_error_falls_back_to_default_ttl(self) -> None:
        """If credential introspection fails, fall back to 12h."""
        import time

        provider = _BedrockApiKeyProvider("us-east-1")

        with (
            patch(
                "botocore.session.Session",
                side_effect=RuntimeError("session broken"),
            ),
            patch(
                "aws_bedrock_token_generator.provide_token",
                return_value="bedrock-api-key-fallback",
            ),
        ):
            provider._do_refresh()

        assert provider._token == "bedrock-api-key-fallback"
        expected_expiry = time.monotonic() + 43200
        assert abs(provider._expires_at - expected_expiry) < 2

    def test_refreshable_credentials_5min_remaining(self) -> None:
        """Creds with only 5 min left → token TTL is 5 min, refresh
        triggers quickly."""
        import time

        from botocore.credentials import RefreshableCredentials

        provider = _BedrockApiKeyProvider("us-east-1")

        mock_creds = MagicMock(spec=RefreshableCredentials)
        mock_creds._seconds_remaining.return_value = 300.0  # 5 minutes

        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_creds

        with (
            patch("botocore.session.Session", return_value=mock_session),
            patch(
                "aws_bedrock_token_generator.provide_token",
                return_value="bedrock-api-key-5min",
            ) as mock_provide,
        ):
            provider._do_refresh()

        assert provider._token == "bedrock-api-key-5min"
        expected_expiry = time.monotonic() + 300
        assert abs(provider._expires_at - expected_expiry) < 2
        call_kwargs = mock_provide.call_args
        assert call_kwargs[1]["expiry"].total_seconds() == 300


# ── Token provider thread safety ────────────────────────────────────


class TestBedrockApiKeyProviderThreadSafety:
    """Verify thread contention scenarios for _BedrockApiKeyProvider."""

    def test_concurrent_calls_only_refresh_once(self) -> None:
        """Multiple threads hitting advisory window simultaneously should
        result in only one refresh call (others return cached token)."""
        import threading
        import time

        provider = _BedrockApiKeyProvider("us-east-1")
        provider._token = "old-token"
        # Put in advisory window (12 min remaining)
        provider._expires_at = time.monotonic() + 720

        refresh_count = 0
        refresh_lock = threading.Lock()

        def _slow_refresh() -> None:
            nonlocal refresh_count
            # Simulate a slow refresh (e.g., network call)
            time.sleep(0.1)
            with refresh_lock:
                refresh_count += 1
            provider._token = "new-token"
            provider._expires_at = time.monotonic() + 43200

        results: list = []

        def _call_provider() -> None:
            token = provider()
            results.append(token)

        with patch.object(provider, "_do_refresh", side_effect=_slow_refresh):
            threads = [threading.Thread(target=_call_provider) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # All threads should get a token (either old or new)
        assert len(results) == 10
        assert all(t in ("old-token", "new-token") for t in results)
        # Only one thread should have performed the refresh
        # (others either got the lock and saw no refresh needed,
        # or returned the old token during advisory window)
        assert refresh_count == 1

    def test_mandatory_window_blocks_all_threads(self) -> None:
        """When in mandatory window, all threads block until refresh
        completes — none return a stale token."""
        import threading
        import time

        provider = _BedrockApiKeyProvider("us-east-1")
        provider._token = "expiring-token"
        # Put in mandatory window (5 min remaining)
        provider._expires_at = time.monotonic() + 300

        def _refresh() -> None:
            time.sleep(0.05)
            provider._token = "fresh-token"
            provider._expires_at = time.monotonic() + 43200

        results: list = []

        def _call_provider() -> None:
            token = provider()
            results.append(token)

        with patch.object(provider, "_do_refresh", side_effect=_refresh):
            threads = [threading.Thread(target=_call_provider) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # All threads must get the fresh token (mandatory = block until done)
        assert all(t == "fresh-token" for t in results)

    def test_no_token_all_threads_block(self) -> None:
        """First call with no cached token — all threads must block
        until the first refresh completes."""
        import threading
        import time

        provider = _BedrockApiKeyProvider("us-east-1")
        assert not provider._token

        def _refresh() -> None:
            time.sleep(0.05)
            provider._token = "first-token"
            provider._expires_at = time.monotonic() + 43200

        results: list = []

        def _call_provider() -> None:
            token = provider()
            results.append(token)

        with patch.object(provider, "_do_refresh", side_effect=_refresh):
            threads = [threading.Thread(target=_call_provider) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # All threads must get the token (none should get None)
        assert len(results) == 5
        assert all(t == "first-token" for t in results)


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


# ── Explicit AWS credentials & configurable TTL ─────────────────────


_AWS_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
)


def _clear_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient AWS env vars so ``secret_from_env`` fields default to None."""
    for key in _AWS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class TestChatBedrockMantleCredentialFields:
    """Field wiring for the credential kwargs added in this revision.

    Mirrors ``ChatBedrockConverse``'s inline field pattern and adds
    ``bedrock_api_key_ttl_seconds`` for configurable auto-refresh TTL.
    """

    # ── Defaults ───────────────────────────────────────────────

    def test_default_ttl_is_12h(self) -> None:
        assert _make_llm().bedrock_api_key_ttl_seconds == 43200

    def test_custom_ttl_field(self) -> None:
        llm = _make_llm(bedrock_api_key_ttl_seconds=3600)
        assert llm.bedrock_api_key_ttl_seconds == 3600

    def test_creds_default_to_none_when_no_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_aws_env(monkeypatch)
        llm = _make_llm()
        assert llm.aws_access_key_id is None
        assert llm.aws_secret_access_key is None
        assert llm.aws_session_token is None
        assert llm.credentials_profile_name is None

    def test_creds_pick_up_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_aws_env(monkeypatch)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-env")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-env")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "token-env")
        llm = _make_llm()
        assert llm.aws_access_key_id is not None
        assert llm.aws_access_key_id.get_secret_value() == "AKIA-env"
        assert llm.aws_secret_access_key is not None
        assert llm.aws_secret_access_key.get_secret_value() == "secret-env"
        assert llm.aws_session_token is not None
        assert llm.aws_session_token.get_secret_value() == "token-env"

    # ── Validation ─────────────────────────────────────────────

    def test_paired_creds_only_access_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_aws_env(monkeypatch)
        with pytest.raises(ValueError, match="both be provided"):
            _make_llm(aws_access_key_id=SecretStr("AKIA..."))

    def test_paired_creds_only_secret_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_aws_env(monkeypatch)
        with pytest.raises(ValueError, match="both be provided"):
            _make_llm(aws_secret_access_key=SecretStr("secret..."))

    def test_paired_creds_both_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_aws_env(monkeypatch)
        llm = _make_llm(
            aws_access_key_id=SecretStr("AKIA..."),
            aws_secret_access_key=SecretStr("secret..."),
        )
        assert llm.aws_access_key_id is not None
        assert llm.aws_secret_access_key is not None

    @pytest.mark.parametrize("bad_ttl", [0, -1, -43200, 43201, 86400])
    def test_ttl_out_of_range_rejected(self, bad_ttl: int) -> None:
        with pytest.raises(ValueError, match="between 1 and 43200"):
            _make_llm(bedrock_api_key_ttl_seconds=bad_ttl)

    @pytest.mark.parametrize("good_ttl", [1, 60, 3600, 43200])
    def test_ttl_in_range_accepted(self, good_ttl: int) -> None:
        llm = _make_llm(bedrock_api_key_ttl_seconds=good_ttl)
        assert llm.bedrock_api_key_ttl_seconds == good_ttl

    # ── Propagation to _BedrockApiKeyProvider ──────────────────

    def test_explicit_creds_propagate_to_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Field values propagate through _validate_environment into the
        provider's internal state."""
        _clear_aws_env(monkeypatch)
        # Force auto-refresh path (no static api_key).
        ChatBedrockMantle(
            model="test",
            region_name="us-east-1",
            aws_access_key_id=SecretStr("AKIA-explicit"),
            aws_secret_access_key=SecretStr("secret-explicit"),
            aws_session_token=SecretStr("session-explicit"),
        )
        # openai.OpenAI is a MagicMock; retrieve api_key from last call kwargs.
        openai_ctor = sys.modules["openai"].OpenAI
        provider = openai_ctor.call_args.kwargs["api_key"]
        assert isinstance(provider, _BedrockApiKeyProvider)
        assert provider._aws_access_key_id == "AKIA-explicit"
        assert provider._aws_secret_access_key == "secret-explicit"
        assert provider._aws_session_token == "session-explicit"

    def test_profile_name_propagates_to_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_aws_env(monkeypatch)
        ChatBedrockMantle(
            model="test",
            region_name="us-east-1",
            credentials_profile_name="my-profile",
        )
        openai_ctor = sys.modules["openai"].OpenAI
        provider = openai_ctor.call_args.kwargs["api_key"]
        assert isinstance(provider, _BedrockApiKeyProvider)
        assert provider._credentials_profile_name == "my-profile"

    def test_custom_ttl_propagates_to_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_aws_env(monkeypatch)
        ChatBedrockMantle(
            model="test",
            region_name="us-east-1",
            bedrock_api_key_ttl_seconds=1800,
        )
        openai_ctor = sys.modules["openai"].OpenAI
        provider = openai_ctor.call_args.kwargs["api_key"]
        assert isinstance(provider, _BedrockApiKeyProvider)
        assert provider._ttl_seconds == 1800

    def test_default_ttl_propagates_to_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_aws_env(monkeypatch)
        ChatBedrockMantle(model="test", region_name="us-east-1")
        openai_ctor = sys.modules["openai"].OpenAI
        provider = openai_ctor.call_args.kwargs["api_key"]
        assert isinstance(provider, _BedrockApiKeyProvider)
        assert provider._ttl_seconds == 43200

    def test_static_api_key_bypasses_provider(self) -> None:
        """When ``bedrock_api_key`` is set, no _BedrockApiKeyProvider is built."""
        llm = _make_llm(api_key=SecretStr("static-key"))
        openai_ctor = sys.modules["openai"].OpenAI
        provider = openai_ctor.call_args.kwargs["api_key"]
        # Should be the raw string, not a provider.
        assert provider == "static-key"
        assert llm.bedrock_api_key is not None


class TestBedrockApiKeyProviderCredentialResolution:
    """Verify _do_refresh picks up explicit creds / profile / default chain
    with the correct precedence and passes them to provide_token.
    """

    def _patch_provide_token(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Patch provide_token in the utils module; return the mock."""
        mock_pt = MagicMock(return_value="bedrock-api-key-fake&Version=1")
        # Provide a mock module for aws_bedrock_token_generator
        mock_mod = MagicMock()
        mock_mod.provide_token = mock_pt
        monkeypatch.setitem(
            __import__("sys").modules, "aws_bedrock_token_generator", mock_mod
        )
        return mock_pt

    def test_explicit_creds_used_by_provide_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit access_key + secret_key + session_token produce a
        credential provider whose .load() returns matching Credentials."""
        from botocore.credentials import Credentials

        from langchain_aws.utils import _BedrockApiKeyProvider

        mock_pt = self._patch_provide_token(monkeypatch)

        p = _BedrockApiKeyProvider(
            "us-east-1",
            aws_access_key_id="AKIA-abc",
            aws_secret_access_key="secret-def",
            aws_session_token="session-ghi",
        )
        p._do_refresh()

        assert mock_pt.called
        call_kwargs = mock_pt.call_args[1]
        assert call_kwargs["region"] == "us-east-1"
        provider_arg = call_kwargs["aws_credentials_provider"]
        assert provider_arg is not None
        loaded = provider_arg.load()
        assert isinstance(loaded, Credentials)
        assert loaded.access_key == "AKIA-abc"
        assert loaded.secret_key == "secret-def"
        assert loaded.token == "session-ghi"

    def test_no_explicit_creds_uses_default_chain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No explicit creds & no profile → default botocore chain."""
        from langchain_aws.utils import _BedrockApiKeyProvider

        mock_pt = self._patch_provide_token(monkeypatch)

        # Mock botocore session to return a benign Credentials-like object.
        from botocore.credentials import Credentials

        default_creds = Credentials(
            access_key="default-key",
            secret_key="default-secret",
            token=None,
        )
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = default_creds

        with patch("botocore.session.Session", return_value=mock_session):
            p = _BedrockApiKeyProvider("us-east-1")
            p._do_refresh()

        provider_arg = mock_pt.call_args[1]["aws_credentials_provider"]
        assert provider_arg is not None
        loaded = provider_arg.load()
        assert loaded.access_key == "default-key"

    def test_profile_only_uses_profile_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only credentials_profile_name set → BotocoreSession(profile=...)."""
        from botocore.credentials import Credentials

        from langchain_aws.utils import _BedrockApiKeyProvider

        mock_pt = self._patch_provide_token(monkeypatch)

        profile_creds = Credentials(
            access_key="profile-key",
            secret_key="profile-secret",
            token=None,
        )
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = profile_creds
        session_cls = MagicMock(return_value=mock_session)

        with patch("botocore.session.Session", session_cls):
            p = _BedrockApiKeyProvider(
                "us-east-1",
                credentials_profile_name="foo-profile",
            )
            p._do_refresh()

        # Session was constructed with the profile kwarg (not the default).
        session_cls.assert_called_once_with(profile="foo-profile")
        loaded = mock_pt.call_args[1]["aws_credentials_provider"].load()
        assert loaded.access_key == "profile-key"

    def test_session_failure_falls_back_to_provide_token_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When BotocoreSession raises, aws_credentials_provider=None so
        provide_token uses its own internal default resolution."""
        from langchain_aws.utils import _BedrockApiKeyProvider

        mock_pt = self._patch_provide_token(monkeypatch)

        # Session construction raises (e.g., malformed ~/.aws/config).
        session_cls = MagicMock(side_effect=OSError("profile not found"))

        with patch("botocore.session.Session", session_cls):
            p = _BedrockApiKeyProvider(
                "us-east-1", credentials_profile_name="missing-profile"
            )
            p._do_refresh()

        # provide_token still called, but with aws_credentials_provider=None.
        assert mock_pt.called
        assert mock_pt.call_args[1]["aws_credentials_provider"] is None

    def test_explicit_wins_over_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When BOTH explicit creds AND profile are set, explicit wins."""
        from botocore.credentials import Credentials

        from langchain_aws.utils import _BedrockApiKeyProvider

        mock_pt = self._patch_provide_token(monkeypatch)

        p = _BedrockApiKeyProvider(
            "us-east-1",
            aws_access_key_id="EXPLICIT-key",
            aws_secret_access_key="EXPLICIT-secret",
            credentials_profile_name="some-profile",
        )
        p._do_refresh()

        loaded = mock_pt.call_args[1]["aws_credentials_provider"].load()
        assert isinstance(loaded, Credentials)
        assert loaded.access_key == "EXPLICIT-key"

    def test_custom_ttl_used_in_provide_token_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ttl_seconds propagates into the expiry= arg of provide_token."""
        import datetime as _dt

        from langchain_aws.utils import _BedrockApiKeyProvider

        mock_pt = self._patch_provide_token(monkeypatch)

        p = _BedrockApiKeyProvider(
            "us-east-1",
            ttl_seconds=1800,
            aws_access_key_id="AKIA-x",
            aws_secret_access_key="secret-x",
        )
        p._do_refresh()

        expiry_arg = mock_pt.call_args[1]["expiry"]
        assert isinstance(expiry_arg, _dt.timedelta)
        assert expiry_arg.total_seconds() == 1800

    def test_ttl_capped_by_refreshable_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When default-chain creds are RefreshableCredentials with less
        remaining lifetime than ttl_seconds, effective TTL is capped."""
        import datetime as _dt

        from botocore.credentials import RefreshableCredentials

        from langchain_aws.utils import _BedrockApiKeyProvider

        mock_pt = self._patch_provide_token(monkeypatch)

        # RefreshableCredentials with 900s remaining, TTL request 12h.
        mock_refreshable = MagicMock(spec=RefreshableCredentials)
        mock_refreshable._seconds_remaining.return_value = 900
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = mock_refreshable

        with patch("botocore.session.Session", return_value=mock_session):
            p = _BedrockApiKeyProvider("us-east-1", ttl_seconds=43200)
            p._do_refresh()

        expiry_arg = mock_pt.call_args[1]["expiry"]
        assert isinstance(expiry_arg, _dt.timedelta)
        assert expiry_arg.total_seconds() == 900


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

    def test_authentication_error_reraised(self) -> None:
        """AuthenticationError is re-raised unchanged (not wrapped)."""
        mock_mod = self._make_mock_openai()
        error = mock_mod.AuthenticationError("invalid key")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(mock_mod.AuthenticationError, match="invalid key"):
                _handle_openai_error(error)

    def test_berm_error_logs_hint_and_reraises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 401 with 'Berm is not enabled' logs a routing hint and re-raises.

        Mantle returns this string when a model that requires the
        ``/openai/v1`` base path is called on ``/v1`` (e.g. gpt-5.x,
        gemma-4-*, grok-4.3). The exception type is preserved so callers
        can still ``except openai.AuthenticationError``; the hint is
        emitted as a WARNING for discoverability.
        """
        mock_mod = self._make_mock_openai()
        error = mock_mod.AuthenticationError("Berm is not enabled for this account")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with caplog.at_level(
                "WARNING", logger="langchain_aws.chat_models.bedrock_mantle"
            ):
                with pytest.raises(mock_mod.AuthenticationError, match="Berm"):
                    _handle_openai_error(error)
        assert any(
            "'/openai/v1' base path" in rec.getMessage() for rec in caplog.records
        ), f"Expected /openai/v1 hint in log records, got: {caplog.records!r}"

    def test_rate_limit_error_reraised(self) -> None:
        """RateLimitError is re-raised unchanged (not wrapped)."""
        mock_mod = self._make_mock_openai()
        error = mock_mod.RateLimitError("rate limited")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(mock_mod.RateLimitError, match="rate limited"):
                _handle_openai_error(error)

    def test_not_found_error_reraised(self) -> None:
        """NotFoundError is re-raised unchanged (not wrapped)."""
        mock_mod = self._make_mock_openai()
        error = mock_mod.NotFoundError("model not found")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(mock_mod.NotFoundError, match="model not found"):
                _handle_openai_error(error)

    def test_bad_request_tool_choice(self) -> None:
        """BadRequestError with tool_choice context is wrapped in ValueError."""
        mock_mod = self._make_mock_openai()
        error = mock_mod.BadRequestError("Invalid 'tool_choice': value did not match")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(ValueError, match="rejected the tool_choice"):
                _handle_openai_error(error)

    def test_bad_request_not_supported(self) -> None:
        """BadRequestError with 'does not support' is wrapped in ValueError."""
        mock_mod = self._make_mock_openai()
        error = mock_mod.BadRequestError("model does not support /v1/responses")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(ValueError, match="not supported"):
                _handle_openai_error(error)

    def test_unknown_error_reraised(self) -> None:
        """Non-OpenAI errors are re-raised unchanged."""
        mock_mod = self._make_mock_openai()
        error = RuntimeError("something else")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(RuntimeError, match="something else"):
                _handle_openai_error(error)

    def test_bad_request_generic_reraised(self) -> None:
        """Generic BadRequestError is re-raised unchanged (not wrapped)."""
        mock_mod = self._make_mock_openai()
        error = mock_mod.BadRequestError("some other bad request")
        with patch.dict(__import__("sys").modules, {"openai": mock_mod}):
            with pytest.raises(
                mock_mod.BadRequestError, match="some other bad request"
            ):
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
        llm._sync_client.responses.create = MagicMock(
            return_value=_MockSyncStream(events)
        )

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
        llm._sync_client.responses.create = MagicMock(
            return_value=_MockSyncStream(events)
        )

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
        llm._sync_client.responses.create = MagicMock(
            return_value=_MockSyncStream(events)
        )

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

        async def _mock_create(**kwargs: object) -> _MockAsyncStream:
            return _MockAsyncStream(events)

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

    def test_stream_responses_closes_context_manager(self) -> None:
        """``_stream_responses`` must consume the SDK stream via ``with``.

        The mock helpers implement both ``__iter__`` and ``__enter__``, so
        naive iteration would silently pass even if the production code
        stopped using ``with``. A tracking subclass surfaces that
        regression by asserting ``__enter__`` / ``__exit__`` actually
        fire — both on happy-path exhaustion and on early exit.

        Tests ``_stream_responses`` directly (rather than the ``_stream``
        router) so ``generator.close()`` propagates synchronously through
        the ``async with`` we care about.
        """

        class _TrackingSyncStream(_MockSyncStream):
            entered: bool = False
            exited: bool = False

            def __enter__(self) -> "_TrackingSyncStream":
                type(self).entered = True
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                type(self).exited = True

        events = [
            self._make_text_delta_event("ok"),
            self._make_completed_event(),
        ]
        llm = _make_llm()
        params = llm._build_params(stream=True)

        # Happy path: exhaust the iterator.
        _TrackingSyncStream.entered = False
        _TrackingSyncStream.exited = False
        llm._sync_client.responses.create = MagicMock(
            return_value=_TrackingSyncStream(events)
        )
        list(llm._stream_responses([HumanMessage(content="hi")], params))
        assert _TrackingSyncStream.entered, "expected __enter__ on happy path"
        assert _TrackingSyncStream.exited, "expected __exit__ on happy path"

        # Early exit: break, then close the generator explicitly. Python's
        # generator + ``break`` alone defers cleanup to GC; well-behaved
        # consumers close the generator to signal early termination.
        _TrackingSyncStream.entered = False
        _TrackingSyncStream.exited = False
        llm._sync_client.responses.create = MagicMock(
            return_value=_TrackingSyncStream(events)
        )
        gen = cast(
            Generator[ChatGenerationChunk, None, None],
            llm._stream_responses([HumanMessage(content="hi")], params),
        )
        for _ in gen:
            break
        gen.close()
        assert _TrackingSyncStream.entered, "expected __enter__ on early exit"
        assert _TrackingSyncStream.exited, "expected __exit__ on early exit"

    def test_astream_responses_closes_context_manager(self) -> None:
        """Async counterpart of ``test_stream_responses_closes_context_manager``.

        Verifies that ``_astream_responses`` consumes the SDK stream via
        ``async with``, firing ``__aenter__`` / ``__aexit__`` on both
        happy-path exhaustion and early exit + ``aclose()``.

        Targets ``_astream_responses`` directly rather than the
        ``_astream`` router. Calling ``aclose()`` on an outer async
        generator that wraps an inner one defers the inner ``__aexit__``
        to garbage collection via ``CancelledError`` propagation, which
        cannot be observed synchronously. Testing the inner method
        directly makes cleanup synchronous.
        """
        import asyncio

        class _TrackingAsyncStream(_MockAsyncStream):
            entered: bool = False
            exited: bool = False

            async def __aenter__(self) -> "_TrackingAsyncStream":
                type(self).entered = True
                return self

            async def __aexit__(
                self, exc_type: object, exc: object, tb: object
            ) -> None:
                type(self).exited = True

        events = [
            self._make_text_delta_event("ok"),
            self._make_completed_event(),
        ]
        llm = _make_llm()
        params = llm._build_params(stream=True)

        async def _run_full() -> None:
            _TrackingAsyncStream.entered = False
            _TrackingAsyncStream.exited = False

            async def _mock_create(**kwargs: object) -> _TrackingAsyncStream:
                return _TrackingAsyncStream(events)

            llm._async_client.responses.create = _mock_create
            async for _ in llm._astream_responses([HumanMessage(content="hi")], params):
                pass
            assert _TrackingAsyncStream.entered, "expected __aenter__ on happy path"
            assert _TrackingAsyncStream.exited, "expected __aexit__ on happy path"

        async def _run_early_break() -> None:
            _TrackingAsyncStream.entered = False
            _TrackingAsyncStream.exited = False

            async def _mock_create(**kwargs: object) -> _TrackingAsyncStream:
                return _TrackingAsyncStream(events)

            llm._async_client.responses.create = _mock_create
            gen = cast(
                AsyncGenerator[ChatGenerationChunk, None],
                llm._astream_responses([HumanMessage(content="hi")], params),
            )
            async for _ in gen:
                break
            await gen.aclose()
            assert _TrackingAsyncStream.entered, "expected __aenter__ on early exit"
            assert _TrackingAsyncStream.exited, "expected __aexit__ on early exit"

        asyncio.run(_run_full())
        asyncio.run(_run_early_break())


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

        llm._sync_client.responses.create = MagicMock(
            return_value=_MockSyncStream([completed])
        )

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
            return_value=_MockSyncStream([added, delta, delta2, completed])
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

    def test_chat_completions_final_usage_only_chunk_surfaced(self) -> None:
        """OpenAI ``include_usage=True`` emits a terminal chunk with
        ``choices=[]`` and populated ``usage``. That chunk must reach the
        caller as an ``AIMessageChunk`` carrying ``usage_metadata`` — the
        previous implementation dropped it via an early ``not chunk.choices``
        return and silently lost token counts on every Chat Completions
        stream.
        """
        llm = _make_llm(use_responses_api=False)

        content_chunk = MagicMock()
        content_chunk.id = "chatcmpl-abc"
        content_chunk.model = "test-model"
        content_chunk.choices = [MagicMock()]
        content_chunk.choices[0].delta.content = "Hi"
        content_chunk.choices[0].delta.tool_calls = None
        content_chunk.choices[0].finish_reason = "stop"
        content_chunk.usage = None

        usage_chunk = MagicMock()
        usage_chunk.id = "chatcmpl-abc"
        usage_chunk.model = "test-model"
        usage_chunk.choices = []
        usage_chunk.usage = MagicMock()
        usage_chunk.usage.prompt_tokens = 15
        usage_chunk.usage.completion_tokens = 12
        usage_chunk.usage.total_tokens = 27

        llm._sync_client.chat.completions.create = MagicMock(
            return_value=_MockSyncStream([content_chunk, usage_chunk])
        )

        chunks = list(llm._stream([HumanMessage(content="hi")]))
        last = chunks[-1]
        assert isinstance(last.message, AIMessageChunk)
        assert last.message.usage_metadata is not None
        assert last.message.usage_metadata["input_tokens"] == 15
        assert last.message.usage_metadata["output_tokens"] == 12
        assert last.message.usage_metadata["total_tokens"] == 27

    @staticmethod
    def _make_cc_delta(
        content: Optional[str] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[list] = None,
    ) -> MagicMock:
        """Build a mock ChatCompletionChunk with a single delta choice."""
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        chunk.choices[0].delta.tool_calls = tool_calls
        chunk.choices[0].finish_reason = finish_reason
        chunk.usage = None
        return chunk

    def test_chat_completions_stream_text_accumulates(self) -> None:
        """Multiple content deltas are each surfaced as AIMessageChunks; the
        caller can concatenate them to reconstruct the full text.
        """
        llm = _make_llm(use_responses_api=False)
        chunks_in = [
            self._make_cc_delta(content="Hel"),
            self._make_cc_delta(content="lo"),
            self._make_cc_delta(content="", finish_reason="stop"),
        ]
        llm._sync_client.chat.completions.create = MagicMock(
            return_value=_MockSyncStream(chunks_in)
        )

        chunks = list(llm._stream([HumanMessage(content="hi")]))
        text = "".join(
            c.message.content for c in chunks if isinstance(c.message.content, str)
        )
        assert text == "Hello"

    def test_chat_completions_stream_finish_reason_surfaced(self) -> None:
        """``finish_reason`` on the terminating chunk lands in
        ``response_metadata``.
        """
        llm = _make_llm(use_responses_api=False)
        chunks_in = [self._make_cc_delta(content="done", finish_reason="stop")]
        llm._sync_client.chat.completions.create = MagicMock(
            return_value=_MockSyncStream(chunks_in)
        )

        chunks = list(llm._stream([HumanMessage(content="hi")]))
        assert chunks[-1].message.response_metadata.get("finish_reason") == "stop"

    def test_chat_completions_stream_tool_call_chunks(self) -> None:
        """Streaming tool_calls emit tool_call_chunks with the fields the
        SDK provides on each incremental delta (name on the first chunk,
        args on subsequent chunks, shared index).
        """
        llm = _make_llm(use_responses_api=False)

        def _tc(
            name: Optional[str],
            args: Optional[str],
            id_: Optional[str],
        ) -> MagicMock:
            tc = MagicMock()
            tc.function = MagicMock()
            tc.function.name = name
            tc.function.arguments = args
            tc.id = id_
            tc.index = 0
            return tc

        chunks_in = [
            self._make_cc_delta(tool_calls=[_tc("get_weather", "", "call_1")]),
            self._make_cc_delta(tool_calls=[_tc(None, '{"location":', None)]),
            self._make_cc_delta(tool_calls=[_tc(None, '"Seattle"}', None)]),
        ]
        llm._sync_client.chat.completions.create = MagicMock(
            return_value=_MockSyncStream(chunks_in)
        )

        chunks = list(llm._stream([HumanMessage(content="weather?")]))
        # Runtime types are AIMessageChunk; narrow so mypy sees .tool_call_chunks.
        first_msg = chunks[0].message
        second_msg = chunks[1].message
        third_msg = chunks[2].message
        assert isinstance(first_msg, AIMessageChunk)
        assert isinstance(second_msg, AIMessageChunk)
        assert isinstance(third_msg, AIMessageChunk)
        # First chunk carries the name and id.
        first_tc_chunks = first_msg.tool_call_chunks
        assert len(first_tc_chunks) == 1
        assert first_tc_chunks[0]["name"] == "get_weather"
        assert first_tc_chunks[0]["id"] == "call_1"
        assert first_tc_chunks[0]["index"] == 0
        # Subsequent chunks carry args deltas at the same index.
        assert second_msg.tool_call_chunks[0]["args"] == '{"location":'
        assert third_msg.tool_call_chunks[0]["args"] == '"Seattle"}'


# ── use_previous_response_id ────────────────────────────────────────


class TestUsePreviousResponseId:
    def test_extracts_id_from_message_history(self) -> None:
        llm = _make_llm(use_previous_response_id=True)
        mock_resp = _mock_responses_response(text="followup", resp_id="resp_f1")
        llm._sync_client.responses.create = MagicMock(return_value=mock_resp)

        messages = [
            HumanMessage(content="Hello"),
            # Mirrors the shape produced by ``_build_responses_result`` — the
            # response id lives in ``response_metadata["id"]``.
            AIMessage(
                content="Hi!",
                response_metadata={"id": "resp_first_123"},
            ),
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
    def test_auto_passes_through(self) -> None:
        llm = _make_llm()
        params = llm._build_params(stream=False, tool_choice="auto")
        resp_params = llm._prepare_responses_params(params)
        assert resp_params["tool_choice"] == "auto"

    def test_none_passes_through(self) -> None:
        llm = _make_llm()
        params = llm._build_params(stream=False, tool_choice="none")
        resp_params = llm._prepare_responses_params(params)
        assert resp_params["tool_choice"] == "none"

    def test_required_passes_through(self) -> None:
        """'required' is a valid Responses API value — no warning."""
        llm = _make_llm()
        params = llm._build_params(stream=False, tool_choice="required")
        resp_params = llm._prepare_responses_params(params)
        assert resp_params["tool_choice"] == "required"

    def test_dict_format_converted(self) -> None:
        """Chat Completions dict format is converted to Responses API format."""
        llm = _make_llm()
        params = llm._build_params(
            stream=False,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        resp_params = llm._prepare_responses_params(params)
        assert resp_params["tool_choice"] == {
            "type": "function",
            "name": "get_weather",
        }


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


# ── disabled_params ─────────────────────────────────────────────────


class TestDisabledParams:
    """_filter_disabled_params silently drops params based on config."""

    def test_no_disabled_params(self) -> None:
        """When disabled_params is None, nothing is filtered."""
        llm = _make_llm()
        result = llm._filter_disabled_params(
            tool_choice="required", parallel_tool_calls=False
        )
        assert result == {"tool_choice": "required", "parallel_tool_calls": False}

    def test_drop_param_entirely(self) -> None:
        """disabled_params={"parallel_tool_calls": None} drops it always."""
        llm = _make_llm(disabled_params={"parallel_tool_calls": None})
        result = llm._filter_disabled_params(
            tool_choice="required", parallel_tool_calls=False
        )
        assert result == {"tool_choice": "required"}
        assert "parallel_tool_calls" not in result

    def test_drop_param_specific_value(self) -> None:
        """disabled_params={"tool_choice": ["required"]} drops only that value."""
        llm = _make_llm(disabled_params={"tool_choice": ["required"]})

        # "required" is disabled — should be dropped
        result = llm._filter_disabled_params(
            tool_choice="required", parallel_tool_calls=False
        )
        assert "tool_choice" not in result
        assert result["parallel_tool_calls"] is False

        # "auto" is not disabled — should pass through
        result2 = llm._filter_disabled_params(
            tool_choice="auto", parallel_tool_calls=False
        )
        assert result2["tool_choice"] == "auto"

    def test_structured_output_respects_disabled_params(self) -> None:
        """with_structured_output drops parallel_tool_calls if disabled."""
        llm = _make_llm(
            disabled_params={"parallel_tool_calls": None},
            use_responses_api=False,
        )

        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.message.tool_calls = [MagicMock()]
        mock_choice.message.tool_calls[0].function.name = "Person"
        mock_choice.message.tool_calls[
            0
        ].function.arguments = '{"name": "Alice", "age": 30}'
        mock_choice.message.tool_calls[0].id = "call_1"
        mock_choice.finish_reason = "tool_calls"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = None
        mock_resp.model = "test"
        mock_resp.id = "chatcmpl_dp1"
        llm._sync_client.chat.completions.create = MagicMock(return_value=mock_resp)

        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        structured = llm.with_structured_output(Person, method="function_calling")
        structured.invoke("Give me a person")

        # Verify parallel_tool_calls was NOT passed to the API
        call_kwargs = llm._sync_client.chat.completions.create.call_args[1]
        assert "parallel_tool_calls" not in call_kwargs


# ── with_structured_output json_schema ──────────────────────────────


class TestWithStructuredOutputJsonSchema:
    """with_structured_output(method='json_schema') binds response_format.

    ``response_format`` is a Chat-Completions-only key so requests route to
    Chat Completions by default. If the request also has Responses-API-only
    keys, ``_prepare_responses_params`` remaps ``response_format`` to
    ``text``.
    """

    def test_json_schema_routes_to_chat_completions(self) -> None:
        """Default: response_format routes to Chat Completions."""
        llm = _make_llm()

        mock_choice = MagicMock()
        mock_choice.message.content = '{"name": "Alice", "age": 30}'
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = None
        mock_resp.model = "test"
        mock_resp.id = "chatcmpl_js1"
        llm._sync_client.chat.completions.create = MagicMock(return_value=mock_resp)
        llm._sync_client.responses.create = MagicMock(
            side_effect=AssertionError("should not call Responses API")
        )

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
        result = structured.invoke("Give me a person")

        # Verify response_format was passed to Chat Completions
        call_kwargs = llm._sync_client.chat.completions.create.call_args[1]
        assert "response_format" in call_kwargs
        rf = call_kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Person"
        assert rf["json_schema"]["strict"] is True
        assert "text" not in call_kwargs

        # Verify the output was parsed
        assert result == {"name": "Alice", "age": 30}

    def test_json_schema_remapped_when_forced_to_responses_api(self) -> None:
        """When use_responses_api=True, response_format is remapped to text."""
        llm = _make_llm(use_responses_api=True)
        mock_resp = _mock_responses_response(
            text='{"name": "Bob", "age": 25}', resp_id="resp_js2"
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
        result = structured.invoke("Give me a person")

        # Verify response_format was remapped to text for Responses API
        call_kwargs = llm._sync_client.responses.create.call_args[1]
        assert "text" in call_kwargs
        text_format = call_kwargs["text"]["format"]
        assert text_format["type"] == "json_schema"
        assert text_format["name"] == "Person"
        assert text_format["strict"] is True
        assert "response_format" not in call_kwargs

        # Verify the output was parsed
        assert result == {"name": "Bob", "age": 25}

    def test_json_schema_with_explicit_false(self) -> None:
        """use_responses_api=False also routes to Chat Completions."""
        llm = _make_llm(use_responses_api=False)

        mock_choice = MagicMock()
        mock_choice.message.content = '{"name": "Carol", "age": 40}'
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = None
        mock_resp.model = "test"
        mock_resp.id = "chatcmpl_js3"
        llm._sync_client.chat.completions.create = MagicMock(return_value=mock_resp)

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
        result = structured.invoke("Give me a person")

        call_kwargs = llm._sync_client.chat.completions.create.call_args[1]
        assert "response_format" in call_kwargs
        assert result == {"name": "Carol", "age": 40}
