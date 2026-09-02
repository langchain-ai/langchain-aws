"""Test the native async paths of `ChatBedrockConverse`."""

import os
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_aws import ChatBedrockConverse

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"


class _AsyncStream:
    """Async iterator over canned Converse stream events."""

    def __init__(self, events: List[Dict[str, Any]]) -> None:
        self._events = events
        self.closed = False

    async def __aiter__(self) -> Any:
        for event in self._events:
            yield event

    async def close(self) -> None:
        self.closed = True


def _converse_response() -> Dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi async"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
        "metrics": {"latencyMs": 1},
        "ResponseMetadata": {"RequestId": "req-1"},
    }


def _async_client(
    response: Optional[Dict[str, Any]] = None,
    stream_events: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Build a stand-in for an entered async bedrock-runtime client."""
    client = mock.AsyncMock()
    client.converse.return_value = response or _converse_response()
    client.converse_stream.return_value = {
        "stream": _AsyncStream(stream_events or []),
        "ResponseMetadata": {"RequestId": "req-2"},
    }
    return client


def _sync_client() -> Any:
    client = mock.MagicMock()
    client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi sync"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        "metrics": {"latencyMs": 1},
    }
    return client


def _llm(**kwargs: Any) -> ChatBedrockConverse:
    kwargs.setdefault("client", mock.MagicMock())
    kwargs.setdefault("bedrock_client", mock.MagicMock())
    return ChatBedrockConverse(model=MODEL_ID, region_name="us-west-2", **kwargs)


async def test_agenerate_uses_async_client() -> None:
    async_client = _async_client()
    llm = _llm(async_client=async_client)

    result = await llm.ainvoke("hello")

    assert result.content == "hi async"
    assert result.usage_metadata is not None
    assert result.usage_metadata["total_tokens"] == 5
    assert result.response_metadata["model_provider"] == "bedrock_converse"
    async_client.converse.assert_awaited_once()
    llm.client.converse.assert_not_called()


async def test_agenerate_falls_back_to_executor_without_async_client() -> None:
    sync_client = _sync_client()
    llm = _llm(client=sync_client)

    result = await llm.ainvoke("hello")

    assert result.content == "hi sync"
    sync_client.converse.assert_called_once()


async def test_astream_uses_async_client() -> None:
    events: List[Dict[str, Any]] = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "he"}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "llo"}, "contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]
    async_client = _async_client(stream_events=events)
    llm = _llm(async_client=async_client)

    chunks = [chunk async for chunk in llm.astream("hello")]

    assert "".join(chunk.text for chunk in chunks) == "hello"
    async_client.converse_stream.assert_awaited_once()
    llm.client.converse_stream.assert_not_called()


async def test_astream_attaches_usage_and_model_metadata() -> None:
    events: List[Dict[str, Any]] = [
        {"contentBlockDelta": {"delta": {"text": "hi"}, "contentBlockIndex": 0}},
        {
            "metadata": {
                "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
                "metrics": {"latencyMs": 2},
            }
        },
    ]
    llm = _llm(async_client=_async_client(stream_events=events))

    chunks = [chunk async for chunk in llm.astream("hello")]

    usage_chunks = [chunk for chunk in chunks if chunk.usage_metadata]
    assert usage_chunks, "expected usage metadata on a chunk"
    assert usage_chunks[0].usage_metadata is not None
    assert usage_chunks[0].usage_metadata["total_tokens"] == 4
    assert usage_chunks[0].response_metadata["model_name"] == MODEL_ID


async def test_astream_falls_back_to_executor_without_async_client() -> None:
    sync_client = mock.MagicMock()
    sync_client.converse_stream.return_value = {
        "stream": iter(
            [{"contentBlockDelta": {"delta": {"text": "sync"}, "contentBlockIndex": 0}}]
        )
    }
    llm = _llm(client=sync_client)

    chunks = [chunk async for chunk in llm.astream("hello")]

    assert "".join(chunk.text for chunk in chunks) == "sync"
    sync_client.converse_stream.assert_called_once()


async def test_astream_closes_the_stream() -> None:
    stream = _AsyncStream([{"messageStart": {"role": "assistant"}}])
    async_client = mock.AsyncMock()
    async_client.converse_stream.return_value = {"stream": stream}
    llm = _llm(async_client=async_client)

    [chunk async for chunk in llm.astream("hello")]

    assert stream.closed


async def test_async_and_sync_send_identical_requests() -> None:
    """The shared request builder must not let the two paths drift apart."""
    sync_client = _sync_client()
    async_client = _async_client()
    llm = _llm(
        client=sync_client,
        async_client=async_client,
        temperature=0.5,
        max_tokens=42,
    )

    messages = [SystemMessage("be terse"), HumanMessage("hello")]
    llm.invoke(messages)
    await llm.ainvoke(messages)

    assert sync_client.converse.call_args.kwargs == (
        async_client.converse.await_args.kwargs
    )


def test_unentered_async_client_is_rejected() -> None:
    class _UnenteredContextManager:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

    with pytest.raises(ValueError, match="unentered async context manager"):
        _llm(async_client=_UnenteredContextManager())


def test_async_client_without_converse_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not provide a `converse` method"):
        _llm(async_client=object())


async def test_aclose_leaves_a_supplied_client_alone() -> None:
    async_client = _async_client()
    llm = _llm(async_client=async_client)

    await llm.aclose()

    async_client.close.assert_not_awaited()
    assert llm.async_client is async_client


@mock.patch.dict(os.environ, {"AWS_REGION": "us-west-2"})
def test_use_async_transport_builds_a_client() -> None:
    llm = ChatBedrockConverse(
        model=MODEL_ID,
        client=mock.MagicMock(),
        bedrock_client=mock.MagicMock(),
        use_async_transport=True,
    )

    assert type(llm.async_client).__name__ == "BedrockAsyncClient"


@mock.patch.dict(os.environ, {"AWS_REGION": "us-west-2"})
async def test_aclose_closes_a_client_this_model_built() -> None:
    llm = ChatBedrockConverse(
        model=MODEL_ID,
        client=mock.MagicMock(),
        bedrock_client=mock.MagicMock(),
        use_async_transport=True,
    )
    llm.async_client = mock.AsyncMock()

    await llm.aclose()

    assert llm.async_client is None


def test_explicit_async_client_wins_over_use_async_transport() -> None:
    async_client = _async_client()
    llm = _llm(async_client=async_client, use_async_transport=True)

    assert llm.async_client is async_client


@mock.patch.dict(os.environ, {"AWS_REGION": "us-west-2"})
def test_bedrock_api_key_with_async_transport_is_rejected() -> None:
    """The built transport signs with SigV4, so a bearer key must not be ignored."""
    with pytest.raises(ValueError, match="does not support bearer-token"):
        ChatBedrockConverse(  # type: ignore[call-arg]
            model=MODEL_ID,
            client=mock.MagicMock(),
            bedrock_client=mock.MagicMock(),
            bedrock_api_key="abc123",
            use_async_transport=True,
        )


@mock.patch.dict(os.environ, {"AWS_REGION": "us-west-2"})
def test_bedrock_api_key_is_fine_without_async_transport() -> None:
    llm = ChatBedrockConverse(  # type: ignore[call-arg]
        model=MODEL_ID,
        client=mock.MagicMock(),
        bedrock_client=mock.MagicMock(),
        bedrock_api_key="abc123",
    )

    assert llm.async_client is None
