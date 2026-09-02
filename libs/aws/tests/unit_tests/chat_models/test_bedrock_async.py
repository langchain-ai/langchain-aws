"""Test the native async paths of `ChatBedrock`."""

import json
import os
from typing import Any, AsyncIterator, Dict, List
from unittest import mock

import pytest

from langchain_aws import ChatBedrock

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"


class _Body:
    """Stands in for botocore's streaming response body."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload


class _AsyncEventBody:
    """Async iterator over canned Invoke response-stream events."""

    def __init__(self, payloads: List[Dict[str, Any]]) -> None:
        self._payloads = payloads
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        for payload in self._payloads:
            yield {"chunk": {"bytes": json.dumps(payload).encode()}}

    async def close(self) -> None:
        self.closed = True


def _anthropic_response() -> Dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi async"}],
        "model": MODEL_ID,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


def _async_client(
    response: Dict[str, Any] | None = None,
    stream_payloads: List[Dict[str, Any]] | None = None,
) -> Any:
    client = mock.AsyncMock()
    client.invoke_model.return_value = {
        "body": _Body(response or _anthropic_response()),
        "ResponseMetadata": {
            "RequestId": "req-1",
            # ChatBedrock reads token counts from response headers, so an async
            # client has to surface them the way botocore does.
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "3",
                "x-amzn-bedrock-output-token-count": "2",
            },
        },
    }
    client.invoke_model_with_response_stream.return_value = {
        "body": _AsyncEventBody(stream_payloads or []),
        "ResponseMetadata": {"RequestId": "req-2"},
    }
    return client


def _llm(**kwargs: Any) -> ChatBedrock:
    kwargs.setdefault("client", mock.MagicMock())
    kwargs.setdefault("bedrock_client", mock.MagicMock())
    return ChatBedrock(model_id=MODEL_ID, region_name="us-west-2", **kwargs)  # type: ignore[call-arg]


async def test_agenerate_uses_async_client() -> None:
    async_client = _async_client()
    llm = _llm(async_client=async_client)

    result = await llm.ainvoke("hello")

    assert result.content == "hi async"
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] == 3
    assert result.response_metadata["model_provider"] == "bedrock"
    async_client.invoke_model.assert_awaited_once()
    llm.client.invoke_model.assert_not_called()


async def test_agenerate_falls_back_to_executor_without_async_client() -> None:
    sync_client = mock.MagicMock()
    sync_client.invoke_model.return_value = {"body": _Body(_anthropic_response())}
    llm = _llm(client=sync_client)

    result = await llm.ainvoke("hello")

    assert result.content == "hi async"
    sync_client.invoke_model.assert_called_once()


async def test_astream_uses_async_client() -> None:
    payloads: List[Dict[str, Any]] = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "he"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "llo"},
        },
        {"type": "message_stop"},
    ]
    async_client = _async_client(stream_payloads=payloads)
    llm = _llm(async_client=async_client)

    chunks = [chunk async for chunk in llm.astream("hello")]

    assert "".join(chunk.text for chunk in chunks) == "hello"
    async_client.invoke_model_with_response_stream.assert_awaited_once()
    llm.client.invoke_model_with_response_stream.assert_not_called()


async def test_astream_closes_the_stream() -> None:
    body = _AsyncEventBody(
        [
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
            {"type": "message_stop"},
        ]
    )
    async_client = mock.AsyncMock()
    async_client.invoke_model_with_response_stream.return_value = {"body": body}
    llm = _llm(async_client=async_client)

    [chunk async for chunk in llm.astream("hello")]

    assert body.closed


async def test_async_and_sync_send_identical_requests() -> None:
    sync_client = mock.MagicMock()
    sync_client.invoke_model.return_value = {"body": _Body(_anthropic_response())}
    async_client = _async_client()
    llm = _llm(client=sync_client, async_client=async_client, temperature=0.5)

    llm.invoke("hello")
    await llm.ainvoke("hello")

    assert sync_client.invoke_model.call_args.kwargs == (
        async_client.invoke_model.await_args.kwargs
    )


async def test_streaming_flag_routes_agenerate_through_astream() -> None:
    payloads: List[Dict[str, Any]] = [
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "streamed"},
        },
        {"type": "message_stop"},
    ]
    async_client = _async_client(stream_payloads=payloads)
    llm = _llm(async_client=async_client, streaming=True)

    result = await llm.ainvoke("hello")

    assert result.text == "streamed"
    async_client.invoke_model.assert_not_awaited()
    async_client.invoke_model_with_response_stream.assert_awaited_once()


def test_unentered_async_client_is_rejected() -> None:
    class _UnenteredContextManager:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

    with pytest.raises(ValueError, match="unentered async context manager"):
        _llm(async_client=_UnenteredContextManager())


@mock.patch.dict(os.environ, {"AWS_REGION": "us-west-2"})
def test_use_async_transport_builds_a_client() -> None:
    llm = ChatBedrock(  # type: ignore[call-arg]
        model_id=MODEL_ID,
        client=mock.MagicMock(),
        bedrock_client=mock.MagicMock(),
        use_async_transport=True,
    )

    assert type(llm.async_client).__name__ == "BedrockAsyncClient"


def test_converse_delegate_inherits_the_async_client() -> None:
    async_client = _async_client()
    llm = _llm(async_client=async_client, beta_use_converse_api=True)

    assert llm._as_converse.async_client is async_client
