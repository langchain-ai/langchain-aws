"""Test the native async Bedrock runtime transport."""

import asyncio
import base64
import binascii
import json
import struct
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest import mock

import pytest
from botocore.credentials import Credentials
from botocore.exceptions import ClientError, EventStreamError

from langchain_aws.async_client import (
    BedrockAsyncClient,
    _b64_encode_blobs,
    _decode_invoke_chunk,
)

CREDENTIALS = Credentials("AKIDEXAMPLE", "secret", "token")


def _encode_event_frame(headers: Dict[str, str], payload: bytes) -> bytes:
    """Encode one `vnd.amazon.eventstream` message the way Bedrock does."""
    encoded_headers = b""
    for key, value in headers.items():
        key_bytes, value_bytes = key.encode(), value.encode()
        encoded_headers += (
            bytes([len(key_bytes)])
            + key_bytes
            + b"\x07"
            + struct.pack(">H", len(value_bytes))
            + value_bytes
        )
    total_length = 16 + len(encoded_headers) + len(payload)
    prelude = struct.pack(">II", total_length, len(encoded_headers))
    prelude += struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message = prelude + encoded_headers + payload
    return message + struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)


def _event(event_type: str, body: Dict[str, Any]) -> bytes:
    return _encode_event_frame(
        {":message-type": "event", ":event-type": event_type},
        json.dumps(body).encode(),
    )


class _FakeResponse:
    """Stands in for an `httpx.Response`."""

    def __init__(
        self,
        status_code: int = 200,
        json_body: Optional[Dict[str, Any]] = None,
        content: bytes = b"",
        headers: Optional[Dict[str, str]] = None,
        stream_chunks: Optional[List[bytes]] = None,
    ) -> None:
        self.status_code = status_code
        self._json_body = json_body
        self.content = content
        self.headers = headers or {}
        self.text = content.decode() if content else ""
        self._stream_chunks = stream_chunks or []
        self.closed = False

    def json(self) -> Dict[str, Any]:
        if self._json_body is None:
            raise ValueError("no json body")
        return self._json_body

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._stream_chunks:
            yield chunk

    async def aread(self) -> bytes:
        return self.content

    async def aclose(self) -> None:
        self.closed = True


class _FakeHTTPClient:
    """Stands in for an `httpx.AsyncClient`, recording what was sent."""

    def __init__(self, *responses: _FakeResponse) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    @property
    def response(self) -> _FakeResponse:
        return self.responses[-1]

    def build_request(
        self, method: str, url: str, *, content: bytes, headers: Dict[str, str]
    ) -> Dict[str, Any]:
        request = {
            "method": method,
            "url": url,
            "content": content,
            "headers": headers,
        }
        self.calls.append(request)
        return request

    async def send(self, request: Dict[str, Any], *, stream: bool = False) -> Any:
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    async def aclose(self) -> None:
        pass


def _client(
    *responses: _FakeResponse, config: Any = None
) -> tuple[BedrockAsyncClient, _FakeHTTPClient]:
    http_client = _FakeHTTPClient(*responses)
    client = BedrockAsyncClient(
        region_name="us-east-1",
        credentials=CREDENTIALS,
        config=config,
        http_client=http_client,
    )
    return client, http_client


def test_b64_encode_blobs_walks_nested_payloads() -> None:
    payload = {
        "messages": [{"content": [{"image": {"source": {"bytes": b"\x00\x01"}}}]}],
        "text": "unchanged",
    }
    encoded = _b64_encode_blobs(payload)
    source = encoded["messages"][0]["content"][0]["image"]["source"]
    assert source["bytes"] == "AAE="
    assert encoded["text"] == "unchanged"


def test_invoke_chunk_decodes_only_its_own_bytes_member() -> None:
    assert _decode_invoke_chunk({"bytes": "AAE="}) == {"bytes": b"\x00\x01"}
    # not base64 and not ours to touch
    assert _decode_invoke_chunk({"text": "AAE="}) == {"text": "AAE="}
    assert _decode_invoke_chunk({"bytes": 5}) == {"bytes": 5}


async def test_converse_signs_and_returns_response() -> None:
    response = _FakeResponse(
        json_body={"output": {"message": {"content": [{"text": "hi"}]}}},
        headers={"x-amzn-requestid": "req-1"},
    )
    client, http_client = _client(response)

    result = await client.converse(
        modelId="my-model",
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        inferenceConfig={"maxTokens": 5},
    )

    assert result["output"]["message"]["content"][0]["text"] == "hi"
    assert result["ResponseMetadata"]["RequestId"] == "req-1"

    call = http_client.calls[0]
    assert call["url"].endswith("/model/my-model/converse")
    assert call["headers"]["Authorization"].startswith(
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE"
    )
    assert call["headers"]["Accept"] == "application/json"
    assert json.loads(call["content"])["inferenceConfig"] == {"maxTokens": 5}


async def test_converse_encodes_image_bytes() -> None:
    client, http_client = _client(_FakeResponse(json_body={"output": {}}))

    await client.converse(
        modelId="my-model",
        messages=[
            {
                "role": "user",
                "content": [{"image": {"source": {"bytes": b"\xff\xd8"}}}],
            }
        ],
    )

    sent = json.loads(http_client.calls[0]["content"])
    assert sent["messages"][0]["content"][0]["image"]["source"]["bytes"] == "/9g="


async def test_model_id_is_url_escaped() -> None:
    client, http_client = _client(_FakeResponse(json_body={"output": {}}))

    await client.converse(modelId="arn:aws:bedrock:us-east-1::foo/bar", messages=[])

    assert (
        "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A%3Afoo%2Fbar/converse"
        in (http_client.calls[0]["url"])
    )


async def test_converse_stream_parses_event_frames() -> None:
    response = _FakeResponse(
        headers={"x-amzn-requestid": "req-2"},
        stream_chunks=[
            _event("messageStart", {"role": "assistant"}),
            _event("contentBlockDelta", {"delta": {"text": "he"}}),
            _event("contentBlockDelta", {"delta": {"text": "llo"}}),
            _event("messageStop", {"stopReason": "end_turn"}),
        ],
    )
    client, http_client = _client(response)

    result = await client.converse_stream(modelId="my-model", messages=[])
    events = [event async for event in result["stream"]]

    assert [next(iter(event)) for event in events] == [
        "messageStart",
        "contentBlockDelta",
        "contentBlockDelta",
        "messageStop",
    ]
    text = "".join(
        event["contentBlockDelta"]["delta"]["text"]
        for event in events
        if "contentBlockDelta" in event
    )
    assert text == "hello"
    assert http_client.calls[0]["headers"]["Accept"] == (
        "application/vnd.amazon.eventstream"
    )


async def test_converse_stream_frames_split_across_chunks() -> None:
    """A frame arriving in pieces must still parse, as it does over the wire."""
    frame = _event("contentBlockDelta", {"delta": {"text": "split"}})
    response = _FakeResponse(
        stream_chunks=[frame[:5], frame[5:12], frame[12:]],
    )
    client, _ = _client(response)

    result = await client.converse_stream(modelId="my-model", messages=[])
    events = [event async for event in result["stream"]]

    assert events == [{"contentBlockDelta": {"delta": {"text": "split"}}}]


async def test_converse_stream_raises_modeled_exception() -> None:
    response = _FakeResponse(
        stream_chunks=[
            _encode_event_frame(
                {
                    ":message-type": "exception",
                    ":exception-type": "ThrottlingException",
                },
                json.dumps({"message": "slow down"}).encode(),
            )
        ]
    )
    client, _ = _client(response)

    result = await client.converse_stream(modelId="my-model", messages=[])

    with pytest.raises(EventStreamError) as excinfo:
        [event async for event in result["stream"]]

    assert excinfo.value.response["Error"]["Code"] == "ThrottlingException"
    assert excinfo.value.response["Error"]["Message"] == "slow down"


async def test_converse_stream_closes_response() -> None:
    response = _FakeResponse(stream_chunks=[])
    client, _ = _client(response)

    result = await client.converse_stream(modelId="my-model", messages=[])
    await result["stream"].close()

    assert response.closed


async def test_error_status_becomes_client_error() -> None:
    response = _FakeResponse(
        status_code=400,
        json_body={"message": "bad input"},
        content=b'{"message": "bad input"}',
        headers={"x-amzn-errortype": "ValidationException:http://internal"},
    )
    client, _ = _client(response)

    with pytest.raises(ClientError) as excinfo:
        await client.converse(modelId="my-model", messages=[])

    error = excinfo.value.response["Error"]
    assert error["Code"] == "ValidationException"
    assert error["Message"] == "bad input"
    assert excinfo.value.response["ResponseMetadata"]["HTTPStatusCode"] == 400


async def test_error_without_json_body_still_raises() -> None:
    response = _FakeResponse(status_code=503, content=b"upstream unavailable")
    client, _ = _client(response)

    # 503 is retryable, so skip the real backoff
    with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
        with pytest.raises(ClientError):
            await client.converse(modelId="my-model", messages=[])


async def test_invoke_model_body_reads_like_boto3() -> None:
    response = _FakeResponse(
        content=b'{"embedding": [0.5]}',
        headers={"content-type": "application/json"},
    )
    client, http_client = _client(response)

    result = await client.invoke_model(
        modelId="titan", body='{"inputText": "hi"}', accept="application/json"
    )

    assert json.loads(result["body"].read())["embedding"] == [0.5]
    assert http_client.calls[0]["content"] == b'{"inputText": "hi"}'


async def test_invoke_model_accepts_bytes_body() -> None:
    client, http_client = _client(_FakeResponse(content=b"{}"))

    await client.invoke_model(modelId="titan", body=b'{"inputText": "hi"}')

    assert http_client.calls[0]["content"] == b'{"inputText": "hi"}'


async def test_missing_region_is_reported() -> None:
    client = BedrockAsyncClient(
        credentials=CREDENTIALS, http_client=_FakeHTTPClient(_FakeResponse())
    )

    with pytest.raises(ValueError, match="region_name"):
        await client.converse(modelId="my-model", messages=[])


async def test_endpoint_url_overrides_default_host() -> None:
    client, http_client = _client(_FakeResponse(json_body={}))
    client._endpoint_url = "https://gateway.internal/bedrock"

    await client.converse(modelId="my-model", messages=[])

    assert http_client.calls[0]["url"].startswith(
        "https://gateway.internal/bedrock/model/"
    )


async def test_close_only_closes_owned_http_client() -> None:
    supplied = _FakeHTTPClient(_FakeResponse())
    with mock.patch.object(supplied, "aclose", new=mock.AsyncMock()) as aclose:
        client = BedrockAsyncClient(region_name="us-east-1", http_client=supplied)
        await client.close()
        aclose.assert_not_awaited()


def test_pool_size_follows_botocore_config() -> None:
    config = mock.MagicMock(max_pool_connections=64, connect_timeout=5, read_timeout=30)
    client = BedrockAsyncClient(region_name="us-east-1", config=config)

    with mock.patch("httpx.AsyncClient") as async_client:
        client._build_http_client()

    limits = async_client.call_args.kwargs["limits"]
    timeout = async_client.call_args.kwargs["timeout"]
    assert limits.max_connections == 64
    assert limits.max_keepalive_connections == 64
    assert timeout.connect == 5
    assert timeout.read == 30


def test_pool_size_defaults_to_botocore_default() -> None:
    """Left unset, the pool matches botocore's default of 10."""
    client = BedrockAsyncClient(region_name="us-east-1")

    with mock.patch("httpx.AsyncClient") as async_client:
        client._build_http_client()

    assert async_client.call_args.kwargs["limits"].max_connections == 10


# --- regressions found in review -------------------------------------------


async def test_invoke_model_sends_guardrail_headers() -> None:
    """Guardrails are header-bound; dropping them disables them silently."""
    client, http_client = _client(_FakeResponse(content=b"{}"))

    await client.invoke_model(
        modelId="m",
        body="{}",
        guardrailIdentifier="gr-1",
        guardrailVersion="3",
        trace="ENABLED",
        serviceTier="flex",
        accept="application/json",
        contentType="application/json",
    )

    headers = http_client.calls[0]["headers"]
    assert headers["X-Amzn-Bedrock-GuardrailIdentifier"] == "gr-1"
    assert headers["X-Amzn-Bedrock-GuardrailVersion"] == "3"
    assert headers["X-Amzn-Bedrock-Trace"] == "ENABLED"
    assert headers["X-Amzn-Bedrock-Service-Tier"] == "flex"
    # header-bound members must not also leak into the body
    assert http_client.calls[0]["content"] == b"{}"


async def test_invoke_stream_uses_the_bedrock_accept_header() -> None:
    """`Accept` itself selects the event stream, so `accept` moves aside."""
    client, http_client = _client(_FakeResponse(stream_chunks=[]))

    await client.invoke_model_with_response_stream(
        modelId="m", body="{}", accept="application/json"
    )

    headers = http_client.calls[0]["headers"]
    assert headers["X-Amzn-Bedrock-Accept"] == "application/json"
    assert headers["Accept"] == "application/vnd.amazon.eventstream"


async def test_guardrail_headers_are_signed() -> None:
    """An unsigned header would be rejected by AWS, so it must be in the scope."""
    client, http_client = _client(_FakeResponse(content=b"{}"))

    await client.invoke_model(modelId="m", body="{}", guardrailIdentifier="gr-1")

    authorization = http_client.calls[0]["headers"]["Authorization"]
    signed_headers = authorization.split("SignedHeaders=")[1].split(",")[0]
    assert "x-amzn-bedrock-guardrailidentifier" in signed_headers


async def test_throttling_is_retried() -> None:
    throttled = _FakeResponse(
        status_code=429,
        json_body={"message": "slow down"},
        content=b'{"message": "slow down"}',
        headers={"x-amzn-errortype": "ThrottlingException"},
    )
    ok = _FakeResponse(json_body={"output": {}})
    client, http_client = _client(
        throttled, ok, config=mock.MagicMock(retries={"max_attempts": 3})
    )

    with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
        result = await client.converse(modelId="m", messages=[])

    assert result["output"] == {}
    assert len(http_client.calls) == 2


async def test_retries_are_exhausted_then_raise() -> None:
    throttled = _FakeResponse(
        status_code=429,
        json_body={"message": "slow down"},
        content=b'{"message": "slow down"}',
        headers={"x-amzn-errortype": "ThrottlingException"},
    )
    client, http_client = _client(
        throttled, config=mock.MagicMock(retries={"max_attempts": 2})
    )

    with mock.patch("asyncio.sleep", new=mock.AsyncMock()):
        with pytest.raises(ClientError):
            await client.converse(modelId="m", messages=[])

    assert len(http_client.calls) == 2


async def test_validation_errors_are_not_retried() -> None:
    bad = _FakeResponse(
        status_code=400,
        json_body={"message": "bad"},
        content=b'{"message": "bad"}',
        headers={"x-amzn-errortype": "ValidationException"},
    )
    client, http_client = _client(
        bad, config=mock.MagicMock(retries={"max_attempts": 5})
    )

    with pytest.raises(ClientError):
        await client.converse(modelId="m", messages=[])

    assert len(http_client.calls) == 1


async def test_stream_error_frame_raises_rather_than_truncating() -> None:
    """An `error` frame carries no `:event-type` and must not be skipped."""
    response = _FakeResponse(
        stream_chunks=[
            _event("contentBlockDelta", {"delta": {"text": "partial"}}),
            _encode_event_frame(
                {
                    ":message-type": "error",
                    ":error-code": "InternalServerException",
                    ":error-message": "boom",
                },
                b"",
            ),
        ]
    )
    client, _ = _client(response)

    result = await client.converse_stream(modelId="m", messages=[])

    with pytest.raises(EventStreamError) as excinfo:
        [event async for event in result["stream"]]

    assert excinfo.value.response["Error"]["Code"] == "InternalServerException"


async def test_truncated_stream_is_reported() -> None:
    frame = _event("contentBlockDelta", {"delta": {"text": "hi"}})
    response = _FakeResponse(stream_chunks=[frame, frame[:12]])
    client, _ = _client(response)

    result = await client.converse_stream(modelId="m", messages=[])

    with pytest.raises(EventStreamError, match="truncated"):
        [event async for event in result["stream"]]


async def test_tool_arguments_named_bytes_are_left_alone() -> None:
    """`toolUse.input` is a free-form document, not a blob."""
    response = _FakeResponse(
        json_body={
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"input": {"path": "/a", "bytes": "hello world"}}}
                    ]
                }
            }
        }
    )
    client, _ = _client(response)

    result = await client.converse(modelId="m", messages=[])

    tool_input = result["output"]["message"]["content"][0]["toolUse"]["input"]
    assert tool_input == {"path": "/a", "bytes": "hello world"}


async def test_invoke_stream_chunk_bytes_are_decoded() -> None:
    """The Invoke stream really does base64 its `chunk.bytes` member."""
    payload = base64.b64encode(b'{"completion": "hi"}').decode()
    response = _FakeResponse(stream_chunks=[_event("chunk", {"bytes": payload})])
    client, _ = _client(response)

    result = await client.invoke_model_with_response_stream(modelId="m", body="{}")
    events = [event async for event in result["body"]]

    assert events[0]["chunk"]["bytes"] == b'{"completion": "hi"}'


async def test_abandoned_stream_releases_the_response() -> None:
    response = _FakeResponse(
        stream_chunks=[
            _event("contentBlockDelta", {"delta": {"text": str(i)}}) for i in range(5)
        ]
    )
    client, _ = _client(response)

    result = await client.converse_stream(modelId="m", messages=[])
    stream = result["stream"].__aiter__()
    await stream.asend(None)
    await stream.aclose()

    assert response.closed


async def test_non_dict_error_body_still_raises_client_error() -> None:
    response = _FakeResponse(
        status_code=400, json_body=None, content=b'"just a string"'
    )
    response._json_body = "just a string"  # type: ignore[assignment]
    client, _ = _client(response)

    with pytest.raises(ClientError):
        await client.converse(modelId="m", messages=[])


async def test_credentials_are_resolved_off_the_event_loop() -> None:
    """A cold SSO/STS resolution takes seconds and must not stall the loop.

    Asserts the observable property — that other coroutines keep running while
    credentials resolve — rather than that a particular helper was called.
    """
    slow_resolve_seconds = 0.3
    frozen = CREDENTIALS.get_frozen_credentials()

    class _SlowCredentials:
        def get_frozen_credentials(self) -> Any:
            time.sleep(slow_resolve_seconds)  # blocking, as botocore's are
            return frozen

    http_client = _FakeHTTPClient(_FakeResponse(json_body={}))
    client = BedrockAsyncClient(
        region_name="us-east-1",
        credentials=_SlowCredentials(),
        http_client=http_client,
    )

    ticks = 0
    stop = False

    async def heartbeat() -> None:
        nonlocal ticks
        while not stop:
            await asyncio.sleep(0.01)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    await client.converse(modelId="m", messages=[])
    stop = True
    await beat

    # A loop frozen for the whole resolution would tick 0-1 times.
    assert ticks > 5, f"event loop stalled during credential resolution ({ticks})"


async def test_closed_client_refuses_further_requests() -> None:
    client, _ = _client(_FakeResponse(json_body={}))
    await client.converse(modelId="m", messages=[])
    await client.close()

    with pytest.raises(RuntimeError, match="has been closed"):
        await client.converse(modelId="m", messages=[])


def test_cross_loop_use_raises_instead_of_hanging() -> None:
    """Sharing one pool across event loops otherwise deadlocks silently."""
    client, _ = _client(_FakeResponse(json_body={}), _FakeResponse(json_body={}))

    asyncio.run(client.converse(modelId="m", messages=[]))

    with pytest.raises(RuntimeError, match="different event loop"):
        asyncio.run(client.converse(modelId="m", messages=[]))
