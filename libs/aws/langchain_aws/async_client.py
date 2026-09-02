"""Native async transport for the Bedrock runtime data plane.

`boto3` is synchronous, so LangChain's async methods have historically bridged to
it with `run_in_executor`, capping concurrency at the size of the thread pool.
`aiobotocore` solves that but pins a narrow, trailing `botocore` window that
conflicts with this package's `boto3` floor, so it cannot be depended on here.

This module instead signs requests with `botocore` and sends them with `httpx`,
the same approach the Anthropic SDK uses for its Bedrock client. `ChatAnthropicBedrock`
already gets native async that way, but only for Anthropic models; this covers the
rest of the Bedrock surface. Only the four runtime operations LangChain needs are
implemented, and their signatures match the `aiobotocore` client's so either can be
supplied as `async_client`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import time
import weakref
from types import TracebackType
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple, Type
from urllib.parse import quote

from botocore.awsrequest import AWSRequest
from botocore.eventstream import EventStreamBuffer
from botocore.exceptions import ClientError, EventStreamError

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_POOL_CONNECTIONS = 10
_DEFAULT_MAX_ATTEMPTS = 5
_CREDENTIAL_CACHE_SECONDS = 60.0
_MAX_BACKOFF_SECONDS = 20.0

# Members of the Invoke operations that travel as headers rather than in the JSON
# body, taken from the `bedrock-runtime` service model. Converse and ConverseStream
# have no header-bound members, so their kwargs all belong in the body. Dropping
# these would silently disable guardrails, so they are mapped explicitly.
_INVOKE_HEADER_MEMBERS: Mapping[str, str] = {
    "contentType": "Content-Type",
    "accept": "Accept",
    "trace": "X-Amzn-Bedrock-Trace",
    "guardrailIdentifier": "X-Amzn-Bedrock-GuardrailIdentifier",
    "guardrailVersion": "X-Amzn-Bedrock-GuardrailVersion",
    "performanceConfigLatency": "X-Amzn-Bedrock-PerformanceConfig-Latency",
    "serviceTier": "X-Amzn-Bedrock-Service-Tier",
    "requestMetadata": "X-Amzn-Bedrock-Request-Metadata",
}

# The streaming Invoke carries the accept header under a different name, because
# `Accept` itself selects the event-stream content type.
_INVOKE_STREAM_HEADER_MEMBERS: Mapping[str, str] = {
    **_INVOKE_HEADER_MEMBERS,
    "accept": "X-Amzn-Bedrock-Accept",
}

_STREAM_OPERATIONS = frozenset({"ConverseStream", "InvokeModelWithResponseStream"})

# Error codes botocore's retry policy treats as transient.
_RETRYABLE_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "Throttling",
        "TooManyRequestsException",
        "ProvisionedThroughputExceededException",
        "RequestLimitExceeded",
        "ServiceUnavailableException",
        "ServiceUnavailable",
        "InternalServerException",
        "InternalFailure",
        "ModelTimeoutException",
        "ModelNotReadyException",
    }
)


def _warn_unclosed(where: str) -> None:
    """Warn that a client was garbage collected without being closed.

    An unclosed `httpx` pool holds its sockets until collection, so a long-lived
    process that leaks these will eventually exhaust its file descriptors.

    Args:
        where: The endpoint the leaked client was pointed at.
    """
    logger.warning(
        "BedrockAsyncClient for %s was garbage collected without being closed; "
        "its connections leaked. Call `await client.close()`, use the client as "
        "an async context manager, or `await model.aclose()`.",
        where,
    )


def _b64_encode_blobs(value: Any) -> Any:
    """Base64-encode every `bytes` leaf in a request payload.

    `botocore` does this during serialization for members typed as blobs. Sending
    JSON directly means doing it here, so that callers can keep passing raw
    `bytes` exactly as they would to `boto3`.

    Args:
        value: An arbitrarily nested request payload.

    Returns:
        The payload with `bytes` replaced by base64 `str`.
    """
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")
    if isinstance(value, dict):
        return {k: _b64_encode_blobs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_b64_encode_blobs(v) for v in value]
    return value


def _decode_invoke_chunk(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Decode the base64 `bytes` member of an Invoke stream chunk.

    Only this one shape is decoded. Walking a whole response for members named
    `bytes` would also rewrite model-authored data, because `toolUse.input` and
    `additionalModelResponseFields` are free-form documents that may contain a
    key of that name.

    Args:
        payload: The decoded JSON payload of one `chunk` event.

    Returns:
        The payload with `bytes` decoded, when present.
    """
    raw = payload.get("bytes")
    if not isinstance(raw, str):
        return payload
    return {**payload, "bytes": base64.b64decode(raw)}


class _AsyncEventStream:
    """Async iterator over an `application/vnd.amazon.eventstream` body.

    Wraps `botocore`'s `EventStreamBuffer` so the binary framing is parsed by
    `botocore` rather than reimplemented here. Yields events shaped like the ones
    `boto3` produces, e.g. `{"contentBlockDelta": {...}}`.
    """

    def __init__(self, response: Any, operation: str) -> None:
        self._response = response
        self._operation = operation

    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        buffer = EventStreamBuffer()
        # Closed here as well as by the callers, so that abandoning the iterator
        # releases the connection even when nobody wraps it in try/finally.
        try:
            async for chunk in self._response.aiter_bytes():
                buffer.add_data(chunk)
                for event in buffer:
                    parsed = self._parse_event(event)
                    if parsed is not None:
                        yield parsed
            self._check_complete(buffer)
        finally:
            await self._response.aclose()

    def _check_complete(self, buffer: Any) -> None:
        """Fail loudly when the stream ended mid-frame.

        A truncated response would otherwise look like a clean, short completion.

        Args:
            buffer: The event stream buffer after the body was exhausted.

        Raises:
            EventStreamError: If undecoded bytes remain in the buffer.
        """
        leftover = getattr(buffer, "_data", b"")
        if not leftover:
            return
        msg = (
            f"Stream ended with {len(leftover)} undecoded bytes; the response "
            "was truncated."
        )
        raise EventStreamError(
            {"Error": {"Code": "IncompleteStream", "Message": msg}}, self._operation
        )

    def _parse_event(self, event: Any) -> Optional[Dict[str, Any]]:
        """Shape one raw event-stream message like a `boto3` stream event.

        Args:
            event: A message from `botocore`'s event stream parser.

        Returns:
            A single-key event dict, or `None` for framing-only messages.

        Raises:
            EventStreamError: If the stream carries a service error or a modeled
                exception, matching what `boto3` raises.
        """
        headers = event.headers
        message_type = headers.get(":message-type")

        if message_type in ("exception", "error"):
            raise self._stream_error(event, headers, message_type)

        event_type = headers.get(":event-type")
        if message_type != "event" or event_type is None:
            return None

        payload = json.loads(event.payload.decode("utf-8") or "{}")
        if event_type == "chunk":
            payload = _decode_invoke_chunk(payload)
        return {event_type: payload}

    def _stream_error(
        self, event: Any, headers: Mapping[str, Any], message_type: str
    ) -> EventStreamError:
        """Build the error raised for an `exception` or `error` frame."""
        if message_type == "error":
            code = headers.get(":error-code", "StreamError")
            message = headers.get(":error-message", "")
        else:
            code = headers.get(":exception-type", "UnknownException")
            try:
                message = json.loads(event.payload.decode("utf-8") or "{}").get(
                    "message", ""
                )
            except ValueError:
                message = event.payload.decode("utf-8", "replace")
        return EventStreamError(
            {"Error": {"Code": code, "Message": message}}, self._operation
        )

    async def close(self) -> None:
        """Release the underlying HTTP response."""
        await self._response.aclose()


class BedrockAsyncClient:
    """An async Bedrock runtime client backed by `httpx`.

    Implements the subset of the `bedrock-runtime` API that LangChain's chat
    models and embeddings use, with the same method names and payload shapes as
    the `boto3`/`aiobotocore` clients so it is a drop-in for `async_client`.

    Credentials are resolved through `botocore` off the event loop; each request
    is signed with SigV4 at send time and retried on transient errors according
    to the `retries` setting of the supplied config.

    Example:
        ```python
        from langchain_aws import ChatBedrockConverse
        from langchain_aws.async_client import BedrockAsyncClient

        async with BedrockAsyncClient(region_name="us-east-1") as client:
            model = ChatBedrockConverse(model="...", async_client=client)
            await model.ainvoke("hi")
        ```

    !!! note
        Concurrency is bounded by the underlying `httpx` connection pool, sized
        from `max_pool_connections` on the supplied `config`.

    !!! warning
        Bearer-token authentication (a Bedrock API key) is not supported; this
        client signs with SigV4 only.
    """

    def __init__(
        self,
        *,
        region_name: Optional[str] = None,
        credentials: Any = None,
        endpoint_url: Optional[str] = None,
        config: Any = None,
        http_client: Any = None,
    ) -> None:
        """Build a client.

        Args:
            region_name: AWS region. Required unless discoverable from the
                environment or the supplied credentials session.
            credentials: A `botocore` credentials object. Defaults to the
                ambient credential chain.
            endpoint_url: Override for the Bedrock runtime endpoint.
            config: A `botocore` `Config`. `max_pool_connections`,
                `connect_timeout`, `read_timeout` and `retries["max_attempts"]`
                are honored. `retries["mode"]` is not: retries here always use
                jittered exponential backoff and share no quota with botocore's
                adaptive mode. Other members are ignored.
            http_client: An `httpx.AsyncClient` to use instead of one built
                here. The caller keeps ownership of its lifecycle.
        """
        self._region_name = region_name
        self._credentials = credentials
        self._endpoint_url = endpoint_url
        self._config = config
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._closed = False
        self._frozen: Any = None
        self._frozen_at = 0.0
        self._client_lock = asyncio.Lock()
        self._loop: Any = None
        self._finalizer: Any = None

    @property
    def _base_url(self) -> str:
        if self._endpoint_url:
            return self._endpoint_url.rstrip("/")
        return f"https://bedrock-runtime.{self._region_name}.amazonaws.com"

    @property
    def _max_attempts(self) -> int:
        """Total attempts per request, from the botocore retry config."""
        retries = getattr(self._config, "retries", None) or {}
        try:
            return max(1, int(retries.get("max_attempts", _DEFAULT_MAX_ATTEMPTS)))
        except (TypeError, ValueError):
            return _DEFAULT_MAX_ATTEMPTS

    def _build_http_client(self) -> Any:
        """Create an `httpx.AsyncClient` sized from the botocore config."""
        import httpx

        max_connections = (
            getattr(self._config, "max_pool_connections", None)
            or _DEFAULT_POOL_CONNECTIONS
        )
        connect_timeout = getattr(self._config, "connect_timeout", None)
        read_timeout = getattr(self._config, "read_timeout", None)
        timeout = httpx.Timeout(
            _DEFAULT_TIMEOUT_SECONDS,
            connect=connect_timeout or _DEFAULT_TIMEOUT_SECONDS,
            read=read_timeout or _DEFAULT_TIMEOUT_SECONDS,
        )
        return httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )

    async def __aenter__(self) -> "BedrockAsyncClient":
        await self._require_http_client()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client, if this instance created it.

        The client is not reusable afterwards; a later request raises rather
        than quietly opening a second connection pool.
        """
        self._closed = True
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self._http_client is not None and self._owns_http_client:
            await self._http_client.aclose()
            self._http_client = None

    def _check_loop(self) -> None:
        """Reject use from an event loop other than the one that opened the pool.

        `httpx` binds its connection pool to the loop that first drives it.
        Reusing one client across loops — a cached model instance driven by
        `asyncio.run` from several threads, for example — otherwise hangs
        forever on I/O that will never be polled, with nothing raised.

        Raises:
            RuntimeError: If called from a different running event loop.
        """
        running = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = running
        elif self._loop is not running:
            msg = (
                "This BedrockAsyncClient is bound to a different event loop. Its "
                "connection pool cannot be shared across loops; build one client "
                "per loop, or per `asyncio.run` call."
            )
            raise RuntimeError(msg)

    async def _require_http_client(self) -> Any:
        """Return the HTTP client, creating it at most once.

        Raises:
            RuntimeError: If the client is closed or bound to another loop.
        """
        if self._closed:
            msg = "This BedrockAsyncClient has been closed."
            raise RuntimeError(msg)
        self._check_loop()
        if self._http_client is not None:
            return self._http_client
        async with self._client_lock:
            if self._http_client is None:
                self._http_client = self._build_http_client()
                self._finalizer = weakref.finalize(
                    self, _warn_unclosed, repr(self._base_url)
                )
        return self._http_client

    def _resolve_credentials(self) -> Any:
        """Resolve credentials from the ambient chain. Blocking.

        Raises:
            ValueError: If no credentials can be loaded.
        """
        if self._credentials is None:
            from botocore.session import get_session

            session = get_session()
            self._credentials = session.get_credentials()
            if self._region_name is None:
                self._region_name = session.get_config_variable("region")
        if self._credentials is None:
            msg = (
                "Could not load AWS credentials. Configure them the same way "
                "you would for boto3, or pass `credentials` explicitly."
            )
            raise ValueError(msg)
        return self._credentials.get_frozen_credentials()

    async def _frozen_credentials(self) -> Any:
        """Return frozen credentials, resolving them off the event loop.

        Resolution and refresh are synchronous `botocore` calls that may reach
        IMDS, SSO or STS over the network — seconds on a cold start — so they
        must not run inline on the loop. The result is cached briefly, well
        inside botocore's own advisory refresh window.
        """
        now = time.monotonic()
        if self._frozen is not None and now - self._frozen_at < (
            _CREDENTIAL_CACHE_SECONDS
        ):
            return self._frozen
        self._frozen = await asyncio.to_thread(self._resolve_credentials)
        self._frozen_at = time.monotonic()
        return self._frozen

    def _require_region(self) -> str:
        if not self._region_name:
            msg = (
                "`region_name` must be set to sign Bedrock requests. Pass it "
                "explicitly or configure AWS_REGION."
            )
            raise ValueError(msg)
        return self._region_name

    async def _sign(
        self, url: str, body: bytes, operation: str, extra_headers: Dict[str, str]
    ) -> Dict[str, str]:
        """Sign a request with SigV4 and return the headers to send.

        Args:
            url: The fully qualified request URL.
            body: The serialized request body.
            operation: The Bedrock operation name.
            extra_headers: Header-bound request members, signed alongside.

        Returns:
            The signed request headers.
        """
        from botocore.auth import SigV4Auth

        headers = {
            "Content-Type": "application/json",
            "Accept": (
                "application/vnd.amazon.eventstream"
                if operation in _STREAM_OPERATIONS
                else "application/json"
            ),
            **extra_headers,
        }
        request = AWSRequest(method="POST", url=url, data=body, headers=headers)
        credentials = await self._frozen_credentials()
        SigV4Auth(credentials, "bedrock", self._require_region()).add_auth(request)
        return dict(request.headers)

    def _url_for(self, model_id: str, path_suffix: str) -> str:
        return f"{self._base_url}/model/{quote(model_id, safe='')}/{path_suffix}"

    @staticmethod
    def _error_from_response(response: Any, operation: str) -> ClientError:
        """Build the `ClientError` a boto3 caller expects from a failed response."""
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        code = response.headers.get("x-amzn-errortype", "").split(":")[0] or (
            str(payload.get("__type", "")).split("#")[-1] or "UnknownError"
        )
        message = payload.get("message") or payload.get("Message") or response.text
        return ClientError(
            {
                "Error": {"Code": code, "Message": message},
                "ResponseMetadata": BedrockAsyncClient._response_metadata(response),
            },
            operation,
        )

    @staticmethod
    def _response_metadata(response: Any) -> Any:
        """Shape response metadata the way botocore does.

        Callers read token counts out of `HTTPHeaders`, and `ClientError`
        expects the full set of keys.

        Args:
            response: The `httpx` response.

        Returns:
            The `ResponseMetadata` member of a `boto3`-shaped response.
        """
        return {
            "RequestId": response.headers.get("x-amzn-requestid", ""),
            "HostId": "",
            "HTTPStatusCode": response.status_code,
            "HTTPHeaders": dict(response.headers),
            "RetryAttempts": 0,
        }

    @staticmethod
    def _is_retryable(error: ClientError) -> bool:
        """Whether botocore would treat this error as transient."""
        code = error.response["Error"]["Code"]
        status = error.response["ResponseMetadata"]["HTTPStatusCode"]
        return code in _RETRYABLE_ERROR_CODES or status == 429 or status >= 500

    @staticmethod
    async def _sleep_before_retry(attempt: int) -> None:
        """Back off with full jitter, as botocore's standard mode does."""
        delay = min(_MAX_BACKOFF_SECONDS, 0.5 * (2**attempt))
        await asyncio.sleep(delay * (0.5 + random.random() / 2))

    @staticmethod
    def _split_header_members(
        kwargs: Dict[str, Any], header_map: Mapping[str, str]
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Separate header-bound request members from body members.

        Args:
            kwargs: The request members supplied by the caller.
            header_map: Member name to header name for this operation.

        Returns:
            A tuple of headers to send and the remaining body members.
        """
        headers = {
            header_map[name]: str(value)
            for name, value in kwargs.items()
            if name in header_map and value is not None
        }
        body = {k: v for k, v in kwargs.items() if k not in header_map}
        return headers, body

    async def _send(
        self,
        url: str,
        body: bytes,
        operation: str,
        headers: Dict[str, str],
        *,
        stream: bool,
    ) -> Any:
        """Send a signed request, retrying transient failures.

        Args:
            url: The request URL.
            body: The serialized request body.
            operation: The Bedrock operation name.
            headers: Header-bound request members.
            stream: Whether to leave the response body unread for streaming.

        Returns:
            The `httpx` response.

        Raises:
            ClientError: If the request fails and is not retryable, or if
                retries are exhausted.
        """
        import httpx

        client = await self._require_http_client()
        for attempt in range(self._max_attempts):
            signed = await self._sign(url, body, operation, headers)
            try:
                request = client.build_request(
                    "POST", url, content=body, headers=signed
                )
                response = await client.send(request, stream=stream)
            except httpx.TransportError:
                if attempt + 1 >= self._max_attempts:
                    raise
                await self._sleep_before_retry(attempt)
                continue

            if response.status_code < 400:
                return response

            await response.aread()
            await response.aclose()
            error = self._error_from_response(response, operation)
            if not self._is_retryable(error) or attempt + 1 >= self._max_attempts:
                raise error
            await self._sleep_before_retry(attempt)

        msg = f"{operation} exhausted {self._max_attempts} attempts."
        raise RuntimeError(msg)

    async def _post_json(
        self, url: str, payload: Dict[str, Any], operation: str, headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Send a signed JSON request and return the decoded response."""
        body = json.dumps(_b64_encode_blobs(payload)).encode("utf-8")
        response = await self._send(url, body, operation, headers, stream=False)
        decoded = response.json()
        decoded["ResponseMetadata"] = self._response_metadata(response)
        return decoded

    async def converse(
        self,
        *,
        modelId: str,  # noqa: N803
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Invoke the Converse API.

        Args:
            modelId: The Bedrock model or inference profile ID.
            messages: Converse-formatted messages.
            system: Converse-formatted system blocks.
            **kwargs: Additional Converse request members, all body-bound.

        Returns:
            The Converse response.
        """
        payload = {"messages": messages, "system": system or [], **kwargs}
        return await self._post_json(
            self._url_for(modelId, "converse"), payload, "Converse", {}
        )

    async def converse_stream(
        self,
        *,
        modelId: str,  # noqa: N803
        messages: List[Dict[str, Any]],
        system: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Invoke the ConverseStream API.

        Args:
            modelId: The Bedrock model or inference profile ID.
            messages: Converse-formatted messages.
            system: Converse-formatted system blocks.
            **kwargs: Additional ConverseStream request members, all body-bound.

        Returns:
            A response whose `"stream"` member is an async iterator of events.
        """
        payload = {"messages": messages, "system": system or [], **kwargs}
        body = json.dumps(_b64_encode_blobs(payload)).encode("utf-8")
        url = self._url_for(modelId, "converse-stream")
        response = await self._send(url, body, "ConverseStream", {}, stream=True)
        return {
            "stream": _AsyncEventStream(response, "ConverseStream"),
            "ResponseMetadata": self._response_metadata(response),
        }

    @staticmethod
    def _invoke_body(body: Any) -> bytes:
        return body.encode("utf-8") if isinstance(body, str) else body

    async def invoke_model(
        self,
        *,
        modelId: str,  # noqa: N803
        body: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Invoke a model with a provider-native request body.

        Args:
            modelId: The Bedrock model ID.
            body: A JSON string or bytes holding the provider-native request.
            **kwargs: Header-bound members such as `accept`, `contentType`,
                `guardrailIdentifier`, `guardrailVersion`, `trace` and
                `serviceTier`.

        Returns:
            A response whose `"body"` member reads like the `boto3` streaming
            body, i.e. exposes `read()`.
        """
        headers, _ = self._split_header_members(kwargs, _INVOKE_HEADER_MEMBERS)
        response = await self._send(
            self._url_for(modelId, "invoke"),
            self._invoke_body(body),
            "InvokeModel",
            headers,
            stream=False,
        )
        return {
            "body": _BytesBody(response.content),
            "contentType": response.headers.get("content-type", "application/json"),
            "ResponseMetadata": self._response_metadata(response),
        }

    async def invoke_model_with_response_stream(
        self,
        *,
        modelId: str,  # noqa: N803
        body: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Invoke a model with a provider-native body and stream the response.

        Args:
            modelId: The Bedrock model ID.
            body: A JSON string or bytes holding the provider-native request.
            **kwargs: Header-bound members, as for `invoke_model`.

        Returns:
            A response whose `"body"` member is an async iterator of events.
        """
        headers, _ = self._split_header_members(kwargs, _INVOKE_STREAM_HEADER_MEMBERS)
        operation = "InvokeModelWithResponseStream"
        response = await self._send(
            self._url_for(modelId, "invoke-with-response-stream"),
            self._invoke_body(body),
            operation,
            headers,
            stream=True,
        )
        return {
            "body": _AsyncEventStream(response, operation),
            "contentType": response.headers.get(
                "x-amzn-bedrock-content-type", "application/json"
            ),
            "ResponseMetadata": self._response_metadata(response),
        }


class _BytesBody:
    """A minimal stand-in for `botocore`'s streaming body."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        """Return the full response body."""
        return self._data
