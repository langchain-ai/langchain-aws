"""Test the native async paths of `BedrockEmbeddings`."""

import json
import os
from typing import Any, Dict, List
from unittest import mock

import pytest

from langchain_aws.embeddings.bedrock import BedrockEmbeddings

TITAN_MODEL = "amazon.titan-embed-text-v1"
COHERE_MODEL = "cohere.embed-english-v3"


class _Body:
    """Stands in for botocore's streaming response body."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._payload


def _async_client(payloads: List[Dict[str, Any]]) -> Any:
    """Build an async client returning one payload per `invoke_model` call."""
    client = mock.AsyncMock()
    client.invoke_model.side_effect = [
        {"body": _Body(payload), "ResponseMetadata": {"RequestId": "req"}}
        for payload in payloads
    ]
    return client


def _embeddings(model_id: str = TITAN_MODEL, **kwargs: Any) -> BedrockEmbeddings:
    kwargs.setdefault("client", mock.MagicMock())
    return BedrockEmbeddings(model_id=model_id, region_name="us-west-2", **kwargs)


async def test_aembed_query_uses_async_client() -> None:
    async_client = _async_client([{"embedding": [0.1, 0.2]}])
    embeddings = _embeddings(async_client=async_client)

    result = await embeddings.aembed_query("hello")

    assert result == [0.1, 0.2]
    async_client.invoke_model.assert_awaited_once()
    embeddings.client.invoke_model.assert_not_called()


async def test_aembed_query_falls_back_to_executor() -> None:
    sync_client = mock.MagicMock()
    sync_client.invoke_model.return_value = {"body": _Body({"embedding": [0.3]})}
    embeddings = _embeddings(client=sync_client)

    result = await embeddings.aembed_query("hello")

    assert result == [0.3]
    sync_client.invoke_model.assert_called_once()


async def test_aembed_documents_fans_out_concurrently() -> None:
    async_client = _async_client(
        [{"embedding": [0.1]}, {"embedding": [0.2]}, {"embedding": [0.3]}]
    )
    embeddings = _embeddings(async_client=async_client)

    result = await embeddings.aembed_documents(["a", "b", "c"])

    assert result == [[0.1], [0.2], [0.3]]
    assert async_client.invoke_model.await_count == 3


async def test_aembed_query_normalizes_when_requested() -> None:
    async_client = _async_client([{"embedding": [3.0, 4.0]}])
    embeddings = _embeddings(async_client=async_client, normalize=True)

    result = await embeddings.aembed_query("hello")

    assert result == pytest.approx([0.6, 0.8])


async def test_aembed_query_uses_cohere_search_query_input_type() -> None:
    async_client = _async_client([{"embeddings": [[0.1]]}])
    embeddings = _embeddings(COHERE_MODEL, async_client=async_client)

    await embeddings.aembed_query("hello")

    sent = json.loads(async_client.invoke_model.await_args.kwargs["body"])
    assert sent["input_type"] == "search_query"


async def test_aembed_documents_uses_cohere_multi_endpoint() -> None:
    async_client = _async_client([{"embeddings": [[0.1], [0.2]]}])
    embeddings = _embeddings(COHERE_MODEL, async_client=async_client)

    result = await embeddings.aembed_documents(["a", "b"])

    assert result == [[0.1], [0.2]]
    # one batched call, not one per text
    async_client.invoke_model.assert_awaited_once()
    sent = json.loads(async_client.invoke_model.await_args.kwargs["body"])
    assert sent["texts"] == ["a", "b"]
    assert sent["input_type"] == "search_document"


async def test_aembed_documents_handles_cohere_v4_schema() -> None:
    async_client = _async_client([{"embeddings": {"float": [[0.1], [0.2]]}}])
    embeddings = _embeddings("us.cohere.embed-v4:0", async_client=async_client)

    result = await embeddings.aembed_documents(["a", "b"])

    assert result == [[0.1], [0.2]]


async def test_aembed_documents_cohere_falls_back_to_executor() -> None:
    sync_client = mock.MagicMock()
    sync_client.invoke_model.return_value = {
        "body": _Body({"embeddings": [[0.1], [0.2]]})
    }
    embeddings = _embeddings(COHERE_MODEL, client=sync_client)

    result = await embeddings.aembed_documents(["a", "b"])

    assert result == [[0.1], [0.2]]
    sync_client.invoke_model.assert_called_once()


async def test_missing_embedding_is_reported() -> None:
    async_client = _async_client([{}])
    embeddings = _embeddings(async_client=async_client)

    with pytest.raises(ValueError, match="No embedding returned from model"):
        await embeddings.aembed_query("hello")


async def test_async_and_sync_send_identical_requests() -> None:
    sync_client = mock.MagicMock()
    sync_client.invoke_model.return_value = {"body": _Body({"embedding": [0.1]})}
    async_client = _async_client([{"embedding": [0.1]}])
    embeddings = _embeddings(client=sync_client, async_client=async_client)

    embeddings.embed_query("hello")
    await embeddings.aembed_query("hello")

    assert sync_client.invoke_model.call_args.kwargs == (
        async_client.invoke_model.await_args.kwargs
    )


def test_unentered_async_client_is_rejected() -> None:
    class _UnenteredContextManager:
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

    with pytest.raises(ValueError, match="unentered async context manager"):
        _embeddings(async_client=_UnenteredContextManager())


@mock.patch.dict(os.environ, {"AWS_REGION": "us-west-2"})
def test_use_async_transport_builds_a_client() -> None:
    embeddings = BedrockEmbeddings(
        model_id=TITAN_MODEL, client=mock.MagicMock(), use_async_transport=True
    )

    assert type(embeddings.async_client).__name__ == "BedrockAsyncClient"
