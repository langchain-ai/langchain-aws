"""A failed DynamoDB read must not look like a missing item or an empty namespace.

``_vector_search`` already says why: "for a memory store that silently drops recall
(throttling/auth/malformed look identical to 'no memories')". The same holds for
``get``, the non-vector ``search`` and ``list_namespaces``. ``get`` matters most,
because ``None`` there means "no such item", and a caller that reads, changes and
writes an item back would then replace the stored one.
"""

import os
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from langgraph_checkpoint_aws import DynamoDBStore


def _throttled(operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {
                "Code": "ProvisionedThroughputExceededException",
                "Message": "Rate of requests exceeds the allowed throughput.",
            }
        },
        operation,
    )


@pytest.fixture
def client():
    mock = Mock()
    mock.get_waiter = Mock(return_value=Mock())
    return mock


@pytest.fixture
def store(client):
    with patch(
        "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
        return_value=client,
    ):
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):
            return DynamoDBStore(table_name="test_table")


def test_get_raises_when_the_read_fails(store, client):
    client.get_item.side_effect = _throttled("GetItem")

    with pytest.raises(ClientError):
        store.get(("users", "123"), "profile")


def test_get_of_a_missing_item_is_still_none(store, client):
    client.get_item.return_value = {}

    assert store.get(("users", "123"), "profile") is None


def test_search_raises_when_the_query_fails(store, client):
    client.query.side_effect = _throttled("Query")

    with pytest.raises(ClientError):
        store.search(("users", "123"))


def test_search_of_an_empty_namespace_is_still_empty(store, client):
    client.query.return_value = {"Items": []}

    assert store.search(("users", "123")) == []


def test_list_namespaces_raises_when_the_scan_fails(store, client):
    client.scan.side_effect = _throttled("Scan")

    with pytest.raises(ClientError):
        store.list_namespaces()


@pytest.mark.asyncio
async def test_aget_raises_when_the_read_fails(store, client):
    client.get_item.side_effect = _throttled("GetItem")

    with pytest.raises(ClientError):
        await store.aget(("users", "123"), "profile")


@pytest.mark.asyncio
async def test_asearch_raises_when_the_query_fails(store, client):
    client.query.side_effect = _throttled("Query")

    with pytest.raises(ClientError):
        await store.asearch(("users", "123"))


@pytest.mark.asyncio
async def test_alist_namespaces_raises_when_the_scan_fails(store, client):
    client.scan.side_effect = _throttled("Scan")

    with pytest.raises(ClientError):
        await store.alist_namespaces()
