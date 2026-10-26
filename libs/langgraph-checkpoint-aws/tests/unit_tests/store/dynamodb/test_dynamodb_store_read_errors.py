"""A failed DynamoDB read must not look like a missing item or an empty namespace.

``_vector_search`` already says why: "for a memory store that silently drops recall
(throttling/auth/malformed look identical to 'no memories')". The same holds for
``get``, the non-vector ``search`` and ``list_namespaces``. ``get`` matters most,
because ``None`` there means "no such item", and a caller that reads, changes and
writes an item back would then replace the stored one.
"""

import os
from typing import Any
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError
from langgraph.store.base import TTLConfig

from langgraph_checkpoint_aws import DynamoDBStore

# ResourceNotFoundException means the table is missing here, not the item, so it
# has to raise like the others.
ERROR_CODES = [
    "ProvisionedThroughputExceededException",
    "ResourceNotFoundException",
    "AccessDeniedException",
]


def _error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


@pytest.fixture
def client():
    mock = Mock()
    mock.get_waiter = Mock(return_value=Mock())
    return mock


def _make_store(client, **kwargs) -> DynamoDBStore:
    with patch(
        "langgraph_checkpoint_aws.store.dynamodb.base.create_dynamodb_client",
        return_value=client,
    ):
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-1"}):
            return DynamoDBStore(table_name="test_table", **kwargs)


@pytest.fixture
def store(client):
    return _make_store(client)


@pytest.mark.parametrize("code", ERROR_CODES)
def test_get_raises_when_the_read_fails(store, client, code):
    err = _error(code, "GetItem")
    client.get_item.side_effect = err

    with pytest.raises(ClientError) as exc_info:
        store.get(("users", "123"), "profile")
    assert exc_info.value is err


def test_get_of_a_missing_item_is_still_none(store, client):
    client.get_item.return_value = {}

    assert store.get(("users", "123"), "profile") is None


@pytest.mark.parametrize("code", ERROR_CODES)
def test_search_raises_when_the_query_fails(store, client, code):
    err = _error(code, "Query")
    client.query.side_effect = err

    with pytest.raises(ClientError) as exc_info:
        store.search(("users", "123"))
    assert exc_info.value is err


def test_search_of_an_empty_namespace_is_still_empty(store, client):
    client.query.return_value = {"Items": []}

    assert store.search(("users", "123")) == []


@pytest.mark.parametrize("code", ERROR_CODES)
def test_list_namespaces_raises_when_the_scan_fails(store, client, code):
    err = _error(code, "Scan")
    client.scan.side_effect = err

    with pytest.raises(ClientError) as exc_info:
        store.list_namespaces()
    assert exc_info.value is err


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ERROR_CODES)
async def test_aget_raises_when_the_read_fails(store, client, code):
    err = _error(code, "GetItem")
    client.get_item.side_effect = err

    with pytest.raises(ClientError) as exc_info:
        await store.aget(("users", "123"), "profile")
    assert exc_info.value is err


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ERROR_CODES)
async def test_asearch_raises_when_the_query_fails(store, client, code):
    err = _error(code, "Query")
    client.query.side_effect = err

    with pytest.raises(ClientError) as exc_info:
        await store.asearch(("users", "123"))
    assert exc_info.value is err


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ERROR_CODES)
async def test_alist_namespaces_raises_when_the_scan_fails(store, client, code):
    err = _error(code, "Scan")
    client.scan.side_effect = err

    with pytest.raises(ClientError) as exc_info:
        await store.alist_namespaces()
    assert exc_info.value is err


PK, SK = "users:123", "profile"


def _fake_table(client, delete_after_read: bool) -> dict[tuple[str, str], Any]:
    """A one-item table behind GetItem, UpdateItem and Query.

    It models only what the TTL refresh depends on: UpdateItem creates a
    missing item unless ``attribute_exists(PK)`` stops it.
    """
    table: dict[tuple[str, str], Any] = {
        (PK, SK): {
            "PK": {"S": PK},
            "SK": {"S": SK},
            "value": {"M": {"name": {"S": "Ada"}}},
            "created_at": {"S": "2026-01-01T00:00:00+00:00"},
            "updated_at": {"S": "2026-01-01T00:00:00+00:00"},
        }
    }

    def get_item(**kwargs):
        key = (kwargs["Key"]["PK"]["S"], kwargs["Key"]["SK"]["S"])
        item = table.get(key)
        if delete_after_read:
            # Another caller deletes the item right after this read returns it.
            table.pop(key, None)
        return {"Item": item} if item else {}

    def update_item(**kwargs):
        key = (kwargs["Key"]["PK"]["S"], kwargs["Key"]["SK"]["S"])
        condition = kwargs.get("ConditionExpression")
        assert condition in (None, "attribute_exists(PK)"), condition
        if condition and key not in table:
            raise _error("ConditionalCheckFailedException", "UpdateItem")
        item = table.setdefault(
            key, {"PK": kwargs["Key"]["PK"], "SK": kwargs["Key"]["SK"]}
        )
        item["expires_at"] = kwargs["ExpressionAttributeValues"][":expires_at"]
        item["updated_at"] = kwargs["ExpressionAttributeValues"][":updated_at"]

    def query(**kwargs):
        wanted = kwargs["ExpressionAttributeValues"][":pk"]["S"]
        return {"Items": [i for (p, _), i in table.items() if p == wanted]}

    client.get_item.side_effect = get_item
    client.update_item.side_effect = update_item
    client.query.side_effect = query
    return table


def test_ttl_refresh_does_not_recreate_an_item_deleted_after_the_read(client):
    """Another caller deletes the item between GetItem and the TTL refresh.

    An unconditional refresh would leave a stub with only PK, SK, expires_at
    and updated_at, and the next get or search would raise KeyError('value').
    """
    table = _fake_table(client, delete_after_read=True)
    store = _make_store(client, ttl=TTLConfig(default_ttl=60, refresh_on_read=True))

    item = store.get(("users", "123"), "profile")

    assert item is not None and item.value == {"name": "Ada"}
    client.update_item.assert_called_once()
    assert table == {}
    assert store.search(("users", "123")) == []
    assert store.get(("users", "123"), "profile") is None


def test_ttl_refresh_still_updates_an_item_that_exists(client):
    table = _fake_table(client, delete_after_read=False)
    store = _make_store(client, ttl=TTLConfig(default_ttl=60, refresh_on_read=True))

    item = store.get(("users", "123"), "profile")

    assert item is not None and item.value == {"name": "Ada"}
    stored = table[(PK, SK)]
    assert stored["value"] == {"M": {"name": {"S": "Ada"}}}
    assert "expires_at" in stored
    assert stored["updated_at"] != {"S": "2026-01-01T00:00:00+00:00"}
