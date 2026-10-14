"""Live vector-search integration test for DynamoDBStore.

Exercises the full write -> index -> semantic-search path against real
DynamoDB. Opt-in (it creates and deletes a table): skipped unless
DDB_VECTOR_INTEG=1 is set and the installed botocore exposes
`search_vectors` (requires botocore >= 1.43.64).

Run:
    DDB_VECTOR_INTEG=1 DDB_REGION=us-east-1 AWS_PROFILE=<profile> \
    pytest tests/integration_tests/store/dynamodb/test_vector_search_integration.py

DDB_VECTOR_ENDPOINT may be set to override the endpoint; it is not required.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid

import pytest

try:
    import boto3

    _client = boto3.client(
        "dynamodb",
        region_name=os.getenv("DDB_REGION", "us-east-1"),
        endpoint_url=os.getenv("DDB_VECTOR_ENDPOINT"),
    )
    _SUPPORTS_VECTORS = hasattr(_client, "search_vectors")
except Exception:  # pragma: no cover - environment dependent
    _SUPPORTS_VECTORS = False

_ENDPOINT = os.getenv("DDB_VECTOR_ENDPOINT")  # optional override
_ENABLED = os.getenv("DDB_VECTOR_INTEG") == "1"

pytestmark = pytest.mark.skipif(
    not (_ENABLED and _SUPPORTS_VECTORS),
    reason="set DDB_VECTOR_INTEG=1 and install botocore >= 1.43.64",
)

_DIMS = 8


class DeterministicEmbeddings:
    """Maps text -> a stable pseudo-random unit-ish vector.

    Identical text yields an identical vector, so an exact-text query is the
    nearest neighbour (distance ~0) of the stored memory with that text.
    """

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [((h[i] / 255.0) * 2.0 - 1.0) for i in range(_DIMS)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def vector_store(monkeypatch):
    from langgraph_checkpoint_aws.store import DynamoDBStore

    # The dir-level conftest autouse fixture injects DUMMY creds for DynamoDB
    # Local; drop them so real profile / credential_process resolution applies
    # for this endpoint-backed test.
    for var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)

    table = f"lg-ddb-vector-integ-{uuid.uuid4().hex[:8]}"
    store = DynamoDBStore(
        table_name=table,
        region_name=os.getenv("DDB_REGION", "us-east-1"),
        endpoint_url=_ENDPOINT,
        index={"embed": DeterministicEmbeddings(), "dims": _DIMS, "fields": ["text"]},
    )
    store.setup()  # creates table + waits for the vector index to be ACTIVE
    try:
        yield store
    finally:
        store.client.delete_table(TableName=table)


def test_semantic_search_round_trip(vector_store):
    ns = ("user", "alice")
    vector_store.put(ns, "m1", {"text": "prefers dark mode"})
    vector_store.put(ns, "m2", {"text": "allergic to peanuts"})
    vector_store.put(ns, "m3", {"text": "lives in Dublin"})

    # brief settle for index ingestion
    results = []
    for _ in range(6):
        results = vector_store.search(ns, query="prefers dark mode", limit=3)
        if results:
            break
        time.sleep(5)

    assert results, "vector search returned no results"
    # exact-text query must surface m1 first, with the highest score
    assert results[0].key == "m1"
    # score is a relevance score (higher == closer); exact match ~1.0
    assert results[0].score >= results[-1].score
    assert results[0].score > 0.9


def test_search_is_partition_scoped(vector_store):
    vector_store.put(("user", "alice"), "m1", {"text": "shared phrase"})
    vector_store.put(("user", "bob"), "m1", {"text": "shared phrase"})

    for _ in range(6):
        alice = vector_store.search(("user", "alice"), query="shared phrase", limit=5)
        if alice:
            break
        time.sleep(5)

    # an empty list would satisfy the all() below vacuously
    assert alice, "vector search returned no results for alice"
    # search on alice's namespace must not return bob's memory
    assert all(r.namespace == ("user", "alice") for r in alice)
    assert all(r.key == "m1" for r in alice)
