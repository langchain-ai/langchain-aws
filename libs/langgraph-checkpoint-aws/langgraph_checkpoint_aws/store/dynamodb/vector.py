"""Vector-search plumbing for the DynamoDB store.

Embedding protocol, callable adapter, and vector-index constants shared by
:class:`~langgraph_checkpoint_aws.store.dynamodb.base.DynamoDBStore`. Kept
separate from ``base.py`` so the store class stays focused on BaseStore
operations.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Embedding protocol — matches LangGraph's standard interface
# ---------------------------------------------------------------------------


@runtime_checkable
class Embeddings(Protocol):
    """Protocol for embedding providers (sync).

    Compatible with any LangChain embeddings object (BedrockEmbeddings,
    OpenAIEmbeddings, CohereEmbeddings, etc.) or any custom implementation.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------------
# Index configuration
# ---------------------------------------------------------------------------

# Default fields to embed (entire document value serialized as text)
_DEFAULT_FIELDS = ["$"]

# Vector index name used on the DynamoDB table
_VECTOR_INDEX_NAME = "memory-vector-index"
_VECTOR_ATTR = "embedding"

# When a client-side value filter is applied to a vector search, over-fetch by
# this factor so filtering doesn't silently drop below the requested limit.
_FILTER_OVERFETCH = 10
# Service maximum for SearchVectors TopK, verified against the live API:
# "Provided TopK value ... must be between 1 and 100 inclusive".
_MAX_TOP_K = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CallableEmbeddingsAdapter:
    """Wraps a callable (texts -> vectors) into the Embeddings protocol."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> list[float]:
        result = self._fn([text])
        return result[0]
