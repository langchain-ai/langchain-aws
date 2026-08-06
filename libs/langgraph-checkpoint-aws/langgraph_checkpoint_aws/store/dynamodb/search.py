"""Search operations for the DynamoDB store.

Split out of ``base.py``: the prefix/filter search path, the semantic
vector-search path (DynamoDB ``SearchVectors``), namespace listing, and the
distance-to-score conversion live here as ``DynamoDBSearchMixin``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from langgraph.store.base import ListNamespacesOp, SearchItem, SearchOp

from .vector import _FILTER_OVERFETCH, _MAX_TOP_K, _VECTOR_INDEX_NAME

logger = logging.getLogger(__name__)


class DynamoDBSearchMixin:
    """Search behaviour mixed into ``DynamoDBStore``."""

    # Provided by DynamoDBStore.__init__
    client: Any
    table_name: str
    ttl_config: Any
    _embeddings: Any
    _distance_function: str

    if TYPE_CHECKING:
        # Implemented on DynamoDBStore / sibling mixins; declared here so the
        # mixin type-checks standalone.
        @property
        def has_vector_index(self) -> bool: ...
        def _deserialize_item(self, item: dict[str, Any]) -> dict[str, Any]: ...
        def _deconstruct_namespace(self, namespace: str) -> tuple[str, ...]: ...
        def _parse_ts(self, value: Any) -> Any: ...
        def _map_to_item(self, result_dict: dict[str, Any], item_type: Any) -> Any: ...
        def _apply_filter(self, items: Any, filter_dict: Any) -> Any: ...
        def _matches_filter(self, value: Any, filter_dict: Any) -> bool: ...
        def _refresh_ttl(self, pk: str, sk: str) -> None: ...

    def _distance_to_score(self, distance: Any) -> float:
        """Convert a Vector Index distance into a LangGraph relevance score.

        DynamoDB Vector Index returns a distance (lower == closer). LangGraph
        expects a score where higher == more relevant. Only COSINE is verified
        live against the service (exact match == 0.0, orthogonal == 1.0).

        - COSINE:      score = 1 - distance   (cosine similarity, range ~[-1, 1])
        - EUCLIDEAN:   score = 1 / (1 + distance)  (monotonic, range (0, 1])
        - DOT_PRODUCT: score = raw service score (already higher == more
          similar; may be negative)
        """
        if distance is None:
            return 0.0
        d = float(distance)
        fn = self._distance_function
        if fn == "EUCLIDEAN":
            return 1.0 / (1.0 + d)
        if fn == "DOT_PRODUCT":
            # The service Score for DOT_PRODUCT is already higher == more
            # similar (and may be negative); return it unchanged so
            # SearchItem.score keeps the higher-is-better convention.
            return d
        # COSINE (default)
        return 1.0 - d

    def _batch_search_op(self, op: SearchOp) -> list[SearchItem]:
        """Execute a SearchOp operation.

        If a query is provided and vector index is configured, performs
        semantic search via DynamoDB SearchVectors. Otherwise falls back
        to a DynamoDB Query with optional filtering.

        Args:
            op: SearchOp operation.

        Returns:
            List of SearchItem instances.
        """
        # Semantic vector search path
        if op.query and self.has_vector_index:
            return self._vector_search(op)
        if op.query and not self.has_vector_index:
            logger.warning(
                "search() received query=%r but no vector index is configured; "
                "falling back to namespace listing (the query is ignored). "
                "Configure index={...} to enable semantic search.",
                op.query,
            )

        # Fallback: standard DynamoDB Query
        namespace_str = ":".join(op.namespace_prefix)

        try:
            response = self.client.query(
                TableName=self.table_name,
                KeyConditionExpression="PK = :pk",
                ExpressionAttributeValues={":pk": {"S": namespace_str}},
                Limit=op.limit + op.offset,
            )

            raw_items = response.get("Items", [])
            items = [self._deserialize_item(raw) for raw in raw_items]

            if op.filter:
                items = self._apply_filter(items, op.filter)

            if op.offset > 0:
                items = items[op.offset :]

            results = [self._map_to_item(item, SearchItem) for item in items]

            if op.refresh_ttl and self.ttl_config:
                for item in items:
                    self._refresh_ttl(item["PK"], item["SK"])

            return results

        except Exception as e:
            logger.error(f"Error searching namespace {op.namespace_prefix}: {e}")
            return []

    def _vector_search(self, op: SearchOp) -> list[SearchItem]:
        """Perform semantic search using DynamoDB SearchVectors.

        Args:
            op: SearchOp with a query string.

        Returns:
            List of SearchItem instances ranked by similarity.
        """
        # Embed the query. _vector_search is only reached when the store has a
        # vector index and op.query is set (see _batch_search_op), so both are
        # non-None here.
        assert self._embeddings is not None
        assert op.query is not None
        query_vector = self._embeddings.embed_query(op.query)
        if not all(math.isfinite(v) for v in query_vector):
            raise ValueError(
                "Query embedding contains non-finite values (nan/inf); "
                "the DynamoDB N type rejects them."
            )

        namespace_str = ":".join(op.namespace_prefix)

        # SearchVectors is a single top-K query: there is no pagination
        # token, and the service caps TopK at _MAX_TOP_K. offset/limit from
        # the BaseStore contract are honored by fetching once and slicing
        # client-side, so results beyond rank _MAX_TOP_K are unreachable by
        # design. Reject those requests loudly instead of silently returning
        # nothing past the cap.
        if op.limit + op.offset > _MAX_TOP_K:
            raise ValueError(
                "DynamoDB vector search returns at most the top "
                f"{_MAX_TOP_K} matches (service TopK limit); requested "
                f"limit+offset={op.limit + op.offset}. Use a smaller "
                "limit/offset or refine the query."
            )
        # The service applies TopK before we can filter on value fields
        # client-side, so over-fetch (capped at _MAX_TOP_K) when a filter is
        # in play. Consequence: a filtered semantic search can return fewer
        # than `limit` matches once the cap truncates the over-fetch. (A
        # future optimization is to push equality filters into the index via
        # INLINE_FILTER SearchSchema elements + SearchConditionExpression.)
        top_k = op.limit + op.offset
        if op.filter:
            top_k = min((op.limit + op.offset) * _FILTER_OVERFETCH, _MAX_TOP_K)

        try:
            # search_vectors requires botocore >= 1.43.64; typeshed stubs lag.
            resp = self.client.search_vectors(  # type: ignore[attr-defined]
                TableName=self.table_name,
                IndexName=_VECTOR_INDEX_NAME,
                SearchVector=[{"N": str(v)} for v in query_vector],
                TopK=top_k,
                SearchConditionExpression="#pk = :pk",
                ExpressionAttributeNames={"#pk": "PK"},
                ExpressionAttributeValues={":pk": {"S": namespace_str}},
            )

            results = []
            for r in resp.get("SearchResults", []):
                raw_item = r["Item"]
                item = self._deserialize_item(raw_item)
                # DynamoDB Vector Index returns a DISTANCE (lower == closer),
                # e.g. COSINE: exact match == 0.0, orthogonal == 1.0. LangGraph's
                # SearchItem.score convention is the opposite (higher == more
                # relevant), so convert. The service already returns results
                # nearest-first, so relative ordering is preserved.
                score = self._distance_to_score(r.get("Score"))

                namespace = self._deconstruct_namespace(item["PK"])
                key = item["SK"]
                value = item.get("value", {})
                created_at = self._parse_ts(item.get("created_at"))
                updated_at = self._parse_ts(item.get("updated_at"))

                search_item = SearchItem(
                    value=value,
                    key=key,
                    namespace=namespace,
                    created_at=created_at,
                    updated_at=updated_at,
                    score=score,
                )
                results.append(search_item)

            # Apply filter post-search if provided
            if op.filter:
                results = [
                    r for r in results if self._matches_filter(r.value, op.filter)
                ]

            # Apply offset + limit (we may have over-fetched above)
            if op.offset > 0:
                results = results[op.offset :]
            results = results[: op.limit]

            # Parity with the fallback path: recalling a memory refreshes
            # its TTL, otherwise actively-used memories can expire.
            if op.refresh_ttl and self.ttl_config:
                for r in results:
                    self._refresh_ttl(":".join(r.namespace), r.key)

            return results

        except Exception as e:
            # Do NOT swallow into an empty result: for a memory store that
            # silently drops recall (throttling/auth/malformed look identical
            # to "no memories"). Surface it so callers can react.
            logger.error(f"Error in vector search for '{op.query}': {e}")
            raise

    def _batch_list_namespaces_op(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        """Execute a ListNamespacesOp operation.

        Args:
            op: ListNamespacesOp operation.

        Returns:
            List of namespace tuples.
        """
        try:
            # Scan the table to get all unique namespaces
            response = self.client.scan(
                TableName=self.table_name,
                ProjectionExpression="PK",
            )

            namespaces_set: set[tuple[str, ...]] = set()
            for raw_item in response.get("Items", []):
                item = self._deserialize_item(raw_item)
                namespace = self._deconstruct_namespace(item["PK"])
                namespaces_set.add(namespace)

            # Handle pagination if more items exist
            while "LastEvaluatedKey" in response:
                response = self.client.scan(
                    TableName=self.table_name,
                    ProjectionExpression="PK",
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                for raw_item in response.get("Items", []):
                    item = self._deserialize_item(raw_item)
                    namespace = self._deconstruct_namespace(item["PK"])
                    namespaces_set.add(namespace)

            # Filter namespaces based on match conditions
            namespaces = list(namespaces_set)
            filtered = self._filter_namespaces(namespaces, op)

            # Apply limit and offset
            start = op.offset
            end = start + op.limit
            return filtered[start:end]

        except Exception as e:
            logger.error(f"Error listing namespaces: {e}")
            return []

    def _filter_namespaces(
        self,
        namespaces: list[tuple[str, ...]],
        op: ListNamespacesOp,
    ) -> list[tuple[str, ...]]:
        """Filter namespaces based on operation criteria.

        Args:
            namespaces: List of namespace tuples.
            op: ListNamespacesOp with filter criteria.

        Returns:
            Filtered list of namespaces.
        """
        filtered = namespaces

        # Apply match conditions (prefix/suffix)
        for condition in op.match_conditions or ():
            if condition.match_type == "prefix":
                prefix = condition.path
                filtered = [ns for ns in filtered if ns[: len(prefix)] == prefix]
            elif condition.match_type == "suffix":
                suffix = condition.path
                filtered = [ns for ns in filtered if ns[-len(suffix) :] == suffix]

        # Apply max_depth
        if op.max_depth is not None:
            filtered = [ns[: op.max_depth] for ns in filtered]
            # Remove duplicates after truncation
            filtered = list(dict.fromkeys(filtered))

        return sorted(filtered)
