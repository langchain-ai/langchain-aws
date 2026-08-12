"""Amazon DynamoDB vector store for LangChain.

Documents live as items in a DynamoDB table; a DynamoDB vector index over an
embedding attribute serves approximate nearest neighbor search. One database
serves both the application's operational data and its retrieval corpus, with
no separate vector database to run.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
import uuid
from decimal import Decimal
from typing import Any, Callable, Iterable, List, Literal, Optional, Sequence

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_aws.utils import create_aws_client

logger = logging.getLogger(__name__)

# Service maximum for SearchVectors TopK, verified against the live API:
# "Provided TopK value ... must be between 1 and 100 inclusive".
_MAX_TOP_K = 100

_PK_ATTR = "id"
_VECTOR_ATTR = "embedding"
_CONTENT_ATTR = "page_content"
_METADATA_ATTR = "metadata"


def _to_dynamodb_compatible(value: Any) -> Any:
    """Recursively convert floats to Decimal for DynamoDB serialization.

    boto3's TypeSerializer rejects Python floats; DynamoDB numbers are
    Decimal. Applied to user metadata before writing.
    """
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _to_dynamodb_compatible(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_dynamodb_compatible(v) for v in value]
    return value


def _from_dynamodb_types(value: Any) -> Any:
    """Recursively convert Decimals back to int/float for user metadata."""
    if isinstance(value, Decimal):
        as_int = int(value)
        return as_int if value == as_int else float(value)
    if isinstance(value, dict):
        return {k: _from_dynamodb_types(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_dynamodb_types(v) for v in value]
    return value


def _cosine_relevance_score_fn(distance: float) -> float:
    """Cosine distance (0 == identical) to relevance score in [0, 1]."""
    return 1.0 - distance / 2.0


def _euclidean_relevance_score_fn(distance: float) -> float:
    """Euclidean distance to a monotonically decreasing relevance score."""
    return 1.0 / (1.0 + distance)


class DynamoDBVectorStore(VectorStore):
    """Amazon DynamoDB vector store using a DynamoDB vector index.

    Documents are stored as regular DynamoDB items and searched via the
    table's vector index (approximate nearest neighbor). Like a global
    secondary index, the vector index is eventually consistent: a search
    issued immediately after ``add_texts`` may not include the just-written
    documents until the index catches up.

    To use, provide a table name and an embedding function. The table and
    vector index are created on first write if they don't exist.

    Limitations: metadata filtering is not supported (a ``filter`` argument
    is rejected rather than silently ignored), and maximal marginal
    relevance (MMR) search is not implemented. SearchVectors returns at
    most the top 100 matches per query.

    Example:
        ```python
        from langchain_aws.embeddings import BedrockEmbeddings
        from langchain_aws.vectorstores.dynamodb import DynamoDBVectorStore

        vector_store = DynamoDBVectorStore.from_texts(
            ["hello", "developer", "wife"],
            embedding=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0"),
            table_name="my-documents",
        )
        docs = vector_store.similarity_search("greeting", k=2)
        ```
    """

    def __init__(
        self,
        *,
        table_name: str,
        embedding: Embeddings,
        index_name: str = "documents-vector-index",
        distance_function: Literal["COSINE", "EUCLIDEAN", "DOT_PRODUCT"] = "COSINE",
        create_table_if_not_exist: bool = True,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        region_name: Optional[str] = None,
        credentials_profile_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        config: Any = None,
        client: Any = None,
        **kwargs: Any,
    ):
        """Create a DynamoDBVectorStore.

        Args:
            table_name: Name of the DynamoDB table holding the documents.
            embedding: Embedding function used for documents and queries.
            index_name: Name of the vector index on the table.
            distance_function: Distance function for the vector index.
                One of "COSINE" (default), "EUCLIDEAN", "DOT_PRODUCT".
            create_table_if_not_exist: Create the table and vector index on
                first use if they don't exist. Default is True.
            relevance_score_fn: Override for the relevance score conversion
                used by ``similarity_search_with_relevance_scores``.
            region_name: AWS region name.
            credentials_profile_name: Name of an AWS credentials profile.
            endpoint_url: Custom endpoint URL for the DynamoDB service.
            config: An optional ``botocore.config.Config`` for the client.
            client: Pre-configured boto3 DynamoDB client.
            kwargs: Additional keyword arguments.
        """
        self.table_name = table_name
        self.index_name = index_name
        self.distance_function = distance_function
        self.create_table_if_not_exist = create_table_if_not_exist
        self.relevance_score_fn = relevance_score_fn
        self._embedding = embedding
        self._serializer = TypeSerializer()
        self._deserializer = TypeDeserializer()
        self.client = client
        if client is None:
            self.client = create_aws_client(
                "dynamodb",
                region_name=region_name,
                credentials_profile_name=credentials_profile_name,
                endpoint_url=endpoint_url,
                config=config,
            )

    @property
    def embeddings(self) -> Embeddings:
        """The embedding function used for documents and queries."""
        return self._embedding

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    def _get_table_vector_indexes(self) -> Optional[list[dict]]:
        """Return the table's vector indexes, or None if the table is absent."""
        try:
            resp = self.client.describe_table(TableName=self.table_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return None
            raise
        return resp["Table"].get("VectorIndexes", [])

    def _create_table(self, *, dimensions: int) -> None:
        """Create the table with the vector index and wait until ACTIVE."""
        self.client.create_table(
            TableName=self.table_name,
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[
                {"AttributeName": _PK_ATTR, "AttributeType": "S"},
            ],
            KeySchema=[{"AttributeName": _PK_ATTR, "KeyType": "HASH"}],
            VectorIndexes=[self._vector_index_spec(dimensions)],
        )
        waiter = self.client.get_waiter("table_exists")
        waiter.wait(TableName=self.table_name)
        self._wait_for_vector_index()

    def _create_vector_index(self, *, dimensions: int) -> None:
        """Retrofit the vector index onto an existing table."""
        self.client.update_table(
            TableName=self.table_name,
            VectorIndexUpdates=[{"Create": self._vector_index_spec(dimensions)}],
        )
        self._wait_for_vector_index()

    def _vector_index_spec(self, dimensions: int) -> dict:
        # No SearchSchema: the index spans the whole table so searches are
        # global over the corpus (unlike a partition-scoped memory store).
        return {
            "IndexName": self.index_name,
            "VectorAttribute": {"AttributeName": _VECTOR_ATTR},
            "Projection": {"ProjectionType": "ALL"},
            "Dimensions": dimensions,
            "DistanceFunction": self.distance_function,
        }

    def _wait_for_vector_index(self, timeout: int = 600) -> None:
        """Poll describe_table until the vector index reports ACTIVE."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            indexes = self._get_table_vector_indexes() or []
            for idx in indexes:
                if idx.get("IndexName") == self.index_name:
                    # ACTIVE alone is not "ready": while Backfilling is true,
                    # SearchVectors returns an error. Require explicit ACTIVE
                    # and Backfilling false/absent; never assume a missing
                    # status means ready.
                    if idx.get("IndexStatus") == "ACTIVE" and not idx.get(
                        "Backfilling", False
                    ):
                        return
            time.sleep(5)
        raise TimeoutError(
            f"Vector index '{self.index_name}' on table '{self.table_name}' "
            f"did not become ACTIVE within {timeout}s."
        )

    def _ensure_provisioned(self, *, dimensions: int) -> None:
        """Create the table and/or vector index if configured to do so."""
        indexes = self._get_table_vector_indexes()
        if indexes is None:
            if not self.create_table_if_not_exist:
                raise ValueError(
                    f"DynamoDB table '{self.table_name}' does not exist and "
                    "create_table_if_not_exist is False."
                )
            self._create_table(dimensions=dimensions)
            return
        for idx in indexes:
            if idx.get("IndexName") == self.index_name:
                existing_dims = idx.get("Dimensions")
                if existing_dims is not None and existing_dims != dimensions:
                    raise ValueError(
                        f"Vector index '{self.index_name}' has Dimensions="
                        f"{existing_dims} but the embedding model returned "
                        f"{dimensions}-dimensional vectors. Use a matching "
                        "embedding model or a different index."
                    )
                # The service scores using the index's DistanceFunction, which
                # is immutable after creation. Relevance conversion keys off
                # self.distance_function, so a mismatch silently returns
                # wrongly-scaled scores rather than failing.
                existing_fn = idx.get("DistanceFunction")
                if existing_fn is not None and existing_fn != self.distance_function:
                    raise ValueError(
                        f"Vector index '{self.index_name}' has DistanceFunction="
                        f"{existing_fn} but this store is configured for "
                        f"{self.distance_function}. The index metric cannot be "
                        "changed after creation, and searches would be scored "
                        f"by {existing_fn} while relevance scores were computed "
                        f"for {self.distance_function}. Construct the store with "
                        f"distance_function='{existing_fn}' or use a different "
                        "index."
                    )
                return
        if not self.create_table_if_not_exist:
            raise ValueError(
                f"Vector index '{self.index_name}' does not exist on table "
                f"'{self.table_name}' and create_table_if_not_exist is False."
            )
        self._create_vector_index(dimensions=dimensions)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        batch_size: int = 25,
        **kwargs: Any,
    ) -> List[str]:
        """Embed texts and write them as DynamoDB items.

        Args:
            texts: Iterable of strings to add.
            metadatas: Optional list of metadata dicts, one per text.
            ids: Optional list of item IDs; generated when omitted.
            batch_size: Items per BatchWriteItem call (service max is 25).
            kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added texts.
        """
        texts_list = list(texts)
        if metadatas is not None and len(metadatas) != len(texts_list):
            raise ValueError("Number of metadatas must match number of texts")
        if ids is not None:
            if len(ids) != len(texts_list):
                raise ValueError("Number of ids must match number of texts")
            if len(set(ids)) != len(ids):
                raise ValueError(
                    "ids must be unique: BatchWriteItem rejects a batch "
                    "containing duplicate keys"
                )
        if not texts_list:
            return []
        if batch_size < 1 or batch_size > 25:
            raise ValueError("batch_size must be between 1 and 25 (service limit)")

        vectors = self._embedding.embed_documents(texts_list)
        dims = len(vectors[0])
        if any(len(v) != dims for v in vectors):
            raise ValueError("Embedding model returned vectors of unequal length")
        if dims < 1 or dims > 4096:
            raise ValueError(
                f"Embedding dimension {dims} outside the supported range "
                "1..4096 for DynamoDB vector indexes"
            )
        self._ensure_provisioned(dimensions=dims)

        result_ids = list(ids) if ids else [uuid.uuid4().hex for _ in texts_list]
        requests = []
        for text, metadata, vec, id_ in zip(
            texts_list,
            metadatas or [{}] * len(texts_list),
            vectors,
            result_ids,
        ):
            item = {
                _PK_ATTR: {"S": id_},
                _CONTENT_ATTR: {"S": text},
                _METADATA_ATTR: self._serializer.serialize(
                    _to_dynamodb_compatible(
                        json.loads(json.dumps(metadata, default=str))
                    )
                ),
                _VECTOR_ATTR: {"L": [{"N": str(v)} for v in vec]},
            }
            requests.append({"PutRequest": {"Item": item}})

        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            resp = self.client.batch_write_item(RequestItems={self.table_name: batch})
            # Retry unprocessed items rather than silently dropping documents.
            unprocessed = resp.get("UnprocessedItems", {}).get(self.table_name)
            attempts = 0
            while unprocessed:
                attempts += 1
                if attempts > 8:
                    raise RuntimeError(
                        f"BatchWriteItem left {len(unprocessed)} unprocessed "
                        f"items after {attempts} retries."
                    )
                delay = min(2**attempts * 0.05, 2.0)
                time.sleep(random.uniform(0, delay))
                resp = self.client.batch_write_item(
                    RequestItems={self.table_name: unprocessed}
                )
                unprocessed = resp.get("UnprocessedItems", {}).get(self.table_name)
        return result_ids

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete. Required; deleting the whole
                table is deliberately not supported through this API.
            kwargs: Additional keyword arguments.

        Returns:
            True when the delete requests were issued.
        """
        if not ids:
            raise ValueError(
                "ids is required; DynamoDBVectorStore does not delete the "
                "table. Use the AWS API to delete the table itself."
            )
        unique_ids = list(dict.fromkeys(ids))
        for i in range(0, len(unique_ids), 25):
            requests = [
                {"DeleteRequest": {"Key": {_PK_ATTR: {"S": id_}}}}
                for id_ in unique_ids[i : i + 25]
            ]
            resp = self.client.batch_write_item(
                RequestItems={self.table_name: requests}
            )
            unprocessed = resp.get("UnprocessedItems", {}).get(self.table_name)
            attempts = 0
            while unprocessed:
                attempts += 1
                if attempts > 8:
                    raise RuntimeError(
                        f"BatchWriteItem left {len(unprocessed)} unprocessed "
                        f"deletes after {attempts} retries."
                    )
                delay = min(2**attempts * 0.05, 2.0)
                time.sleep(random.uniform(0, delay))
                resp = self.client.batch_write_item(
                    RequestItems={self.table_name: unprocessed}
                )
                unprocessed = resp.get("UnprocessedItems", {}).get(self.table_name)
        return True

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: Sequence of document IDs.

        Returns:
            List of ``Document`` objects; IDs that don't exist are skipped.
        """
        docs: list[Document] = []
        # De-dupe while preserving order: the base contract requires
        # tolerating duplicate ids (returning fewer documents), and
        # BatchGetItem rejects duplicate keys within one request.
        unique_ids = list(dict.fromkeys(ids))
        for i in range(0, len(unique_ids), 100):
            chunk = unique_ids[i : i + 100]
            resp = self.client.batch_get_item(
                RequestItems={
                    self.table_name: {"Keys": [{_PK_ATTR: {"S": id_}} for id_ in chunk]}
                }
            )
            found = {
                item[_PK_ATTR]["S"]: item
                for item in resp.get("Responses", {}).get(self.table_name, [])
            }
            for id_ in chunk:
                if id_ in found:
                    docs.append(self._item_to_document(found[id_]))
        return docs

    # ------------------------------------------------------------------
    # Search path
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Return the documents most similar to the query text.

        Args:
            query: Input text.
            k: Number of documents to return (service max is 100).
            kwargs: Additional keyword arguments.

        Returns:
            List of ``Document`` objects most similar to the query.
        """
        return [
            doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Similarity search returning (document, score) tuples.

        The score is the raw ``Score`` returned by the vector index for the
        configured ``distance_function``: for COSINE and EUCLIDEAN it is a
        distance (lower is closer); for DOT_PRODUCT it is a similarity
        (higher is more similar, and may be negative). Results are always
        ordered most-similar first.

        Args:
            query: Input text.
            k: Number of documents to return (service max is 100).
            kwargs: Additional keyword arguments.

        Returns:
            List of tuples of (document, distance).
        """
        query_vector = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(query_vector, k=k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """Return the documents most similar to the given embedding."""
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding, k=k, **kwargs
            )
        ]

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Similarity search by vector returning (document, score) tuples.

        See ``similarity_search_with_score`` for the score semantics.
        """
        if kwargs.get("filter") is not None:
            raise ValueError(
                "DynamoDBVectorStore does not support metadata filtering: the "
                "vector index is created without a filterable SearchSchema. "
                "Remove the 'filter' argument (silently ignoring it would "
                "return unfiltered results)."
            )
        if k < 1 or k > _MAX_TOP_K:
            raise ValueError(
                f"k must be between 1 and {_MAX_TOP_K} (SearchVectors TopK limit)"
            )
        resp = self.client.search_vectors(
            TableName=self.table_name,
            IndexName=self.index_name,
            SearchVector=[{"N": str(v)} for v in embedding],
            TopK=k,
        )
        results = []
        for r in resp.get("SearchResults", []):
            doc = self._item_to_document(r["Item"])
            results.append((doc, float(r["Score"])))
        return results

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Distance-to-relevance conversion for the configured metric."""
        if self.relevance_score_fn:
            return self.relevance_score_fn
        if self.distance_function == "EUCLIDEAN":
            return _euclidean_relevance_score_fn
        if self.distance_function == "DOT_PRODUCT":
            # For DOT_PRODUCT the service Score is already higher == more
            # similar (and can be negative/unbounded). Squash monotonically
            # into (0, 1) so relevance ordering is preserved and LangChain's
            # [0, 1] range expectations hold.
            return lambda score: 1.0 / (1.0 + math.exp(-score))
        return _cosine_relevance_score_fn

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(  # type: ignore[override]
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        table_name: str,
        **kwargs: Any,
    ) -> DynamoDBVectorStore:
        """Create a store, provision the table/index, and add the texts.

        Args:
            texts: Texts to add.
            embedding: Embedding function.
            metadatas: Optional list of metadata dicts.
            ids: Optional list of IDs.
            table_name: Name of the DynamoDB table.
            kwargs: Passed through to the constructor.

        Returns:
            An initialized ``DynamoDBVectorStore`` containing the texts.
        """
        store = cls(table_name=table_name, embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def _item_to_document(self, item: dict[str, Any]) -> Document:
        metadata = item.get(_METADATA_ATTR)
        return Document(
            id=item[_PK_ATTR]["S"],
            page_content=item.get(_CONTENT_ATTR, {}).get("S", ""),
            metadata=_from_dynamodb_types(self._deserializer.deserialize(metadata))
            if metadata
            else {},
        )
