"""Wrapper around Valkey vector database."""

from __future__ import annotations

import logging
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict

from langchain_aws.utilities.redis import (
    _array_to_buffer,
)
from langchain_aws.utilities.valkey import get_client

logger = logging.getLogger(__name__)

try:
    from valkey import Valkey as ValkeyType  # type: ignore[import-untyped]
except ImportError:
    ValkeyType = Any  # type: ignore


def _default_relevance_score(val: float) -> float:
    return 1 - val


def check_index_exists(client: ValkeyType, index_name: str) -> bool:
    """Check if Valkey index exists."""
    try:
        client.ft(index_name).info()
    except:  # noqa: E722
        logger.debug("Index does not exist")
        return False
    logger.debug("Index already exists")
    return True


class ValkeyVectorStore(VectorStore):
    """Valkey vector database.

    To use, you should have the `valkey` python package installed:

        ```bash
        pip install valkey
        ```

    Connection URL schemas:
    - valkey://<host>:<port> # simple connection
    - valkey://<username>:<password>@<host>:<port> # connection with authentication
    - valkeyss://<host>:<port> # connection with SSL
    - valkeyss://<username>:<password>@<host>:<port> # connection with SSL and auth

    Examples:

        Initialize and load documents:
        ```python
        from langchain_aws.vectorstores import ValkeyVectorStore
        from langchain_aws.embeddings import BedrockEmbeddings

        embeddings = BedrockEmbeddings()
        vds = ValkeyVectorStore.from_documents(
            documents,
            embeddings,
            valkey_url="valkey://cluster_endpoint:6379",
        )
        ```

        Initialize with texts and metadata:
        ```python
        vds = ValkeyVectorStore.from_texts(
            texts,
            metadata,
            embeddings,
            valkey_url="valkey://cluster_endpoint:6379",
        )
        ```
    """

    DEFAULT_VECTOR_SCHEMA = {
        "name": "content_vector",
        "algorithm": "FLAT",
        "dims": 1536,
        "distance_metric": "COSINE",
        "datatype": "FLOAT32",
    }

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        valkey_url: str,
        index_name: str,
        embedding: Embeddings,
        vector_schema: Optional[Dict[str, Union[str, int]]] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        key_prefix: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Valkey vector store.

        Args:
            valkey_url: Connection URL for Valkey server.
            index_name: Name of the index.
            embedding: Embeddings object.
            vector_schema: Vector schema configuration.
            relevance_score_fn: Function to compute relevance score.
            key_prefix: Prefix for document keys.
            **kwargs: Additional arguments to pass to Valkey client.
        """
        self.index_name = index_name
        self._embeddings = embedding
        try:
            valkey_client = get_client(valkey_url=valkey_url, **kwargs)
        except ValueError as e:
            raise ValueError(f"Valkey failed to connect: {e}")

        self.client = valkey_client
        self.relevance_score_fn = relevance_score_fn or _default_relevance_score
        self.vector_schema = vector_schema or self.DEFAULT_VECTOR_SCHEMA.copy()
        self.key_prefix = key_prefix if key_prefix is not None else f"doc:{index_name}"

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embeddings

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
            embeddings: Optional pre-computed embeddings.
            keys: Optional keys for the documents.
            **kwargs: Additional arguments.

        Returns:
            List of document IDs.
        """
        texts_list = list(texts)
        if embeddings is None:
            embeddings = self._embeddings.embed_documents(texts_list)

        if keys is None:
            keys = [f"{self.key_prefix}:{uuid.uuid4().hex}" for _ in texts_list]

        pipeline = self.client.pipeline(transaction=False)
        for i, text in enumerate(texts_list):
            key = keys[i]
            metadata = metadatas[i] if metadatas else {}
            
            doc_dict = {
                "content": text,
                "content_vector": _array_to_buffer(
                    embeddings[i], dtype=np.float32
                ),
            }
            doc_dict.update(metadata)
            
            pipeline.hset(key, mapping=doc_dict)  # type: ignore

        pipeline.execute()
        return keys

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional filter expression.
            **kwargs: Additional arguments.

        Returns:
            List of similar documents.
        """
        embedding = self._embeddings.embed_query(query)
        return self.similarity_search_by_vector(embedding, k=k, filter=filter, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents by vector.

        Args:
            embedding: Query embedding.
            k: Number of results to return.
            filter: Optional filter expression.
            **kwargs: Additional arguments.

        Returns:
            List of similar documents.
        """
        results = self.similarity_search_with_score_by_vector(
            embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional filter expression.
            **kwargs: Additional arguments.

        Returns:
            List of (document, score) tuples.
        """
        embedding = self._embeddings.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding, k=k, filter=filter, **kwargs
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents by vector with scores.

        Args:
            embedding: Query embedding.
            k: Number of results to return.
            filter: Optional filter expression.
            **kwargs: Additional arguments.

        Returns:
            List of (document, score) tuples.
        """
        from valkey.commands.search.query import Query  # type: ignore

        vector_field = self.vector_schema.get("name", "content_vector")
        
        base_query = f"*=>[KNN {k} @{vector_field} $vector AS score]"
        if filter:
            base_query = f"({filter})=>[KNN {k} @{vector_field} $vector AS score]"

        query = Query(base_query).dialect(2)

        params = {
            "vector": _array_to_buffer(embedding, dtype=np.float32)
        }

        results = self.client.ft(self.index_name).search(query, query_params=params)

        docs = []
        for result in results.docs:
            content = result.content if hasattr(result, "content") else ""
            score = float(result.score) if hasattr(result, "score") else 0.0
            
            metadata = {}
            for key, value in result.__dict__.items():
                if key not in ["id", "content", "content_vector", "score", "payload"]:
                    metadata[key] = value

            doc = Document(page_content=content, metadata=metadata)
            docs.append((doc, self.relevance_score_fn(score)))

        return docs

    @classmethod
    def from_texts(
        cls: Type[ValkeyVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        valkey_url: Optional[str] = None,
        **kwargs: Any,
    ) -> ValkeyVectorStore:
        """Create a ValkeyVectorStore from texts.

        Args:
            texts: List of texts.
            embedding: Embeddings object.
            metadatas: Optional metadata for each text.
            index_name: Name of the index.
            valkey_url: Connection URL for Valkey server.
            **kwargs: Additional arguments.

        Returns:
            ValkeyVectorStore instance.
        """
        if valkey_url is None:
            raise ValueError("valkey_url must be provided")

        if index_name is None:
            index_name = f"valkey_index_{uuid.uuid4().hex[:8]}"

        instance = cls(
            valkey_url=valkey_url,
            index_name=index_name,
            embedding=embedding,
            **kwargs,
        )

        instance.add_texts(texts, metadatas=metadatas)
        return instance

    @classmethod
    def from_existing_index(
        cls: Type[ValkeyVectorStore],
        embedding: Embeddings,
        index_name: str,
        valkey_url: str,
        **kwargs: Any,
    ) -> ValkeyVectorStore:
        """Connect to an existing Valkey index.

        Args:
            embedding: Embeddings object.
            index_name: Name of the existing index.
            valkey_url: Connection URL for Valkey server.
            **kwargs: Additional arguments.

        Returns:
            ValkeyVectorStore instance.
        """
        return cls(
            valkey_url=valkey_url,
            index_name=index_name,
            embedding=embedding,
            **kwargs,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete.
            **kwargs: Additional arguments.

        Returns:
            True if successful.
        """
        if ids is None:
            return False

        pipeline = self.client.pipeline(transaction=False)
        for doc_id in ids:
            pipeline.delete(doc_id)
        pipeline.execute()
        return True
