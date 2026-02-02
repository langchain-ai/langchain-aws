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
    from glide_sync import GlideClient, GlideClusterClient

    GlideClientType = Union[GlideClient, GlideClusterClient]
except ImportError:
    GlideClientType = Any  # type: ignore


def _default_relevance_score(val: float) -> float:
    return 1 - val


def check_index_exists(client: GlideClientType, index_name: str) -> bool:
    """Check if Valkey index exists."""
    try:
        from glide_sync import ft

        ft.info(client, index_name)
    except Exception:
        logger.debug("Index does not exist")
        return False
    logger.debug("Index already exists")
    return True


class ValkeyVectorStore(VectorStore):
    """Valkey vector database.

    To use, you should have the `valkey-glide-sync` python package installed:

        ```bash
        pip install valkey-glide-sync
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
            **kwargs: Additional arguments to pass to GLIDE client.
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

        for i, text in enumerate(texts_list):
            key = keys[i]
            metadata = metadatas[i] if metadatas else {}
            
            # Build field-value mapping
            field_value_map = {
                "content": text,
                "content_vector": _array_to_buffer(
                    embeddings[i], dtype=np.float32
                ),
            }
            # Add metadata fields
            for meta_key, meta_value in metadata.items():
                # Convert values to strings for storage
                if isinstance(meta_value, (int, float, bool)):
                    field_value_map[meta_key] = str(meta_value)
                else:
                    field_value_map[meta_key] = meta_value
            
            self.client.hset(key, field_value_map)

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
        from glide_sync import ft
        from glide_shared.commands.server_modules.ft_options.ft_search_options import (
            FtSearchOptions,
        )

        vector_field = self.vector_schema.get("name", "content_vector")
        
        base_query = f"*=>[KNN {k} @{vector_field} $vector AS score]"
        if filter:
            base_query = f"({filter})=>[KNN {k} @{vector_field} $vector AS score]"

        params = {
            "vector": _array_to_buffer(embedding, dtype=np.float32)
        }

        results = ft.search(
            self.client,
            self.index_name,
            base_query,
            options=FtSearchOptions(params=params),
        )

        # results format: [count, {key: {field: value}}]
        docs = []
        if len(results) > 1 and isinstance(results[1], dict):
            for doc_key, doc_data in results[1].items():
                if isinstance(doc_data, dict):
                    content = doc_data.get(b"content", b"")
                    if isinstance(content, bytes):
                        content = content.decode("utf-8")
                    
                    score_val = doc_data.get(b"score", b"0")
                    if isinstance(score_val, bytes):
                        score_val = score_val.decode("utf-8")
                    score = float(score_val)
                    
                    metadata = {}
                    for key, value in doc_data.items():
                        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                        if key_str not in ["content", "content_vector", "score"]:
                            if isinstance(value, bytes):
                                value = value.decode("utf-8")
                            metadata[key_str] = value

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

        for doc_id in ids:
            self.client.delete(doc_id)

        return True
