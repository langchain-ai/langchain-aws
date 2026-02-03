from __future__ import annotations

import copy
import logging
import math
import os
import uuid
from typing import Any, Callable, Iterable, List, Literal, Optional, Sequence

from botocore.exceptions import ClientError
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import SecretStr

from langchain_aws.retrievers import AmazonS3VectorsRetriever
from langchain_aws.utils import create_aws_client

logger = logging.getLogger(__name__)


class AmazonS3Vectors(VectorStore):
    """S3Vectors is Amazon S3 Vectors database.

    To use, you MUST first manually create a S3 vector bucket.
    There is no need to create a vector index.
    See: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html

    Pay attention to s3 vectors limitations and restrictions.
    By default, metadata for s3 vectors includes page_content and metadata
    for the Document.
    See: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html

    Examples:

    The following examples show various ways to use the AmazonS3Vectors with
    LangChain.

    For all the following examples assume we have the following:

        ```python
        from langchain_aws.embeddings import BedrockEmbeddings
        from langchain_aws.vectorstores.s3_vectors import AmazonS3Vectors

        embedding = BedrockEmbeddings()
        ```

    Initialize, create vector index if it does not exist, and add texts:

        ```python
        vector_store = AmazonS3Vectors.from_texts(
            ["hello", "developer", "wife"],
            vector_bucket_name="<vector bucket name>",
            index_name="<vector index name>",
            embedding=embedding,
        )
        ```

    Initialize, create vector index if it does not exist, and add Documents:
        ```python
        from langchain_core.documents import Document

        vector_store = AmazonS3Vectors(
            vector_bucket_name="<vector bucket name>",
            index_name="<vector index name>",
            embedding=embedding,
        )
        vector_store.add_documents(
            [
                Document("Star Wars", id="key1", metadata={"genre": "scifi"}),
                Document("Jurassic Park", id="key2", metadata={"genre": "scifi"}),
                Document("Finding Nemo", id="key3", metadata={"genre": "family"}),
            ]
        )
        ```

    Search with score(distance) and metadata filter:
        ```python
        vector_store.similarity_search_with_score(
            "adventures in space", filter={"genre": {"$eq": "family"}}
        )
        ```

    """

    def __init__(
        self,
        *,
        vector_bucket_name: str,
        index_name: str,
        data_type: Literal["float32"] = "float32",
        distance_metric: Literal["euclidean", "cosine"] = "cosine",
        non_filterable_metadata_keys: list[str] | None = None,
        page_content_metadata_key: Optional[str] = "_page_content",
        create_index_if_not_exist: bool = True,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        embedding: Optional[Embeddings] = None,
        query_embedding: Optional[Embeddings] = None,
        region_name: Optional[str] = None,
        credentials_profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        config: Any = None,
        client: Any = None,
        **kwargs: Any,
    ):
        """Create a AmazonS3Vectors.

        Args:
            vector_bucket_name: The name of an existing S3 vector bucket
            index_name: The name of the S3 vector index. The index names must be
                3 to 63 characters long, start and end with a letter or number,
                and contain only lowercase letters, numbers, hyphens and dots.
            data_type (Literal["float32"]): The data type of the vectors to be inserted
                into the vector index. Default is "float32".
            distance_metric (Literal["euclidean","cosine"]): The distance metric to be
                used for similarity search. Default is "cosine".
            non_filterable_metadata_keys (list[str] | None): Non-filterable metadata
                keys
            page_content_metadata_key (Optional[str]): Key of metadata to store
                page_content in Document. If None, embedding page_content
                but stored as an empty string. Default is `_page_content`.
            create_index_if_not_exist: Automatically create vector index if it
                does not exist. Default is True.
            relevance_score_fn (Optional[Callable[[float], float]]): The 'correct'
                relevance function.
            embedding (Optional[Embeddings]): Embedding function to use for indexing
                documents.
            query_embedding (Optional[Embeddings]): Separate embedding function to use
                for queries. If not provided, the `embedding` parameter is used for
                both indexing and querying. This is useful for embedding providers
                that require different task types for documents vs queries.
            region_name (Optional[str]): The aws region where the Sagemaker model is
                deployed, eg. `us-west-2`.
            credentials_profile_name (Optional[str]): The name of the profile in the
                ~/.aws/credentials or ~/.aws/config files, which has either access keys
                or role information specified.
                If not specified, the default credential profile or,
                if on an EC2 instance, credentials from IMDS will be used.
                See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
            aws_access_key_id (Optional[str]): AWS access key id.
                If provided, aws_secret_access_key must also be provided.
                If not specified, the default credential profile or,
                if on an EC2 instance, credentials from IMDS will be used.
                See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
                If not provided, will be read from `AWS_ACCESS_KEY_ID`
                environment variable.
            aws_secret_access_key (Optional[str]): AWS secret_access_key.
                If provided, aws_access_key_id must also be provided.
                If not specified, the default credential profile or,
                if on an EC2 instance, credentials from IMDS will be used.
                See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
                If not provided, will be read from `AWS_SECRET_ACCESS_KEY`
                environment variable.
            aws_session_token (Optional[str]): AWS session token.
                If provided, aws_access_key_id and
                aws_secret_access_key must also be provided.
                Not required unless using temporary credentials.
                See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
                If not provided, will be read from `AWS_SESSION_TOKEN`
                environment variable.
            endpoint_url (Optional[str]): Needed if you don't want to default to
                us-east-1 endpoint
            config: An optional `botocore.config.Config` instance to pass to
                the client.
            client: Boto3 client for s3vectors
            kwargs: Additional keyword arguments.

        """
        self.vector_bucket_name = vector_bucket_name
        self.index_name = index_name
        self.data_type = data_type
        self.distance_metric = distance_metric
        self.non_filterable_metadata_keys = non_filterable_metadata_keys
        self.page_content_metadata_key = page_content_metadata_key
        self.create_index_if_not_exist = create_index_if_not_exist
        self.relevance_score_fn = relevance_score_fn
        self._embedding = embedding
        self._query_embedding = query_embedding
        self.client = client
        if client is None:
            aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = aws_secret_access_key or os.getenv(
                "AWS_SECRET_ACCESS_KEY"
            )
            aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
            self.client = create_aws_client(
                "s3vectors",
                region_name=region_name,
                credentials_profile_name=credentials_profile_name,
                aws_access_key_id=SecretStr(aws_access_key_id)
                if aws_access_key_id
                else None,
                aws_secret_access_key=SecretStr(aws_secret_access_key)
                if aws_secret_access_key
                else None,
                aws_session_token=SecretStr(aws_session_token)
                if aws_session_token
                else None,
                endpoint_url=endpoint_url,
                config=config,
            )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        batch_size: int = 200,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the `VectorStore`.

        Args:
            texts: Iterable of strings/text to add to the `VectorStore`.
            metadatas: Optional list of metadatas.
            ids: Optional list of IDs associated with the texts.
            batch_size: Batch size for `put_vectors`.
            kwargs: Additional keyword arguments.

        Returns:
            List of IDs added to the `VectorStore`.

        """
        # Convert iterable to list to allow indexing and len operations
        texts_list = list(texts)

        # type check for metadata
        if metadatas:
            if isinstance(metadatas, list) and len(metadatas) != len(texts_list):
                raise ValueError("Number of metadatas must match number of texts")
            if not (isinstance(metadatas, list) and isinstance(metadatas[0], dict)):
                raise ValueError("Metadatas must be a list of dicts")
        # check for ids
        if isinstance(ids, list) and len(ids) != len(texts_list):
            raise ValueError("Number of IDs must match number of texts")

        result_ids = []
        for i in range(0, len(texts_list), batch_size):
            vectors = []
            sliced_texts = texts_list[i : i + batch_size]
            if self.embeddings is None:
                raise ValueError("Embeddings object is required for adding texts")
            sliced_data = self.embeddings.embed_documents(sliced_texts)
            if i == 0 and self.create_index_if_not_exist:
                if self._get_index() is None:
                    self._create_index(dimension=len(sliced_data[0]))

            for j, text in enumerate(sliced_texts):
                result_ids.append(ids and ids[i + j] or uuid.uuid4().hex)

                if metadatas:
                    if self.page_content_metadata_key:
                        # mixin page_content
                        metadata = copy.copy(metadatas[i + j])
                        metadata[self.page_content_metadata_key] = text
                    else:
                        metadata = metadatas[i + j]
                else:
                    if self.page_content_metadata_key:
                        metadata = {self.page_content_metadata_key: text}
                    else:
                        metadata = {}

                vectors.append(
                    {
                        "key": result_ids[i + j],
                        "data": {self.data_type: sliced_data[j]},
                        "metadata": metadata,
                    }
                )
            self.client.put_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                vectors=vectors,
            )
        return result_ids

    def delete(
        self, ids: Optional[list[str]] = None, *, batch_size: int = 500, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or delete index.

        Args:
            ids: List of IDs to delete vectors. If `None`, delete index with all
                vectors.
            batch_size: Batch size for `delete_vectors`.
            **kwargs: Additional keyword arguments.

        Returns:
            Always `True`.
        """

        if ids is None:
            self.client.delete_index(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
            )
        else:
            for i in range(0, len(ids), batch_size):
                self.client.delete_vectors(
                    vectorBucketName=self.vector_bucket_name,
                    indexName=self.index_name,
                    keys=ids[i : i + batch_size],
                )
        return True

    def get_by_ids(
        self, ids: Sequence[str], /, *, batch_size: int = 100
    ) -> list[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of id.
            batch_size: Batch size for get_vectors.

        Returns:
            List of `Document` objects.

        """

        docs = []
        for i in range(0, len(ids), batch_size):
            # get_vectors does not maintain order and ignores duplicates
            # and non-existent keys.
            response = self.client.get_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                keys=ids[i : i + batch_size],
                returnData=False,
                returnMetadata=True,
            )
            vector_map = {vector["key"]: vector for vector in response["vectors"]}
            for id_ in ids[i : i + batch_size]:
                if id_ not in vector_map:
                    error_msg = f"Id '{id_}' not found in vector store."
                    raise ValueError(error_msg)
            has_duplicated_id = len(vector_map) < len(ids[i : i + batch_size])
            docs.extend(
                [
                    self._create_document(
                        vector_map[id_], deepcopy_metadata=has_duplicated_id
                    )
                    for id_ in ids[i : i + batch_size]
                ]
            )
        return docs

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """The 'correct' relevance function."""
        if self.relevance_score_fn:
            return self.relevance_score_fn

        if self.distance_metric == "euclidean":
            return _euclidean_relevance_score_fn
        if self.distance_metric == "cosine":
            return _cosine_relevance_score_fn

        msg = "distance_metric must be euclidean or cosine in relevance_score."
        raise ValueError(msg)

    def _get_query_embedding(self) -> Embeddings:
        """Get the embedding to use for queries."""
        query_emb = self._query_embedding or self._embedding
        if query_emb is None:
            raise ValueError(
                "`query_embedding` arg missing at init, but cannot fall back to "
                "using the `embedding` arg because it's also not provided. "
                "Provide at least the `embedding` arg to be used for query embeddings."
            )
        return query_emb

    def similarity_search(
        self, query: str, k: int = 4, *, filter: Optional[dict] = None, **kwargs: Any
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Metadata filter to apply during the query.
                See: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-metadata-filtering.html
            **kwargs: Arguments to pass to the search method.

        Returns:
            List of `Document` objects most similar to the query.

        """
        query_vector = self._get_query_embedding().embed_query(query)
        return self.similarity_search_by_vector(
            query_vector, k=k, filter=filter, **kwargs
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Run similarity search with score(distance).

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            filter: Metadata filter to apply during the query.
                See: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-metadata-filtering.html
            **kwargs: Additional keyword arguments.

        Returns:
            List of Tuples of (doc, distance).

        """
        query_vector = self._get_query_embedding().embed_query(query)
        response = self.client.query_vectors(
            vectorBucketName=self.vector_bucket_name,
            indexName=self.index_name,
            topK=k,
            queryVector={self.data_type: query_vector},
            filter=filter,
            returnMetadata=True,
            returnDistance=True,
        )
        docs = [self._create_document(vector) for vector in response["vectors"]]
        distances = [vector["distance"] for vector in response["vectors"]]
        return list(zip(docs, distances))

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Metadata filter to apply during the query.
                See: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-metadata-filtering.html
            **kwargs: Additional keyword arguments.

        Returns:
            List of `Document` objects most similar to the query vector.

        """
        response = self.client.query_vectors(
            vectorBucketName=self.vector_bucket_name,
            indexName=self.index_name,
            topK=k,
            queryVector={self.data_type: embedding},
            filter=filter,
            returnMetadata=True,
            returnDistance=False,
        )
        return [self._create_document(vector) for vector in response["vectors"]]

    def as_retriever(self, **kwargs: Any) -> AmazonS3VectorsRetriever:
        """Return AmazonS3VectorsRetriever initialized from this AmazonS3Vectors."""

        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return AmazonS3VectorsRetriever(vectorstore=self, **kwargs, tags=tags)

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: type[AmazonS3Vectors],
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        vector_bucket_name: str,
        index_name: str,
        data_type: Literal["float32"] = "float32",
        distance_metric: Literal["euclidean", "cosine"] = "cosine",
        non_filterable_metadata_keys: list[str] | None = None,
        page_content_metadata_key: Optional[str] = "_page_content",
        create_index_if_not_exist: bool = True,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        region_name: Optional[str] = None,
        credentials_profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        config: Any = None,
        client: Any = None,
        **kwargs: Any,
    ) -> AmazonS3Vectors:
        """Return AmazonS3Vectors initialized from texts and embeddings.

        Args:
            texts: Texts to add to the `VectorStore`.
            embedding: Embedding function to use.
            metadatas: Optional list of metadatas associated with the texts.
                Default is None.
            ids: Optional list of IDs associated with the texts.
            vector_bucket_name: The name of an existing S3 vector bucket
            index_name: The name of the S3 vector index. The index names must be
                3 to 63 characters long, start and end with a letter or number,
                and contain only lowercase letters, numbers, hyphens and dots.
            data_type (Literal["float32"]): The data type of the vectors to be inserted
                into the vector index. Default is "float32".
            distance_metric (Literal["euclidean","cosine"]): The distance metric to be
                used for similarity search. Default is "cosine".
            non_filterable_metadata_keys (list[str] | None): Non-filterable metadata
                keys
            page_content_metadata_key (Optional[str]): Key of metadata to store
                page_content in Document. If None, embedding page_content
                but stored as an empty string. Default is "_page_content".
            create_index_if_not_exist: Automatically create vector index if it
                does not exist. Default is True.
            relevance_score_fn (Optional[Callable[[float], float]]): The 'correct'
                relevance function.
            region_name (Optional[str]): The aws region where the Sagemaker model is
                deployed, eg. `us-west-2`.
            credentials_profile_name (Optional[str]): The name of the profile in the
                ~/.aws/credentials or ~/.aws/config files, which has either access keys
                or role information specified.
                If not specified, the default credential profile or,
                if on an EC2 instance, credentials from IMDS will be used.
                See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
            aws_access_key_id (Optional[str]): AWS access key id.
                If provided, aws_secret_access_key must also be provided.
                If not specified, the default credential profile or,
                if on an EC2 instance, credentials from IMDS will be used.
                See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
                If not provided, will be read from `AWS_ACCESS_KEY_ID`
                environment variable.
            aws_secret_access_key (Optional[str]): AWS secret_access_key.
                If provided, aws_access_key_id must also be provided.
                If not specified, the default credential profile or,
                if on an EC2 instance, credentials from IMDS will be used.
                See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
                If not provided, will be read from `AWS_SECRET_ACCESS_KEY`
                environment variable.
            aws_session_token (Optional[str]): AWS session token.
                If provided, aws_access_key_id and
                aws_secret_access_key must also be provided.
                Not required unless using temporary credentials.
                See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
                If not provided, will be read from `AWS_SESSION_TOKEN`
                environment variable.
            endpoint_url (Optional[str]): Needed if you don't want to default to
                us-east-1 endpoint
            config: An optional `botocore.config.Config` instance to pass to
                the client.
            client: Boto3 client for s3vectors
            kwargs: Arguments to pass to AmazonS3Vectors.

        Returns:
            AmazonS3Vectors initialized from texts and embeddings.

        """

        instance = cls(
            embedding=embedding,
            vector_bucket_name=vector_bucket_name,
            index_name=index_name,
            data_type=data_type,
            distance_metric=distance_metric,
            non_filterable_metadata_keys=non_filterable_metadata_keys,
            page_content_metadata_key=page_content_metadata_key,
            create_index_if_not_exist=create_index_if_not_exist,
            relevance_score_fn=relevance_score_fn,
            region_name=region_name,
            credentials_profile_name=credentials_profile_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            endpoint_url=endpoint_url,
            config=config,
            client=client,
            **kwargs,
        )
        instance.add_texts(texts, metadatas, ids=ids)
        return instance

    def _get_index(self) -> dict | None:
        try:
            return self.client.get_index(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                return None
            raise e

    def _create_index(self, *, dimension: int) -> None:
        if self.non_filterable_metadata_keys:
            self.client.create_index(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                dataType=self.data_type,
                dimension=dimension,
                distanceMetric=self.distance_metric,
                metadataConfiguration={
                    "nonFilterableMetadataKeys": self.non_filterable_metadata_keys,
                },
            )
        else:
            self.client.create_index(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                dataType=self.data_type,
                dimension=dimension,
                distanceMetric=self.distance_metric,
            )

    def _create_document(
        self, vector: dict, *, deepcopy_metadata: bool = False
    ) -> Document:
        page_content = ""
        metadata = vector.get("metadata", {})
        if deepcopy_metadata:
            metadata = copy.deepcopy(metadata)
        if self.page_content_metadata_key and isinstance(metadata, dict):
            page_content = metadata.pop(self.page_content_metadata_key, "")
        return Document(page_content=page_content, id=vector["key"], metadata=metadata)


def _euclidean_relevance_score_fn(distance: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    max_dimension = 4096
    return 1.0 - distance / math.sqrt(max_dimension)


def _cosine_relevance_score_fn(distance: float) -> float:
    """Normalize the distance to a score on a scale [0, 1]."""
    return 1.0 - distance
