import os
import uuid

import pytest
from langchain_core.documents import Document

from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.vectorstores.s3_vectors import AmazonS3Vectors

vector_bucket_name = os.getenv("INTEGRATION_TEST_S3_VECTORS_VECTOR_BUCKET_NAME")


@pytest.mark.skipif(not vector_bucket_name, reason="S3 vector bucket name not set")
def test_amazon_s3_vectors_from_texts() -> None:
    embedding = BedrockEmbeddings()
    vector_store = AmazonS3Vectors.from_texts(
        ["hello", "developer", "wife"],
        vector_bucket_name=vector_bucket_name,
        index_name=uuid.uuid4().hex,
        embedding=embedding,
    )
    try:
        result = vector_store.similarity_search("hey")
        assert len(result) == 3
    finally:
        vector_store.delete()


@pytest.mark.skipif(not vector_bucket_name, reason="S3 vector bucket name not set")
def test_amazon_s3_vectors_documents() -> None:
    embedding = BedrockEmbeddings()
    vector_store = AmazonS3Vectors(
        vector_bucket_name=vector_bucket_name,
        index_name=uuid.uuid4().hex,
        embedding=embedding,
    )
    try:
        vector_store.add_documents(
            [
                Document("Star Wars", id="key1", metadata={"genre": "scifi"}),
                Document("Jurassic Park", id="key2", metadata={"genre": "scifi"}),
                Document("Finding Nemo", id="key3", metadata={"genre": "family"}),
            ]
        )
        result_get_by_ids = vector_store.get_by_ids(["key1", "key2"])
        assert result_get_by_ids == [
            Document("Star Wars", id="key1", metadata={"genre": "scifi"}),
            Document("Jurassic Park", id="key2", metadata={"genre": "scifi"}),
        ]

        result_search = vector_store.similarity_search_with_score(
            "adventures in space", filter={"genre": {"$eq": "family"}}
        )
        assert len(result_search) == 1
        assert result_search[0][0] == Document(
            "Finding Nemo", id="key3", metadata={"genre": "family"}
        )
    finally:
        vector_store.delete()
