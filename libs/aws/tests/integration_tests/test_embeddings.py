"""Test Bedrock embeddings."""
from langchain_aws.embeddings import BedrockEmbeddings


def test_langchain_aws_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = BedrockEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_aws_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = BedrockEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
