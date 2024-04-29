#  type: ignore
import numpy as np
import pytest

from langchain_aws import BedrockEmbeddings


@pytest.fixture
def bedrock_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")


@pytest.mark.scheduled
def test_bedrock_embedding_documents(bedrock_embeddings) -> None:
    documents = ["foo bar"]
    output = bedrock_embeddings.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


@pytest.mark.scheduled
def test_bedrock_embedding_documents_multiple(bedrock_embeddings) -> None:
    documents = ["foo bar", "bar foo", "foo"]
    output = bedrock_embeddings.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


@pytest.mark.scheduled
async def test_bedrock_embedding_documents_async_multiple(bedrock_embeddings) -> None:
    documents = ["foo bar", "bar foo", "foo"]
    output = await bedrock_embeddings.aembed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1536
    assert len(output[1]) == 1536
    assert len(output[2]) == 1536


@pytest.mark.scheduled
def test_bedrock_embedding_query(bedrock_embeddings) -> None:
    document = "foo bar"
    output = bedrock_embeddings.embed_query(document)
    assert len(output) == 1536


@pytest.mark.scheduled
async def test_bedrock_embedding_async_query(bedrock_embeddings) -> None:
    document = "foo bar"
    output = await bedrock_embeddings.aembed_query(document)
    assert len(output) == 1536


@pytest.mark.skip(reason="Unblock scheduled testing. TODO: fix.")
@pytest.mark.scheduled
def test_bedrock_embedding_with_empty_string(bedrock_embeddings) -> None:
    document = ["", "abc"]
    output = bedrock_embeddings.embed_documents(document)
    assert len(output) == 2
    assert len(output[0]) == 1536


@pytest.mark.scheduled
def test_embed_documents_normalized(bedrock_embeddings) -> None:
    bedrock_embeddings.normalize = True
    output = bedrock_embeddings.embed_documents(["foo walked to the market"])
    assert np.isclose(np.linalg.norm(output[0]), 1.0)


@pytest.mark.scheduled
def test_embed_query_normalized(bedrock_embeddings) -> None:
    bedrock_embeddings.normalize = True
    output = bedrock_embeddings.embed_query("foo walked to the market")
    assert np.isclose(np.linalg.norm(output), 1.0)
