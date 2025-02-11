#  type: ignore
import numpy as np
import pytest

from langchain_aws import BedrockEmbeddings


@pytest.fixture
def bedrock_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")


@pytest.fixture
def bedrock_embeddings_v2() -> BedrockEmbeddings:
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        model_kwargs={"dimensions": 256, "normalize": True},
    )


@pytest.fixture
def cohere_embeddings_v3() -> BedrockEmbeddings:
    return BedrockEmbeddings(
        model_id="cohere.embed-english-v3",
    )


@pytest.mark.scheduled
def test_bedrock_embedding_documents(bedrock_embeddings) -> None:
    documents = ["foo bar"]
    output = bedrock_embeddings.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1536


@pytest.mark.scheduled
def test_bedrock_embedding_documents_with_v2(bedrock_embeddings_v2) -> None:
    documents = ["foo bar"]
    output = bedrock_embeddings_v2.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 256


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


@pytest.mark.scheduled
def test_embed_query_with_size(bedrock_embeddings_v2) -> None:
    prompt_data = """Priority should be funding retirement through ROTH/IRA/401K 
    over HSA extra. You need to fund your HSA for reasonable and expected medical 
    expenses. 
    """
    response = bedrock_embeddings_v2.embed_documents([prompt_data])
    output = bedrock_embeddings_v2.embed_query(prompt_data)
    assert len(response[0]) == 256
    assert len(output) == 256


@pytest.mark.scheduled
def test_bedrock_cohere_embedding_documents(cohere_embeddings_v3) -> None:
    documents = ["foo bar"]
    output = cohere_embeddings_v3.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024


@pytest.mark.scheduled
def test_bedrock_cohere_embedding_documents_multiple(cohere_embeddings_v3) -> None:
    documents = ["foo bar", "bar foo", "foo"]
    output = cohere_embeddings_v3.embed_documents(documents)
    assert len(output) == 3
    assert len(output[0]) == 1024
    assert len(output[1]) == 1024
    assert len(output[2]) == 1024
