"""Integration tests for multimodal embedding support."""

import base64

import pytest

from langchain_aws import BedrockEmbeddings

# A minimal 1x1 red JPEG image (smallest valid JPEG)
TEST_IMAGE_BASE64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAAR"
    "CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAA"
    "AAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMB"
    "AAIRAxEAPwCwAB//2Q=="
)

TEST_IMAGE_BYTES = base64.b64decode(TEST_IMAGE_BASE64)


@pytest.fixture
def titan_multimodal() -> BedrockEmbeddings:
    """Titan Embed Image fixture with explicit dimensions for determinism."""
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-image-v1",
        dimensions=256,
    )


@pytest.fixture
def nova_multimodal() -> BedrockEmbeddings:
    """Nova multimodal fixture with explicit dimensions for determinism."""
    return BedrockEmbeddings(
        model_id="amazon.nova-2-multimodal-embeddings-v1:0",
        dimensions=256,
    )


@pytest.fixture
def cohere_v4() -> BedrockEmbeddings:
    """Cohere v4 fixture with explicit dimensions for determinism."""
    return BedrockEmbeddings(
        model_id="us.cohere.embed-v4:0",
        dimensions=256,
    )


@pytest.mark.scheduled
def test_titan_embed_image_bytes(titan_multimodal: BedrockEmbeddings) -> None:
    """Test Titan image embedding with raw bytes."""
    output = titan_multimodal.embed_image(TEST_IMAGE_BYTES)
    assert len(output) == 256
    assert all(isinstance(x, float) for x in output)


@pytest.mark.scheduled
def test_titan_embed_image_base64(titan_multimodal: BedrockEmbeddings) -> None:
    """Test Titan image embedding with base64 string."""
    output = titan_multimodal.embed_image(TEST_IMAGE_BASE64)
    assert len(output) == 256


@pytest.mark.scheduled
def test_titan_embed_images_batch(titan_multimodal: BedrockEmbeddings) -> None:
    """Test Titan batch image embedding."""
    output = titan_multimodal.embed_images([TEST_IMAGE_BYTES, TEST_IMAGE_BYTES])
    assert len(output) == 2
    assert len(output[0]) == 256
    assert len(output[1]) == 256


@pytest.mark.scheduled
def test_titan_embed_image_normalized(titan_multimodal: BedrockEmbeddings) -> None:
    """Test Titan image embedding with normalization."""
    titan_multimodal.normalize = True
    output = titan_multimodal.embed_image(TEST_IMAGE_BYTES)
    assert len(output) == 256
    # Check unit vector (sum of squares ≈ 1)
    import numpy as np

    norm = np.linalg.norm(output)
    assert abs(norm - 1.0) < 0.001


@pytest.mark.scheduled
def test_nova_embed_image_bytes(nova_multimodal: BedrockEmbeddings) -> None:
    """Test Nova image embedding with raw bytes."""
    output = nova_multimodal.embed_image(TEST_IMAGE_BYTES)
    assert len(output) == 256
    assert all(isinstance(x, float) for x in output)


@pytest.mark.scheduled
def test_nova_embed_image_base64(nova_multimodal: BedrockEmbeddings) -> None:
    """Test Nova image embedding with base64 string."""
    output = nova_multimodal.embed_image(TEST_IMAGE_BASE64)
    assert len(output) == 256


@pytest.mark.scheduled
def test_cohere_v4_embed_image_bytes(cohere_v4: BedrockEmbeddings) -> None:
    """Test Cohere v4 image embedding with raw bytes."""
    output = cohere_v4.embed_image(TEST_IMAGE_BYTES)
    assert len(output) == 256
    assert all(isinstance(x, float) for x in output)


@pytest.mark.scheduled
def test_text_embedding_still_works(titan_multimodal: BedrockEmbeddings) -> None:
    """Verify text embeddings still work after multimodal additions."""
    # Use text model for this test
    text_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        dimensions=256,
    )
    output = text_embeddings.embed_query("test query")
    assert len(output) == 256
    assert all(isinstance(x, float) for x in output)
