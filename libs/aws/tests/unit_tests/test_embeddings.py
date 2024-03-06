"""Test embedding model integration."""


from langchain_aws.embeddings import BedrockEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    BedrockEmbeddings()
