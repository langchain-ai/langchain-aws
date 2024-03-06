from langchain_aws.vectorstores import BedrockVectorStore


def test_initialization() -> None:
    """Test integration vectorstore initialization."""
    BedrockVectorStore()
