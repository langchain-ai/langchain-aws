"""Test Bedrock Chat API wrapper."""
from langchain_aws import BedrockLLM


def test_initialization() -> None:
    """Test integration initialization."""
    BedrockLLM()
