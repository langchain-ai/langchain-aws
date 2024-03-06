"""Test chat model integration."""


from langchain_aws.chat_models import ChatBedrock


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatBedrock()
