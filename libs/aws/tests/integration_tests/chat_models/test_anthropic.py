from langchain_aws import ChatAnthropicBedrock

MODEL_NAME = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_invoke() -> None:
    model = ChatAnthropicBedrock(model=MODEL_NAME)  # type: ignore[call-arg]
    result = model.invoke("Hello")
    assert result


def test_stream_usage_metadata() -> None:
    model = ChatAnthropicBedrock(model=MODEL_NAME, streaming=True)  # type: ignore[call-arg]
    result = model.invoke("Hello")
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] > 0
