from langchain_aws import ChatAnthropicBedrock


def test_invoke() -> None:
    model = ChatAnthropicBedrock(model="us.anthropic.claude-haiku-4-5-20251001-v1:0")  # type: ignore[call-arg]
    result = model.invoke("Hello")
    assert result
