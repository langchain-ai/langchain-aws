from langchain_aws import BedrockLLM


def test_bedrock_llm() -> None:
    llm = BedrockLLM(model_id="anthropic.claude-v2:1")  # type: ignore[call-arg]
    response = llm.invoke("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
