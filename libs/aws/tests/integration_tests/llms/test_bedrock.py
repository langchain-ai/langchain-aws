from langchain_aws import BedrockLLM


def test_bedrock_llm() -> None:
    llm = BedrockLLM(
        model="us.meta.llama4-scout-17b-instruct-v1:0"
    )  # type: ignore[call-arg]
    response = llm.invoke("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
