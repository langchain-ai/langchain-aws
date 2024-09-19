from langchain_aws import BedrockLLM

def test_bedrock_llm():
    llm = BedrockLLM(model_id="anthropic.claude-v2:1")
    response = llm.invoke("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
