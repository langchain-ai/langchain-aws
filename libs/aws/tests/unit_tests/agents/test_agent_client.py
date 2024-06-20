from langchain_aws.agents.bedrock.agent_client import (
    bedrock_agent,
    bedrock_agent_runtime,
    iam,
    sts
)


def test_create_bedrock_agent_runtime():
    """
    Test bedrock_agent_runtime creation
    """
    agent_runtime = bedrock_agent_runtime()
    assert agent_runtime is not None
    agent_runtime.close()


def test_create_bedrock_agent():
    """
    Test bedrock_agent creation
    """
    agent = bedrock_agent()
    assert agent is not None
    agent.close()


def test_create_iam_client():
    """
    Test iam client creation
    """
    iam_client = iam()
    assert iam_client is not None
    iam_client.close()


def test_create_sts_client():
    """
    Test sts client creation
    """
    sts_client = sts()
    assert sts_client is not None
    sts_client.close()
