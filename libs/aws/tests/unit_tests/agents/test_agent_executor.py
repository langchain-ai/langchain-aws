from unittest import mock

from langchain_aws.agents.bedrock.agent_base import agent_tool
from langchain_aws.agents.bedrock.agent_executor import BedrockAgentExecutor
from langchain_aws.agents.bedrock.bedrock_agent import BedrockAgent


@agent_tool(
    action_group='testActionGroup',
    action='testAction',
    method='testMethod')
def getTestFunction1(
        parameter: str = ' '
) -> str:
    """
    Test function for agent tools with an action group, action and method

    Args:
        parameter: argument of the test function
    Returns:
        str
    """
    return f"Test function executed with parameter {parameter}"


@mock.patch("boto3.client")
def test_create_bedrock_agent_executor(
    mock_client
):
    """
    Test create bedrock agent executor with valid inputs
    """
    mock_create = mock_client.return_value.create_agent
    mock_create.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agent":
        {
            "agentId": "test_agent_id"
        }
    }
    mock_create.return_valu
    mock_prepare = mock_client.return_value.prepare_agent
    mock_prepare.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.side_effect = [
        {
            "agent":
            {
                "agentStatus": "NOT_PREPARED"
            }
        },
        {
            "agent":
            {
                "agentStatus": "PREPARED"
            }
        }
    ]

    agent = BedrockAgent(
        agent_name='testing_agent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_region='us-west-2',
        agent_tools=[getTestFunction1]
    )

    agentExecutor = BedrockAgentExecutor(agent)

    assert agentExecutor is not None
    assert agentExecutor.agent == agent
    assert agentExecutor.agent.bedrock_agent.name == 'testing_agent'
    assert agentExecutor.agent.bedrock_agent.agent_id == 'test_agent_id'
    assert agentExecutor.agent.bedrock_agent.agent_alias_id == 'TSTALIASID'
    assert agentExecutor.agent.bedrock_agent.agent_region == 'us-west-2'
    assert agentExecutor.agent.bedrock_agent.agent_tools == [getTestFunction1]
    assert agentExecutor.agent.agent_executor.agent.name == 'testing_agent'
    assert agentExecutor.agent.agent_executor.agent.agent_id == 'test_agent_id'
    assert agentExecutor.agent.agent_executor.agent.agent_alias_id == 'TSTALIASID'
    assert agentExecutor.agent.agent_executor.agent.agent_region == 'us-west-2'
    assert agentExecutor.agent.agent_executor.agent.agent_tools == [getTestFunction1]


@mock.patch("boto3.client")
def test_create_bedrock_agent_executor_from_agent_and_tools(
    mock_client
):
    """
    Test create bedrock agent executor with valid inputs from agent and tools
    """
    mock_create = mock_client.return_value.create_agent
    mock_create.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agent":
        {
            "agentId": "test_agent_id"
        }
    }
    mock_prepare = mock_client.return_value.prepare_agent
    mock_prepare.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.side_effect = [
        {
            "agent":
            {
                "agentStatus": "NOT_PREPARED"
            }
        },
        {
            "agent":
            {
                "agentStatus": "PREPARED"
            }
        }
    ]

    agent = BedrockAgent(
        agent_name='testing_agent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_region='us-west-2',
        agent_tools=[getTestFunction1]
    )

    agentExecutor = BedrockAgentExecutor(agent)

    outputAgentExecutor = agentExecutor.from_agent_and_tools(
        agent,
        [getTestFunction1]
    )
    assert agentExecutor == outputAgentExecutor


@mock.patch("boto3.client")
def test_bedrock_agent_executor_input_keys_method(
    mock_client
):
    """
    Test bedrock agent executor input keys method
    """
    mock_create = mock_client.return_value.create_agent
    mock_create.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agent":
        {
            "agentId": "test_agent_id"
        }
    }
    mock_prepare = mock_client.return_value.prepare_agent
    mock_prepare.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.side_effect = [
        {
            "agent":
            {
                "agentStatus": "NOT_PREPARED"
            }
        },
        {
            "agent":
            {
                "agentStatus": "PREPARED"
            }
        }
    ]

    agent = BedrockAgent(
        agent_name='testing_agent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_region='us-west-2',
        agent_tools=[getTestFunction1]
    )

    agentExecutor = BedrockAgentExecutor(agent)

    # Call input_keys method
    agentExecutor.input_keys.append('testInputKey')


@mock.patch("boto3.client")
def test_bedrock_agent_executor_output_keys_method(
    mock_client
):
    """
    Test bedrock agent executor output keys method
    """
    mock_create = mock_client.return_value.create_agent
    mock_create.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agent":
        {
            "agentId": "test_agent_id"
        }
    }
    mock_prepare = mock_client.return_value.prepare_agent
    mock_prepare.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.side_effect = [
        {
            "agent":
            {
                "agentStatus": "NOT_PREPARED"
            }
        },
        {
            "agent":
            {
                "agentStatus": "PREPARED"
            }
        }
    ]

    agent = BedrockAgent(
        agent_name='testing_agent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_region='us-west-2',
        agent_tools=[getTestFunction1]
    )

    agentExecutor = BedrockAgentExecutor(agent)

    # Call output_keys method
    agentExecutor.output_keys.append('testOutputKeys')


@mock.patch("boto3.client")
def test_calling_bedrock_agent_executor_and_validate_response(
    mock_client
):
    """
    Test calling bedrock agent executor with valid inputs and validate the response
    """
    mock_create = mock_client.return_value.create_agent
    mock_create.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agent":
        {
            "agentId": "test_agent_id"
        }
    }
    mock_prepare = mock_client.return_value.prepare_agent
    mock_prepare.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        }
    }
    mock_get = mock_client.return_value.get_agent
    mock_get.side_effect = [
        {
            "agent":
            {
                "agentStatus": "NOT_PREPARED"
            }
        },
        {
            "agent":
            {
                "agentStatus": "PREPARED"
            }
        }
    ]
    mock_invoke = mock_client.return_value.invoke_agent
    mock_invoke.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "completion":
        [
            {
                "chunk":
                {
                    "bytes": bytes(b'Hello from Bedrock')
                },
                "trace":
                {
                    'agentAliasId': 'string',
                    'agentId': 'string',
                    'agentVersion': 'string',
                    'sessionId': 'string',
                    'trace': {
                        "orchestrationTrace": {
                            "observation": {
                                "finalResponse": {
                                    "text": "Final response text"
                                }
                            },
                            "traceId": "efc43f51-28bd-43c0-b5d2-96c64b085af6-1",
                            "type": "FINISH"
                        }
                    }
                }
            }
        ]
    }

    agent = BedrockAgent(
        agent_name='testing_agent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_region='us-west-2',
        agent_tools=[getTestFunction1]
    )

    agentExecutor = BedrockAgentExecutor(agent)

    # Call return_values method
    agentResponse = agentExecutor._call(
        inputs={'testInputKey': 'testInputValue'}
    )

    assert agentResponse is not None
    assert agentResponse['output'] == 'Hello from Bedrock'
