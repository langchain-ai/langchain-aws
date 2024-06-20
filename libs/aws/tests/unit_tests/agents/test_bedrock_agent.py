import json
import uuid
from unittest import mock


from langchain_aws.agents.bedrock.agent_base import agent_tool
from langchain_aws.agents.bedrock.bedrock_agent import (
    BedrockAgent,
    BedrockAgentResponseParser,
    BedrockAgentInputFormatter
)
from langchain_core.agents import AgentAction
from langchain_core.tools import StructuredTool


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


def getAssetFunction(
        asset_holder_id: str = ' '
) -> str:
    """
    Get the asset value for an owner id

    Args:
        asset_holder_id: id of the owner holding the asset
    Returns:
        str -> the valuation of the asset
    """
    return f"The total asset value for {asset_holder_id} is 100K"


def getMortgageRateFunction(
        asset_holder_id: str = ' ',
        asset_value: str = ' '
) -> str:
    """
    Get the mortgage rate based on asset value

    Args:
        asset_holder_id: id of the owner holding the asset
        asset_value: asset value which is used to get the mortgage rate
    Returns:
        str -> the calculated mortgage rate based on the asset value
    """
    return f"The mortgage rate for {asset_holder_id} with asset value of {asset_value} is 8.87%"


def test_parse_bedrock_agent_response_with_roc():
    """
    Test the parse method of the Bedrock Agent Response Parser with a return
    of control and verify that it succeeds.
    """
    agent_response_parser = BedrockAgentResponseParser()
    parse_string = json.dumps(
        {
            "returnControl": {
                "invocationInputs": [{
                    "functionInvocationInput": {
                        "actionGroup": "WeatherAPIs",
                        "function": "getWeather",
                        "parameters": [
                            {
                                "name": "location",
                                "type": "string",
                                "value": "seattle"
                            },
                            {
                                "name": "date",
                                "type": "string",
                                "value": "2024-09-15"
                            }
                        ]
                    }
                }],
                "invocationId": "79e0feaa-c6f7-49bf-814d-b7c498505172"
            }
        }
    )
    parse_response = agent_response_parser.parse(parse_string)
    assert parse_response is not None
    assert parse_response.tool == "WeatherAPIs::getWeather"
    assert parse_response.tool_input == {
        "location": "seattle",
        "date": "2024-09-15"
    }
    assert parse_response.log == parse_string


def test_parse_bedrock_agent_response_with_default_ag():
    """
    Test the parse method of the Bedrock Agent Response Parser with a default
    action group and verify that it succeeds.
    """
    agent_response_parser = BedrockAgentResponseParser()
    parse_string = json.dumps(
        {
            "returnControl": {
                "invocationInputs": [{
                    "functionInvocationInput": {
                        "actionGroup": "DEFAULT_AG_WEATHER",
                        "function": "getWeather",
                        "parameters": [
                            {
                                "name": "location",
                                "type": "string",
                                "value": "seattle"
                            },
                            {
                                "name": "date",
                                "type": "string",
                                "value": "2024-09-15"
                            }
                        ]
                    }
                }],
                "invocationId": "79e0feaa-c6f7-49bf-814d-b7c498505172"
            }
        }
    )
    parse_response = agent_response_parser.parse(parse_string)
    assert parse_response is not None
    assert parse_response.tool == "getWeather"
    assert parse_response.tool_input == {
        "location": "seattle",
        "date": "2024-09-15"
    }
    assert parse_response.log == parse_string


def test_parse_bedrock_agent_response_with_empty_response():
    """
    Test the parse method of the Bedrock Agent Response Parser with an empty
    input and verify that it succeeds.
    """
    agent_response_parser = BedrockAgentResponseParser()
    parse_response = agent_response_parser.parse('')
    assert parse_response is not None
    assert parse_response.return_values['output'] == ''
    assert parse_response.log == ''


def test_parse_bedrock_agent_response_with_nonempty_response():
    """
    Test the parse method of the Bedrock Agent Response Parser with a non-empty
    input and verify that it succeeds.
    """
    agent_response_parser = BedrockAgentResponseParser()
    parse_response = agent_response_parser.parse('testing')
    assert parse_response is not None
    assert parse_response.return_values['output'] == 'testing'
    assert parse_response.log == 'testing'


def test_parse_bedrock_agent_response_fails_with_invalid_type_input():
    """
    Test the parse method of the Bedrock Agent Response Parser with an invalid
    type input and verify that it fails.
    """
    agent_response_parser = BedrockAgentResponseParser()
    try:
        agent_response_parser.parse(123)
    except Exception:
        assert True


def test_bedrock_agent_input_formatter_format_method():
    """
    Test the format method of the Bedrock Agent Input Formatter
    """
    agent_input_formatter = BedrockAgentInputFormatter()
    test_message = json.dumps(
        {
            "returnControl": {
                "invocationInputs": [{
                    "functionInvocationInput": {
                        "actionGroup": "WeatherAPIs",
                        "function": "getWeather",
                        "parameters": [
                            {
                                "name": "location",
                                "type": "string",
                                "value": "seattle"
                            },
                            {
                                "name": "date",
                                "type": "string",
                                "value": "2024-09-15"
                            }
                        ]
                    }
                }],
                "invocationId": "79e0feaa-c6f7-49bf-814d-b7c498505172"
            }
        }
    )
    formatted_input = agent_input_formatter.format(
        intermediate_steps=[
            (
                AgentAction(
                    tool='testAction',
                    tool_input='testInput',
                    log=test_message
                ),
                'testAgentActionStep'
            )
        ],
        callbacks=None
    )
    assert formatted_input is not None
    assert formatted_input['session_state']['invocationId'] == '79e0feaa-c6f7-49bf-814d-b7c498505172'
    assert formatted_input['session_state']['returnControlInvocationResults'][0]['functionResult']['actionGroup'] == 'DEFAULT_AG_'
    assert formatted_input['session_state']['returnControlInvocationResults'][0]['functionResult']['function'] == 'testAction'
    assert formatted_input['session_state']['returnControlInvocationResults'][0]['functionResult']['responseBody']['TEXT']['body'] == 'testAgentActionStep'


def test_bedrock_agent_input_formatter_format_prompt_method():
    """
    Test the format prompt method of the Bedrock Agent Input Formatter
    """
    agent_input_formatter = BedrockAgentInputFormatter()
    formatted_input = agent_input_formatter.format_prompt(
        input='testInput'
    )
    assert formatted_input is not None
    assert formatted_input == """{'input': 'testInput'}"""


@mock.patch("boto3.client")
def test_create_bedrock_agent(
    mock_client
):
    """
    Test the creation of a Bedrock Agent
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

    asset_value_tool = StructuredTool.from_function(getAssetFunction)
    mortgage_rate_tool = StructuredTool.from_function(getMortgageRateFunction)
    agent = BedrockAgent(
        agent_name='testAgent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_region='us-west-2',
        agent_tools=[asset_value_tool, mortgage_rate_tool]
    )

    assert agent is not None
    assert agent.bedrock_agent.name == 'testAgent'
    assert agent.bedrock_agent.agent_id == 'test_agent_id'
    assert agent.bedrock_agent.agent_alias_id == 'TSTALIASID'
    assert agent.bedrock_agent.agent_region == 'us-west-2'
    assert agent.bedrock_agent.agent_tools == [asset_value_tool, mortgage_rate_tool]
    assert agent.agent_executor.agent.name == 'testAgent'
    assert agent.agent_executor.agent.agent_id == 'test_agent_id'
    assert agent.agent_executor.agent.agent_alias_id == 'TSTALIASID'
    assert agent.agent_executor.agent.agent_region == 'us-west-2'
    assert agent.agent_executor.agent.agent_tools == [asset_value_tool, mortgage_rate_tool]
    assert agent.agent_executor.agent.agent_tools[0].name == 'getAssetFunction'
    assert agent.agent_executor.agent.agent_tools[0].description == """getAssetFunction(asset_holder_id: str = ' ') -> str - Get the asset value for an owner id

Args:
    asset_holder_id: id of the owner holding the asset
Returns:
    str -> the valuation of the asset"""
    assert agent.agent_executor.agent.agent_tools[1].name == 'getMortgageRateFunction'
    assert agent.agent_executor.agent.agent_tools[1].description == """getMortgageRateFunction(asset_holder_id: str = ' ', asset_value: str = ' ') -> str - Get the mortgage rate based on asset value

Args:
    asset_holder_id: id of the owner holding the asset
    asset_value: asset value which is used to get the mortgage rate
Returns:
    str -> the calculated mortgage rate based on the asset value"""


@mock.patch("boto3.client")
def test_run_bedrock_agent(
    mock_client
):
    """
    This test will create a bedrock agent, prepare it, then run it.
    The agent will be run with a test input, and the response will be checked
    to make sure input was successfully captured.
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
        agent_name='testAgent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_region='us-west-2',
        agent_tools=[getTestFunction1]
    )

    agentRunResponse = agent.run('Test input', 'Test session id')

    assert agentRunResponse is not None
    assert agentRunResponse['input'] == 'Test input'
    assert agentRunResponse['session_id'] == 'Test session id'
    assert agentRunResponse['trace_enabled'] is True


@mock.patch("boto3.client")
def test_invoke_bedrock_agent(
    mock_client
):
    """
    This test will create a bedrock agent, prepare it, then invoke it.
    The agent will be invoked with a test input, and the response will be checked.
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
        agent_name='testAgent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_region='us-west-2',
        agent_tools=[getTestFunction1]
    )

    invoke_agent_request = {
        "input": 'Test input',
        "session_id": str(uuid.uuid4()),
        "trace_enabled": True
    }

    agentRunResponse = agent.invoke_agent(invoke_agent_request)

    assert agentRunResponse is not None
    assert agentRunResponse['input'] == 'Test input'
    assert agentRunResponse['session_id'] is not None
    assert agentRunResponse['trace_enabled'] is True


@mock.patch("boto3.client")
def test_delete_bedrock_agent(
    mock_client
):
    """This test will create a bedrock agent, prepare it, then delete it"""
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
    mock_delete = mock_client.return_value.delete_agent
    mock_delete.return_value = {
        "ResponseMetadata":
        {
            "RequestId": "test_request_id"
        },
        "agentStatus": "DELETING"
    }

    agent = BedrockAgent(
        agent_name='testAgent',
        agent_instruction='You are a testing agent that has an instruction with a sufficient length',
        agent_region='us-west-2',
        agent_tools=[getTestFunction1]
    )

    agent.delete()

    assert agent is not None
