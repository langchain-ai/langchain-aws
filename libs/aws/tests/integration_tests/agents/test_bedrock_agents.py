import json
import time
import uuid

import boto3
from langchain.agents import AgentExecutor
from langchain_aws.agents.base import BedrockAgentsRunnable
from langchain_core.tools import tool


def _create_iam_client():
    return boto3.client('iam')


def _create_agent_role(
        agent_region,
        foundational_model
) -> str:
    """
    Create agent resource role prior to creation of agent, at this point we do not have agentId, keep it as wildcard

    Args:
        agent_region: AWS region in which is the Agent if available
        foundational_model: The model used for inference in AWS BedrockAgents
    Returns:
       Agent execution role arn
    """
    try:
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        assume_role_policy_document = json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:bedrock:{agent_region}:{account_id}:agent/*"
                        }
                    }
                }
            ]
        })
        managed_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AmazonBedrockAgentBedrockFoundationModelStatement",
                    "Effect": "Allow",
                    "Action": "bedrock:InvokeModel",
                    "Resource": [
                        f"arn:aws:bedrock:{agent_region}::foundation-model/{foundational_model}"
                    ]
                }
            ]
        }
        role_name = f'bedrock_agent_{uuid.uuid4()}'
        iam_client = _create_iam_client()
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=assume_role_policy_document,
            Description='Role for Bedrock Agent'
        )
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=f'AmazonBedrockAgentBedrockFoundationModelPolicy_{uuid.uuid4()}',
            PolicyDocument=json.dumps(managed_policy)
        )
        time.sleep(2)
        return response.get('Role', {}).get('Arn', '')

    except Exception as exception:
        raise exception


def _delete_agent_role(agent_resource_role_arn: str):
    """
    Delete agent resource role

    Args:
       agent_resource_role_arn: Associated Agent execution role arn
    """
    try:
        iam_client = _create_iam_client()
        role_name = agent_resource_role_arn.split('/')[-1]
        inline_policies = iam_client.list_role_policies(RoleName=role_name)
        for inline_policy_name in inline_policies.get('PolicyNames', []):
            iam_client.delete_role_policy(
                RoleName=role_name,
                PolicyName=inline_policy_name
            )
        iam_client.delete_role(
            RoleName=role_name
        )
    except Exception as exception:
        raise exception


def _delete_agent(agent_id):
    bedrock_client = boto3.client('bedrock-agent')
    bedrock_client.delete_agent(agentId=agent_id, skipResourceInUseCheck=True)


# --------------------------------------------------------------------------------------------------------#

@tool("AssetDetail::getAssetValue")
def getAssetValue(asset_holder_id: str) -> str:
    """Get the asset value for an owner id"""
    return f"The total asset value for {asset_holder_id} is 100K"


@tool("AssetDetail::getMortgageRate")
def getMortgageRate(asset_holder_id: str, asset_value: str) -> str:
    """Get the mortgage rate based on asset value"""
    return (
        f"The mortgage rate for {asset_holder_id} "
        f"with asset value of {asset_value} is 8.87%"
    )


def test_mortgage_bedrock_agent() -> None:
    foundational_model = 'anthropic.claude-3-sonnet-20240229-v1:0'
    tools = [getAssetValue, getMortgageRate]
    agent_resource_role_arn = None
    agent = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region='us-west-2',
            foundational_model=foundational_model)
        agent = BedrockAgentsRunnable.create_agent(
            agent_name="mortgage_interest_rate_agent",
            agent_resource_role_arn=agent_resource_role_arn,
            model=foundational_model,
            instructions="""
            You are an agent who helps with getting the mortgage rate based on the current asset valuation""",
            tools=tools
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore[arg-type]
        output = agent_executor.invoke(
            {"input": "what is my mortgage rate for id AVC-1234"}
        )

        assert output["output"] == ("The mortgage rate for the asset holder id AVC-1234 "
                                    "with an asset value of 100K is 8.87%.")
    except Exception as ex:
        raise ex
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent:
            _delete_agent(agent.agent_id)


# --------------------------------------------------------------------------------------------------------#
@tool
def getWeather(location: str = '') -> str:
    """
        Get the weather of a location

        Args:
            location: location of the place
    """
    if location.lower() == 'seattle':
        return f"It is raining in {location}"
    return f"It is hot and humid in {location}"


def test_weather_agent():
    foundational_model = 'anthropic.claude-3-sonnet-20240229-v1:0'
    tools = [getWeather]
    agent_resource_role_arn = None
    agent = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region='us-west-2',
            foundational_model=foundational_model)
        agent = BedrockAgentsRunnable.create_agent(
            agent_name="weather_agent",
            agent_resource_role_arn=agent_resource_role_arn,
            model=foundational_model,
            instructions="""
                You are an agent who helps with getting weather for a given location""",
            tools=tools
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore[arg-type]
        output = agent_executor.invoke(
            {"input": "what is the weather in Seattle?"}
        )

        assert output["output"] == "It is raining in Seattle"
    except Exception as ex:
        raise ex
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent:
            _delete_agent(agent.agent_id)


# # --------------------------------------------------------------------------------------------------------#

@tool("AssetDetail::getAssetValue")
def getTotalAssetValue(asset_holder_id: str = '') -> str:
    """
        Get the asset value for an owner id

        Args:
            asset_holder_id: id of the owner holding the asset
        Returns:
            str -> the valuation of the asset
        """
    return f"The total asset value for {asset_holder_id} is 100K"


@tool("MortgateEvaluation::getMortgateEvaluation")
def getMortgateEvaluation(asset_holder_id: str = '', asset_value: int = 0) -> str:
    """
        Get the mortgage rate based on asset value

        Args:
            asset_holder_id: id of the owner holding the asset
            asset_value: asset value which is used to get the mortgage rate
        Returns:
            str -> the calculated mortgage rate based on the asset value
        """
    return f"The mortgage rate for {asset_holder_id} with asset value of {asset_value} is 8.87%"


def test_multi_serial_actions_agent():
    foundational_model = 'anthropic.claude-3-sonnet-20240229-v1:0'
    tools = [getTotalAssetValue, getMortgateEvaluation]
    agent_resource_role_arn = None
    agent = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region='us-west-2',
            foundational_model=foundational_model)
        agent = BedrockAgentsRunnable.create_agent(
            agent_name="weather_agent",
            agent_resource_role_arn=agent_resource_role_arn,
            model=foundational_model,
            instructions="""
                    You are an agent who helps with getting weather for a given location""",
            tools=tools
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore[arg-type]
        output = agent_executor.invoke(
            {"input": "what is my mortgage rate for id AVC-1234?"}
        )

        assert output["output"] == "The mortgage rate for the asset holder id AVC-1234 is 8.87%"
    except Exception as ex:
        raise ex
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent:
            _delete_agent(agent.agent_id)


# # --------------------------------------------------------------------------------------------------------#
#
# @tool("CensusAgent::city_census")
# def city_census(
#         city_name: str = ' '
# ) -> str:
#     """
#     Gets the census of a city
#     """
#     return f"The population in {city_name} is 2 million"
#
#
# @tool("StatusAgent::getStatus")
# def city_status(
#         city_name: str = ' ',
#         city_population: int = 0
# ) -> str:
#     """
#     Gets the status of a city based on population
#     """
#     if city_population > 20000:
#         return f"{city_name} with population of {city_population} is BIG"
#     return f"{city_name} with population of {city_population} is SMALL"
#
#
# # Define logic that will be used to determine which conditional edge to go down
# def should_continue(data):
#     return "continue"
#
#
# def final_response(agent_output=None):
#     return agent_output
#
# def test_bedrock_agent_lang_graph():
#     from langgraph.graph import StateGraph, END
#
#     workflow = StateGraph(AgentState)
#
#     # Add nodes for each agent
#     workflow.add_node("get_census", get_census_agent().invoke)
#     workflow.add_node("get_status", get_status_agent().invoke)
#     workflow.add_node("final_output", final_response)
#
#     # Add entry point
#     workflow.set_entry_point("get_census")
#
#     # Add exit point
#     workflow.set_finish_point("final_output")
#
#     # Add transitions
#     workflow.add_edge("get_census", "get_status")
#     workflow.add_edge("get_status", "final_output")
#
#     workflow.add_conditional_edges(
#         "get_census",
#         should_continue,
#         path_map={
#             "continue": "get_status",
#             "end": END,
#         }
#     )
#
#     chain = workflow.compile()
#
#     final_state = chain.invoke({
#         "input": "get the city status for seattle?",
#         "output": ""
#     })
#
#
# def get_census_agent() -> AgentExecutor:
#     agent = BedrockAgentsRunnable.create_agent(
#         {"agent_id": "W2DF1EPGZM", "enable_trace": True}
#     )
#     tools = [city_census]
#     agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore[arg-type]
#
#     return agent_executor
#
#
# def get_status_agent() -> AgentExecutor:
#     agent = BedrockAgentsRunnable.create_agent(
#         {"agent_id": "UGWHI76MCI", "enable_trace": True}
#     )
#     tools = [city_status]
#     agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore[arg-type]
#
#     return agent_executor
#
#
# from typing import TypedDict, Annotated
# from operator import add
#
# class AgentState(TypedDict):
#     # The input string
#     input: str
#
#     # The outcome of a given call to the agent
#     # Needs `None` as a valid type, since this is what this will start as
#     # BedrockAgent final response will be either in string format or ROC structured dict
#     output: Annotated[str, add]
