import json
import logging
import uuid
from typing import List

from langchain.agents.agent import AgentExecutor
from langchain_core.tools import Tool, StructuredTool

from langchain_aws.agents.bedrock.agent_base import agent_tool, BedrockAgentMetadata, BedrockAgentBase
from langchain_aws.agents.bedrock.agent_executor import BedrockAgentExecutor
from langchain_aws.agents.bedrock.bedrock_agent import BedrockAgent

logging.basicConfig(format='%(message)s', level=logging.INFO)


def test_run_chain_agent_executor_tools_create_delete():
    agent_name: str = "TestAgent"
    agent_instruction: str = "You are a test agent who will help me get weather information"
    agent_region: str = "us-east-1"
    agent_tools: List[Tool] = [getWeatherScenario1]

    bedrock_agent_metadata = BedrockAgentMetadata(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_tools=agent_tools
    )

    agent = BedrockAgentBase()
    try:
        agent.create(bedrock_agent_metadata)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            verbose=True,
            tools=agent_tools,
            return_intermediate_steps=True,
            max_iterations=8)

        invoke_agent_request = {
            "input": "What is the weather in seattle?",
            "session_id": "dutsudip1234",
            "trace_enabled": True
        }
        response = agent_executor.invoke(invoke_agent_request)
        logging.info(f"Bedrock Agent Response : \n {json.dumps(response, indent=4)} \n")
        assert isinstance(response, dict)

    finally:
        agent.delete()


@agent_tool(action_group='GetWeatherActionGroup', action='getWeather')
def getWeatherScenario1(
        location: str = ' '
) -> str:
    """
    Get the weather of a location

    Args:
        location: location of the place
    Returns:
        str -> the weather at the requested location
    """
    if location.lower() == 'seattle':
        return f"It is raining in {location}"
    return f"It is hot and humid in {location}"


# --------------------------------------------------------------------------------------------------------#

def test_run_chain_bedrock_agent_executor_tools_create_delete():
    agent_name: str = "TestAgent"
    agent_instruction: str = "You are a test agent who will help me get weather information"
    agent_region: str = "us-east-1"
    agent_tools: List[Tool] = [getWeatherScenario1_1]

    agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_tools=agent_tools
    )
    try:
        agent_executor = BedrockAgentExecutor.from_agent_and_tools(
            agent=agent,
            verbose=True,
            tools=agent_tools,
            return_intermediate_steps=True,
            max_iterations=8)

        invoke_agent_request = {
            "input": "What is the weather in seattle?",
            "session_id": "dutsudip1234",
            "trace_enabled": True
        }
        response = agent_executor.invoke(invoke_agent_request)
        logging.info(f"Bedrock Agent Response : \n {json.dumps(response, indent=4)} \n")
        assert isinstance(response, dict)
    finally:
        agent.delete()


@agent_tool(action_group='GetWeatherActionGroup', action='getWeather')
def getWeatherScenario1_1(
        location: str = ' '
) -> str:
    """
    Get the weather of a location

    Args:
        location: location of the place
    Returns:
        str -> the weather at the requested location
    """
    if location.lower() == 'seattle':
        return f"It is raining in {location}"
    return f"It is hot and humid in {location}"


# --------------------------------------------------------------------------------------------------------#

def test_run_chain_bedrock_agent_executor_tools_create_delete_simplified():
    agent_name: str = "TestAgent"
    agent_instruction: str = "You are a test agent who will help me get weather information"
    agent_region: str = "us-east-1"
    agent_tools: List[Tool] = [getWeatherScenario1_1]

    agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_tools=agent_tools,
        agent_region=agent_region,
        trace_handler=handle_trace
    )
    try:
        agent_executor = BedrockAgentExecutor.from_agent_and_tools(
            agent=agent,
            verbose=True,
            tools=agent_tools,
            return_intermediate_steps=True,
            max_iterations=8)

        response = agent_executor.invoke({"input": "What is the weather in seattle?"})
        logging.info(f"Bedrock Agent Response : \n {json.dumps(response, indent=4)} \n")
        assert isinstance(response, dict)
    finally:
        agent.delete()


@agent_tool(action_group='GetWeatherActionGroup', action='getWeather')
def getWeatherScenario1_1(
        location: str = ' '
) -> str:
    """
    Get the weather of a location

    Args:
        location: location of the place
    Returns:
        str -> the weather at the requested location
    """
    if location.lower() == 'seattle':
        return f"It is raining in {location}"
    return f"It is hot and humid in {location}"


# --------------------------------------------------------------------------------------------------------#

def test_run_chain_agent_executor_tools_invoke_create_delete():
    agent_name: str = "TestAgent"
    agent_instruction: str = "You are a test agent who will help me understand the basics of electrical engineering"
    agent_region: str = "us-east-1"
    agent_tools: List[Tool] = [getWeatherScenario2]

    bedrock_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_tools=agent_tools
    )
    try:
        session_id = str(uuid.uuid4())
        invoke_agent_request = {
            "input": "what is the weather in Seattle and Vancouver?",
            "session_id": session_id,
            "trace_enabled": True
        }
        response = bedrock_agent.invoke_agent(invoke_agent_request)
        assert isinstance(response, dict)
    finally:
        bedrock_agent.delete()


@agent_tool(action_group='GetWeatherActionGroup', action='getWeather')
def getWeatherScenario2(
        location: str = ' '
) -> str:
    """
    Get the weather of a location

    Args:
        location: location of the place
    Returns:
        str -> the weather at the requested location
    """
    if location.lower() == 'seattle':
        return f"It is raining in {location}"
    return f"It is hot and humid in {location}"


# --------------------------------------------------------------------------------------------------------#

def test_run_chain_agent_executor_tools_invoke_multi_serial_actions():
    agent_name: str = "MortgageEvaluatorAgent1"
    agent_instruction: str = "You are an agent who helps with getting the mortgage rate based on the current asset " \
                             "valuation "
    agent_region: str = "us-east-1"
    agent_tools: List[Tool] = [getAssetValueScenario3, getMortgageRateScenario3]

    bedrock_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_tools=agent_tools
    )
    try:
        session_id = str(uuid.uuid4())
        invoke_agent_request = {
            "input": "what is my mortgage rate for id AVC-1234",
            "session_id": session_id,
            "trace_enabled": True
        }
        response = bedrock_agent.invoke_agent(invoke_agent_request)
        assert isinstance(response, dict)
    finally:
        bedrock_agent.delete()


@agent_tool(action_group='AssetDetail', action='getAssetValue')
def getAssetValueScenario3(
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


@agent_tool(action_group='MortgageEvaluation', action='getMortgageRate')
def getMortgageRateScenario3(
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


# # --------------------------------------------------------------------------------------------------------#

def test_run_chain_agent_executor_tools_invoke_multi_serial_actions_simplified():
    agent_name: str = "MortgageEvaluatorAgent1"
    agent_instruction: str = "You are an agent who helps with getting the mortgage rate based on the current asset " \
                             "valuation "
    agent_region: str = "us-east-1"
    agent_tools: List[Tool] = [getAssetValueScenario4, getMortgageRateScenario4]

    bedrock_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_tools=agent_tools
    )
    try:
        response = bedrock_agent.invoke(agent_input="what is my mortgage rate for id AVC-1234")
        assert isinstance(response, dict)
    finally:
        bedrock_agent.delete()


@agent_tool(action_group='AssetDetail', action='getAssetValue')
def getAssetValueScenario4(
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


@agent_tool(action_group='MortgageEvaluation', action='getMortgageRate')
def getMortgageRateScenario4(
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


# --------------------------------------------------------------------------------------------------------#

def test_run_chain_agent_executor_tools_invoke_multi_serial_actions_simplified_tool():
    agent_name: str = "MortgageEvaluatorAgent2"
    agent_instruction: str = "You are an agent who helps with getting the mortgage rate based on the current asset " \
                             "valuation "
    agent_region: str = "us-east-1"
    agent_tools: List[Tool] = [getAssetValue, getMortgageRate]

    bedrock_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_tools=agent_tools,
        trace_handler=handle_trace
    )
    try:
        response = bedrock_agent.invoke(agent_input="what is my mortgage rate for id AVC-1234")
        assert isinstance(response, dict)
    finally:
        bedrock_agent.delete()


# --------------------------------------------------------------------------------------------------------#

def test_run_chain_agent_executor_tools_invoke_multi_serial_actions_structured_tool():
    agent_name: str = "MortgageEvaluatorAgent"
    agent_instruction: str = "You are an agent who helps with getting the mortgage rate based on the current asset " \
                             "valuation "
    agent_region: str = "us-east-1"
    asset_value_tool = StructuredTool.from_function(getAssetFunction)
    mortgage_rate_tool = StructuredTool.from_function(getMortgageRateFunction)
    agent_tools: List[Tool] = [asset_value_tool, mortgage_rate_tool]

    bedrock_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_tools=agent_tools,
        trace_handler=handle_trace
    )
    try:
        response = bedrock_agent.invoke(agent_input="what is my mortgage rate for id AVC-1234")
        assert isinstance(response, dict)
    finally:
        bedrock_agent.delete()


@agent_tool
def getAssetValue(
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


@agent_tool
def getMortgageRate(
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


def handle_trace(event=dict):
    logging.info("\n********************** Event **********************\n")
    logging.info(f"{json.dumps(event, indent=4)} \n")


# --------------------------------------------------------------------------------------------------------#

def test_run_chain_agent_executor_tools_invoke_multi_serial_actions_structured_tool_claude2_1():
    agent_name: str = "MortgageEvaluatorAgent3"
    agent_instruction: str = "You are an agent who helps with getting the mortgage rate based on the current asset " \
                             "valuation "
    agent_region: str = "us-east-1"
    asset_value_tool = StructuredTool.from_function(getAssetFunction)
    mortgage_rate_tool = StructuredTool.from_function(getMortgageRateFunction)
    agent_tools: List[Tool] = [asset_value_tool, mortgage_rate_tool]

    bedrock_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_foundation_model='anthropic.claude-v2:1',
        agent_tools=agent_tools,
        trace_handler=handle_trace
    )
    try:
        response = bedrock_agent.invoke(agent_input="what is my mortgage rate for id AVC-1234")
        assert isinstance(response, dict)
    finally:
        bedrock_agent.delete()


# --------------------------------------------------------------------------------------------------------#

def test_run_chain_agent_executor_tools_invoke_multi_serial_actions_structured_tool_session_parameter():
    agent_name: str = "MortgageEvaluatorAgent3"
    agent_instruction: str = "You are an agent who helps with getting the mortgage rate based on the current asset " \
                             "valuation "
    agent_region: str = "us-east-1"
    mortgage_rate_tool = StructuredTool.from_function(getMortgageRateFunction)
    agent_tools: List[Tool] = [mortgage_rate_tool]

    agent_session_state = {
        "promptSessionAttributes": {
            "assetValue": "300K"
        }
    }

    bedrock_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_foundation_model='anthropic.claude-v2:1',
        agent_tools=agent_tools,
        trace_handler=handle_trace
    )
    try:
        response = bedrock_agent.invoke(
            agent_input="what is my mortgage rate for id AVC-1234",
            **{'session_state': agent_session_state}
        )
        assert isinstance(response, dict)
    finally:
        bedrock_agent.delete()


# --------------------------------------------------------------------------------------------------------#

def test_run_chain_bedrock_agent_executor_tools_session_parameter():
    agent_name: str = "MortgageEvaluatorAgent3"
    agent_instruction: str = "You are an agent who helps in calculating previous dates"
    agent_region: str = "us-east-1"

    agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        trace_handler=handle_trace
    )
    try:
        agent_executor = BedrockAgentExecutor.from_agent_and_tools(
            agent=agent,
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=8
        )

        agent_session_state = {
            "promptSessionAttributes": {
                "dateToday": "05/22/2024"
            }
        }
        response = agent_executor.invoke(
            {
                "input": "What is the date 10 days before today's date in MM-DD-YYYY format?",
                "session_state": agent_session_state
            }
        )
        logging.info(f"Bedrock Agent Response : \n {json.dumps(response, indent=4)} \n")
    finally:
        agent.delete()
