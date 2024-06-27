import json
import logging
import uuid
from typing import List
from typing import TypedDict, Union

from langchain_core.tools import Tool, StructuredTool
from langgraph.constants import END
from langgraph.graph import StateGraph

from langchain_aws.agents.bedrock.agent_base import agent_tool
from langchain_aws.agents.bedrock.agent_executor import BedrockAgentExecutor
from langchain_aws.agents.bedrock.bedrock_agent import BedrockAgent

logging.basicConfig(format='%(message)s', level=logging.INFO)


# --------------------------------------------------------------------------------------------------------#

def test_native_bedrock_agent():
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
        response = agent.invoke(
            agent_input="What is the weather in seattle?"
        )
        logging.info(f"Bedrock Agent Response : \n {json.dumps(response, indent=4)} \n")
        assert isinstance(json.loads(response), dict)
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

# def test_run_chain_agent_executor_tools_create_delete():
#     agent_name: str = "TestAgent"
#     agent_instruction: str = "You are a test agent who will help me get weather information"
#     agent_region: str = "us-east-1"
#     agent_tools: List[Tool] = [getWeatherScenario1]
#
#     agent = BedrockAgent(
#         agent_name=agent_name,
#         agent_instruction=agent_instruction,
#         agent_region=agent_region,
#         agent_tools=agent_tools
#     )
#     try:
#         agent_executor = AgentExecutor.from_agent_and_tools(
#             agent=agent,
#             verbose=True,
#             tools=agent_tools,
#             return_intermediate_steps=True,
#             max_iterations=8
#         )
#
#         invoke_agent_request = {
#             "input": "What is the weather in seattle?",
#             "session_id": "dutsudip1234",
#             "trace_enabled": True
#         }
#         response = agent_executor.invoke(invoke_agent_request)
#         logging.info(f"Bedrock Agent Response : \n {json.dumps(response, indent=4)} \n")
#         assert isinstance(response, dict)
#
#     finally:
#         agent.delete_agent()


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
            max_iterations=8
        )

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
            max_iterations=8
        )

        response = agent_executor.invoke(agent_input="What is the weather in seattle?")
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
        response = bedrock_agent.execute(
            agent_input="what is the weather in Seattle and Vancouver?",
            session_id=session_id
        )
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
        response = bedrock_agent.execute(
            agent_input="what is my mortgage rate for id AVC-1234",
            session_id=session_id
        )
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

def test_run_chain_bedrock_agent_executor_tools_invoke_multi_serial_actions():
    agent_name: str = "MortgageEvaluatorAgent1"
    agent_instruction: str = "You are an agent who helps with getting the mortgage rate based on the current asset " \
                             "valuation "
    agent_region: str = "us-east-1"
    agent_tools: List[Tool] = [getAssetValueScenario3, getMortgageRateScenario3]

    bedrock_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_tools=agent_tools,
        trace_handler=handle_trace
    )

    agent_executor = BedrockAgentExecutor.from_agent_and_tools(
        agent=bedrock_agent,
        verbose=True,
        tools=agent_tools,
        return_intermediate_steps=True,
        max_iterations=8
    )
    try:
        response = agent_executor.invoke(agent_input="what is my mortgage rate for id AVC-1234")
        assert isinstance(response, dict)
    finally:
        bedrock_agent.delete()


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
        response = bedrock_agent.execute(agent_input="what is my mortgage rate for id AVC-1234")
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
        response = bedrock_agent.execute(agent_input="what is my mortgage rate for id AVC-1234")
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
        response = bedrock_agent.execute(agent_input="what is my mortgage rate for id AVC-1234")
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
        response = bedrock_agent.execute(agent_input="what is my mortgage rate for id AVC-1234")
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
        response = bedrock_agent.execute(
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
            agent_input="What is the date 10 days before today's date in MM-DD-YYYY format?",
            **{"session_state": agent_session_state}
        )
        logging.info(f"Bedrock Agent Response : \n {json.dumps(response, indent=4)} \n")
    finally:
        agent.delete()


# --------------------------------------------------------------------------------------------------------#

def test_bedrock_agent_lang_graph():
    agent_name: str = "Census"
    agent_instruction: str = "You are an agent who helps in calculating the population of a city using the provided tool"
    agent_region: str = "us-east-1"
    census_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_foundation_model="anthropic.claude-3-haiku-20240307-v1:0",
        trace_handler=handle_trace,
        agent_tools=[StructuredTool.from_function(city_census)],
        input_handler=agent_input_handler,
        output_handler=agent_output_handler
    )

    agent_name: str = "CityStatus"
    agent_instruction: str = "You are an agent who helps in getting the status of a city based on its population"
    agent_region: str = "us-east-1"
    city_status_agent = BedrockAgent(
        agent_name=agent_name,
        agent_instruction=agent_instruction,
        agent_region=agent_region,
        agent_foundation_model="anthropic.claude-3-haiku-20240307-v1:0",
        trace_handler=handle_trace,
        agent_tools=[StructuredTool.from_function(city_status)],
        input_handler=agent_input_handler,
        output_handler=agent_output_handler
    )

    try:
        # Define a lang chain graph
        workflow = StateGraph(AgentState)

        # Add nodes for each agent
        workflow.add_node("get_census", census_agent.run)
        workflow.add_node("get_status", city_status_agent.run)
        workflow.add_node("final_output", final_response)

        # Add entry point
        workflow.set_entry_point("get_census")
        # Add exit point
        workflow.set_finish_point("final_output")

        # Add transition
        workflow.add_edge("get_census", "get_status")
        workflow.add_edge("get_status", "final_output")

        # We now add a conditional edge; for this example we don't need a conditional flow
        workflow.add_conditional_edges(
            # First, we define the start node. We use `get_census`.
            # This means these are the edges taken after the `get_census` node is called.
            "get_census",
            # Next, we pass in the function that will determine which node is called next.
            should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `continue`, then we call the get_status Agent.
                "continue": "get_status",
                # Otherwise we finish.
                "end": END,
            },
        )

        # compile the graph
        chain = workflow.compile()

        chain.invoke({"input": "get the city status for seattle"})

    finally:
        census_agent.delete()
        city_status_agent.delete()


def city_census(
        city_name: str = ' '
) -> str:
    """
    Gets the census of a city
    """
    return f"The population in {city_name} is 2 million"


def city_status(
        city_name: str = ' ',
        city_population: int = 0
) -> str:
    """
    Gets the status of a city based on population
    """
    if city_population > 20000:
        return f"{city_name} with population of {city_population} is BIG"
    return f"{city_name} with population of {city_population} is SMALL"


# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    return "continue"


def agent_input_handler(agent_input=None):
    if type(agent_input) is dict:
        _agent_outcome = agent_input.get("agent_outcome")
        if agent_input.get("input"):
            agent_input = agent_input.get("input")
        if _agent_outcome:
            agent_input = "{}. {}".format(agent_input, _agent_outcome)
    return agent_input


def agent_output_handler(agent_output=None):
    return {"agent_outcome": agent_output.get("output", "")}


def final_response(agent_output=None):
    return agent_output


# https://github.com/langchain-ai/langgraph/blob/main/examples/agent_executor/base.ipynb?ref=blog.langchain.dev
# https://langchain-ai.github.io/langgraph/#overview
# https://medium.com/@rotemweiss/discover-the-power-of-langgraph-my-adventure-in-building-gpt-newspaper-f59c7fbcf039
# https://github.com/langchain-ai/langgraph/issues/54
class AgentState(TypedDict):
    # The input string
    input: str

    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    # BedrockAgent final response will be either in string format or ROC structured dict
    agent_outcome: Union[str, dict, None]

# --------------------------------------------------------------------------------------------------------#
