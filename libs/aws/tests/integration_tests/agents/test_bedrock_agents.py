# type: ignore

import json
import operator
import time
import uuid
from typing import Annotated, Any, Tuple, TypedDict, Union

import boto3
import pytest
from langchain.agents import AgentExecutor
from langchain_core.tools import tool

import langchain_aws.agents.base
from langchain_aws.agents.base import (
    BedrockAgentAction,
    BedrockAgentFinish,
    BedrockAgentsRunnable,
)


def _create_iam_client() -> Any:
    return boto3.client("iam")


def _create_agent_role(agent_region: str, foundational_model: str) -> str:
    """Create agent resource role

    Args:
        agent_region: AWS region in which agent should be created
        foundational_model: The model id of the foundation model to use for the agent
    Returns:
       Agent execution role arn"""

    try:
        account_id = boto3.client("sts").get_caller_identity().get("Account")
        assume_role_policy_document = json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "bedrock.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                        "Condition": {
                            "ArnLike": {
                                "aws:SourceArn": f"arn:aws:bedrock:{agent_region}:{account_id}:agent/*"  # noqa: E501
                            }
                        },
                    }
                ],
            }
        )
        managed_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AmazonBedrockAgentBedrockFoundationModelStatement",
                    "Effect": "Allow",
                    "Action": "bedrock:InvokeModel",
                    "Resource": [
                        f"arn:aws:bedrock:{agent_region}::foundation-model/{foundational_model}"
                    ],
                }
            ],
        }
        role_name = f"bedrock_agent_{uuid.uuid4()}"
        iam_client = _create_iam_client()
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=assume_role_policy_document,
            Description="Role for Bedrock Agent",
        )
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=f"AmazonBedrockAgentBedrockFoundationModelPolicy_{uuid.uuid4()}",
            PolicyDocument=json.dumps(managed_policy),
        )
        time.sleep(2)
        return response.get("Role", {}).get("Arn", "")

    except Exception as exception:
        raise exception


def _delete_agent_role(agent_resource_role_arn: str) -> None:
    """
    Delete agent resource role

    Args:
       agent_resource_role_arn: Associated Agent execution role arn
    """
    try:
        iam_client = _create_iam_client()
        role_name = agent_resource_role_arn.split("/")[-1]
        inline_policies = iam_client.list_role_policies(RoleName=role_name)
        for inline_policy_name in inline_policies.get("PolicyNames", []):
            iam_client.delete_role_policy(
                RoleName=role_name, PolicyName=inline_policy_name
            )
        iam_client.delete_role(RoleName=role_name)
    except Exception as exception:
        raise exception


def _delete_agent(agent_id: str) -> None:
    bedrock_client = boto3.client("bedrock-agent")
    bedrock_client.delete_agent(agentId=agent_id, skipResourceInUseCheck=True)


def _delete_guardrail(guardrail_id: str) -> None:
    bedrock_client = boto3.client("bedrock")
    bedrock_client.delete_guardrail(guardrailIdentifier=guardrail_id)


def create_stock_advice_guardrail() -> None:
    # create a guard rail
    bedrock_client = boto3.client("bedrock")
    create_guardrail_response = bedrock_client.create_guardrail(
        name="block_financial_advice_guardrail",
        blockedInputMessaging="Sorry, the prompt is not allowed for this agent",
        blockedOutputsMessaging="Sorry, the model cannot answer this question.",
        topicPolicyConfig={
            "topicsConfig": [
                {
                    "name": "StockAdvice",
                    "definition": (
                        "Any questions related to stock market investments or any "
                        "questions related to investment in stocks, bonds, or "
                        "commodities"
                    ),
                    "examples": [
                        "what stocks should i invest in?",
                    ],
                    "type": "DENY",
                },
            ]
        },
    )

    return create_guardrail_response["guardrailId"], create_guardrail_response[
        "version"
    ]


@pytest.mark.skip
def test_mortgage_bedrock_agent():
    # define tools
    @tool("AssetDetail::getAssetValue")
    def get_asset_value(asset_holder_id: str) -> str:
        """Get the asset value for an owner id"""
        return f"The total asset value for {asset_holder_id} is 100K"

    @tool("AssetDetail::getMortgageRate")
    def get_mortgage_rate(asset_holder_id: str, asset_value: str) -> str:
        """Get the mortgage rate based on asset value"""
        return (
            f"The mortgage rate for the asset holder id {asset_holder_id}"
            f"with asset value of {asset_value} is 8.87%"
        )

    foundational_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    tools = [get_asset_value, get_mortgage_rate]
    agent_resource_role_arn = None
    agent = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region="us-west-2", foundational_model=foundational_model
        )
        agent = BedrockAgentsRunnable.create_agent(
            agent_name="mortgage_interest_rate_agent",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundational_model,
            instruction=(
                "You are an agent who helps with getting the mortgage rate based on "
                "the current asset valuation"
            ),
            tools=tools,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore[arg-type]
        output = agent_executor.invoke(
            {"input": "what is my mortgage rate for id AVC-1234"}
        )

        assert output["output"] == (
            "The mortgage rate for the asset holder id AVC-1234 "
            "with an asset value of 100K is 8.87%."
        )
    except Exception as ex:
        raise ex
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent:
            _delete_agent(agent.agent_id)


@pytest.mark.skip
def test_weather_agent():
    @tool
    def get_weather(location: str = "") -> str:
        """
        Get the weather of a location

        Args:
            location: location of the place
        """
        if location.lower() == "seattle":
            return f"It is raining in {location}"
        return f"It is hot and humid in {location}"

    foundational_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    tools = [get_weather]
    agent_resource_role_arn = None
    agent = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region="us-west-2", foundational_model=foundational_model
        )
        agent = BedrockAgentsRunnable.create_agent(
            agent_name="weather_agent",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundational_model,
            instruction="""
                You are an agent who helps with getting weather for a given location""",
            tools=tools,
        )
        agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore[arg-type]
        output = agent_executor.invoke({"input": "what is the weather in Seattle?"})

        assert output["output"] == "It is raining in Seattle"
    except Exception as ex:
        raise ex
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent:
            _delete_agent(agent.agent_id)


@pytest.mark.skip
def test_agent_with_guardrail():
    guardrail_id, guardrail_version = create_stock_advice_guardrail()
    foundational_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    agent_resource_role_arn = None
    agent_with_guardrail = None
    agent_without_guardrail = None
    try:
        agent_resource_role_arn = _create_agent_role(
            agent_region="us-west-2", foundational_model=foundational_model
        )
        agent_with_guardrail = BedrockAgentsRunnable.create_agent(
            agent_name="agent_with_financial_advice_guardrail",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundational_model,
            instruction="You are a test agent which will respond to user query",
            guardrail_configuration=langchain_aws.agents.base.GuardrailConfiguration(
                guardrail_identifier=guardrail_id, guardrail_version=guardrail_version
            ),
            description="Sample agent",
        )

        agent_without_guardrail = BedrockAgentsRunnable.create_agent(
            agent_name="agent_without_financial_advice_guardrail",
            agent_resource_role_arn=agent_resource_role_arn,
            foundation_model=foundational_model,
            instruction="You are a test agent which will respond to user query",
            memory_storage_days=30,
        )
        agent_executor_1 = AgentExecutor(agent=agent_with_guardrail, tools=[])  # type: ignore[arg-type]
        agent_executor_2 = AgentExecutor(agent=agent_without_guardrail, tools=[])  # type: ignore[arg-type]

        with pytest.raises(Exception):
            agent_executor_1.invoke(
                {"input": "can you help me invest in share market?"}
            )

        no_guardrail_output = agent_executor_2.invoke(
            {"input": "can you help me invest in share market?"}
        )

        assert no_guardrail_output["output"] is not None

    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn)
        if agent_with_guardrail:
            _delete_agent(agent_with_guardrail.agent_id)
        if agent_without_guardrail:
            _delete_agent(agent_without_guardrail.agent_id)
        if guardrail_id:
            _delete_guardrail(guardrail_id=guardrail_id)


@pytest.mark.skip
def test_bedrock_agent_langgraph():
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt.tool_executor import ToolExecutor

    @tool
    def get_weather(location: str = "") -> str:
        """
        Get the weather of a location

        Args:
            location: location of the place
        """
        if location.lower() == "seattle":
            return f"It is raining in {location}"
        return f"It is hot and humid in {location}"

    class AgentState(TypedDict):
        input: str
        output: Union[BedrockAgentAction, BedrockAgentFinish, None]
        intermediate_steps: Annotated[
            list[tuple[BedrockAgentAction, str]], operator.add
        ]

    def get_weather_agent_node() -> Tuple[BedrockAgentsRunnable, str]:
        foundational_model = "anthropic.claude-3-sonnet-20240229-v1:0"
        tools = [get_weather]
        try:
            agent_resource_role_arn = _create_agent_role(
                agent_region="us-west-2", foundational_model=foundational_model
            )
            agent = BedrockAgentsRunnable.create_agent(
                agent_name="weather_agent",
                agent_resource_role_arn=agent_resource_role_arn,
                foundation_model=foundational_model,
                instruction=(
                    "You are an agent who helps with getting weather for a given "
                    "location"
                ),
                tools=tools,
            )

            return agent, agent_resource_role_arn
        except Exception as e:
            raise e

    agent_runnable, agent_resource_role_arn = get_weather_agent_node()

    def run_agent(data):
        agent_outcome = agent_runnable.invoke(data)
        return {"output": agent_outcome}

    tool_executor = ToolExecutor([get_weather])

    # Define the function to execute tools
    def execute_tools(data):
        # Get the most recent output - this is the key added in the `agent` above
        agent_action = data["output"]
        output = tool_executor.invoke(agent_action[0])
        tuple_output = agent_action[0], output
        return {"intermediate_steps": [tuple_output]}

    def should_continue(data):
        output_ = data["output"]

        # If the agent outcome is a list of BedrockAgentActions,
        # then we continue to tool execution
        if (
            isinstance(output_, list)
            and len(output_) > 0
            and isinstance(output_[0], BedrockAgentAction)
        ):
            return "continue"

        # If the agent outcome is an AgentFinish, then we return `exit` string
        # This will be used when setting up the graph to define the flow
        if isinstance(output_, BedrockAgentFinish):
            return "end"

        # Unknown output from the agent, end the graph
        return "end"

    try:
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.add_edge(START, "agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node
            # will be called next.
            should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output
            # of that will be matched against the keys in this mapping.
            # The matched node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "action",
                # Otherwise we finish.
                "end": END,
            },
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge("action", "agent")

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        app = workflow.compile()

        inputs = {"input": "what is the weather in seattle?"}
        final_state = app.invoke(inputs)

        assert isinstance(final_state.get("output", {}), BedrockAgentFinish)
        assert (
            final_state.get("output").return_values["output"]
            == "It is raining in Seattle"
        )
    finally:
        if agent_resource_role_arn:
            _delete_agent_role(agent_resource_role_arn=agent_resource_role_arn)
        if agent_runnable:
            _delete_agent(agent_id=agent_runnable.agent_id)
