from langchain_core.tools import tool

from langchain_aws.agents.base import BedrockAgentsRunnable


@tool("AssetDetail::getAssetValue")
def getAssetValue(asset_holder_id: str) -> str:
    """Get the asset value for an owner id"""
    return f"The total asset value for {asset_holder_id} is 100K"


@tool
def getMortgageRate(asset_holder_id: str, asset_value: str) -> str:
    """Get the mortgage rate based on asset value"""
    return (
        f"The mortgage rate for {asset_holder_id} "
        f"with asset value of {asset_value} is 8.87%"
    )


def test_bedrock_agent() -> None:
    from langchain.agents import AgentExecutor

    agent = BedrockAgentsRunnable.create_agent(
        {"agent_id": "UKYYJIV1O1", "enable_trace": True}
    )
    tools = [getAssetValue, getMortgageRate]
    agent_executor = AgentExecutor(agent=agent, tools=tools)  # type: ignore[arg-type]

    output = agent_executor.invoke(
        {"input": "what is my mortgage rate for id AVC-1234"}
    )

    assert output["output"] == "The mortgage rate for id AVC-1234 is 8.87%"
