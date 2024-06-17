"""Test chat model integration."""

from typing import cast

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBinding

from langchain_aws import ChatBedrockConverse


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


def test_anthropic_bind_tools_tool_choice() -> None:
    chat_model = ChatBedrockConverse(
        model_id="anthropic.claude-3-opus-20240229", region_name="us-west-2"
    )  # type: ignore[call-arg]
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], tool_choice={"tool": {"name": "GetWeather"}}
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "tool": {"name": "GetWeather"}
    }
    chat_model_with_tools = chat_model.bind_tools(
        [GetWeather], tool_choice="GetWeather"
    )
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "tool": {"name": "GetWeather"}
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="auto")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "auto": {}
    }
    chat_model_with_tools = chat_model.bind_tools([GetWeather], tool_choice="any")
    assert cast(RunnableBinding, chat_model_with_tools).kwargs["tool_choice"] == {
        "any": {}
    }
