"""Test chat model integration."""

from typing import Type, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBinding
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_aws import ChatBedrockConverse


class TestBedrockStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        }

    @pytest.mark.xfail()
    def test_chat_model_init_api_key(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        super().test_chat_model_init_api_key(chat_model_class, chat_model_params)

    @pytest.mark.xfail()
    def test_chat_model_init_streaming(
        self, chat_model_class: Type[BaseChatModel], chat_model_params: dict
    ) -> None:
        super().test_chat_model_init_streaming(chat_model_class, chat_model_params)


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
