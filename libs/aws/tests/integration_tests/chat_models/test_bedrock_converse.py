"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_aws import ChatBedrockConverse


class TestBedrockStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @property
    def chat_model_params(self) -> dict:
        return {"model_id": "anthropic.claude-3-sonnet-20240229-v1:0"}

    @property
    def standard_chat_model_params(self) -> dict:
        return {
            "temperature": 0,
            "max_tokens": 100,
            "stop_sequences": [],
        }
