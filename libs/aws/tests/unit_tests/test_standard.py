"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_aws.chat_models.bedrock import ChatBedrock


class TestBedrockStandard(ChatModelUnitTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrock

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region_name": "us-east-1",
        }

    @pytest.mark.xfail(reason="Not implemented.")
    def test_chat_model_init_api_key(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_chat_model_init_api_key(
            chat_model_class,
            chat_model_params,
        )

    @pytest.mark.xfail(reason="Not implemented.")
    def test_standard_params(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_standard_params(
            chat_model_class,
            chat_model_params,
        )
