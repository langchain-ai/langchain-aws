"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_aws.chat_models.bedrock import ChatBedrock


class TestBedrockStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrock

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region_name": "us-east-1",
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {}

    @pytest.mark.xfail(reason="Not implemented.")
    def test_standard_params(self, model: BaseChatModel) -> None:
        super().test_standard_params(model)
