"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_aws.chat_models.bedrock import ChatBedrock


class TestBedrockStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrock

    @property
    def chat_model_params(self) -> dict:
        return {"model_id": "anthropic.claude-3-sonnet-20240229-v1:0"}

    @property
    def standard_chat_model_params(self) -> dict:
        return {}

    @pytest.mark.xfail(reason="Not implemented.")
    def test_usage_metadata(self, model: BaseChatModel) -> None:
        super().test_usage_metadata(model)

    @pytest.mark.xfail(reason="Not implemented.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_string_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_string_content(model)

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_list_content(model)

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_structured_few_shot_examples(
        self,
        model: BaseChatModel,
    ) -> None:
        super().test_structured_few_shot_examples(model)


class TestBedrockUseConverseStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrock

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "beta_use_converse_api": True,
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {
            "model_kwargs": {
                "temperature": 0,
                "max_tokens": 100,
                "stop": [],
            }
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @pytest.mark.xfail(reason="Not implemented.")
    def test_stop_sequence(self, model: BaseChatModel) -> None:
        super().test_stop_sequence(model)
