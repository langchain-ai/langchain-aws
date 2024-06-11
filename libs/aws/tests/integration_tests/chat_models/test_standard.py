"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_aws.chat_models.bedrock import ChatBedrock


class TestBedrockStandard(ChatModelIntegrationTests):
    @pytest.fixture
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrock

    @pytest.fixture
    def chat_model_params(self) -> dict:
        return {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        }

    @pytest.mark.xfail(reason="Not implemented.")
    def test_usage_metadata(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_usage_metadata(
            chat_model_class,
            chat_model_params,
        )

    @pytest.mark.xfail(reason="Not implemented.")
    def test_stop_sequence(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
    ) -> None:
        super().test_stop_sequence(
            chat_model_class,
            chat_model_params,
        )

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_string_content(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
        chat_model_has_tool_calling: bool,
    ) -> None:
        super().test_tool_message_histories_string_content(
            chat_model_class, chat_model_params, chat_model_has_tool_calling
        )

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_tool_message_histories_list_content(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
        chat_model_has_tool_calling: bool,
    ) -> None:
        super().test_tool_message_histories_list_content(
            chat_model_class, chat_model_params, chat_model_has_tool_calling
        )

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_structured_few_shot_examples(
        self,
        chat_model_class: Type[BaseChatModel],
        chat_model_params: dict,
        chat_model_has_tool_calling: bool,
    ) -> None:
        super().test_structured_few_shot_examples(
            chat_model_class, chat_model_params, chat_model_has_tool_calling
        )
