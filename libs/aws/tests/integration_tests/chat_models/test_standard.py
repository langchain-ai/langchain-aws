"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_aws.chat_models.bedrock import ChatBedrock


class TestBedrockStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrock

    @property
    def chat_model_params(self) -> dict:
        return {"model_id": "us.anthropic.claude-sonnet-5"}

    @property
    def standard_chat_model_params(self) -> dict:
        return {"max_tokens": 100}

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @pytest.mark.xfail(reason="Not implemented.")
    def test_double_messages_conversation(self, model: BaseChatModel) -> None:
        super().test_double_messages_conversation(model)


class TestBedrockUseConverseStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrock

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": "us.anthropic.claude-sonnet-5",
            "beta_use_converse_api": True,
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {
            "max_tokens": 100,
            "stop_sequences": [],
            "model_kwargs": {
                "stop": [],
            },
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True


try:
    from langchain_aws import ChatBedrockMantle

    _MANTLE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _MANTLE_AVAILABLE = False


@pytest.mark.skipif(
    not _MANTLE_AVAILABLE,
    reason="Mantle deps not installed or CI lacks Mantle API permissions. "
    'Run: pip install "langchain-aws[mantle]"',
)
class TestBedrockMantleStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockMantle

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "openai.gpt-5.5",
            "region_name": "us-east-1",
            "base_url": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
            # gpt-5.x is Responses-API-only.
            "use_responses_api": True,
        }

    @property
    def standard_chat_model_params(self) -> dict:
        # 5000 matches ChatBedrockConverse's standard config; frontier
        # OpenAI models (gpt-5.x) allocate a large slice of the budget to
        # internal reasoning before emitting response text.
        return {"max_tokens": 5000}
