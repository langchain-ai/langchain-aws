"""Standard LangChain interface tests"""

from typing import Dict, Tuple, Type

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
        return {"max_tokens": 100}

    @property
    def init_from_env_params(
        self,
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Return env vars, init args, and expected instance attrs for initializing
        from env vars."""
        return (
            {
                "AWS_REGION": "region",
            },
            {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "region_name": "us-east-1",
            },
            {},
        )


class TestBedrockAsConverseStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrock

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
            "region_name": "us-east-1",
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
    def init_from_env_params(
        self,
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Return env vars, init args, and expected instance attrs for initializing
        from env vars."""
        return (
            {
                "AWS_REGION": "region",
            },
            {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "region_name": "us-east-1",
                "beta_use_converse_api": "True",
            },
            {},
        )
