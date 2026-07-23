"""Standard LangChain interface tests"""

import sys
from typing import Dict, Tuple, Type
from unittest.mock import MagicMock

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
                "AWS_ACCESS_KEY_ID": "key_id",
                "AWS_SECRET_ACCESS_KEY": "secret_key",
                "AWS_SESSION_TOKEN": "token",
                "AWS_REGION": "region",
            },
            {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "region_name": "us-east-1",
            },
            {
                "aws_access_key_id": "key_id",
                "aws_secret_access_key": "secret_key",
                "aws_session_token": "token",
            },
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
                "AWS_ACCESS_KEY_ID": "key_id",
                "AWS_SECRET_ACCESS_KEY": "secret_key",
                "AWS_SESSION_TOKEN": "token",
                "AWS_REGION": "region",
            },
            {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "region_name": "us-east-1",
                "beta_use_converse_api": "True",
            },
            {
                "aws_access_key_id": "key_id",
                "aws_secret_access_key": "secret_key",
                "aws_session_token": "token",
            },
        )


# Mock openai and aws_bedrock_token_generator before importing ChatBedrockMantle
_mock_openai = MagicMock()
_mock_token_gen = MagicMock()
_mock_token_gen.provide_token.return_value = "bedrock-api-key-dGVzdA=="
sys.modules.setdefault("openai", _mock_openai)
sys.modules.setdefault("aws_bedrock_token_generator", _mock_token_gen)

from langchain_aws.chat_models.bedrock_mantle import ChatBedrockMantle  # noqa: E402


class TestBedrockMantleStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockMantle

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "openai.gpt-oss-20b",
            "region_name": "us-east-1",
            "api_key": "test-static-key",
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {"max_tokens": 100}

    @property
    def init_from_env_params(
        self,
    ) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        return (
            {
                "AWS_REGION": "us-east-1",
            },
            {
                "model": "openai.gpt-oss-20b",
                "api_key": "test-static-key",
            },
            {},
        )
