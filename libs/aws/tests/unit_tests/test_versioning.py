from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from langchain_aws import (
    BedrockLLM,
    ChatAnthropicBedrock,
    ChatBedrock,
    ChatBedrockConverse,
)
from langchain_aws._version import __version__
from langchain_aws.chat_models.sagemaker_endpoint import (
    ChatModelContentHandler,
    ChatSagemakerEndpoint,
)
from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint


class TestLLMContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict[str, Any]) -> bytes:
        return prompt.encode()

    def transform_output(self, output: bytes) -> str:
        return output.decode()


class TestChatModelContentHandler(ChatModelContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(
        self, prompt: list[dict[str, Any]], model_kwargs: dict[str, Any]
    ) -> bytes:
        return str(prompt).encode()

    def transform_output(self, output: bytes) -> AIMessage:
        return AIMessage(content=output.decode())


def _assert_langchain_aws_version(model: Any) -> None:
    assert model.metadata is not None
    assert model.metadata["lc_versions"]["langchain-aws"] == __version__
    assert model.metadata["lc_versions"]["user-package"] == "1.2.3"


def test_bedrock_models_add_langchain_aws_version_metadata() -> None:
    metadata = {"lc_versions": {"user-package": "1.2.3"}}

    _assert_langchain_aws_version(
        ChatBedrock(
            model_id="anthropic.claude-v2",
            client=MagicMock(),
            bedrock_client=MagicMock(),
            metadata=metadata,
        )
    )
    _assert_langchain_aws_version(
        ChatBedrockConverse(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            client=MagicMock(),
            bedrock_client=MagicMock(),
            metadata=metadata,
        )
    )
    _assert_langchain_aws_version(
        BedrockLLM(
            model_id="amazon.titan-text-express-v1",
            client=MagicMock(),
            bedrock_client=MagicMock(),
            metadata=metadata,
        )
    )


def test_sagemaker_models_add_langchain_aws_version_metadata() -> None:
    metadata = {"lc_versions": {"user-package": "1.2.3"}}

    _assert_langchain_aws_version(
        SagemakerEndpoint(
            endpoint_name="endpoint",
            content_handler=TestLLMContentHandler(),
            client=MagicMock(),
            metadata=metadata,
        )
    )
    _assert_langchain_aws_version(
        ChatSagemakerEndpoint(
            endpoint_name="endpoint",
            content_handler=TestChatModelContentHandler(),
            client=MagicMock(),
            metadata=metadata,
        )
    )


def test_anthropic_bedrock_adds_langchain_aws_version_metadata() -> None:
    model = ChatAnthropicBedrock(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        metadata={"lc_versions": {"user-package": "1.2.3"}},
    )

    _assert_langchain_aws_version(model)
