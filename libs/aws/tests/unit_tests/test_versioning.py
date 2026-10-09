from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from langchain_aws import (
    BedrockLLM,
    ChatAnthropicBedrock,
    ChatBedrock,
    ChatBedrockConverse,
)
from langchain_aws._version import FRAMEWORK_UA_TOKEN, __version__, _tag_user_agent
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
            model="anthropic.claude-v2",
            client=MagicMock(),
            bedrock_client=MagicMock(),
            metadata=metadata,
        )
    )
    _assert_langchain_aws_version(
        ChatBedrockConverse(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            client=MagicMock(),
            bedrock_client=MagicMock(),
            metadata=metadata,
        )
    )
    _assert_langchain_aws_version(
        BedrockLLM(
            model="amazon.titan-text-express-v1",
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
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        metadata={"lc_versions": {"user-package": "1.2.3"}},
        timeout=None,
        stop=None,
    )

    _assert_langchain_aws_version(model)


class TestTagUserAgent:
    """Unit tests for the ``_tag_user_agent`` source-marker helper."""

    def test_appends_token_to_sdk_user_agent_when_no_headers(self) -> None:
        result = _tag_user_agent(None, "OpenAI/Python 3.22.1")
        assert result["User-Agent"] == f"OpenAI/Python 3.22.1 {FRAMEWORK_UA_TOKEN}"

    def test_appends_token_when_headers_have_no_user_agent(self) -> None:
        result = _tag_user_agent({"X-Custom": "v"}, "AnthropicBedrock/Python 1.9.0")
        assert result["X-Custom"] == "v"
        assert (
            result["User-Agent"]
            == f"AnthropicBedrock/Python 1.9.0 {FRAMEWORK_UA_TOKEN}"
        )

    def test_respects_caller_supplied_user_agent(self) -> None:
        # A caller who set User-Agent themselves gets the token appended to THAT,
        # not to the SDK's base UA.
        result = _tag_user_agent({"User-Agent": "MyApp/1.0"}, "OpenAI/Python 3.22.1")
        assert result["User-Agent"] == f"MyApp/1.0 {FRAMEWORK_UA_TOKEN}"

    def test_tags_caller_ua_that_merely_contains_the_token(self) -> None:
        # A caller UA containing the token as a substring (not a distinct part)
        # must still be tagged -- the idempotency check is exact-part, not substring.
        caller = "my-x-client-framework:langchain-aws-proxy/2.0"
        result = _tag_user_agent({"User-Agent": caller}, "OpenAI/Python 3.22.1")
        assert result["User-Agent"] == f"{caller} {FRAMEWORK_UA_TOKEN}"

    def test_idempotent_when_token_already_present(self) -> None:
        already = f"OpenAI/Python 3.22.1 {FRAMEWORK_UA_TOKEN}"
        result = _tag_user_agent({"User-Agent": already}, "OpenAI/Python 3.22.1")
        assert result["User-Agent"] == already
        assert result["User-Agent"].count(FRAMEWORK_UA_TOKEN) == 1

    def test_does_not_mutate_input_headers(self) -> None:
        original = {"X-Custom": "v"}
        _tag_user_agent(original, "OpenAI/Python 3.22.1")
        assert original == {"X-Custom": "v"}
