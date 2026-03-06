# type:ignore

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware


def test_bedrock_skips_non_chatbedrock() -> None:
    middleware = BedrockPromptCachingMiddleware(unsupported_model_behavior="ignore")

    request = MagicMock()
    request.model = MagicMock()
    request.system_prompt = "You are helpful."
    request.messages = []

    assert middleware._should_apply_caching(request) is False


def test_bedrock_skips_non_anthropic() -> None:
    middleware = BedrockPromptCachingMiddleware(unsupported_model_behavior="ignore")

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "amazon.titan-text-express-v1"
    request.system_prompt = "You are helpful."
    request.messages = []

    assert middleware._should_apply_caching(request) is False


def test_bedrock_passes_cache_control_via_model_settings() -> None:
    middleware = BedrockPromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [HumanMessage(content="Hello")]
    request.model_settings = {}

    captured_request = None

    def mock_handler(req):
        nonlocal captured_request
        captured_request = req
        return MagicMock()

    middleware.wrap_model_call(request, mock_handler)

    request.override.assert_called_once()
    call_kwargs = request.override.call_args[1]
    assert "model_settings" in call_kwargs
    assert call_kwargs["model_settings"]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }


def test_bedrock_does_not_modify_messages() -> None:
    middleware = BedrockPromptCachingMiddleware()

    original_message = HumanMessage(content="Hello")
    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [original_message]
    request.model_settings = {}

    def mock_handler(req):
        return MagicMock()

    middleware.wrap_model_call(request, mock_handler)

    assert original_message.content == "Hello"
    assert request.messages[0].content == "Hello"


def test_bedrock_does_not_modify_system_prompt() -> None:
    middleware = BedrockPromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [HumanMessage(content="Hello")]
    request.model_settings = {}

    def mock_handler(req):
        return MagicMock()

    middleware.wrap_model_call(request, mock_handler)

    assert request.system_prompt == "You are helpful."


def test_bedrock_respects_min_messages_to_cache() -> None:
    middleware = BedrockPromptCachingMiddleware(min_messages_to_cache=3)

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [HumanMessage(content="Hello")]

    assert middleware._should_apply_caching(request) is False

    request.messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi"),
        HumanMessage(content="How are you?"),
    ]
    assert middleware._should_apply_caching(request) is True


def test_bedrock_custom_ttl() -> None:
    middleware = BedrockPromptCachingMiddleware(ttl="1h")

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [HumanMessage(content="Hello")]
    request.model_settings = {}

    def mock_handler(req):
        return MagicMock()

    middleware.wrap_model_call(request, mock_handler)

    call_kwargs = request.override.call_args[1]
    assert call_kwargs["model_settings"]["cache_control"]["ttl"] == "1h"


def test_converse_model_accepted() -> None:
    middleware = BedrockPromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrockConverse)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [HumanMessage(content="Hello")]

    assert middleware._should_apply_caching(request) is True


def test_nova_model_accepted_converse() -> None:
    middleware = BedrockPromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrockConverse)
    request.model.model_id = "amazon.nova-pro-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [HumanMessage(content="Hello")]

    assert middleware._should_apply_caching(request) is True


def test_nova_model_accepted_chatbedrock() -> None:
    middleware = BedrockPromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "amazon.nova-pro-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [HumanMessage(content="Hello")]

    assert middleware._should_apply_caching(request) is True


def test_unsupported_model_id_with_converse() -> None:
    middleware = BedrockPromptCachingMiddleware(unsupported_model_behavior="ignore")

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrockConverse)
    request.model.model_id = "amazon.titan-text-express-v1"
    request.system_prompt = "You are helpful."
    request.messages = []

    assert middleware._should_apply_caching(request) is False


def test_converse_passes_cache_control_via_model_settings() -> None:
    middleware = BedrockPromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrockConverse)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [HumanMessage(content="Hello")]
    request.model_settings = {}

    def mock_handler(req):
        return MagicMock()

    middleware.wrap_model_call(request, mock_handler)

    request.override.assert_called_once()
    call_kwargs = request.override.call_args[1]
    assert "model_settings" in call_kwargs
    assert call_kwargs["model_settings"]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }
