# type:ignore

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_aws.middleware.prompt_caching import (
    BedrockConversePromptCachingMiddleware,
    BedrockPromptCachingMiddleware,
)

# BedrockPromptCachingMiddleware tests


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
    """Verify cache_control is passed through model_settings, not message modification."""
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

    # Verify override was called with cache_control in model_settings
    request.override.assert_called_once()
    call_kwargs = request.override.call_args[1]
    assert "model_settings" in call_kwargs
    assert call_kwargs["model_settings"]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }


def test_bedrock_does_not_modify_messages() -> None:
    """Verify messages are NOT modified directly (prevents checkpoint accumulation)."""
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

    # Original message should remain unchanged
    assert original_message.content == "Hello"
    assert request.messages[0].content == "Hello"


def test_bedrock_does_not_modify_system_prompt() -> None:
    """Verify system prompt is NOT modified."""
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

    # System prompt should remain unchanged
    assert request.system_prompt == "You are helpful."


def test_bedrock_respects_min_messages_to_cache() -> None:
    middleware = BedrockPromptCachingMiddleware(min_messages_to_cache=3)

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = [HumanMessage(content="Hello")]

    # 1 message + 1 system = 2, which is less than min_messages_to_cache=3
    assert middleware._should_apply_caching(request) is False

    request.messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi"),
        HumanMessage(content="How are you?"),
    ]
    # 3 messages + 1 system = 4, which is >= min_messages_to_cache=3
    assert middleware._should_apply_caching(request) is True


def test_bedrock_custom_ttl() -> None:
    """Verify custom TTL is passed correctly."""
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


# BedrockConversePromptCachingMiddleware tests


def test_converse_applies_cache_point_to_string() -> None:
    middleware = BedrockConversePromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrockConverse)
    request.model.model = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = []

    middleware._apply_cache_point(request)

    assert isinstance(request.system_prompt, list)
    assert request.system_prompt[0] == {"text": "You are helpful."}
    assert request.system_prompt[1] == {"cachePoint": {"type": "default"}}


def test_converse_preserves_existing_cache_point() -> None:
    middleware = BedrockConversePromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrockConverse)
    request.model.model = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = [
        {"text": "Block 1."},
        {"cachePoint": {"type": "default"}},
    ]
    request.messages = []

    middleware._apply_cache_point(request)

    # Should not add another cachePoint
    assert len(request.system_prompt) == 2


def test_converse_skips_non_chatbedrockconverse() -> None:
    middleware = BedrockConversePromptCachingMiddleware(
        unsupported_model_behavior="ignore"
    )

    request = MagicMock()
    request.model = MagicMock()
    request.system_prompt = "You are helpful."
    request.messages = []

    assert middleware._should_apply_caching(request) is False


def test_converse_skips_non_anthropic() -> None:
    middleware = BedrockConversePromptCachingMiddleware(
        unsupported_model_behavior="ignore"
    )

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrockConverse)
    request.model.model = "amazon.titan-text-express-v1"
    request.system_prompt = "You are helpful."
    request.messages = []

    assert middleware._should_apply_caching(request) is False