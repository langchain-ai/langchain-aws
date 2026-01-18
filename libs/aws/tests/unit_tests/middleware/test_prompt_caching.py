# type:ignore

from unittest.mock import MagicMock

import pytest

from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_aws.middleware.prompt_caching import (
    BedrockConversePromptCachingMiddleware,
    BedrockPromptCachingMiddleware,
)


# BedrockPromptCachingMiddleware tests

def test_bedrock_applies_cache_control_to_string() -> None:
    middleware = BedrockPromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = "You are helpful."
    request.messages = []

    middleware._apply_cache_control(request)

    assert isinstance(request.system_prompt, list)
    assert request.system_prompt[0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_bedrock_preserves_existing_cache_control() -> None:
    middleware = BedrockPromptCachingMiddleware()

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrock)
    request.model.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request.system_prompt = [
        {"type": "text", "text": "Block 1.", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    request.messages = []

    middleware._apply_cache_control(request)

    assert request.system_prompt[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


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
    middleware = BedrockConversePromptCachingMiddleware(unsupported_model_behavior="ignore")

    request = MagicMock()
    request.model = MagicMock()
    request.system_prompt = "You are helpful."
    request.messages = []

    assert middleware._should_apply_caching(request) is False


def test_converse_skips_non_anthropic() -> None:
    middleware = BedrockConversePromptCachingMiddleware(unsupported_model_behavior="ignore")

    request = MagicMock()
    request.model = MagicMock(spec=ChatBedrockConverse)
    request.model.model = "amazon.titan-text-express-v1"
    request.system_prompt = "You are helpful."
    request.messages = []

    assert middleware._should_apply_caching(request) is False