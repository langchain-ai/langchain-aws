"""Integration tests for prompt caching with ChatBedrockConverse."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_aws import ChatBedrockConverse


@pytest.mark.integration
def test_prompt_caching_with_1h_ttl():
    """Test prompt caching with 1-hour TTL."""
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        additional_model_request_fields={
            "anthropicBeta": ["extended-cache-ttl-2025-04-11"]
        }
    )
    
    # Create cache points
    cache_1h = ChatBedrockConverse.create_cache_point(ttl="1h")
    cache_5m = ChatBedrockConverse.create_cache_point()
    
    messages = [
        HumanMessage(content=[
            "You are a helpful assistant.",
            cache_1h,
            "Current context.",
            cache_5m,
            "What is 2+2?"
        ])
    ]
    
    # First invocation - should create cache
    response1 = llm.invoke(messages)
    assert response1.content
    assert hasattr(response1, 'usage_metadata')
    
    # Check if cache was created
    if response1.usage_metadata and 'input_token_details' in response1.usage_metadata:
        cache_creation = response1.usage_metadata['input_token_details'].get('cache_creation', 0)
        assert cache_creation > 0, "Expected cache creation on first call"
    
    # Second invocation - should use cache
    response2 = llm.invoke(messages)
    assert response2.content
    
    # Check if cache was used
    if response2.usage_metadata and 'input_token_details' in response2.usage_metadata:
        cache_read = response2.usage_metadata['input_token_details'].get('cache_read', 0)
        assert cache_read > 0, "Expected cache read on second call"


@pytest.mark.integration
def test_cache_ordering_validation():
    """Test that 1-hour cache entries must appear before 5-minute entries."""
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        additional_model_request_fields={
            "anthropicBeta": ["extended-cache-ttl-2025-04-11"]
        }
    )
    
    cache_1h = ChatBedrockConverse.create_cache_point(ttl="1h")
    cache_5m = ChatBedrockConverse.create_cache_point()
    
    # Correct ordering: 1h before 5m
    messages_correct = [
        HumanMessage(content=[
            "Long-term context",
            cache_1h,
            "Short-term context",
            cache_5m,
            "Question"
        ])
    ]
    
    # This should work fine
    response = llm.invoke(messages_correct)
    assert response.content
    
    # Incorrect ordering: 5m before 1h (might fail based on Anthropic's requirements)
    messages_incorrect = [
        HumanMessage(content=[
            "Short-term context",
            cache_5m,
            "Long-term context",
            cache_1h,  # This violates the ordering requirement
            "Question"
        ])
    ]
    
    # Note: Whether this fails depends on Anthropic's validation
    # The documentation states 1h cache must come before 5m cache
    try:
        response = llm.invoke(messages_incorrect)
        # If it succeeds, we should at least verify response is valid
        assert response.content
    except Exception as e:
        # Expected behavior if Anthropic enforces ordering
        assert "cache" in str(e).lower() or "order" in str(e).lower()


@pytest.mark.integration
def test_mixed_message_types_with_caching():
    """Test caching with SystemMessage and HumanMessage."""
    llm = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        additional_model_request_fields={
            "anthropicBeta": ["extended-cache-ttl-2025-04-11"]
        }
    )
    
    cache_1h = ChatBedrockConverse.create_cache_point(ttl="1h")
    
    messages = [
        SystemMessage(content=[
            "You are an expert Python programmer.",
            cache_1h
        ]),
        HumanMessage("Write a function to calculate factorial.")
    ]
    
    response = llm.invoke(messages)
    assert response.content
    assert "def" in response.content or "factorial" in response.content.lower()