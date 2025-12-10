"""Integration tests for Nova Lite 2.0 nova_grounding system tool."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_aws import ChatBedrockConverse
from langchain_aws.tools import NovaGroundingTool


def test_nova_grounding_tool_basic() -> None:
    """Test nova_grounding system tool with real API call."""
    model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0",
        max_tokens=10000,
    )

    model_with_tools = model.bind_tools([NovaGroundingTool()])
    response = model_with_tools.invoke("Who won the 2024 Nobel Prize in Physics?")

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify the response contains text
    assert any(
        isinstance(block, dict) and block.get("type") == "text"
        for block in response.content
    )


def test_nova_grounding_tool_with_string() -> None:
    """Test nova_grounding using direct string instead of helper class."""
    model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0",
        max_tokens=10000,
    )

    model_with_tools = model.bind_tools(["nova_grounding"])
    response = model_with_tools.invoke("What is the current weather in Seattle?")

    assert isinstance(response, AIMessage)
    assert len(response.content) > 0


def test_nova_grounding_with_reasoning_enabled() -> None:
    """Test nova_grounding with reasoning enabled."""
    model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0",
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "low",
            }
        },
    )

    model_with_tools = model.bind_tools([NovaGroundingTool()])
    response = model_with_tools.invoke("Who won the Oscar for best actress in 2024?")

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify content blocks include reasoning
    content_blocks = response.content_blocks
    assert any(block["type"] == "reasoning" for block in content_blocks)


def test_nova_grounding_content_blocks() -> None:
    """Test that content blocks are properly parsed for nova_grounding."""
    model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0",
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "low",
            }
        },
    )

    model_with_tools = model.bind_tools([NovaGroundingTool()])
    response = model_with_tools.invoke(
        "What are the latest developments in quantum computing?"
    )

    # Verify response structure
    assert isinstance(response, AIMessage)

    # Get content blocks
    content_blocks = response.content_blocks
    assert len(content_blocks) > 0

    # Verify we have reasoning content
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0

    # Verify we have text content
    text_blocks = [b for b in content_blocks if b["type"] == "text"]
    assert len(text_blocks) > 0


def test_nova_grounding_streaming() -> None:
    """Test streaming with nova_grounding system tool."""
    model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0",
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "low",
            }
        },
    )

    model_with_tools = model.bind_tools([NovaGroundingTool()])

    chunks = []
    for chunk in model_with_tools.stream("What's the latest news about AI?"):
        chunks.append(chunk)

    # Verify we received chunks
    assert len(chunks) > 0

    # Verify final message by accumulating chunks
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message += chunk

    assert isinstance(full_message, AIMessage)
    assert len(full_message.content) > 0


def test_nova_grounding_with_different_reasoning_levels() -> None:
    """Test nova_grounding with different reasoning effort levels."""
    for effort in ["low", "medium", "high"]:
        model = ChatBedrockConverse(
            model="amazon.nova-2-lite-v1:0",
            max_tokens=10000,
            additional_model_request_fields={
                "reasoningConfig": {
                    "type": "enabled",
                    "maxReasoningEffort": effort,
                }
            },
        )

        model_with_tools = model.bind_tools([NovaGroundingTool()])
        response = model_with_tools.invoke("What is the capital of France?")

        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        # Verify reasoning content exists
        content_blocks = response.content_blocks
        reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
        assert len(reasoning_blocks) > 0


def test_nova_grounding_multi_turn_conversation() -> None:
    """Test nova_grounding in a multi-turn conversation."""
    model = ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0",
        max_tokens=10000,
    )

    model_with_tools = model.bind_tools([NovaGroundingTool()])

    # First turn
    message1 = HumanMessage("What's the population of Tokyo?")
    response1 = model_with_tools.invoke([message1])
    assert isinstance(response1, AIMessage)

    # Second turn - follow-up question
    message2 = HumanMessage("How does that compare to New York City?")
    response2 = model_with_tools.invoke([message1, response1, message2])
    assert isinstance(response2, AIMessage)
    assert len(response2.content) > 0
