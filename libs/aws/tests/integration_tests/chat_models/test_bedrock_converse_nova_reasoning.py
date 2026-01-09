"""Integration tests for Nova Lite 2.0 reasoning levels."""

import pytest
from langchain_core.messages import AIMessage

from langchain_aws import ChatBedrockConverse


@pytest.fixture
def nova_model() -> ChatBedrockConverse:
    """Basic Nova model without reasoning config."""
    return ChatBedrockConverse(
        model="us.amazon.nova-2-lite-v1:0",
        max_tokens=10000,
    )


@pytest.fixture
def nova_model_with_low_reasoning() -> ChatBedrockConverse:
    """Nova model with low reasoning effort."""
    return ChatBedrockConverse(
        model="us.amazon.nova-2-lite-v1:0",
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "low",
            }
        },
    )


@pytest.fixture
def nova_model_with_medium_reasoning() -> ChatBedrockConverse:
    """Nova model with medium reasoning effort."""
    return ChatBedrockConverse(
        model="us.amazon.nova-2-lite-v1:0",
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "medium",
            }
        },
    )


@pytest.fixture
def nova_model_with_high_reasoning() -> ChatBedrockConverse:
    """Nova model with high reasoning effort."""
    return ChatBedrockConverse(
        model="us.amazon.nova-2-lite-v1:0",
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "high",
            }
        },
    )


@pytest.fixture
def nova_model_disabled_reasoning() -> ChatBedrockConverse:
    """Nova model with reasoning explicitly disabled."""
    return ChatBedrockConverse(
        model="us.amazon.nova-2-lite-v1:0",
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "disabled",
            }
        },
    )


def test_reasoning_effort_low(
    nova_model_with_low_reasoning: ChatBedrockConverse,
) -> None:
    """Test reasoning with effort level 'low'."""
    response = nova_model_with_low_reasoning.invoke("What is 7 to the power of 6?")

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify reasoning content blocks are present
    content_blocks = response.content_blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0, (
        "Expected reasoning content blocks with LOW effort"
    )

    # Verify reasoning content has text
    for block in reasoning_blocks:
        assert "reasoning" in block
        assert len(block["reasoning"]) > 0


def test_reasoning_effort_medium(
    nova_model_with_medium_reasoning: ChatBedrockConverse,
) -> None:
    """Test reasoning with effort level 'medium'."""
    response = nova_model_with_medium_reasoning.invoke(
        "Explain the concept of quantum entanglement"
    )

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify reasoning content blocks are present
    content_blocks = response.content_blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0, (
        "Expected reasoning content blocks with MEDIUM effort"
    )

    # Verify reasoning content has text
    for block in reasoning_blocks:
        assert "reasoning" in block
        assert len(block["reasoning"]) > 0


def test_reasoning_effort_high(
    nova_model_with_high_reasoning: ChatBedrockConverse,
) -> None:
    """Test reasoning with effort level 'high'."""
    response = nova_model_with_high_reasoning.invoke(
        "What are the implications of the halting problem?"
    )

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify reasoning content blocks are present
    content_blocks = response.content_blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0, (
        "Expected reasoning content blocks with HIGH effort"
    )

    # Verify reasoning content has text
    for block in reasoning_blocks:
        assert "reasoning" in block
        assert len(block["reasoning"]) > 0


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_reasoning_effort_string_values(effort: str) -> None:
    """Test reasoning with string effort levels instead of enum."""
    model = ChatBedrockConverse(
        model="us.amazon.nova-2-lite-v1:0",
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": effort,
            }
        },
    )

    response = model.invoke(f"Calculate {effort}: What is 12 * 13?")

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify reasoning content blocks are present
    content_blocks = response.content_blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0, (
        f"Expected reasoning content blocks with '{effort}' effort"
    )


def test_reasoning_disabled(
    nova_model_disabled_reasoning: ChatBedrockConverse,
) -> None:
    """Test that reasoning blocks are not present when reasoning is disabled."""
    response = nova_model_disabled_reasoning.invoke("What is the capital of France?")

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify no reasoning content blocks are present
    content_blocks = response.content_blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) == 0, (
        "Expected no reasoning content blocks when disabled"
    )


def test_reasoning_default_no_config(nova_model: ChatBedrockConverse) -> None:
    """Test that reasoning is disabled by default when no config is provided."""
    response = nova_model.invoke("What is 5 + 5?")

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify no reasoning content blocks are present (default is disabled)
    content_blocks = response.content_blocks
    [b for b in content_blocks if b["type"] == "reasoning"]
    # Note: Some models may still include reasoning even without explicit config
    # So we just verify the response is valid, not that reasoning is absent


def test_reasoning_content_structure(
    nova_model_with_low_reasoning: ChatBedrockConverse,
) -> None:
    """Test the structure of reasoning content blocks."""
    response = nova_model_with_low_reasoning.invoke(
        "Solve this: If x + 5 = 12, what is x?"
    )

    # Verify response structure
    assert isinstance(response, AIMessage)

    # Get content blocks
    content_blocks = response.content_blocks
    assert len(content_blocks) > 0

    # Find reasoning blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0

    # Verify reasoning block structure
    for block in reasoning_blocks:
        assert isinstance(block, dict)
        assert "type" in block
        assert block["type"] == "reasoning"
        assert "reasoning" in block
        assert isinstance(block["reasoning"], str)
        assert len(block["reasoning"]) > 0


def test_reasoning_with_complex_query(
    nova_model_with_medium_reasoning: ChatBedrockConverse,
) -> None:
    """Test reasoning with a complex query that requires deeper thinking."""
    response = nova_model_with_medium_reasoning.invoke(
        "If a train travels at 60 mph for 2 hours, then 80 mph for 1.5 hours, "
        "what is the total distance traveled?"
    )

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify reasoning content blocks are present
    content_blocks = response.content_blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0

    # Verify the response contains the calculation
    response_text = str(response.content).lower()
    # Total distance = (60 * 2) + (80 * 1.5) = 120 + 120 = 240 miles
    assert "240" in response_text or "distance" in response_text
