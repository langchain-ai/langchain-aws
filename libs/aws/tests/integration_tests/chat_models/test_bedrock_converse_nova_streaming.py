"""Integration tests for Nova Lite 2.0 streaming with system tools."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_aws import ChatBedrockConverse
from langchain_aws.chat_models.system_tools.nova import (
    NovaCodeInterpreterTool,
    NovaGroundingTool,
)


@pytest.fixture
def nova_model() -> ChatBedrockConverse:
    """Basic Nova model without reasoning."""
    return ChatBedrockConverse(
        model="us.amazon.nova-2-lite-v1:0",
        max_tokens=10000,
    )


@pytest.fixture
def nova_model_with_reasoning() -> ChatBedrockConverse:
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


def test_streaming_with_nova_grounding(nova_model: ChatBedrockConverse) -> None:
    """Test streaming with nova_grounding system tool."""
    model_with_tools = nova_model.bind_tools([NovaGroundingTool()])

    chunks = []
    for chunk in model_with_tools.stream("What's the latest news about SpaceX?"):
        chunks.append(chunk)
        # Verify each chunk is an AIMessageChunk
        assert isinstance(chunk, AIMessageChunk)

    # Verify we received chunks
    assert len(chunks) > 0

    # Verify final message by accumulating chunks
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    assert isinstance(full_message, AIMessage)
    assert len(full_message.content) > 0


def test_streaming_with_nova_code_interpreter(nova_model: ChatBedrockConverse) -> None:
    """Test streaming with nova_code_interpreter system tool."""
    model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

    chunks = []
    for chunk in model_with_tools.stream("Calculate 456 * 789 using Python"):
        chunks.append(chunk)
        # Verify each chunk is an AIMessageChunk
        assert isinstance(chunk, AIMessageChunk)

    # Verify we received chunks
    assert len(chunks) > 0

    # Verify final message by accumulating chunks
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    assert isinstance(full_message, AIMessage)
    assert len(full_message.content) > 0

    # Verify the calculation result (456 * 789 = 359784)
    response_text = str(full_message.content)
    assert "359784" in response_text or "359,784" in response_text


def test_streaming_reasoning_content_chunks(
    nova_model_with_reasoning: ChatBedrockConverse,
) -> None:
    """Test that reasoning content chunks stream correctly."""
    model_with_tools = nova_model_with_reasoning.bind_tools([NovaGroundingTool()])

    chunks = []
    reasoning_chunks = []

    for chunk in model_with_tools.stream("What are the latest AI breakthroughs?"):
        chunks.append(chunk)

        # Check if this chunk contains reasoning content
        if hasattr(chunk, "content") and isinstance(chunk.content, list):
            for block in chunk.content:
                if isinstance(block, dict) and block.get("type") == "reasoning":
                    reasoning_chunks.append(block)

    # Verify we received chunks
    assert len(chunks) > 0

    # Accumulate full message
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    assert isinstance(full_message, AIMessage)

    # Verify reasoning content exists in the full message
    content_blocks = full_message.content_blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0, (
        "Expected reasoning content blocks in streamed message"
    )


def test_streaming_tool_use_and_result_events(
    nova_model: ChatBedrockConverse,
) -> None:
    """Test that tool use and tool result events stream correctly."""
    model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

    chunks = []
    tool_use_chunks = []

    for chunk in model_with_tools.stream(
        "Use Python to calculate the square root of 144"
    ):
        chunks.append(chunk)

        # Check if this chunk contains tool use content
        if hasattr(chunk, "content") and isinstance(chunk.content, list):
            for block in chunk.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_use_chunks.append(block)

    # Verify we received chunks
    assert len(chunks) > 0

    # Accumulate full message
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    assert isinstance(full_message, AIMessage)
    assert len(full_message.content) > 0

    # Verify the calculation result (sqrt(144) = 12)
    response_text = str(full_message.content)
    assert "12" in response_text


def test_streaming_accumulating_chunks_into_full_message(
    nova_model_with_reasoning: ChatBedrockConverse,
) -> None:
    """Test accumulating chunks into full message with all content types."""
    model_with_tools = nova_model_with_reasoning.bind_tools([NovaGroundingTool()])

    chunks = []
    for chunk in model_with_tools.stream("What is the current population of Tokyo?"):
        chunks.append(chunk)
        assert isinstance(chunk, AIMessageChunk)

    # Verify we received chunks
    assert len(chunks) > 0

    # Accumulate chunks into full message
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    # Verify full message structure
    assert isinstance(full_message, AIMessage)
    assert len(full_message.content) > 0

    # Verify content blocks are properly accumulated
    content_blocks = full_message.content_blocks
    assert len(content_blocks) > 0

    # Verify we have reasoning blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0

    # Verify we have text blocks
    text_blocks = [b for b in content_blocks if b["type"] == "text"]
    assert len(text_blocks) > 0


def test_streaming_with_both_system_tools(nova_model: ChatBedrockConverse) -> None:
    """Test streaming with both nova_grounding and nova_code_interpreter."""
    model_with_tools = nova_model.bind_tools(
        [
            NovaGroundingTool(),
            NovaCodeInterpreterTool(),
        ]
    )

    chunks = []
    for chunk in model_with_tools.stream(
        "Search for the current price of Bitcoin and calculate 10% of that value"
    ):
        chunks.append(chunk)
        assert isinstance(chunk, AIMessageChunk)

    # Verify we received chunks
    assert len(chunks) > 0

    # Accumulate full message
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    assert isinstance(full_message, AIMessage)
    assert len(full_message.content) > 0


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_streaming_reasoning_with_different_effort_levels(effort: str) -> None:
    """Test streaming with different reasoning effort levels."""
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

    model_with_tools = model.bind_tools([NovaCodeInterpreterTool()])

    chunks = []
    for chunk in model_with_tools.stream("Calculate 15 factorial"):
        chunks.append(chunk)

    # Verify we received chunks
    assert len(chunks) > 0

    # Accumulate full message
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    assert isinstance(full_message, AIMessage)

    # Verify reasoning content exists
    content_blocks = full_message.content_blocks
    reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
    assert len(reasoning_blocks) > 0, f"Expected reasoning blocks with {effort} effort"


def test_streaming_chunk_content_structure(
    nova_model_with_reasoning: ChatBedrockConverse,
) -> None:
    """Test the structure of streaming chunks."""
    model_with_tools = nova_model_with_reasoning.bind_tools([NovaGroundingTool()])

    chunks = []
    for chunk in model_with_tools.stream("What is the weather in Paris?"):
        chunks.append(chunk)

        # Verify chunk structure
        assert isinstance(chunk, AIMessageChunk)
        assert hasattr(chunk, "content")

        # Content should be a list or string
        assert isinstance(chunk.content, (list, str))

    # Verify we received chunks
    assert len(chunks) > 0


def test_streaming_with_string_tool_names(nova_model: ChatBedrockConverse) -> None:
    """Test streaming with system tools specified as strings."""
    model_with_tools = nova_model.bind_tools(
        ["nova_grounding", "nova_code_interpreter"]
    )

    chunks = []
    for chunk in model_with_tools.stream("What is 25 * 25?"):
        chunks.append(chunk)
        assert isinstance(chunk, AIMessageChunk)

    # Verify we received chunks
    assert len(chunks) > 0

    # Accumulate full message
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    assert isinstance(full_message, AIMessage)
    assert len(full_message.content) > 0

    # Verify the calculation result (25 * 25 = 625)
    response_text = str(full_message.content)
    assert "625" in response_text


def test_streaming_empty_chunks_handling(nova_model: ChatBedrockConverse) -> None:
    """Test that empty or minimal chunks are handled correctly."""
    model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

    chunks = []
    for chunk in model_with_tools.stream("What is 2 + 2?"):
        chunks.append(chunk)
        # Each chunk should be valid even if content is minimal
        assert isinstance(chunk, AIMessageChunk)

    # Verify we received chunks
    assert len(chunks) > 0

    # Accumulate full message
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    assert isinstance(full_message, AIMessage)
    # Even for simple queries, we should get a response
    assert len(full_message.content) > 0
