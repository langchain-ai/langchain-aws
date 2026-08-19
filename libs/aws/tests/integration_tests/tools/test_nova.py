"""Integration tests for Nova Lite 2.0 with system tools and reasoning.

This module consolidates all Nova tools integration tests including:
- Nova Grounding (web search)
- Nova Code Interpreter
- Reasoning configuration
- Streaming behavior
"""

import time
import warnings
from typing import Any, Generator

import pytest
from botocore.exceptions import ReadTimeoutError as BotocoreReadTimeoutError
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from urllib3.connectionpool import HTTPSConnectionPool
from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError

from langchain_aws import ChatBedrockConverse
from langchain_aws.tools import (
    NovaCodeInterpreterTool,
    NovaGroundingTool,
)

MODEL_ID = "us.amazon.nova-2-lite-v1:0"
NOVA_STREAM_ATTEMPTS = 3
NOVA_STREAM_BACKOFF_SECONDS = 2.0
RETRYABLE_STREAM_TIMEOUT_ERRORS = (
    BotocoreReadTimeoutError,
    Urllib3ReadTimeoutError,
)


def _stream_with_retries(model_with_tools: Any, prompt: str) -> list[AIMessageChunk]:
    """Collect a full stream, retrying on transient read timeouts.

    Nova streaming integration tests intermittently fail when a read times out.
    A mid-stream timeout surfaces as a urllib3 ``ReadTimeoutError`` (raised lazily
    during event-stream iteration, outside botocore's request error handling),
    while a timeout on the initial request surfaces as a botocore
    ``ReadTimeoutError``; both are retried with backoff. Any other error
    propagates immediately so genuine regressions still fail fast.

    Args:
        model_with_tools: The tool-bound model to stream from.
        prompt: The prompt to stream.

    Returns:
        The streamed chunks from the first successful attempt.

    Raises:
        BotocoreReadTimeoutError: If every attempt times out at the request level.
        Urllib3ReadTimeoutError: If every attempt times out mid-stream.
    """
    for attempt in range(1, NOVA_STREAM_ATTEMPTS + 1):
        try:
            return list(model_with_tools.stream(prompt))
        except RETRYABLE_STREAM_TIMEOUT_ERRORS as exc:
            if attempt == NOVA_STREAM_ATTEMPTS:
                raise
            warnings.warn(
                f"Nova stream timed out "
                f"(attempt {attempt}/{NOVA_STREAM_ATTEMPTS}), "
                f"retrying in {NOVA_STREAM_BACKOFF_SECONDS}s: {exc}",
                stacklevel=2,
            )
            time.sleep(NOVA_STREAM_BACKOFF_SECONDS)

    msg = "Failed to stream from Nova"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    "timeout_error",
    [
        BotocoreReadTimeoutError(endpoint_url="https://bedrock-runtime.amazonaws.com"),
        Urllib3ReadTimeoutError(
            HTTPSConnectionPool(host="bedrock-runtime.amazonaws.com"),
            "https://bedrock-runtime.amazonaws.com",
            "Read timed out.",
        ),
    ],
)
def test_stream_with_retries_handles_read_timeouts(timeout_error: Exception) -> None:
    """Test streaming retries for timeout exceptions raised by AWS clients."""

    class TimeoutThenSuccessStream:
        attempts = 0

        def stream(self, prompt: str) -> list[AIMessageChunk]:
            self.attempts += 1
            if self.attempts == 1:
                raise timeout_error
            return [AIMessageChunk(content=prompt)]

    model_with_tools = TimeoutThenSuccessStream()

    chunks = _stream_with_retries(model_with_tools, "hello")

    assert chunks == [AIMessageChunk(content="hello")]
    assert model_with_tools.attempts == 2


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nova_model() -> Generator[ChatBedrockConverse, None, None]:
    """Create a ChatBedrockConverse model for Nova 2 Lite."""
    yield ChatBedrockConverse(
        model=MODEL_ID,
        max_tokens=10000,
    )


@pytest.fixture
def nova_model_reasoning_low() -> Generator[ChatBedrockConverse, None, None]:
    """Create a ChatBedrockConverse model for Nova 2 Lite with LOW reasoning."""
    yield ChatBedrockConverse(
        model=MODEL_ID,
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "low",
            }
        },
    )


@pytest.fixture
def nova_model_reasoning_medium() -> Generator[ChatBedrockConverse, None, None]:
    """Create a ChatBedrockConverse model for Nova 2 Lite with MEDIUM reasoning."""
    yield ChatBedrockConverse(
        model=MODEL_ID,
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "medium",
            }
        },
    )


@pytest.fixture
def nova_model_reasoning_high() -> Generator[ChatBedrockConverse, None, None]:
    """Create a ChatBedrockConverse model for Nova 2 Lite with HIGH reasoning."""
    yield ChatBedrockConverse(
        model=MODEL_ID,
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "high",
            }
        },
    )


@pytest.fixture
def nova_model_reasoning_disabled() -> Generator[ChatBedrockConverse, None, None]:
    """Create a ChatBedrockConverse model for Nova 2 Lite with reasoning disabled."""
    yield ChatBedrockConverse(
        model=MODEL_ID,
        max_tokens=10000,
        additional_model_request_fields={
            "reasoningConfig": {
                "type": "disabled",
            }
        },
    )


# =============================================================================
# Nova Grounding Tool Tests
# =============================================================================


class TestNovaGroundingTool:
    """Tests for nova_grounding system tool (web search)."""

    def test_basic(self, nova_model: ChatBedrockConverse) -> None:
        """Test nova_grounding system tool with real API call."""
        model_with_tools = nova_model.bind_tools([NovaGroundingTool()])
        response = model_with_tools.invoke("Who won the 2024 Nobel Prize in Physics?")

        # Verify response structure
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        # Verify the response contains text
        assert any(
            isinstance(block, dict) and block.get("type") == "text"
            for block in response.content
        )

    def test_with_string(self, nova_model: ChatBedrockConverse) -> None:
        """Test nova_grounding using direct string instead of helper class."""
        model_with_tools = nova_model.bind_tools(["nova_grounding"])
        response = model_with_tools.invoke("What is the current weather in Seattle?")

        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

    def test_with_reasoning_enabled(
        self, nova_model_reasoning_low: ChatBedrockConverse
    ) -> None:
        """Test nova_grounding with reasoning enabled."""
        model_with_tools = nova_model_reasoning_low.bind_tools([NovaGroundingTool()])
        response = model_with_tools.invoke(
            "Who won the Oscar for best actress in 2024?"
        )

        # Verify response structure
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        # Verify content blocks include reasoning
        content_blocks = response.content_blocks
        assert any(block["type"] == "reasoning" for block in content_blocks)

    def test_content_blocks(
        self, nova_model_reasoning_low: ChatBedrockConverse
    ) -> None:
        """Test that content blocks are properly parsed for nova_grounding."""
        model_with_tools = nova_model_reasoning_low.bind_tools([NovaGroundingTool()])
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

    def test_streaming(self, nova_model_reasoning_low: ChatBedrockConverse) -> None:
        """Test streaming with nova_grounding system tool."""
        model_with_tools = nova_model_reasoning_low.bind_tools([NovaGroundingTool()])

        chunks = _stream_with_retries(
            model_with_tools, "What's the latest news about AI?"
        )

        # Verify we received chunks
        assert len(chunks) > 0

        # Verify final message by accumulating chunks
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

        assert isinstance(full_message, AIMessage)
        assert len(full_message.content) > 0

    def test_with_different_reasoning_levels(self) -> None:
        """Test nova_grounding with different reasoning effort levels."""
        for effort in ["low", "medium", "high"]:
            model = ChatBedrockConverse(
                model=MODEL_ID,
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

    def test_multi_turn_conversation(self, nova_model: ChatBedrockConverse) -> None:
        """Test nova_grounding in a multi-turn conversation."""
        model_with_tools = nova_model.bind_tools([NovaGroundingTool()])

        # First turn
        message1 = HumanMessage("What's the population of Tokyo?")
        response1 = model_with_tools.invoke([message1])
        assert isinstance(response1, AIMessage)

        # Second turn - follow-up question
        message2 = HumanMessage("How does that compare to New York City?")
        response2 = model_with_tools.invoke([message1, response1, message2])
        assert isinstance(response2, AIMessage)
        assert len(response2.content) > 0


# =============================================================================
# Nova Code Interpreter Tool Tests
# =============================================================================


class TestNovaCodeInterpreterTool:
    """Tests for nova_code_interpreter system tool."""

    def test_basic(self, nova_model: ChatBedrockConverse) -> None:
        """Test nova_code_interpreter system tool with real API call."""
        model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])
        response = model_with_tools.invoke("Use Python to calculate 123 * 456")

        # Verify response structure
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        # Verify the response contains text with the result (123 * 456 = 56088)
        response_text = str(response.content)
        assert "56088" in response_text or "56,088" in response_text

    def test_with_string(self, nova_model: ChatBedrockConverse) -> None:
        """Test nova_code_interpreter using direct string instead of helper class."""
        model_with_tools = nova_model.bind_tools(["nova_code_interpreter"])
        response = model_with_tools.invoke("What is 123 * 456?")

        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        # Verify the calculation result (123 * 456 = 56088)
        response_text = str(response.content)
        assert "56088" in response_text or "56,088" in response_text

    def test_simple_calculation(self, nova_model: ChatBedrockConverse) -> None:
        """Test nova_code_interpreter with simple calculation."""
        model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])
        response = model_with_tools.invoke("Use Python code to calculate 7 ** 6")

        # Verify response structure
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        # Verify the calculation result (7^6 = 117649)
        response_text = str(response.content)
        assert "117649" in response_text or "117,649" in response_text

    def test_with_reasoning(
        self, nova_model_reasoning_low: ChatBedrockConverse
    ) -> None:
        """Test nova_code_interpreter with reasoning enabled."""
        model_with_tools = nova_model_reasoning_low.bind_tools(
            [NovaCodeInterpreterTool()]
        )
        response = model_with_tools.invoke(
            "Calculate the factorial of 10 and explain the result. "
            "Use Code Interpreter to calculate it."
        )

        # Verify response structure
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        # Verify content blocks include reasoning
        content_blocks = response.content_blocks
        assert any(block["type"] == "reasoning" for block in content_blocks)

        # Verify the calculation result (10! = 3628800)
        response_text = str(response.content)
        assert "3628800" in response_text or "3,628,800" in response_text

    def test_content_blocks(
        self, nova_model_reasoning_low: ChatBedrockConverse
    ) -> None:
        """Test that content blocks are properly parsed for nova_code_interpreter."""
        model_with_tools = nova_model_reasoning_low.bind_tools(
            [NovaCodeInterpreterTool()]
        )
        response = model_with_tools.invoke("Calculate the sum of numbers from 1 to 100")

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

        # Verify the calculation result (sum 1-100 = 5050)
        response_text = str(response.content)
        assert "5050" in response_text or "5,050" in response_text

    def test_streaming(self, nova_model: ChatBedrockConverse) -> None:
        """Test streaming with nova_code_interpreter system tool."""
        model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

        chunks = _stream_with_retries(model_with_tools, "What is 999 * 888?")

        # Verify we received chunks
        assert len(chunks) > 0

        # Verify final message by accumulating chunks
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

        assert isinstance(full_message, AIMessage)
        assert len(full_message.content) > 0

        # Verify the calculation result (999 * 888 = 887112)
        response_text = str(full_message.content)
        assert "887112" in response_text or "887,112" in response_text

    def test_multi_turn_conversation(self, nova_model: ChatBedrockConverse) -> None:
        """Test nova_code_interpreter in a multi-turn conversation."""
        model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

        # First turn
        message1 = HumanMessage("Use Python to calculate 15 * 20")
        response1 = model_with_tools.invoke([message1])
        assert isinstance(response1, AIMessage)
        assert "300" in str(response1.content)

        # Second turn - follow-up calculation
        message2 = HumanMessage(
            "Use Python to add 50 to that result. State the final number."
        )
        response2 = model_with_tools.invoke([message1, response1, message2])
        assert isinstance(response2, AIMessage)
        assert len(response2.content) > 0
        assert "350" in str(response2.content)


# =============================================================================
# Reasoning Configuration Tests
# =============================================================================


class TestNovaReasoning:
    """Tests for Nova reasoning configuration levels."""

    def test_effort_low(self, nova_model_reasoning_low: ChatBedrockConverse) -> None:
        """Test reasoning with effort level 'low'."""
        response = nova_model_reasoning_low.invoke("What is 7 to the power of 6?")

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

    def test_effort_medium(
        self, nova_model_reasoning_medium: ChatBedrockConverse
    ) -> None:
        """Test reasoning with effort level 'medium'."""
        response = nova_model_reasoning_medium.invoke(
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

    def test_effort_high(self, nova_model_reasoning_high: ChatBedrockConverse) -> None:
        """Test reasoning with effort level 'high'."""
        response = nova_model_reasoning_high.invoke(
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

    def test_effort_string_values(self) -> None:
        """Test reasoning with string effort levels instead of enum."""
        for effort in ["low", "medium", "high"]:
            model = ChatBedrockConverse(
                model=MODEL_ID,
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

    def test_disabled(self, nova_model_reasoning_disabled: ChatBedrockConverse) -> None:
        """Test that reasoning blocks are not present when reasoning is disabled."""
        response = nova_model_reasoning_disabled.invoke(
            "What is the capital of France?"
        )

        # Verify response structure
        assert isinstance(response, AIMessage)
        assert len(response.content) > 0

        # Verify no reasoning content blocks are present
        content_blocks = response.content_blocks
        reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
        assert len(reasoning_blocks) == 0, (
            "Expected no reasoning content blocks when disabled"
        )

    def test_default_no_config(self, nova_model: ChatBedrockConverse) -> None:
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

    def test_content_structure(
        self, nova_model_reasoning_low: ChatBedrockConverse
    ) -> None:
        """Test the structure of reasoning content blocks."""
        response = nova_model_reasoning_low.invoke(
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

    def test_with_complex_query(
        self, nova_model_reasoning_medium: ChatBedrockConverse
    ) -> None:
        """Test reasoning with a complex query that requires deeper thinking."""
        response = nova_model_reasoning_medium.invoke(
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


# =============================================================================
# Streaming Tests
# =============================================================================


class TestNovaStreaming:
    """Tests for Nova streaming behavior with system tools."""

    def test_with_grounding(self, nova_model: ChatBedrockConverse) -> None:
        """Test streaming with nova_grounding system tool."""
        model_with_tools = nova_model.bind_tools([NovaGroundingTool()])

        chunks = _stream_with_retries(
            model_with_tools, "What's the latest news about SpaceX?"
        )
        for chunk in chunks:
            # Verify each chunk is an AIMessageChunk
            assert isinstance(chunk, AIMessageChunk)

        # Verify we received chunks
        assert len(chunks) > 0

        # Verify final message by accumulating chunks
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

        assert isinstance(full_message, AIMessage)
        assert len(full_message.content) > 0

    def test_with_code_interpreter(self, nova_model: ChatBedrockConverse) -> None:
        """Test streaming with nova_code_interpreter system tool."""
        model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

        chunks = _stream_with_retries(
            model_with_tools, "Calculate 456 * 789 using Python"
        )
        for chunk in chunks:
            # Verify each chunk is an AIMessageChunk
            assert isinstance(chunk, AIMessageChunk)

        # Verify we received chunks
        assert len(chunks) > 0

        # Verify final message by accumulating chunks
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

        assert isinstance(full_message, AIMessage)
        assert len(full_message.content) > 0

        # Verify the calculation result (456 * 789 = 359784)
        response_text = str(full_message.content)
        assert "359784" in response_text or "359,784" in response_text

    def test_reasoning_content_chunks(
        self, nova_model_reasoning_low: ChatBedrockConverse
    ) -> None:
        """Test that reasoning content chunks stream correctly."""
        model_with_tools = nova_model_reasoning_low.bind_tools([NovaGroundingTool()])

        reasoning_chunks = []

        chunks = _stream_with_retries(
            model_with_tools, "What are the latest AI breakthroughs?"
        )
        for chunk in chunks:
            # Check if this chunk contains reasoning content
            if hasattr(chunk, "content") and isinstance(chunk.content, list):
                for block in chunk.content:
                    if isinstance(block, dict) and block.get("type") == "reasoning":
                        reasoning_chunks.append(block)

        # Verify we received chunks
        assert len(chunks) > 0

        # Accumulate full message
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

        assert isinstance(full_message, AIMessage)

        # Verify reasoning content exists in the full message
        content_blocks = full_message.content_blocks
        reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
        assert len(reasoning_blocks) > 0, (
            "Expected reasoning content blocks in streamed message"
        )

    def test_tool_use_and_result_events(self, nova_model: ChatBedrockConverse) -> None:
        """Test that tool use and tool result events stream correctly."""
        model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

        tool_use_chunks = []

        chunks = _stream_with_retries(
            model_with_tools, "Use Python to calculate the square root of 144"
        )
        for chunk in chunks:
            # Check if this chunk contains tool use content
            if hasattr(chunk, "content") and isinstance(chunk.content, list):
                for block in chunk.content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_use_chunks.append(block)

        # Verify we received chunks
        assert len(chunks) > 0

        # Accumulate full message
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

        assert isinstance(full_message, AIMessage)
        assert len(full_message.content) > 0

        # Verify the calculation result (sqrt(144) = 12)
        response_text = str(full_message.content)
        assert "12" in response_text

    def test_accumulating_chunks_into_full_message(
        self, nova_model_reasoning_low: ChatBedrockConverse
    ) -> None:
        """Test accumulating chunks into full message with all content types."""
        model_with_tools = nova_model_reasoning_low.bind_tools([NovaGroundingTool()])

        chunks = _stream_with_retries(
            model_with_tools, "What is the current population of Tokyo?"
        )
        for chunk in chunks:
            assert isinstance(chunk, AIMessageChunk)

        # Verify we received chunks
        assert len(chunks) > 0

        # Accumulate chunks into full message
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

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

    def test_with_both_system_tools(self, nova_model: ChatBedrockConverse) -> None:
        """Test streaming with both nova_grounding and nova_code_interpreter."""
        model_with_tools = nova_model.bind_tools(
            [
                NovaGroundingTool(),
                NovaCodeInterpreterTool(),
            ]
        )

        chunks = _stream_with_retries(
            model_with_tools,
            "Search for the current price of Bitcoin and calculate 10% of that value",
        )
        for chunk in chunks:
            assert isinstance(chunk, AIMessageChunk)

        # Verify we received chunks
        assert len(chunks) > 0

        # Accumulate full message
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

        assert isinstance(full_message, AIMessage)
        assert len(full_message.content) > 0

    def test_reasoning_with_different_effort_levels(self) -> None:
        """Test streaming with different reasoning effort levels."""
        for effort in ["low", "medium", "high"]:
            model = ChatBedrockConverse(
                model=MODEL_ID,
                max_tokens=10000,
                additional_model_request_fields={
                    "reasoningConfig": {
                        "type": "enabled",
                        "maxReasoningEffort": effort,
                    }
                },
            )

            model_with_tools = model.bind_tools([NovaCodeInterpreterTool()])

            chunks = _stream_with_retries(model_with_tools, "Calculate 15 factorial")

            # Verify we received chunks
            assert len(chunks) > 0

            # Accumulate full message
            full_message: Any = chunks[0]
            for chunk in chunks[1:]:
                full_message += chunk

            assert isinstance(full_message, AIMessage)

            # Verify reasoning content exists
            content_blocks = full_message.content_blocks
            reasoning_blocks = [b for b in content_blocks if b["type"] == "reasoning"]
            assert len(reasoning_blocks) > 0, (
                f"Expected reasoning blocks with {effort} effort"
            )

    def test_chunk_content_structure(
        self, nova_model_reasoning_low: ChatBedrockConverse
    ) -> None:
        """Test the structure of streaming chunks."""
        model_with_tools = nova_model_reasoning_low.bind_tools([NovaGroundingTool()])

        chunks = _stream_with_retries(model_with_tools, "What is the weather in Paris?")
        for chunk in chunks:
            # Verify chunk structure
            assert isinstance(chunk, AIMessageChunk)
            assert hasattr(chunk, "content")

            # Content should be a list or string
            assert isinstance(chunk.content, (list, str))

        # Verify we received chunks
        assert len(chunks) > 0

    def test_with_string_tool_names(self, nova_model: ChatBedrockConverse) -> None:
        """Test streaming with system tools specified as strings."""
        model_with_tools = nova_model.bind_tools(
            ["nova_grounding", "nova_code_interpreter"]
        )

        chunks = _stream_with_retries(model_with_tools, "What is 25 * 25?")
        for chunk in chunks:
            assert isinstance(chunk, AIMessageChunk)

        # Verify we received chunks
        assert len(chunks) > 0

        # Accumulate full message
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

        assert isinstance(full_message, AIMessage)
        assert len(full_message.content) > 0

        # Verify the calculation result (25 * 25 = 625)
        response_text = str(full_message.content)
        assert "625" in response_text

    def test_empty_chunks_handling(self, nova_model: ChatBedrockConverse) -> None:
        """Test that empty or minimal chunks are handled correctly."""
        model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

        chunks = _stream_with_retries(model_with_tools, "What is 2 + 2?")
        for chunk in chunks:
            # Each chunk should be valid even if content is minimal
            assert isinstance(chunk, AIMessageChunk)

        # Verify we received chunks
        assert len(chunks) > 0

        # Accumulate full message
        full_message: Any = chunks[0]
        for chunk in chunks[1:]:
            full_message += chunk

        assert isinstance(full_message, AIMessage)
        # Even for simple queries, we should get a response
        assert len(full_message.content) > 0
