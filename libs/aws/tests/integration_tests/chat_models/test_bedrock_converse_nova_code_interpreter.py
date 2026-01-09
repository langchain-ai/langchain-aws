"""Integration tests for Nova Lite 2.0 nova_code_interpreter system tool."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_aws import ChatBedrockConverse
from langchain_aws.tools import NovaCodeInterpreterTool


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


def test_nova_code_interpreter_basic(nova_model: ChatBedrockConverse) -> None:
    """Test nova_code_interpreter system tool with real API call."""
    model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])
    response = model_with_tools.invoke("Use Python to calculate 123 * 456")

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify the response contains text with the result (123 * 456 = 56088)
    response_text = str(response.content)
    assert "56088" in response_text or "56,088" in response_text


def test_nova_code_interpreter_with_string(nova_model: ChatBedrockConverse) -> None:
    """Test nova_code_interpreter using direct string instead of helper class."""
    model_with_tools = nova_model.bind_tools(["nova_code_interpreter"])
    response = model_with_tools.invoke("What is 123 * 456?")

    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify the calculation result (123 * 456 = 56088)
    response_text = str(response.content)
    assert "56088" in response_text or "56,088" in response_text


def test_nova_code_interpreter_simple_calculation(
    nova_model: ChatBedrockConverse,
) -> None:
    """Test nova_code_interpreter with simple calculation."""
    model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])
    response = model_with_tools.invoke("Use Python code to calculate 7 ** 6")

    # Verify response structure
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

    # Verify the calculation result (7^6 = 117649)
    response_text = str(response.content)
    assert "117649" in response_text or "117,649" in response_text


def test_nova_code_interpreter_with_reasoning(
    nova_model_with_reasoning: ChatBedrockConverse,
) -> None:
    """Test nova_code_interpreter with reasoning enabled."""
    model_with_tools = nova_model_with_reasoning.bind_tools([NovaCodeInterpreterTool()])
    response = model_with_tools.invoke(
        "Calculate the factorial of 10, return and explain the result. "
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


def test_nova_code_interpreter_content_blocks(
    nova_model_with_reasoning: ChatBedrockConverse,
) -> None:
    """Test that content blocks are properly parsed for nova_code_interpreter."""
    model_with_tools = nova_model_with_reasoning.bind_tools([NovaCodeInterpreterTool()])
    response = model_with_tools.invoke(
        "Calculate the sum of numbers from 1 to 100. "
        "Use Code Interpreter to calculate it."
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

    # Verify the calculation result (sum 1-100 = 5050)
    response_text = str(response.content)
    assert "5050" in response_text or "5,050" in response_text


def test_nova_code_interpreter_streaming(nova_model: ChatBedrockConverse) -> None:
    """Test streaming with nova_code_interpreter system tool."""
    model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

    chunks = []
    for chunk in model_with_tools.stream("What is 999 * 888?"):
        chunks.append(chunk)

    # Verify we received chunks
    assert len(chunks) > 0

    # Verify final message by accumulating chunks
    full_message = chunks[0]
    for chunk in chunks[1:]:
        full_message = full_message + chunk  # type: ignore[assignment]

    assert isinstance(full_message, AIMessage)
    assert len(full_message.content) > 0

    # Verify the calculation result (999 * 888 = 887112)
    response_text = str(full_message.content)
    assert "887112" in response_text or "887,112" in response_text


def test_nova_code_interpreter_multi_turn_conversation(
    nova_model: ChatBedrockConverse,
) -> None:
    """Test nova_code_interpreter in a multi-turn conversation."""
    model_with_tools = nova_model.bind_tools([NovaCodeInterpreterTool()])

    # First turn
    message1 = HumanMessage("Calculate 15 * 20")
    response1 = model_with_tools.invoke([message1])
    assert isinstance(response1, AIMessage)
    assert "300" in str(response1.content)

    # Second turn - follow-up calculation
    message2 = HumanMessage("Now add 50 to that result")
    response2 = model_with_tools.invoke([message1, response1, message2])
    assert isinstance(response2, AIMessage)
    assert len(response2.content) > 0
    assert "350" in str(response2.content)
