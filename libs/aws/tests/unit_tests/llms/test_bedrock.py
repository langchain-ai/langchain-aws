# type:ignore

import json
from typing import AsyncGenerator, Dict
from unittest.mock import MagicMock, patch

import pytest

from langchain_aws import BedrockLLM
from langchain_aws.llms.bedrock import (
    ALTERNATION_ERROR,
    LLMInputOutputAdapter,
    _human_assistant_format,
)

mock_boto3 = MagicMock()
# Mocking the client method of the Session object
mock_boto3.Session.return_value.client.return_value = MagicMock()


async def async_gen_mock_streaming_response() -> AsyncGenerator[Dict, None]:
    # Sample mock streaming response data

    MOCK_STREAMING_RESPONSE = [
        {"chunk": {"bytes": b'{"text": "nice"}'}},
        {"chunk": {"bytes": b'{"text": " to meet"}'}},
        {"chunk": {"bytes": b'{"text": " you"}'}},
    ]
    for item in MOCK_STREAMING_RESPONSE:
        yield item


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        (
            """Hey""",
            """\n\nHuman: Hey\n\nAssistant:""",
        ),
        (
            """\n\nHuman: Hello\n\nAssistant:""",
            """\n\nHuman: Hello\n\nAssistant:""",
        ),
        (
            """Human: Hello\n\nAssistant:""",
            """\n\nHuman: Hello\n\nAssistant:""",
        ),
        (
            """\nHuman: Hello\n\nAssistant:""",
            """\n\nHuman: Hello\n\nAssistant:""",
        ),
        (
            """\n\nHuman: Human: Hello\n\nAssistant:""",
            "Error: Prompt must alternate between '\n\nHuman:' and '\n\nAssistant:'.",
        ),
        (
            """Human: Hello\n\nAssistant: Hello\n\nHuman: Hello\n\nAssistant:""",
            """\n\nHuman: Hello\n\nAssistant: Hello\n\nHuman: Hello\n\nAssistant:""",
        ),
        (
            """\n\nHuman: Hello\n\nAssistant: Hello\n\nHuman: Hello\n\nAssistant:""",
            """\n\nHuman: Hello\n\nAssistant: Hello\n\nHuman: Hello\n\nAssistant:""",
        ),
        (
            """\n\nHuman: Hello\n\nAssistant: Hello\n\nHuman: """
            """Hello\n\nAssistant: Hello\n\nAssistant: Hello""",
            ALTERNATION_ERROR,
        ),
        (
            """\n\nHuman: Hi.\n\nAssistant: Hi.\n\nHuman: Hi.\n\nHuman: Hi."""
            """\n\nAssistant:""",
            ALTERNATION_ERROR,
        ),
        (
            """\nHuman: Hello""",
            """\n\nHuman: Hello\n\nAssistant:""",
        ),
        (
            """\n\nHuman: Hello\nHello\n\nAssistant""",
            """\n\nHuman: Hello\nHello\n\nAssistant\n\nAssistant:""",
        ),
        (
            """Hello\n\nAssistant:""",
            """\n\nHuman: Hello\n\nAssistant:""",
        ),
        (
            """Hello\n\nHuman: Hello\n\n""",
            """Hello\n\nHuman: Hello\n\n\n\nAssistant:""",
        ),
        (
            """\n\nHuman: Assistant: Hello""",
            """\n\nHuman: \n\nAssistant: Hello""",
        ),
        (
            """\n\nHuman: Human\n\nAssistant: Assistant\n\nHuman: Assistant\n\n"""
            """Assistant: Human""",
            """\n\nHuman: Human\n\nAssistant: Assistant\n\nHuman: Assistant\n\n"""
            """Assistant: Human""",
        ),
        (
            """\n\nAssistant: Hello there, your name is:\n\nHuman.\n\nHuman: """
            """Hello there, your name is: Assistant.""",
            """\n\nHuman: \n\nAssistant: Hello there, your name is:\n\nHuman."""
            """\n\nHuman: Hello there, your name is: Assistant.\n\nAssistant:""",
        ),
        ("""\n\nHuman: Human: Hi\n\nAssistant: Hi""", ALTERNATION_ERROR),
        (
            """Human: Hi\n\nHuman: Hi""",
            ALTERNATION_ERROR,
        ),
        (
            """\n\nAssistant: Hi\n\nHuman: Hi""",
            """\n\nHuman: \n\nAssistant: Hi\n\nHuman: Hi\n\nAssistant:""",
        ),
        (
            """\n\nHuman: Hi\n\nAssistant: Yo\n\nHuman: Hey\n\nAssistant: Sup"""
            """\n\nHuman: Hi\n\nAssistant: Hi\n\nHuman: Hi\n\nAssistant:""",
            """\n\nHuman: Hi\n\nAssistant: Yo\n\nHuman: Hey\n\nAssistant: Sup"""
            """\n\nHuman: Hi\n\nAssistant: Hi\n\nHuman: Hi\n\nAssistant:""",
        ),
        (
            """\n\nHello.\n\nHuman: Hello.\n\nAssistant:""",
            """\n\nHello.\n\nHuman: Hello.\n\nAssistant:""",
        ),
    ],
)
def test_human_assistant_format(input_text, expected_output) -> None:
    if expected_output == ALTERNATION_ERROR:
        with pytest.warns(UserWarning, match=ALTERNATION_ERROR):
            _human_assistant_format(input_text)
    else:
        output = _human_assistant_format(input_text)
        assert output == expected_output


@pytest.mark.asyncio
async def test_bedrock_async_streaming_call() -> None:
    # Mock boto3 import
    mock_boto3 = MagicMock()
    mock_boto3.Session.return_value.client.return_value = (
        MagicMock()
    )  # Mocking the client method of the Session object

    with patch.dict(
        "sys.modules", {"boto3": mock_boto3}
    ):  # Mocking boto3 at the top level using patch.dict
        # Mock the `BedrockLLM` class's method that invokes the model
        mock_invoke_method = MagicMock(return_value=async_gen_mock_streaming_response())
        with patch.object(
            BedrockLLM, "_aprepare_input_and_invoke_stream", mock_invoke_method
        ):
            # Instantiate the Bedrock LLM
            llm = BedrockLLM(
                client=None,
                model_id="anthropic.claude-v2",
                streaming=True,
            )
            # Call the _astream method
            chunks = [
                json.loads(chunk["chunk"]["bytes"])["text"]  # type: ignore
                async for chunk in llm._astream("Hey, how are you?")
            ]

    # Assertions
    assert len(chunks) == 3
    assert chunks[0] == "nice"
    assert chunks[1] == " to meet"
    assert chunks[2] == " you"


def test_prepare_output_for_mistral(mistral_response):
    result = LLMInputOutputAdapter.prepare_output("mistral", mistral_response)
    assert result["text"] == "This is the Mistral output text."
    assert result["usage"]["prompt_tokens"] == 18
    assert result["usage"]["completion_tokens"] == 28
    assert result["usage"]["total_tokens"] == 46
    assert result["stop_reason"] is None


def test_prepare_output_for_cohere(cohere_response):
    result = LLMInputOutputAdapter.prepare_output("cohere", cohere_response)
    assert result["text"] == "This is the Cohere output text."
    assert result["usage"]["prompt_tokens"] == 12
    assert result["usage"]["completion_tokens"] == 22
    assert result["usage"]["total_tokens"] == 34
    assert result["stop_reason"] is None


def test_prepare_output_with_stop_reason(response_with_stop_reason):
    result = LLMInputOutputAdapter.prepare_output(
        "anthropic", response_with_stop_reason
    )
    assert result["text"] == "This is the output text."
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 20
    assert result["usage"]["total_tokens"] == 30
    assert result["stop_reason"] == "length"


def test_prepare_output_for_anthropic(anthropic_response):
    result = LLMInputOutputAdapter.prepare_output("anthropic", anthropic_response)
    assert result["text"] == "This is the output text."
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 20
    assert result["usage"]["total_tokens"] == 30
    assert result["stop_reason"] is None


def test_prepare_output_for_ai21(ai21_response):
    result = LLMInputOutputAdapter.prepare_output("ai21", ai21_response)
    assert result["text"] == "This is the AI21 output text."
    assert result["usage"]["prompt_tokens"] == 15
    assert result["usage"]["completion_tokens"] == 25
    assert result["usage"]["total_tokens"] == 40
    assert result["stop_reason"] is None


@pytest.mark.parametrize(
    "error_state, guardrail_input",
    [
        (True, {}),
        (True, {"guardrailIdentifier": "some-id"}),
        (True, {"guardrailVersion": "some-version"}),
        (True, {"guardrailConfig": {"config": "value"}}),
        (
            False,
            {
                "guardrailIdentifier": "some-id",
                "guardrailVersion": "some-version",
                "trace": True,
            },
        ),
        (
            False,
            {
                "guardrailIdentifier": "some-id",
                "guardrailVersion": "some-version",
                "guardrailConfig": {"streamProcessingMode": "SYNCHRONOUS"},
            },
        ),
    ],
)
async def test_guardrail_input(
    error_state,
    guardrail_input,
):
    llm = BedrockLLM(
        client=mock_boto3,
        model_id="anthropic.claude-v2",
        guardrails=guardrail_input,
    )
    is_valid = False
    if error_state:
        with pytest.raises(TypeError) as error:
            is_valid = llm._guardrails_enabled
        assert error.value.args[0] == (
            "Guardrails must be a dictionary with 'guardrailIdentifier' and "
            "'guardrailVersion' mandatory keys."
        )
        assert is_valid is False
    else:
        llm = BedrockLLM(
            client=mock_boto3,
            model_id="anthropic.claude-v2",
            guardrails=guardrail_input,
        )
        is_valid = llm._guardrails_enabled
        assert is_valid is True
