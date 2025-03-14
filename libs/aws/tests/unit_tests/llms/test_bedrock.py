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

TEST_CASES = {
    """Hey""": """

Human: Hey

Assistant:""",
    """

Human: Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """Human: Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """
Human: Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """

Human: Human: Hello

Assistant:""": (
        "Error: Prompt must alternate between '\n\nHuman:' and '\n\nAssistant:'."
    ),
    """Human: Hello

Assistant: Hello

Human: Hello

Assistant:""": """

Human: Hello

Assistant: Hello

Human: Hello

Assistant:""",
    """

Human: Hello

Assistant: Hello

Human: Hello

Assistant:""": """

Human: Hello

Assistant: Hello

Human: Hello

Assistant:""",
    """

Human: Hello

Assistant: Hello

Human: Hello

Assistant: Hello

Assistant: Hello""": ALTERNATION_ERROR,
    """

Human: Hi.

Assistant: Hi.

Human: Hi.

Human: Hi.

Assistant:""": ALTERNATION_ERROR,
    """
Human: Hello""": """

Human: Hello

Assistant:""",
    """

Human: Hello
Hello

Assistant""": """

Human: Hello
Hello

Assistant

Assistant:""",
    """Hello

Assistant:""": """

Human: Hello

Assistant:""",
    """Hello

Human: Hello

""": """Hello

Human: Hello



Assistant:""",
    """

Human: Assistant: Hello""": """

Human: 

Assistant: Hello""",
    """

Human: Human

Assistant: Assistant

Human: Assistant

Assistant: Human""": """

Human: Human

Assistant: Assistant

Human: Assistant

Assistant: Human""",
    """
Assistant: Hello there, your name is:

Human.

Human: Hello there, your name is: 

Assistant.""": """

Human: 

Assistant: Hello there, your name is:

Human.

Human: Hello there, your name is: 

Assistant.

Assistant:""",
    """

Human: Human: Hi

Assistant: Hi""": ALTERNATION_ERROR,
    """Human: Hi

Human: Hi""": ALTERNATION_ERROR,
    """

Assistant: Hi

Human: Hi""": """

Human: 

Assistant: Hi

Human: Hi

Assistant:""",
    """

Human: Hi

Assistant: Yo

Human: Hey

Assistant: Sup

Human: Hi

Assistant: Hi
Human: Hi
Assistant:""": """

Human: Hi

Assistant: Yo

Human: Hey

Assistant: Sup

Human: Hi

Assistant: Hi

Human: Hi

Assistant:""",
    """

Hello.

Human: Hello.

Assistant:""": """

Hello.

Human: Hello.

Assistant:""",
}


def test__human_assistant_format() -> None:
    for input_text, expected_output in TEST_CASES.items():
        if expected_output == ALTERNATION_ERROR:
            with pytest.warns(UserWarning, match=ALTERNATION_ERROR):
                _human_assistant_format(input_text)
        else:
            output = _human_assistant_format(input_text)
            assert output == expected_output


# Sample mock streaming response data
MOCK_STREAMING_RESPONSE = [
    {"chunk": {"bytes": b'{"text": "nice"}'}},
    {"chunk": {"bytes": b'{"text": " to meet"}'}},
    {"chunk": {"bytes": b'{"text": " you"}'}},
]

MOCK_STREAMING_RESPONSE_MISTRAL = [
    {"chunk": {"bytes": b'{"outputs": [{"text": "Thank","stop_reason": null}]}'}},
    {"chunk": {"bytes": b'{"outputs": [{"text": "you.","stop_reason": "stop"}]}'}},
]

MOCK_STREAMING_RESPONSE_DEEPSEEK = [
    {"chunk": {"bytes": b'{"choices": [{"text": "Thank","stop_reason": null}]}'}},
    {"chunk": {"bytes": b'{"choices": [{"text": "you.","stop_reason": "stop"}]}'}},
]


async def async_gen_mock_streaming_response() -> AsyncGenerator[Dict, None]:
    for item in MOCK_STREAMING_RESPONSE:
        yield item


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
                region_name="us-west-2",
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


@pytest.fixture
def mistral_response():
    body = MagicMock()
    body.read.return_value = json.dumps(
        {"outputs": [{"text": "This is the Mistral output text."}]}
    ).encode()
    response = dict(
        body=body,
        ResponseMetadata={
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "18",
                "x-amzn-bedrock-output-token-count": "28",
            }
        },
    )

    return response


@pytest.fixture
def mistral_streaming_response():
    response = dict(body=MOCK_STREAMING_RESPONSE_MISTRAL)
    return response


@pytest.fixture
def deepseek_response():
    body = MagicMock()
    body.read.return_value = json.dumps(
        {"choices": [{"text": "This is the DeepSeek output text."}]}
    ).encode()
    response = dict(
        body=body,
        ResponseMetadata={
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "41",
                "x-amzn-bedrock-output-token-count": "51",
            }
        },
    )

    return response


@pytest.fixture
def deepseek_streaming_response():
    response = dict(body=MOCK_STREAMING_RESPONSE_DEEPSEEK)
    return response


@pytest.fixture
def cohere_response():
    body = MagicMock()
    body.read.return_value = json.dumps(
        {"generations": [{"text": "This is the Cohere output text."}]}
    ).encode()
    response = dict(
        body=body,
        ResponseMetadata={
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "12",
                "x-amzn-bedrock-output-token-count": "22",
            }
        },
    )
    return response


@pytest.fixture
def anthropic_response():
    body = MagicMock()
    body.read.return_value = json.dumps(
        {"completion": "This is the output text."}
    ).encode()
    response = dict(
        body=body,
        ResponseMetadata={
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "10",
                "x-amzn-bedrock-output-token-count": "20",
            }
        },
    )
    return response


@pytest.fixture
def ai21_response():
    body = MagicMock()
    body.read.return_value = json.dumps(
        {"completions": [{"data": {"text": "This is the AI21 output text."}}]}
    ).encode()
    response = dict(
        body=body,
        ResponseMetadata={
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "15",
                "x-amzn-bedrock-output-token-count": "25",
            }
        },
    )
    return response


@pytest.fixture
def response_with_stop_reason():
    body = MagicMock()
    body.read.return_value = json.dumps(
        {"completion": "This is the output text.", "stop_reason": "length"}
    ).encode()
    response = dict(
        body=body,
        ResponseMetadata={
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "10",
                "x-amzn-bedrock-output-token-count": "20",
            }
        },
    )
    return response


def test_prepare_output_for_mistral(mistral_response):
    result = LLMInputOutputAdapter.prepare_output("mistral", mistral_response)
    assert result["text"] == "This is the Mistral output text."
    assert result["usage"]["prompt_tokens"] == 18
    assert result["usage"]["completion_tokens"] == 28
    assert result["usage"]["total_tokens"] == 46
    assert result["stop_reason"] is None


def test_prepare_output_stream_for_mistral(mistral_streaming_response) -> None:
    results = [
        chunk.text
        for chunk in LLMInputOutputAdapter.prepare_output_stream(
            "mistral", mistral_streaming_response
        )
    ]

    assert results[0] == "Thank"
    assert results[1] == "you."


def test_prepare_output_for_deepseek(deepseek_response):
    result = LLMInputOutputAdapter.prepare_output("deepseek", deepseek_response)
    assert result["text"] == "This is the DeepSeek output text."
    assert result["usage"]["prompt_tokens"] == 41
    assert result["usage"]["completion_tokens"] == 51
    assert result["usage"]["total_tokens"] == 92
    assert result["stop_reason"] is None


def test_prepare_output_stream_for_deepseek(deepseek_streaming_response) -> None:
    results = [
        chunk.text
        for chunk in LLMInputOutputAdapter.prepare_output_stream(
            "deepseek", deepseek_streaming_response
        )
    ]

    assert results[0] == "Thank"
    assert results[1] == "you."


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


def test_standard_tracing_params():
    llm = BedrockLLM(model_id="foo", region_name="us-west-2")
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "amazon_bedrock",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
    }


@pytest.fixture
def anthropic_response_with_thinking():
    body = MagicMock()
    body.read.return_value = json.dumps(
        {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Let me think through this step by step...",
                    "signature": "SIGNATURE123",
                },
                {"type": "text", "text": "This is the output text."},
            ]
        }
    ).encode()
    response = dict(
        body=body,
        ResponseMetadata={
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "10",
                "x-amzn-bedrock-output-token-count": "30",
            }
        },
    )
    return response


@pytest.fixture
def anthropic_response_with_thinking_and_tool_use():
    body = MagicMock()
    body.read.return_value = json.dumps(
        {
            "content": [
                {
                    "type": "thinking",
                    "thinking": "I need to use a tool to answer this question...",
                    "signature": "SIGNATURE456",
                },
                {"type": "text", "text": "Let me check that for you."},
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "get_weather",
                    "input": {"city": "nyc"},
                },
            ],
            "stop_reason": "tool_use",
        }
    ).encode()
    response = dict(
        body=body,
        ResponseMetadata={
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "15",
                "x-amzn-bedrock-output-token-count": "40",
            }
        },
    )
    return response


@pytest.fixture
def anthropic_response_after_tool_use():
    body = MagicMock()
    body.read.return_value = json.dumps(
        {
            "content": [
                {"type": "text", "text": "Based on the data, it's cloudy in NYC."},
            ],
            "stop_reason": "end_turn",
        }
    ).encode()
    response = dict(
        body=body,
        ResponseMetadata={
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "60",
                "x-amzn-bedrock-output-token-count": "20",
            }
        },
    )
    return response


def test_prepare_output_with_thinking(anthropic_response_with_thinking):
    """Test that thinking blocks are extracted properly from the response."""
    result = LLMInputOutputAdapter.prepare_output(
        "anthropic", anthropic_response_with_thinking
    )

    # Check that the text content was extracted correctly
    assert result["text"] == "This is the output text."

    # Check that the thinking block was extracted correctly
    assert "thinking" in result
    assert isinstance(result["thinking"], dict)
    assert result["thinking"]["text"] == "Let me think through this step by step..."
    assert result["thinking"]["signature"] == "SIGNATURE123"

    # Check that token counts are correct
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 30
    assert result["usage"]["total_tokens"] == 40


def test_prepare_output_with_thinking_and_tool_use(
    anthropic_response_with_thinking_and_tool_use,
):
    """Test that thinking blocks and tool use are extracted properly from the response."""
    result = LLMInputOutputAdapter.prepare_output(
        "anthropic", anthropic_response_with_thinking_and_tool_use
    )

    # Check that the text content was extracted correctly
    assert result["text"] == "Let me check that for you."

    # Check that the thinking block was extracted correctly
    assert "thinking" in result
    assert isinstance(result["thinking"], dict)
    assert (
        result["thinking"]["text"] == "I need to use a tool to answer this question..."
    )
    assert result["thinking"]["signature"] == "SIGNATURE456"

    # Check that tool calls are extracted correctly
    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "get_weather"
    assert result["tool_calls"][0]["args"] == {"city": "nyc"}
    assert result["tool_calls"][0]["id"] == "tool_1"

    # Check that stop reason is correctly extracted
    assert result["stop_reason"] == "tool_use"

    # Check that token counts are correct
    assert result["usage"]["prompt_tokens"] == 15
    assert result["usage"]["completion_tokens"] == 40
    assert result["usage"]["total_tokens"] == 55


def test_prepare_output_after_tool_use(anthropic_response_after_tool_use):
    """Test that responses after tool use (which don't have thinking blocks) are handled correctly."""
    result = LLMInputOutputAdapter.prepare_output(
        "anthropic", anthropic_response_after_tool_use
    )

    # Check that the text content was extracted correctly
    assert result["text"] == "Based on the data, it's cloudy in NYC."

    # Check that thinking is an empty dictionary when no thinking blocks are present
    assert "thinking" in result
    assert result["thinking"] == {}

    # Check that stop reason is correctly extracted
    assert result["stop_reason"] == "end_turn"

    # Check that token counts are correct
    assert result["usage"]["prompt_tokens"] == 60
    assert result["usage"]["completion_tokens"] == 20
    assert result["usage"]["total_tokens"] == 80
