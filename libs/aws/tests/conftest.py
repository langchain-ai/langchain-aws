import json
from typing import Dict
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mistral_response() -> Dict:
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
def cohere_response() -> Dict:
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
def anthropic_response() -> Dict:
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
def ai21_response() -> Dict:
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
def response_with_stop_reason() -> Dict:
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
