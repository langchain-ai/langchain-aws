import json
import unittest
from base64 import b64encode
from typing import Iterator, Dict, Any

from langchain_aws.agents.base import parse_agent_response

from langchain_aws.agents.base import BedrockAgentFinish


class MockStreamingResponse:
    """Simulates a streaming response from Bedrock Agent"""

    def __init__(self, session_id: str, events: list):
        self.session_id = session_id
        self.events = events

    def __getitem__(self, key):
        if key == "sessionId":
            return self.session_id
        elif key == "completion":
            return self.get_completion()
        raise KeyError(key)

    def get_completion(self) -> Iterator[Dict]:
        for event in self.events:
            yield event

    def get(self, key: str) -> Any:
        return self[key]

class TestBedrockAgentResponseParser(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        # Mock successful response with function invocation
        self.mock_success_response = {
            "sessionId": "123",
            "completion": [
                {
                    "trace": {
                        "orchestrationTrace": {
                            "modelInvocationInput": {
                                "text": "What stocks should I look up?",
                                "inferenceConfiguration": {
                                    "temperature": 0.0,
                                    "topP": 1.0
                                }
                            }
                        }
                    }
                },
                {
                    "returnControl": {
                        "invocationInputs": [
                            {
                                "functionInvocationInput": {
                                    "actionGroup": "price_tool_action_group",
                                    "function": "PriceTool",
                                    "parameters": [
                                        {
                                            "name": "Symbol",
                                            "value": "XYZ"
                                        },
                                        {
                                            "name": "Start_Date",
                                            "value": "20241020"
                                        },
                                        {
                                            "name": "End_Date",
                                            "value": "20241020"
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                },
                {
                    "chunk": {
                        "bytes": b64encode(
                            "XYZ prices for October 20, 2024: Open: 180, High: 185, Low: 179, Close: 183".encode()
                        ).decode()
                    }
                }
            ]
        }

        # Mock response with only chunks (no function call)
        self.mock_chunk_only_response = {
            "sessionId": "123",
            "completion": [
                {
                    "trace": {
                        "orchestrationTrace": {
                            "modelInvocationInput": {
                                "text": "What is the current price of XYZ?",
                            }
                        }
                    }
                },
                {
                    "chunk": {
                        "bytes": b64encode(
                            "The current price of XYZ is $185".encode()
                        ).decode()
                    }
                }
            ]
        }

        # Mock empty response
        self.mock_empty_response = {
            "sessionId": "123",
            "completion": [
                {
                    "trace": {
                        "orchestrationTrace": {}
                    }
                }
            ]
        }

    def test_parse_function_invocation(self):
        response = MockStreamingResponse("123", [self.mock_success_response])
        parsed_response = parse_agent_response(response)
        self.assertIsInstance(parsed_response, BedrockAgentFinish)

def run_tests():
    unittest.main(argv=[''], verbosity=2)


if __name__ == '__main__':
    run_tests()