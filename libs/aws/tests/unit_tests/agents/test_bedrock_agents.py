import unittest
from base64 import b64encode
from typing import Union

from langchain_aws.agents.base import (
    BedrockAgentAction,
    BedrockAgentFinish,
    parse_agent_response,
)


class TestBedrockAgentResponseParser(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        # Mock successful response with function invocation
        self.mock_success_return_of_control_response = {
            "sessionId": "123",
            "completion": [
                {
                    "returnControl": {
                        "invocationInputs": [
                            {
                                "functionInvocationInput": {
                                    "actionGroup": "price_tool_action_group",
                                    "function": "PriceTool",
                                    "parameters": [
                                        {"name": "Symbol", "value": "XYZ"},
                                        {"name": "Start_Date", "value": "20241020"},
                                        {"name": "End_Date", "value": "20241020"},
                                    ],
                                }
                            }
                        ]
                    }
                }
            ],
        }

        self.mock_success_finish_response = {
            "sessionId": "123",
            "completion": [
                {"chunk": {"bytes": b64encode("FAKE DATA HERE".encode())}},
                {"trace": "This is a fake trace event."},
            ],
        }

    def test_parse_return_of_control_invocation(self) -> None:
        response = self.mock_success_return_of_control_response
        parsed_response: Union[list[BedrockAgentAction], BedrockAgentFinish]
        parsed_response = parse_agent_response(response)
        self.assertIsInstance(
            parsed_response, list, "Expected a list of BedrockAgentAction."
        )

    def test_parse_finish_invocation(self) -> None:
        response = self.mock_success_finish_response
        parsed_response: Union[list[BedrockAgentAction], BedrockAgentFinish]
        parsed_response = parse_agent_response(response)
        # Type narrowing - now TypeScript knows parsed_response is BedrockAgentFinish
        assert isinstance(parsed_response, BedrockAgentFinish)
        assert parsed_response.trace_log is not None, "Expected trace_log"

        self.assertGreater(
            len(parsed_response.trace_log), 0, "Expected a trace log, none received."
        )
