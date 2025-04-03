import json
import unittest
from base64 import b64encode
from typing import List, Union
from unittest.mock import Mock, patch

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import tool

from langchain_aws.agents import BedrockInlineAgentsRunnable
from langchain_aws.agents.base import (
    BedrockAgentAction,
    BedrockAgentFinish,
    parse_agent_response,
)
from langchain_aws.agents.types import InlineAgentConfiguration


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


class TestBedrockInlineAgentsRunnable(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

        # Create mock tools
        @tool("TestGroup::testTool")
        def test_tool(param: str) -> str:
            """Test tool"""
            return f"Result for {param}"

        self.tools = [test_tool]

        # Create mock configuration
        self.inline_agent_config: InlineAgentConfiguration = {
            "foundation_model": "anthropic.claude-v3",
            "instruction": "Test instruction",
            "tools": self.tools,
            "enable_trace": True,
            "enable_human_input": False,
            "enable_code_interpreter": False,
        }

        # Create mock client
        self.mock_client = Mock()
        self.runnable = BedrockInlineAgentsRunnable(
            client=self.mock_client,
            region_name="us-west-2",
            inline_agent_config=self.inline_agent_config,
        )

    def test_create_method(self) -> None:
        """Test the create class method"""
        with patch("boto3.Session") as mock_session:
            mock_client = Mock()
            mock_session.return_value.client.return_value = mock_client

            runnable = BedrockInlineAgentsRunnable.create(
                region_name="us-west-2", inline_agent_config=self.inline_agent_config
            )

            self.assertIsInstance(runnable, BedrockInlineAgentsRunnable)
            self.assertEqual(runnable.region_name, "us-west-2")
            self.assertEqual(runnable.inline_agent_config, self.inline_agent_config)

    def test_get_action_groups(self) -> None:
        """Test _get_action_groups method"""
        action_groups = self.runnable._get_action_groups(
            tools=self.tools, enableHumanInput=True, enableCodeInterpreter=True
        )

        # Check if action groups are correctly formatted
        self.assertEqual(len(action_groups), 3)  # Tools + User Input + Code Interpreter

        # Check tool action group
        tool_group = next(
            group for group in action_groups if group["actionGroupName"] == "TestGroup"
        )
        self.assertEqual(
            tool_group["actionGroupExecutor"], {"customControl": "RETURN_CONTROL"}
        )

        # Check human input action group
        human_input_group = next(
            group
            for group in action_groups
            if group["actionGroupName"] == "UserInputAction"
        )
        self.assertEqual(
            human_input_group["parentActionGroupSignature"], "AMAZON.UserInput"
        )

        # Check code interpreter action group
        code_interpreter_group = next(
            group
            for group in action_groups
            if group["actionGroupName"] == "CodeInterpreterAction"
        )
        self.assertEqual(
            code_interpreter_group["parentActionGroupSignature"],
            "AMAZON.CodeInterpreter",
        )

    def test_invoke_with_new_session(self) -> None:
        """Test invoke method with a new session"""
        # Mock the client response
        mock_response = {
            "sessionId": "test-session",
            "completion": [{"chunk": {"bytes": b64encode(b"Test response")}}],
        }
        self.mock_client.invoke_inline_agent.return_value = mock_response

        # Create test messages
        messages: List[BaseMessage] = [
            SystemMessage(content="System instruction"),
            HumanMessage(content="Test input"),
        ]

        # Test invoke
        result = self.runnable.invoke(messages)

        # Verify the client was called with correct parameters
        self.mock_client.invoke_inline_agent.assert_called_once()

        call_args = self.mock_client.invoke_inline_agent.call_args[1]

        self.assertTrue(isinstance(result, AIMessage))
        self.assertEqual(call_args["foundationModel"], "anthropic.claude-v3")
        self.assertEqual(call_args["instruction"], "Test instruction")
        self.assertTrue(call_args["enableTrace"])
        self.assertEqual(
            call_args["inputText"], "System: System instruction\nHuman: Test input"
        )

    def test_invoke_with_runtime_config_override(self) -> None:
        """Test invoke method with runtime configuration override"""
        mock_response = {
            "sessionId": "test-session",
            "completion": [{"chunk": {"bytes": b64encode(b"Test response")}}],
        }
        self.mock_client.invoke_inline_agent.return_value = mock_response

        # Create test messages
        messages: List[BaseMessage] = [
            SystemMessage(content="System instruction"),
            HumanMessage(content="Test input"),
        ]

        # Runtime configuration override
        config = {"instruction": "Override instruction", "enable_trace": False}

        new_session_id = "new_session_id"

        self.runnable.invoke(
            messages, inline_agent_config=config, session_id=new_session_id
        )
        # Verify the override was applied
        call_args = self.mock_client.invoke_inline_agent.call_args[1]
        self.assertEqual(call_args["sessionId"], new_session_id)
        self.assertEqual(call_args["instruction"], "Override instruction")
        self.assertFalse(call_args["enableTrace"])

    def test_invoke_with_tool_message_response(self) -> None:
        """Test invoke method with runtime configuration override"""
        mock_response = {
            "sessionId": "test-session",
            "completion": [{"chunk": {"bytes": b64encode(b"Test response")}}],
        }
        self.mock_client.invoke_inline_agent.return_value = mock_response

        roc_block = {
            "returnControl": {
                "invocationId": "fake_tool_call_id",
                "invocationInputs": [
                    {
                        "functionInvocationInput": {
                            "actionGroup": "fake_group",
                            "actionInvocationType": "RESULT",
                            "agentId": "INLINE_AGENT",
                            "function": "my_tool_name",
                            "parameters": [
                                {
                                    "name": "fake_parameter",
                                    "type": "string",
                                    "value": "fake value",
                                }
                            ],
                        }
                    }
                ],
            }
        }

        my_tool_call_id = "my_tool_call_id"
        # Create test messages
        messages: List[BaseMessage] = [
            SystemMessage(content="System instruction"),
            HumanMessage(content="Test input"),
            AIMessage(
                content="Tool Response",
                tool_calls=[
                    ToolCall(
                        name="my_tool_name",
                        args={"args": "value"},
                        id=my_tool_call_id,
                        type="tool_call",
                    )
                ],
                additional_kwargs={
                    "session_id": "test-session",
                    "roc_log": json.dumps(roc_block),
                },
            ),
            ToolMessage(
                "tool response", tool_call_id=my_tool_call_id, name="my_tool_name"
            ),
        ]

        self.runnable.invoke(messages)

        # verify the call took place with the tool result
        call_args = self.mock_client.invoke_inline_agent.call_args[1]
        # verify the tool call id is the ToolMessageId we supplied
        self.assertEqual(
            call_args["inlineSessionState"]["invocationId"], my_tool_call_id
        )

    def test_error_handling(self) -> None:
        """Test error handling in invoke method"""
        self.mock_client.invoke_inline_agent.side_effect = Exception("Test error")

        messages: List[BaseMessage] = [
            SystemMessage(content="System instruction"),
            HumanMessage(content="Test input"),
        ]

        with self.assertRaises(Exception) as context:
            self.runnable.invoke(messages)
        self.assertEqual(str(context.exception), "Test error")
