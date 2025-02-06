from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall
from pydantic import Field
from langchain_core.outputs import ChatResult, ChatGeneration

from .types import (
    BedrockAgentAction,
    BedrockAgentFinish,
    InlineAgentConfiguration,
)

from .utils import (
    _get_action_group_and_function_names,
    _tool_to_function,
    parse_agent_response,
    get_boto_session,
)

class BedrockInlineAgentsChatModel(BaseChatModel):
    """Invoke a Bedrock Inline Agent as a chat model."""

    client: Any = Field(default=None)
    """Boto3 client"""
    
    region_name: Optional[str] = None
    """Region"""
    
    credentials_profile_name: Optional[str] = None
    """Credentials to use to invoke the agent"""
    
    endpoint_url: Optional[str] = None
    """Endpoint URL"""
    
    inline_agent_config: Optional[InlineAgentConfiguration] = None
    """Configuration for the inline agent"""
    
    session_id: Optional[str] = None
    """Session identifier to be used with requests"""

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "bedrock-inline-agent"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "region_name": self.region_name,
            "endpoint_url": self.endpoint_url,
            "inline_agent_config": self.inline_agent_config,
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def create(
        cls,
        *,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        inline_agent_config: Optional[InlineAgentConfiguration] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> BedrockInlineAgentsChatModel:
        """Create a new instance of BedrockInlineAgentsChatModel."""
        try:
            client_params, session = get_boto_session(
                credentials_profile_name=credentials_profile_name,
                region_name=region_name,
                endpoint_url=endpoint_url,
            )
            client = session.client("bedrock-agent-runtime", **client_params)

            return cls(
                client=client,
                region_name=region_name,
                credentials_profile_name=credentials_profile_name,
                endpoint_url=endpoint_url,
                inline_agent_config=inline_agent_config,
                session_id=session_id,
                **kwargs,
            )
        except Exception as e:
            raise ValueError(f"Error creating BedrockInlineAgentsChatModel: {str(e)}") from e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion."""
        input_text = self._convert_messages_to_text(messages)
        input_dict = {
            "input_text": input_text,
            **kwargs,
        }

        response = self._invoke_inline_agent(input_dict, run_manager=run_manager)
        if isinstance(response, BedrockAgentFinish):
            message = AIMessage(
                content=response.return_values["output"],
                additional_kwargs={
                    "session_id": response.session_id,
                    "trace_log": response.trace_log
                }
            )
        else:  # BedrockAgentAction
            # Handle tool use response
            tool_calls:list[ToolCall] = []
            # parse_agent_response() returns BedrockAgentAction list
            for action in response:
                tool_calls.append({
                    "name": action.tool,
                    "args": action.tool_input,
                    "id": str(uuid.uuid4())
                })

            message = AIMessage(
                content="",  # Empty content for tool calls
                additional_kwargs={
                    "session_id": response[0].session_id,
                    "trace_log": response[0].trace_log,
                    "roc_log": response[0].log
                },
                tool_calls=tool_calls
            )
        
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        """Convert a list of messages to a single text input for the agent."""
        text_parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                text_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                text_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                text_parts.append(f"Assistant: {message.content}")
            else:
                text_parts.append(str(message.content))
        return "\n".join(text_parts)

    def _invoke_inline_agent(
        self,
        input_dict: Dict[str, Any],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Union[BedrockAgentAction, BedrockAgentFinish]:
        """Invoke the inline agent with the given input."""
        # Merge configurations
        runtime_config = input_dict.get("inline_agent_config", {})
        self.inline_agent_config = self.inline_agent_config or {}
        effective_config = {**self.inline_agent_config, **runtime_config}
        
        # Convert tools to action groups format
        action_groups = self._get_action_groups(
            tools=effective_config.get("tools", []) or [],
            enableHumanInput=effective_config.get("enable_human_input", False),
            enableCodeInterpreter=effective_config.get("enable_code_interpreter", False),
        )

        # Prepare the invoke_inline_agent request
        agent_input: Dict[str, Any] = {
            "foundationModel": effective_config.get("foundation_model"),
            "instruction": effective_config.get("instruction"),
            "actionGroups": action_groups,
            "enableTrace": effective_config.get("enable_trace", False),
            "endSession": bool(input_dict.get("end_session", False)),
            "inputText": input_dict.get("input_text", ""),
        }

        # Add optional configurations
        optional_params = {
            "customerEncryptionKeyArn": "customer_encryption_key_arn",
            "idleSessionTTLInSeconds": "idle_session_ttl_in_seconds",
            "guardrailConfiguration": "guardrail_configuration",
            "knowledgeBases": "knowledge_bases",
            "promptOverrideConfiguration": "prompt_override_configuration",
            "inlineSessionState": "inline_session_state",
        }

        for param_name, config_key in optional_params.items():
            if effective_config.get(config_key):
                agent_input[param_name] = effective_config[config_key]

        # Use existing session_id from input, or from intermediate steps, or generate new one
        self.session_id = input_dict.get("session_id") or self.session_id or str(uuid.uuid4())
        
        output = self.client.invoke_inline_agent(
            sessionId=self.session_id,
            **agent_input
        )
        return parse_agent_response(output)


    def _get_action_groups(self, tools: List[Any], enableHumanInput: bool, enableCodeInterpreter: bool) -> List:
        """Convert tools to Bedrock action groups format."""
        action_groups = []
        tools_by_action_group = defaultdict(list)

        for tool in tools:
            action_group_name, _ = _get_action_group_and_function_names(tool)
            tools_by_action_group[action_group_name].append(tool)

        for action_group_name, functions in tools_by_action_group.items():
            action_groups.append({
                "actionGroupName": action_group_name,
                "actionGroupExecutor": {"customControl": "RETURN_CONTROL"},
                "functionSchema": {
                    "functions": [_tool_to_function(function) for function in functions]
                },
            })

        if enableHumanInput:
            action_groups.append({
                "actionGroupName": "UserInputAction",
                "parentActionGroupSignature": "AMAZON.UserInput",
            })
        
        if enableCodeInterpreter:
            action_groups.append({
                "actionGroupName": "CodeInterpreterAction",
                "parentActionGroupSignature": "AMAZON.CodeInterpreter",
            })

        return action_groups