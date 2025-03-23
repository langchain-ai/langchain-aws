from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from botocore.exceptions import UnknownServiceError
from langchain_core.callbacks import CallbackManager
from langchain_core.load import dumpd
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.tools import BaseTool
from pydantic import Field, model_validator

from langchain_aws.agents.types import (
    _DEFAULT_ACTION_GROUP_NAME,
    _TEST_AGENT_ALIAS_ID,
    BedrockAgentAction,
    BedrockAgentFinish,
    GuardrailConfiguration,
    InlineAgentConfiguration,
    OutputType,
)
from langchain_aws.agents.utils import (
    _create_bedrock_action_groups,
    _create_bedrock_agent,
    _get_action_group_and_function_names,
    _get_bedrock_agent,
    _prepare_agent,
    _tool_to_function,
    get_boto_session,
    parse_agent_response,
)

logger = logging.getLogger(__name__)


class BedrockAgentsRunnable(RunnableSerializable[Dict, OutputType]):
    """
    Invoke a Bedrock Agent
    """

    agent_id: Optional[str]
    """Bedrock Agent Id"""
    agent_alias_id: Optional[str] = _TEST_AGENT_ALIAS_ID
    """Bedrock Agent Alias Id"""
    client: Any
    """Boto3 client"""
    region_name: Optional[str] = None
    """Region"""
    credentials_profile_name: Optional[str] = None
    """Credentials to use to invoke the agent"""
    endpoint_url: Optional[str] = None
    """Endpoint URL"""
    enable_trace: Optional[bool] = False
    """Boolean flag to enable trace when invoking Bedrock Agent"""

    @model_validator(mode="before")
    @classmethod
    def validate_agent(cls, values: dict) -> Any:
        if values.get("client") is not None:
            return values

        try:
            client_params, session = get_boto_session(
                credentials_profile_name=values["credentials_profile_name"],
                region_name=values["region_name"],
                endpoint_url=values["endpoint_url"],
            )

            values["client"] = session.client("bedrock-agent-runtime", **client_params)

            return values
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except UnknownServiceError as e:
            raise ModuleNotFoundError(
                "Ensure that you have installed the latest boto3 package "
                "that contains the API for `bedrock-runtime-agent`."
            ) from e
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e

    @classmethod
    def create_agent(
        cls,
        agent_name: str,
        agent_resource_role_arn: str,
        foundation_model: str,
        instruction: str,
        tools: List[BaseTool] = [],
        *,
        client_token: Optional[str] = None,
        customer_encryption_key_arn: Optional[str] = None,
        description: Optional[str] = None,
        guardrail_configuration: Optional[GuardrailConfiguration] = None,
        idle_session_ttl_in_seconds: Optional[int] = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        bedrock_endpoint_url: Optional[str] = None,
        runtime_endpoint_url: Optional[str] = None,
        enable_trace: Optional[bool] = False,
        enable_human_input: Optional[bool] = False,
        enable_code_interpreter: Optional[bool] = False,
        **kwargs: Any,
    ) -> BedrockAgentsRunnable:
        """
        Creates a Bedrock Agent Runnable that can be used with an AgentExecutor
        or with LangGraph.

        This also sets up the Bedrock agent, actions and action groups infrastructure
        if they don't exist, ensures the agent is in PREPARED state so that it is
        ready to be called.

        Args:
            agent_name: Name of the agent
            agent_resource_role_arn: The Amazon Resource Name (ARN) of the IAM role with
                permissions to invoke API operations on the agent.
            foundation_model: The foundation model to be used for orchestration by the
                agent you create
            instruction: Instructions that tell the agent what it should do and how it
                should interact with users
            tools: List of tools. Accepts LangChain's BaseTool format
            client_token: A unique, case-sensitive identifier to ensure that the API
                request completes no more than one time. If this token matches a
                previous request, Amazon Bedrock ignores the request, but does not
                return an error
            customer_encryption_key_arn: The Amazon Resource Name (ARN) of the KMS key
                with which to encrypt the agent
            description: A description of the agent
            guardrail_configuration: The unique Guardrail configuration assigned to the
                agent when it is created.
            idle_session_ttl_in_seconds: The number of seconds for which Amazon Bedrock
                keeps information about a user's conversation with the agent. A user
                interaction remains active for the amount of time specified. If no
                conversation occurs during this time, the session expires and Amazon
                Bedrock deletes any data provided before the timeout
            credentials_profile_name: The profile name to use if different from default
            region_name: Region for the Bedrock agent
            bedrock_endpoint_url: Endpoint URL for bedrock agent
            runtime_endpoint_url: Endpoint URL for bedrock agent runtime
            enable_trace: Boolean flag to specify whether trace should be enabled when
                invoking the agent
            enable_human_input: Boolean flag to specify whether a human as a tool should
                 be enabled for the agent.
            enable_code_interpreter: Boolean flag to specify whether a code interpreter
            should be enabled for this session.
            **kwargs: Additional arguments
        Returns:
            BedrockAgentsRunnable configured to invoke the Bedrock agent
        """
        client_params, session = get_boto_session(
            credentials_profile_name=credentials_profile_name,
            region_name=region_name,
            endpoint_url=bedrock_endpoint_url,
        )
        bedrock_client = session.client("bedrock-agent", **client_params)
        bedrock_agent = _get_bedrock_agent(
            bedrock_client=bedrock_client, agent_name=agent_name
        )

        if bedrock_agent:
            agent_id = bedrock_agent["agentId"]
            agent_status = bedrock_agent["agentStatus"]
            if agent_status != "PREPARED":
                _prepare_agent(bedrock_client, agent_id)
        else:
            try:
                agent_id = _create_bedrock_agent(
                    bedrock_client=bedrock_client,
                    agent_name=agent_name,
                    agent_resource_role_arn=agent_resource_role_arn,
                    instruction=instruction,
                    foundation_model=foundation_model,
                    client_token=client_token,
                    customer_encryption_key_arn=customer_encryption_key_arn,
                    description=description,
                    guardrail_configuration=guardrail_configuration,
                    idle_session_ttl_in_seconds=idle_session_ttl_in_seconds,
                )
                _create_bedrock_action_groups(
                    bedrock_client,
                    agent_id,
                    tools,
                    enable_human_input,
                    enable_code_interpreter,
                )
                _prepare_agent(bedrock_client, agent_id)
            except Exception as exception:
                logger.exception("Error in create agent call")
                raise exception

        return cls(
            agent_id=agent_id,
            region_name=region_name,
            credentials_profile_name=credentials_profile_name,
            endpoint_url=runtime_endpoint_url,
            enable_trace=enable_trace,
            **kwargs,
        )

    def invoke(
        self, input: Dict, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> OutputType:
        """
        Invoke the Bedrock agent.

        Args:
            input: The LangChain Runnable input dictionary that can include:
                input: The input text to the agent
                memory_id: The memory id to use for an agent with memory enabled
                session_id: The session id to use. If not provided, a new session will
                    be started
                end_session: Boolean indicating whether to end a session or not
                intermediate_steps: The intermediate steps that are used to provide RoC
                    invocation details
            config: The optional RunnableConfig

        Returns:
            Union[List[BedrockAgentAction], BedrockAgentFinish]
        """
        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

        try:
            agent_input: Dict[str, Any] = {
                "agentId": self.agent_id,
                "agentAliasId": self.agent_alias_id,
                "enableTrace": self.enable_trace,
                "endSession": bool(input.get("end_session", False)),
            }

            if input.get("memory_id"):
                agent_input["memoryId"] = input.get("memory_id")

            if input.get("intermediate_steps"):
                session_id, session_state = self._parse_intermediate_steps(
                    input.get("intermediate_steps")  # type: ignore[arg-type]
                )

                if session_id is not None:
                    agent_input["sessionId"] = session_id

                if session_state is not None:
                    agent_input["sessionState"] = session_state
            else:
                agent_input["inputText"] = input.get("input", "")
                agent_input["sessionId"] = input.get("session_id", str(uuid.uuid4()))

            output = self.client.invoke_agent(**agent_input)
        except Exception as e:
            run_manager.on_chain_error(e)
            raise e

        try:
            response = parse_agent_response(output)
        except Exception as e:
            run_manager.on_chain_error(e)
            raise e
        else:
            run_manager.on_chain_end(response)
            return response

    def _parse_intermediate_steps(
        self, intermediate_steps: List[Tuple[BedrockAgentAction, str]]
    ) -> Tuple[Union[str, None], Union[Dict[str, Any], None]]:
        last_step = max(0, len(intermediate_steps) - 1)
        action = intermediate_steps[last_step][0]
        tool_invoked = action.tool
        messages = action.messages
        session_id = action.session_id

        if tool_invoked:
            action_group_name = _DEFAULT_ACTION_GROUP_NAME
            function_name = tool_invoked
            tool_name_split = tool_invoked.split("::")
            if len(tool_name_split) > 1:
                action_group_name = tool_name_split[0]
                function_name = tool_name_split[1]

            if messages:
                last_message = max(0, len(messages) - 1)
                message = messages[last_message]
                if type(message) is AIMessage:
                    response = intermediate_steps[last_step][1]
                    session_state = {
                        "invocationId": json.loads(message.content)  # type: ignore[arg-type]
                        .get("returnControl", {})
                        .get("invocationId", ""),
                        "returnControlInvocationResults": [
                            {
                                "functionResult": {
                                    "actionGroup": action_group_name,
                                    "function": function_name,
                                    "responseBody": {"TEXT": {"body": response}},
                                }
                            }
                        ],
                    }

                    return session_id, session_state

        return None, None


class BedrockInlineAgentsRunnable(RunnableSerializable[List[BaseMessage], BaseMessage]):
    """Invoke Bedrock Inline Agent as a Runnable."""

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

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            if values.get("client") is None:
                client_params, session = get_boto_session(
                    credentials_profile_name=values.get("credentials_profile_name"),
                    region_name=values.get("region_name"),
                    endpoint_url=values.get("endpoint_url"),
                )
                values["client"] = session.client(
                    "bedrock-agent-runtime", **client_params
                )
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except UnknownServiceError as e:
            raise ModuleNotFoundError(
                "Ensure that you have installed the latest boto3 package "
                "that contains the API for `bedrock-runtime-agent`."
            ) from e
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e
        return values

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
    ) -> BedrockInlineAgentsRunnable:
        """Create a new instance of BedrockInlineAgentsRunnable."""
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
            raise ValueError(
                f"Error creating BedrockInlineAgentsRunnable: {str(e)}"
            ) from e

    def invoke(
        self,
        input: List[BaseMessage],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        """Call InvokeInlineAgent to generate a chat completion"""

        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_tags=config.get("tags"),
            inheritable_metadata=config.get("metadata"),
        )
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

        input_text = self._convert_messages_to_text(input)
        last_message = input[-1]
        if isinstance(last_message, ToolMessage):
            roc_bloc = self._get_roc_block(input)
            function_name = roc_bloc["invocationInputs"][0]["functionInvocationInput"][
                "function"
            ]
            action_group = roc_bloc["invocationInputs"][0]["functionInvocationInput"][
                "actionGroup"
            ]
            roc_input = {
                "invocationId": last_message.tool_call_id,
                "returnControlInvocationResults": [
                    {
                        "functionResult": {
                            "actionGroup": action_group,
                            "function": function_name,
                            "responseBody": {"TEXT": {"body": last_message.content}},
                        }
                    }
                ],
            }
            self.inline_agent_config["inline_session_state"] = roc_input

        input_dict = {
            "input_text": input_text,
            **kwargs,
        }

        try:
            response = self._invoke_inline_agent(input_dict)
        except Exception as e:
            run_manager.on_chain_error(e)
            raise e

        if isinstance(response, BedrockAgentFinish):
            message = AIMessage(
                content=response.return_values["output"],
                additional_kwargs={
                    "session_id": response.session_id,
                    "trace_log": response.trace_log,
                    **(
                        {"files": response.return_values["files"]}
                        if response.return_values.get("files")
                        else {}
                    ),
                },
            )
        else:  # BedrockAgentAction
            # Handle tool use response: parse_agent_response()
            # returns BedrockAgentAction list
            tool_calls: list[ToolCall] = [
                {
                    "name": action.tool,
                    "args": action.tool_input,
                    "id": action.invocation_id
                    if action.invocation_id is not None
                    else str(uuid.uuid4()),
                }
                for action in response
            ]

            message = AIMessage(
                content="",  # Empty content for tool calls
                additional_kwargs={
                    "session_id": response[0].session_id,
                    "trace_log": response[0].trace_log,
                    "roc_log": response[0].log,
                },
                tool_calls=tool_calls,
            )
        run_manager.on_chain_end(message)
        return message

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

    def _get_roc_block(self, messages: List[BaseMessage]) -> Optional[Dict[str, Any]]:
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue

            roc_log = message.additional_kwargs.get("roc_log")
            if not roc_log:
                continue

            try:
                return json.loads(roc_log).get("returnControl", {})
            except json.JSONDecodeError:
                # log error
                continue

        return None

    def _invoke_inline_agent(
        self,
        input_dict: Dict[str, Any],
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
            enableCodeInterpreter=effective_config.get(
                "enable_code_interpreter", False
            ),
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

        # Use existing session_id from input, or from intermediate steps,
        # or generate new one
        self.session_id = (
            input_dict.get("session_id") or self.session_id or str(uuid.uuid4())
        )

        output = self.client.invoke_inline_agent(
            sessionId=self.session_id, **agent_input
        )
        return parse_agent_response(output)

    def _get_action_groups(
        self, tools: List[Any], enableHumanInput: bool, enableCodeInterpreter: bool
    ) -> List:
        """Convert tools to Bedrock action groups format."""
        action_groups = []
        tools_by_action_group = defaultdict(list)

        for tool in tools:
            action_group_name, _ = _get_action_group_and_function_names(tool)
            tools_by_action_group[action_group_name].append(tool)

        for action_group_name, functions in tools_by_action_group.items():
            action_groups.append(
                {
                    "actionGroupName": action_group_name,
                    "actionGroupExecutor": {"customControl": "RETURN_CONTROL"},
                    "functionSchema": {
                        "functions": [
                            _tool_to_function(function) for function in functions
                        ]
                    },
                }
            )

        if enableHumanInput:
            action_groups.append(
                {
                    "actionGroupName": "UserInputAction",
                    "parentActionGroupSignature": "AMAZON.UserInput",
                }
            )

        if enableCodeInterpreter:
            action_groups.append(
                {
                    "actionGroupName": "CodeInterpreterAction",
                    "parentActionGroupSignature": "AMAZON.CodeInterpreter",
                }
            )

        return action_groups

    # Serialization helpers
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the runnable to a dictionary."""
        return {
            "class_name": self.__class__.__name__,
            "region_name": self.region_name,
            "endpoint_url": self.endpoint_url,
            "inline_agent_config": self.inline_agent_config,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BedrockInlineAgentsRunnable":
        """Deserialize the runnable from a dictionary."""
        return cls(**data)
