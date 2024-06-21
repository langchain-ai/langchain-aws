import json
import uuid
from abc import ABC
from typing import Any, List, Tuple, Union, Callable

from langchain.agents import AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.manager import (
    Callbacks,
)
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.tools import Tool

from langchain_aws.agents.bedrock.agent_base import BedrockAgentBase, _DEFAULT_ACTION_GROUP_NAME
from langchain_aws.agents.bedrock.agent_base import BedrockAgentMetadata
from langchain_aws.agents.bedrock.agent_executor import BedrockAgentExecutor


def parse_response(
        agent_output: str
) -> Union[AgentAction, AgentFinish]:
    """
    Custom Parser to parse BedrockAgents response

    Args:
        agent_output: Text response from BedrockAgents

    Returns:
            Either the iteration of a Bedrock Agent can be AgentFinish or AgentAction
            AgentFinish - Final answer returned by BedrockAgents
            AgentAction - When a tool/API is returned by BedrockAgents
    """

    try:
        if agent_output and "returnControl" in agent_output:
            return_of_control = json.loads(agent_output).get('returnControl', {})
            if return_of_control:
                return_of_control_inputs = return_of_control.get('invocationInputs', [])
                if return_of_control_inputs:
                    invocation_input = return_of_control_inputs[0].get('functionInvocationInput', {})
                    action_group = invocation_input.get('actionGroup', '')
                    function = invocation_input.get('function', '')
                    parameters = invocation_input.get('parameters', [])
                    parameters_json = {}
                    for parameter in parameters:
                        parameters_json[parameter.get('name')] = parameter.get('value', '')

                    tool = f"{action_group}::{function}"
                    if _DEFAULT_ACTION_GROUP_NAME in action_group:
                        tool = f"{function}"
                    return AgentAction(
                        tool=tool,
                        tool_input=parameters_json,
                        log=agent_output
                    )
        if not agent_output:
            agent_output = ''
        return AgentFinish({"output": agent_output}, log=agent_output)
    except Exception as ex:
        raise Exception("Parse exception encountered {}".format(repr(ex)))


class BedrockAgentResponseParser(BaseOutputParser, ABC):
    """
    Custom parser class for BedrockAgents, to parse Agent output
    """

    def parse(
            self,
            agent_output: str
    ) -> Union[AgentAction, AgentFinish]:
        """
        Custom Parser to parse BedrockAgents response

        Args:
            agent_output: Text response from BedrockAgents

        Returns:
                Either the iteration of a Bedrock Agent can be AgentFinish or AgentAction
                AgentFinish - Final answer returned by BedrockAgents
                AgentAction - When a tool/API is returned by BedrockAgents
        """
        return parse_response(agent_output)


class BedrockAgentInputFormatter(BasePromptTemplate, ABC):
    """
    Custom formatter class for BedrockAgents, to format input prior to sending to the Agent
    """

    input_variables: List[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def format_prompt(self, **kwargs) -> str:
        return format(kwargs)

    def format(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any
    ) -> Any:
        """
        Format request sent to BedrockAgents

        Args:
            intermediate_steps: List of intermediary steps traced by Langchain
            callbacks: Callbacks provided for additional formatting

        Returns:
                Supports sending 2 forms of output -
                1. Text response, which is sent if a final answer or further questions is received from BedrockAgents
                2. Session state, if tool invocation was returned from BedrockAgents, the response to the tool
                    invocation can be added as a structured response `returnControlInvocationResults` inside session
                    state and invoke BedrockAgent.
                    Session state is also used to add "PromptSessionAttributes" or "SessionAttributes" to BedrockAgents
                        Refer - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_SessionState.html
        """

        if not intermediate_steps:
            return kwargs.get('input', '')

        last_step = max(0, len(intermediate_steps) - 1)
        action = intermediate_steps[last_step][0]
        tool_invoked = action.tool
        messages = action.messages
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
                        "invocationId": json.loads(message.content).get('returnControl', {}).get('invocationId', ''),
                        "returnControlInvocationResults": [{
                            "functionResult": {
                                "actionGroup": action_group_name,
                                "function": function_name,
                                "responseBody": {
                                    "TEXT": {
                                        "body": response
                                    }
                                }
                            }
                        }],
                    }
                    kwargs.__setitem__('session_state', {**session_state, **kwargs.get('session_state', {})})
        return kwargs


class BedrockAgent(BedrockAgentBase, ABC):
    bedrock_agent_base: BedrockAgentBase = None
    agent_executor: BedrockAgentExecutor = None
    trace_handler: Callable = None
    input_handler: Callable = None
    output_handler: Callable = None

    def __init__(
            self,
            agent_name: str,
            agent_instruction: str,
            agent_region: str,
            agent_foundation_model: str = 'anthropic.claude-3-sonnet-20240229-v1:0',
            agent_tools: List[Tool] = None,
            trace_handler: Callable = None,
            input_handler: Callable = lambda _input: _input,
            output_handler: Callable = lambda _output: _output,
            **kwargs
    ):
        super().__init__(**kwargs)
        bedrock_agent_metadata = BedrockAgentMetadata(
            agent_name=agent_name,
            agent_instruction=agent_instruction,
            agent_foundation_model=agent_foundation_model,
            agent_region=agent_region,
            agent_tools=agent_tools
        )
        self.trace_handler = trace_handler
        self.input_handler = input_handler
        self.output_handler = output_handler
        self.bedrock_agent_base = self._activate_agent(bedrock_agent_metadata)
        self.agent_executor = self._activate_agent_executor(bedrock_agent_metadata)

    def invoke_agent(
            self,
            invoke_agent_request: dict,
            **kwargs: Any,
    ):
        """
        Invoke Bedrock Agent and return the output.

        Args:
            invoke_agent_request: Json structure invoke agent request
            **kwargs: Containing session state

        Returns:
            The response from BedrockAgents
        """
        return self.invoke(
            invoke_agent_request.get('input'),
            kwargs.get('session_id', None),
            **kwargs
        )

    def invoke(
            self,
            agent_input: str,
            session_id: str = None,
            **kwargs: Any,
    ):
        """
        Invoke Bedrock Agent and return the output.
        Create `invoke_agent_request` with input text, session_id and defaulting to enabling trace

        Args:
            agent_input: Input text to BedrockAgents
            session_id: Session Ids are unique values, identifying a list of turns pertaining to a single conversation
            **kwargs: Containing session state

        Returns:
            The response from BedrockAgents
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        invoke_agent_request = {
            "input": agent_input,
            "session_id": session_id,
            "trace_enabled": True
        }
        return self.agent_executor.invoke({**invoke_agent_request, **kwargs})

    def run(
            self,
            agent_input: dict,
            session_id: str = None,
            **kwargs: Any,
    ):
        """
        Invoke Bedrock Agent and return the output.
        Overloaded function to sync with LangGraph

        Args:
           agent_input: Input text to BedrockAgents, as dict wth format {'input': ''}
           session_id: Session Ids are unique values, identifying a list of turns pertaining to a single conversation
           **kwargs: Containing session state

        Returns:
           The response from BedrockAgents
        """
        if type(agent_input) is dict and agent_input.get("input"):
            _input = self.input_handler(agent_input)
            return self.output_handler(self.invoke(_input, session_id, **kwargs))

        raise Exception("Input to Bedrock Agents should be a dict with 'input' field")

    def delete(
            self,
    ):
        """
        Delete an agent
        """
        return self.bedrock_agent_base.delete()

    def _activate_agent(
            self,
            bedrock_agent_metadata: BedrockAgentMetadata
    ) -> BedrockAgentBase:
        """
        Activate an agent, i.e -
         1. Create the agent execution role
         2. Create the agent using the execution role
         2. Attach actions
         3. Prepare the agent for inference

        Args:
            bedrock_agent_metadata: Meta data for Bedrock Agents

        Returns:
            BedrockAgentExecutor - Instantiates BedrockAgentExecutor to run inference on BedrockAgents
        """
        output_parser = BedrockAgentResponseParser()
        request_formatter = BedrockAgentInputFormatter()

        agent = BedrockAgentBase()
        agent.create(
            bedrock_agent_metadata=bedrock_agent_metadata,
            output_parser=output_parser,
            prompt_template=request_formatter,
            trace_handler=self.trace_handler
        )
        return agent

    def _activate_agent_executor(
            self,
            bedrock_agent_metadata: BedrockAgentMetadata
    ) -> BedrockAgentExecutor:

        return BedrockAgentExecutor.from_agent_and_tools(
            agent=self.bedrock_agent_base,
            verbose=False,
            tools=bedrock_agent_metadata.agent_tools,
            return_intermediate_steps=True,
            max_iterations=8
        )