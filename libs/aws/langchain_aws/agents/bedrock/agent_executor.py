from __future__ import annotations

import inspect
from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union, Tuple)

from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    CallbackManagerForChainRun,
    Callbacks,
    CallbackManager)
from langchain_core.load import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables import RunnableConfig, ensure_config, RunnableSerializable
from langchain_core.tools import BaseTool
from pydantic.v1 import validator, Field

from langchain_aws.agents.bedrock.agent_base import BedrockAgentBase


# class BedrockAgentExecutor(RunnableSerializable[Dict[str, Any], Dict[str, Any]], ABC):
# class BedrockAgentExecutor(Chain, ABC):
class BedrockAgentExecutor:
    """
    BedrockAgentExecutor handling the orchestration
    """

    name: str = 'BedrockAgentExecutor'
    agent: BedrockAgentBase = None
    name_to_tool_map: dict = {}

    def __init__(
            self,
            agent: BedrockAgentBase,
            **kwargs: {}
    ):
        # super().__init__(**kwargs)
        self.agent = agent
        self.name_to_tool_map = {tool.name: tool for tool in self.agent.agent_tools}

    @classmethod
    def from_agent_and_tools(
            cls,
            agent: BedrockAgentBase,
            tools: Sequence[BaseTool] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> BedrockAgentExecutor:
        """
        Create an Bedrock Agent executor from BedrockAgentBase and tools.

        Args:
            agent: Bedrock Agent to use
            tools: Function which can be used in conjunction with BedrockAgentBase
            callbacks: Chain callbacks

        Returns:
           BedrockAgentBase Executor instance
        """

        return cls(
            agent=agent,
            tools=tools,
            callbacks=callbacks,
            **kwargs,
        )

    # def invoke(
    #         self,
    #         inputs: Dict[str, Any],
    #         **kwargs,
    # ) -> str:
    #     return self._call({**inputs, **kwargs}).get('output', {})

    def invoke(
            self,
            agent_input: Dict[str, Any],
            intermediate_steps: List[Tuple[AgentAction, str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Union[AgentFinish | AgentAction]:

        agent_response = self.agent.plan(intermediate_steps, callbacks, **agent_input)
        if type(agent_response) is AgentFinish:
            return agent_response.return_values

        if type(agent_response) is AgentAction:
            tool_response = self._perform_agent_action(agent_response)
            observation = [(tool_response.action, tool_response.observation)]
            return self.invoke(agent_input, observation, callbacks, **kwargs)

        # return outputs

    def _perform_agent_action(
            self,
            agent_action: AgentAction
    ) -> AgentStep:
        if agent_action.tool in self.name_to_tool_map:
            tool = self.name_to_tool_map[agent_action.tool]
            observation = tool.run(
                agent_action.tool_input
            )
        else:
            raise Exception(f"Invalid tool name {agent_action.tool} provided !")
        return AgentStep(action=agent_action, observation=observation)

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Execute the chain.

        Args:
            inputs: A dict of named inputs to the chain. Assumed to contain all inputs
                specified in `Chain.input_keys`, including any inputs added by memory.
            run_manager: The callbacks manager that contains the callback handlers for
                this run of the chain.

        Returns:
            A dict of named outputs. Should contain all outputs specified in
                `Chain.output_keys`.
        """

        agent_input = {'input': inputs.get('input', '')}
        kwargs = {'session_state': inputs.get('session_state', {})}
        agent_response = self.agent._invoke_agent_base(**{**agent_input, **kwargs})
        return {
            'output': agent_response
        }

    @property
    def input_keys(self) -> List[str]:
        """
        Return the input keys.

        :meta private: input_keys
        """
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        """
        Return the singular output key.

        :meta private: output_keys
        """
        return ['output']

    # @validator("agent", pre=True, always=True)
    # def validate(self, agent: BedrockAgentBase):
    #     return agent
