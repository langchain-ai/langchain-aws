from __future__ import annotations

from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
)

from langchain.chains.base import Chain
from langchain_core.callbacks import (
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.tools import BaseTool

from langchain_aws.agents.bedrock.bedrock_agent import BedrockAgent


class BedrockAgentExecutor(Chain, ABC):
    """
    BedrockAgentExecutor handling the orchestration
    """

    name: str = 'BedrockAgentExecutor'
    agent: BedrockAgent = None

    def __init__(
            self,
            agent: BedrockAgent,
            **kwargs: {}
    ):
        super().__init__(**kwargs)
        self.agent = agent

    @classmethod
    def from_agent_and_tools(
            cls,
            agent: BedrockAgent,
            tools: Sequence[BaseTool] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> BedrockAgentExecutor:
        """
        Create an Bedrock Agent executor from BedrockAgent and tools.

        Args:
            agent: Bedrock Agent to use
            tools: Function which can be used in conjunction with BedrockAgent
            callbacks: Chain callbacks

        Returns:
           BedrockAgent Executor instance
        """

        return cls(
            agent=agent,
            tools=tools,
            callbacks=callbacks,
            **kwargs,
        )

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
        agent_response = self.agent.invoke_agent(agent_input, **kwargs).get('output', {})
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
        return self.agent.return_values
