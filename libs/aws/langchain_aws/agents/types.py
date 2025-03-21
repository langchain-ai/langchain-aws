from __future__ import annotations

from typing import Dict, List, Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

_DEFAULT_ACTION_GROUP_NAME = "DEFAULT_AG_"
_TEST_AGENT_ALIAS_ID = "TSTALIASID"


class BedrockAgentFinish(AgentFinish):
    """AgentFinish with session id information.

    Parameters:
        session_id: Session id
        trace_log: trace log as string when enable_trace flag is set
    """

    session_id: str
    trace_log: Optional[str]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Check if the class is serializable by LangChain.

        Returns:
            False
        """
        return False


class BedrockAgentAction(AgentAction):
    """AgentAction with session id information.

    Parameters:
        session_id: session id
        trace_log: trace log as string when enable_trace flag is set
    """

    session_id: str
    trace_log: Optional[str]
    invocation_id: Optional[str]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Check if the class is serializable by LangChain.

        Returns:
            False
        """
        return False


OutputType = Union[List[BedrockAgentAction], BedrockAgentFinish]


class GuardrailConfiguration(TypedDict):
    guardrail_identifier: str
    guardrail_version: str


class KnowledgebaseConfiguration(TypedDict, total=False):
    description: str
    knowledgeBaseId: str
    retrievalConfiguration: Dict


class InlineAgentConfiguration(TypedDict, total=False):
    """Configurations for an Inline Agent."""

    foundation_model: str
    instruction: str
    enable_trace: Optional[bool]
    tools: List[BaseTool]
    enable_human_input: Optional[bool]
    enable_code_interpreter: Optional[bool]
    customer_encryption_key_arn: Optional[str]
    idle_session_ttl_in_seconds: Optional[int]
    guardrail_configuration: Optional[GuardrailConfiguration]
    knowledge_bases: Optional[KnowledgebaseConfiguration]
    prompt_override_configuration: Optional[Dict]
    inline_session_state: Optional[Dict]
