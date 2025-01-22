from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
    Dict,
    Callable,
    Literal,
    Type,
    TypeVar,
    Tuple,
    TypedDict,
    cast,
    Mapping
)

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ChatMessage,
    ToolCall
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel
from langchain_aws.utils import enforce_stop_tokens
from langchain_aws.function_calling import _tools_in_params, _lc_tool_calls_to_anthropic_tool_use_blocks
from langchain_core.messages.tool import tool_call, tool_call_chunk
from pydantic import BaseModel
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from abc import ABC, abstractmethod
import re
import json
import logging
import warnings
# ModelAdapter might also need access to the data that the wrapper ChatModel class has
# for example, the provider or custom inputs passed in by the user


class ModelAdapter(ABC):
    """Abstract base class for model-specific adaptation strategies"""

    @abstractmethod
    def convert_messages_to_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Convert LangChain messages to model-specific payload"""
        pass

    @abstractmethod
    def convert_response_to_chat_result(self, response: Any) -> ChatResult:
        """Convert model-specific response to LangChain ChatResult"""
        pass

    @abstractmethod
    def convert_stream_response_to_chunks(
        self, response: Any
    ) -> Iterator[ChatGenerationChunk]:
        """Convert model-specific stream response to LangChain chunks"""
        pass

    @abstractmethod
    def format_tools(
        self, tools: Sequence[Union[Dict[str, Any], TypeBaseModel, Callable, BaseTool]]
    ) -> Any:
        """Format tools for the specific model"""
        pass




