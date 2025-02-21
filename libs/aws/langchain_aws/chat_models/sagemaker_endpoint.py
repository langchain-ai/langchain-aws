import logging
import re
import warnings
from collections import defaultdict
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import (
    BaseChatModel,
    LangSmithParams,
    LanguageModelInput,
)
from nibernie_pr.llms.sagemaker_endpoint import (
    SagemakerEndpoint
)
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from pydantic import BaseModel, ConfigDict, model_validator
from langchain_aws.function_calling import (
    ToolsOutputParser,
    _lc_tool_calls_to_anthropic_tool_use_blocks,
    convert_to_anthropic_tool,
    get_system_message,
)
from langchain_aws.utils import (
    anthropic_tokens_supported,
    get_num_tokens_anthropic,
    get_token_ids_anthropic,
)


logger = logging.getLogger(__name__)


class ChatSagemakerEndpoint(BaseChatModel, SagemakerEndpoint):
    """A chat model that uses a HugguingFace TGI compatible SageMaker Endpoint."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "amazon_sagemaker_chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "sagemaker"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.region_name:
            attributes["region_name"] = self.region_name

        return attributes

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs = {**_model_kwargs, **kwargs}
        _endpoint_kwargs = self.endpoint_kwargs or {}
        completion = ""
        llm_output: Dict[str, Any] = {}
        tool_calls: List[ToolCall] = []

        sagemaker_messages = _messages_to_sagemaker(messages)
        logger.debug(f"input message to sagemaker: {sagemaker_messages}")
        invocation_params = {
            "EndpointName": self.endpoint_name,
            "Body": self.content_handler.transform_input(sagemaker_messages, _model_kwargs),
            "ContentType": self.content_handler.content_type,
            "Accept": self.content_handler.accepts,
            **_endpoint_kwargs,
        }
        
        # If inference_compoent_name is specified, append it to invocation_params
        if self.inference_component_name:
            invocation_params["InferenceComponentName"] = self.inference_component_name

        try:
            response = self.client.invoke_endpoint(**invocation_params)
        except Exception as e:
            logging.error(f"Error raised by inference endpoint: {e}")
            if run_manager is not None:
                run_manager.on_llm_error(e)
            raise e
        logger.info(f"The message received from SageMaker: {response['Body']}")

        response_message = self.content_handler.transform_output(response["Body"]) 

        return ChatResult(generations=[ChatGeneration(message=response_message)])
        

def _messages_to_sagemaker(
    messages: List[BaseMessage],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Merge system, human, ai message runs because Anthropic expects (at most) 1
    # system message then alternating human/ai messages.
    sagemaker_messages: List[Dict[str, Any]] = []
    if not isinstance(messages, list):
        messages = [messages]
        
    messages = merge_message_runs(messages)
    for msg in messages:
        content = msg.content
        if isinstance(msg, HumanMessage):
            # If there's a human, tool, human message sequence, the
            # tool message will be merged with the first human message, so the second
            # human message will now be preceded by a human message and should also
            # be merged with it.
            if sagemaker_messages and sagemaker_messages[-1]["role"] == "user":
                sagemaker_messages[-1]["content"].extend(content)
            else:
                sagemaker_messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            sagemaker_messages.append({"role": "assistant", "content": content})
        elif isinstance(msg, SystemMessage):
            sagemaker_messages.insert(0, {"role": "system", "content": content})
        else:
            raise ValueError(f"Unsupported message type {type(msg)}")
    return sagemaker_messages