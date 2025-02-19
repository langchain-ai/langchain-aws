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
from langchain_aws.llms.sagemaker_endpoint import (
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


class ChatSageMakerEndpointConfig(BaseModel):
    stop_reason: str


class ChatSagemakerEndpoint(BaseChatModel, SagemakerEndpoint):
    """A chat model that uses a HugguingFace TGI compatible SageMaker Endpoint."""
    system_prompt_with_tools: str = ""
    """Use SageMaker invoke endpoint API which provides a standardized interface models"""

    def __init__(self, config: ChatSageMakerEndpointConfig):
        self.config = config

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

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        continue

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        continue

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        completion = ""
        llm_output: Dict[str, Any] = {}
        tool_calls: List[ToolCall] = []

        prompt, system, formatted_messages = None, None, None
        params: Dict[str, Any] = {**kwargs}

        completion, tool_calls, llm_output = self._prepare_input_and_invoke(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                system=system,
                messages=formatted_messages,
                **params,
            )
        logger.info(f"The message received from SageMaker: {completion}")
        msg = AIMessage(
            content=completion,
            additional_kwargs=llm_output,
            tool_calls=cast(List[ToolCall], tool_calls)
        )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=msg,
                )
            ],
            llm_output=llm_output,
        )

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        continue

    def get_num_tokens(self, text: str) -> int:
        continue

    def get_token_ids(self, text: str) -> List[int]:
        continue

    def set_system_prompt_with_tools(self, xml_tools_system_prompt: str) -> None:
        """Workaround to bind. Sets the system prompt with tools"""
        continue

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], TypeBaseModel, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model has a tool calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        continue

    def with_structured_output(
        self,
        schema: Union[Dict, TypeBaseModel],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input. The output type depends on
            include_raw and schema.

            If include_raw is True then output is a dict with keys:
                raw: BaseMessage,
                parsed: Optional[_DictOrPydantic],
                parsing_error: Optional[BaseException],

            If include_raw is False and schema is a Dict then the runnable outputs a Dict.
            If include_raw is False and schema is a Type[BaseModel] then the runnable
            outputs a BaseModel.

        Example: Pydantic schema (include_raw=False):
            .. code-block:: python

                from langchain_aws.chat_models.bedrock import ChatBedrock
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm =ChatBedrock(
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                    model_kwargs={"temperature": 0.001},
                )  # type: ignore[call-arg]
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example:  Pydantic schema (include_raw=True):
            .. code-block:: python

                from langchain_aws.chat_models.bedrock import ChatBedrock
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm =ChatBedrock(
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                    model_kwargs={"temperature": 0.001},
                )  # type: ignore[call-arg]
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Dict schema (include_raw=False):
            .. code-block:: python

                from langchain_aws.chat_models.bedrock import ChatBedrock

                schema = {
                    "name": "AnswerWithJustification",
                    "description": "An answer to the user question along with justification for the answer.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                            "justification": {"type": "string"},
                        },
                        "required": ["answer", "justification"]
                    }
                }
                llm =ChatBedrock(
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                    model_kwargs={"temperature": 0.001},
                )  # type: ignore[call-arg]
                structured_llm = llm.with_structured_output(schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        """  # noqa: E501
        continue