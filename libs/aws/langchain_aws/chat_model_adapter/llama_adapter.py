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
from langchain_aws.chat_model_adapter.demo_chat_adapter import ModelAdapter
from pydantic import BaseModel
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
import re
import json
import logging
import warnings


class AnthropicTool(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]

HUMAN_PROMPT = "\n\nHuman:"
ASSISTANT_PROMPT = "\n\nAssistant:"
ALTERNATION_ERROR = (
    "Error: Prompt must alternate between '\n\nHuman:' and '\n\nAssistant:'."
)

# Example concrete implementation for a specific model
class BedrockLlamaAdapter(ModelAdapter):

    _message_type_lookups = {
        "human": "user",
        "ai": "assistant",
        "AIMessageChunk": "assistant",
        "HumanMessageChunk": "user",
    }

    def convert_messages_to_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Specific implementation for converting LC messages to Claude payload
        prompt = self._convert_messages_to_prompt(messages=messages, model=model)

        return {"prompt": prompt}


    def convert_response_to_chat_result(self, response: Any) -> ChatResult:
        pass

    def convert_stream_response_to_chunks(
        self, response: Any
    ) -> Iterator[ChatGenerationChunk]:
        """Convert model-specific stream response to LangChain chunks"""
        pass

    def format_tools(
        self, tools: Sequence[Union[Dict[str, Any], TypeBaseModel, Callable, BaseTool]]
    ) -> Any:
        """Format tools for the specific model"""
        pass
    

    def _convert_messages_to_prompt(
        self, messages: List[BaseMessage], model: str
    ) -> str:
        if "llama3" in model:
            prompt = self._convert_messages_to_prompt_llama3(messages=messages)
        else:
            prompt = self._convert_messages_to_prompt_llama(messages=messages)
        return prompt

    def _convert_one_message_to_text_llama(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            message_text = f"[INST] {message.content} [/INST]"
        elif isinstance(message, AIMessage):
            message_text = f"{message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"<<SYS>> {message.content} <</SYS>>"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text


    def _convert_messages_to_prompt_llama(self, messages: List[BaseMessage]) -> str:
        """Convert a list of messages to a prompt for llama."""

        return "\n".join(
            [self._convert_one_message_to_text_llama(message) for message in messages]
        )


    def _convert_one_message_to_text_llama3(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = (
                f"<|start_header_id|>{message.role}"
                f"<|end_header_id|>{message.content}<|eot_id|>"
            )
        elif isinstance(message, HumanMessage):
            message_text = (
                f"<|start_header_id|>user" f"<|end_header_id|>{message.content}<|eot_id|>"
            )
        elif isinstance(message, AIMessage):
            message_text = (
                f"<|start_header_id|>assistant"
                f"<|end_header_id|>{message.content}<|eot_id|>"
            )
        elif isinstance(message, SystemMessage):
            message_text = (
                f"<|start_header_id|>system" f"<|end_header_id|>{message.content}<|eot_id|>"
            )
        else:
            raise ValueError(f"Got unknown type {message}")

        return message_text


    def _convert_messages_to_prompt_llama3(self, messages: List[BaseMessage]) -> str:
        """Convert a list of messages to a prompt for llama."""
        return "\n".join(
            ["<|begin_of_text|>"]
            + [self._convert_one_message_to_text_llama3(message) for message in messages]
            + ["<|start_header_id|>assistant<|end_header_id|>\n\n"]
        )

    def _prepare_input(
        self,
        model_kwargs: Dict[str, Any],
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        tools: Optional[List[AnthropicTool]] = None,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        
        input_body = {**model_kwargs}
        input_body["prompt"] = prompt
        if max_tokens:
            input_body["max_gen_len"] = max_tokens
        if temperature is not None:
            input_body["temperature"] = temperature
        return input_body

    def _prepare_output(self, response: Any) -> dict:
        text = ""
        tool_calls = []
        response_body = json.loads(response.get("body").read().decode())
        text = response_body.get("generation")

        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        prompt_tokens = int(headers.get("x-amzn-bedrock-input-token-count", 0))
        completion_tokens = int(headers.get("x-amzn-bedrock-output-token-count", 0))
        return {
            "text": text,
            "tool_calls": tool_calls,
            "body": response_body,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "stop_reason": response_body.get("stop_reason"),
        }


    def prepare_input_and_invoke(
        self,
        client: Any,
        model_id: str,
        request_options: Dict[str, Any],
        input_params: Dict[str, Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[
        str,
        List[ToolCall],
        Dict[str, Any],
    ]:
        _model_kwargs = model_kwargs or {}
        params = {**_model_kwargs, **kwargs}
        
        input_body = self._prepare_input(
            model_kwargs=params,
            prompt=input_params["prompt"],
            system=input_params["system"],
            messages=input_params["messages"],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        body = json.dumps(input_body)
        request_options["body"] = body

        try:
            print("Meta adapter used for invoke")
            response = client.invoke_model(**request_options)

            (
                text,
                tool_calls,
                body,
                usage_info,
                stop_reason,
            ) = self._prepare_output(response).values()

        except Exception as e:
            logging.error(f"Error raised by bedrock service: {e}")
            if run_manager is not None:
                run_manager.on_llm_error(e)
            raise e

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        llm_output = {"usage": usage_info, "stop_reason": stop_reason}


        ''' TODO: checking for intervention is body should be done in ChatBedrock'''
        # Verify and raise a callback error if any intervention occurs or a signal is
        # sent from a Bedrock service,
        # such as when guardrails are triggered.
        # services_trace = self._get_bedrock_services_signal(body)  # type: ignore[arg-type]

        # if run_manager is not None and services_trace.get("signal"):
        #     run_manager.on_llm_error(
        #         Exception(
        #             f"Error raised by bedrock service: {services_trace.get('reason')}"
        #         ),
        #         **services_trace,
        #     )

        return text, tool_calls, llm_output
    # Implement other abstract methods similarly...