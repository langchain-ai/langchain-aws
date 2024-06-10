import json
import re
from collections import defaultdict
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
    Type,
    Union,
    cast,
)

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Extra
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_aws.function_calling import (
    _lc_tool_calls_to_anthropic_tool_use_blocks,
    _tools_in_params,
    convert_to_anthropic_tool,
    get_system_message,
)
from langchain_aws.llms.bedrock import (
    BedrockBase,
    _combine_generation_info_for_llm_result,
)
from langchain_aws.utils import (
    get_num_tokens_anthropic,
    get_token_ids_anthropic,
)


def _convert_one_message_to_text_llama(message: BaseMessage) -> str:
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


def convert_messages_to_prompt_llama(messages: List[BaseMessage]) -> str:
    """Convert a list of messages to a prompt for llama."""

    return "\n".join(
        [_convert_one_message_to_text_llama(message) for message in messages]
    )


def _convert_one_message_to_text_llama3(message: BaseMessage) -> str:
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


def convert_messages_to_prompt_llama3(messages: List[BaseMessage]) -> str:
    """Convert a list of messages to a prompt for llama."""

    return "\n".join(
        ["<|begin_of_text|>"]
        + [_convert_one_message_to_text_llama3(message) for message in messages]
        + ["<|start_header_id|>assistant<|end_header_id|>\n\n"]
    )


def _convert_one_message_to_text_anthropic(
    message: BaseMessage,
    human_prompt: str,
    ai_prompt: str,
) -> str:
    content = cast(str, message.content)
    if isinstance(message, ChatMessage):
        message_text = f"\n\n{message.role.capitalize()}: {content}"
    elif isinstance(message, HumanMessage):
        message_text = f"{human_prompt} {content}"
    elif isinstance(message, AIMessage):
        message_text = f"{ai_prompt} {content}"
    elif isinstance(message, SystemMessage):
        message_text = content
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_text


def convert_messages_to_prompt_anthropic(
    messages: List[BaseMessage],
    *,
    human_prompt: str = "\n\nHuman:",
    ai_prompt: str = "\n\nAssistant:",
) -> str:
    """Format a list of messages into a full prompt for the Anthropic model
    Args:
        messages (List[BaseMessage]): List of BaseMessage to combine.
        human_prompt (str, optional): Human prompt tag. Defaults to "\n\nHuman:".
        ai_prompt (str, optional): AI prompt tag. Defaults to "\n\nAssistant:".
    Returns:
        str: Combined string with necessary human_prompt and ai_prompt tags.
    """

    messages = messages.copy()  # don't mutate the original list
    if not isinstance(messages[-1], AIMessage):
        messages.append(AIMessage(content=""))

    text = "".join(
        _convert_one_message_to_text_anthropic(message, human_prompt, ai_prompt)
        for message in messages
    )

    # trim off the trailing ' ' that might come from the "Assistant: "
    return text.rstrip()


def _convert_one_message_to_text_mistral(message: BaseMessage) -> str:
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


def convert_messages_to_prompt_mistral(messages: List[BaseMessage]) -> str:
    """Convert a list of messages to a prompt for mistral."""
    return "\n".join(
        [_convert_one_message_to_text_mistral(message) for message in messages]
    )


def _format_image(image_url: str) -> Dict:
    """
    Formats an image of format data:image/jpeg;base64,{b64_string}
    to a dict for anthropic api

    {
      "type": "base64",
      "media_type": "image/jpeg",
      "data": "/9j/4AAQSkZJRg...",
    }

    And throws an error if it's not a b64 image
    """
    regex = r"^data:(?P<media_type>image/.+);base64,(?P<data>.+)$"
    match = re.match(regex, image_url)
    if match is None:
        raise ValueError(
            "Anthropic only supports base64-encoded images currently."
            " Example: data:image/png;base64,'/9j/4AAQSk'..."
        )
    return {
        "type": "base64",
        "media_type": match.group("media_type"),
        "data": match.group("data"),
    }


def _merge_messages(
    messages: Sequence[BaseMessage],
) -> List[Union[SystemMessage, AIMessage, HumanMessage]]:
    """Merge runs of human/tool messages into single human messages with content blocks."""  # noqa: E501
    merged: list = []
    for curr in messages:
        curr = curr.copy(deep=True)
        if isinstance(curr, ToolMessage):
            if isinstance(curr.content, str):
                curr = HumanMessage(  # type: ignore[misc]
                    [
                        {
                            "type": "tool_result",
                            "content": curr.content,
                            "tool_use_id": curr.tool_call_id,
                        }
                    ]
                )
            else:
                curr = HumanMessage(curr.content)  # type: ignore[misc]
        last = merged[-1] if merged else None
        if isinstance(last, HumanMessage) and isinstance(curr, HumanMessage):
            if isinstance(last.content, str):
                new_content: List = [{"type": "text", "text": last.content}]
            else:
                new_content = last.content
            if isinstance(curr.content, str):
                new_content.append({"type": "text", "text": curr.content})
            else:
                new_content.extend(curr.content)
            last.content = new_content
        else:
            merged.append(curr)
    return merged


def _format_anthropic_messages(
    messages: List[BaseMessage],
) -> Tuple[Optional[str], List[Dict]]:
    """Format messages for anthropic."""

    """
    [
                {
                    "role": _message_type_lookups[m.type],
                    "content": [_AnthropicMessageContent(text=m.content).dict()],
                }
                for m in messages
            ]
    """
    system: Optional[str] = None
    formatted_messages: List[Dict] = []

    merged_messages = _merge_messages(messages)
    for i, message in enumerate(merged_messages):
        if message.type == "system":
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            if not isinstance(message.content, str):
                raise ValueError(
                    "System message must be a string, "
                    f"instead was: {type(message.content)}"
                )
            system = message.content
            continue

        role = _message_type_lookups[message.type]
        content: Union[str, List]

        if not isinstance(message.content, str):
            # parse as dict
            assert isinstance(
                message.content, list
            ), "Anthropic message content must be str or list of dicts"

            # populate content
            content = []
            for item in message.content:
                if isinstance(item, str):
                    content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError("Dict content item must have a type key")
                    elif item["type"] == "image_url":
                        # convert format
                        source = _format_image(item["image_url"]["url"])
                        content.append({"type": "image", "source": source})
                    elif item["type"] == "tool_use":
                        # If a tool_call with the same id as a tool_use content block
                        # exists, the tool_call is preferred.
                        if isinstance(message, AIMessage) and item["id"] in [
                            tc["id"] for tc in message.tool_calls
                        ]:
                            overlapping = [
                                tc
                                for tc in message.tool_calls
                                if tc["id"] == item["id"]
                            ]
                            content.extend(
                                _lc_tool_calls_to_anthropic_tool_use_blocks(overlapping)
                            )
                        else:
                            item.pop("text", None)
                            content.append(item)
                    elif item["type"] == "text":
                        text = item.get("text", "")
                        # Only add non-empty strings for now as empty ones are not
                        # accepted.
                        # https://github.com/anthropics/anthropic-sdk-python/issues/461
                        if text.strip():
                            content.append({"type": "text", "text": text})
                    else:
                        content.append(item)
                else:
                    raise ValueError(
                        f"Content items must be str or dict, instead was: {type(item)}"
                    )
        elif isinstance(message, AIMessage) and message.tool_calls:
            content = (
                []
                if not message.content
                else [{"type": "text", "text": message.content}]
            )
            # Note: Anthropic can't have invalid tool calls as presently defined,
            # since the model already returns dicts args not JSON strings, and invalid
            # tool calls are those with invalid JSON for args.
            content += _lc_tool_calls_to_anthropic_tool_use_blocks(message.tool_calls)
        else:
            content = message.content

        formatted_messages.append({"role": role, "content": content})
    return system, formatted_messages


class ChatPromptAdapter:
    """Adapter class to prepare the inputs from Langchain to prompt format
    that Chat model expects.
    """

    @classmethod
    def convert_messages_to_prompt(
        cls, provider: str, messages: List[BaseMessage], model: str
    ) -> str:
        if provider == "anthropic":
            prompt = convert_messages_to_prompt_anthropic(messages=messages)
        elif provider == "meta":
            if "llama3" in model:
                prompt = convert_messages_to_prompt_llama3(messages=messages)
            else:
                prompt = convert_messages_to_prompt_llama(messages=messages)
        elif provider == "mistral":
            prompt = convert_messages_to_prompt_mistral(messages=messages)
        elif provider == "amazon":
            prompt = convert_messages_to_prompt_anthropic(
                messages=messages,
                human_prompt="\n\nUser:",
                ai_prompt="\n\nBot:",
            )
        else:
            raise NotImplementedError(
                f"Provider {provider} model does not support chat."
            )
        return prompt

    @classmethod
    def format_messages(
        cls, provider: str, messages: List[BaseMessage]
    ) -> Tuple[Optional[str], List[Dict]]:
        if provider == "anthropic":
            return _format_anthropic_messages(messages)

        raise NotImplementedError(
            f"Provider {provider} not supported for format_messages"
        )


_message_type_lookups = {"human": "user", "ai": "assistant"}


class ChatBedrock(BaseChatModel, BedrockBase):
    """A chat model that uses the Bedrock API."""

    system_prompt_with_tools: str = ""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "amazon_bedrock_chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "bedrock"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.region_name:
            attributes["region_name"] = self.region_name

        return attributes

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    # def _format_anthropic_params(
    #     self,
    #     *,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     **kwargs: Dict,
    # ) -> Dict:
    #     # get system prompt if any
    #     system, formatted_messages = _format_anthropic_messages(messages)
    #     stop_sequences = stop or self.stop_sequences
    #     rtn = {
    #         "model": self.model,
    #         "max_tokens": self.max_tokens,
    #         "messages": formatted_messages,
    #         "temperature": self.temperature,
    #         "top_k": self.top_k,
    #         "top_p": self.top_p,
    #         "stop_sequences": stop_sequences,
    #         "system": system,
    #         **self.model_kwargs,
    #         **kwargs,
    #     }
    #     rtn = {k: v for k, v in rtn.items() if v is not None}

    #     return rtn

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        provider = self._get_provider()
        prompt, system, formatted_messages = None, None, None

        if "claude-3" in self._get_model():
            if _tools_in_params({**kwargs}):
                result = self._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                message = result.generations[0].message
                if isinstance(message, AIMessage) and message.tool_calls is not None:
                    tool_call_chunks = [
                        {
                            "name": tool_call["name"],
                            "args": json.dumps(tool_call["args"]),
                            "id": tool_call["id"],
                            "index": idx,
                        }
                        for idx, tool_call in enumerate(message.tool_calls)
                    ]
                    message_chunk = AIMessageChunk(
                        content=message.content,
                        tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
                        usage_metadata=message.usage_metadata,
                    )
                    yield ChatGenerationChunk(message=message_chunk)
                else:
                    yield cast(ChatGenerationChunk, result.generations[0])
                return
        if provider == "anthropic":
            system, formatted_messages = ChatPromptAdapter.format_messages(
                provider, messages
            )
            # use tools the new way with claude 3
            # if "claude-3" in self._get_model():
            #     if _tools_in_params()
            if self.system_prompt_with_tools:
                if system:
                    system = self.system_prompt_with_tools + f"\n{system}"
                else:
                    system = self.system_prompt_with_tools
        else:
            prompt = ChatPromptAdapter.convert_messages_to_prompt(
                provider=provider, messages=messages, model=self._get_model()
            )

        for chunk in self._prepare_input_and_invoke_stream(
            prompt=prompt,
            system=system,
            messages=formatted_messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            delta = chunk.text
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=delta, response_metadata=chunk.generation_info
                )
                if chunk.generation_info is not None
                else AIMessageChunk(content=delta)
            )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        completion = ""
        llm_output: Dict[str, Any] = {}
        tool_calls: List[Dict[str, Any]] = []
        provider_stop_reason_code = self.provider_stop_reason_key_map.get(
            self._get_provider(), "stop_reason"
        )
        if self.streaming:
            response_metadata: List[Dict[str, Any]] = []
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                completion += chunk.text
                response_metadata.append(chunk.message.response_metadata)
                if "tool_calls" in chunk.message.additional_kwargs.keys():
                    tool_calls = chunk.message.additional_kwargs["tool_calls"]
            llm_output = _combine_generation_info_for_llm_result(
                response_metadata, provider_stop_reason_code
            )
        else:
            provider = self._get_provider()
            prompt, system, formatted_messages = None, None, None
            params: Dict[str, Any] = {**kwargs}

            if provider == "anthropic":
                system, formatted_messages = ChatPromptAdapter.format_messages(
                    provider, messages
                )
                # use tools the new way with claude 3
                if self.system_prompt_with_tools:
                    if system:
                        system = self.system_prompt_with_tools + f"\n{system}"
                    else:
                        system = self.system_prompt_with_tools
            else:
                prompt = ChatPromptAdapter.convert_messages_to_prompt(
                    provider=provider, messages=messages, model=self._get_model()
                )

            if stop:
                params["stop_sequences"] = stop

            completion, tool_calls, llm_output = self._prepare_input_and_invoke(
                prompt=prompt,
                stop=stop,
                run_manager=run_manager,
                system=system,
                messages=formatted_messages,
                **params,
            )

        llm_output["model_id"] = self.model_id
        if len(tool_calls) > 0:
            msg = AIMessage(
                content=completion,
                additional_kwargs=llm_output,
                tool_calls=cast(List[ToolCall], tool_calls),
            )
        else:
            msg = AIMessage(content=completion, additional_kwargs=llm_output)
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=msg,
                )
            ],
            llm_output=llm_output,
        )

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        final_usage: Dict[str, int] = defaultdict(int)
        final_output = {}
        for output in llm_outputs:
            output = output or {}
            usage = output.get("usage", {})
            for token_type, token_count in usage.items():
                final_usage[token_type] += token_count
            final_output.update(output)
        final_output["usage"] = final_usage
        return final_output

    def get_num_tokens(self, text: str) -> int:
        if self._model_is_anthropic:
            return get_num_tokens_anthropic(text)
        else:
            return super().get_num_tokens(text)

    def get_token_ids(self, text: str) -> List[int]:
        if self._model_is_anthropic:
            return get_token_ids_anthropic(text)
        else:
            return super().get_token_ids(text)

    def set_system_prompt_with_tools(self, xml_tools_system_prompt: str) -> None:
        """Workaround to bind. Sets the system prompt with tools"""
        self.system_prompt_with_tools = xml_tools_system_prompt

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
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
        provider = self._get_provider()

        if provider == "anthropic":
            formatted_tools = [convert_to_anthropic_tool(tool) for tool in tools]

            # true if the model is a claude 3 model
            if "claude-3" in self._get_model():
                if not tool_choice:
                    pass
                elif isinstance(tool_choice, dict):
                    kwargs["tool_choice"] = tool_choice
                elif isinstance(tool_choice, str) and tool_choice in ("any", "auto"):
                    kwargs["tool_choice"] = {"type": tool_choice}
                elif isinstance(tool_choice, str):
                    kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
                else:
                    raise ValueError(
                        f"Unrecognized 'tool_choice' type {tool_choice=}."
                        f"Expected dict, str, or None."
                    )
                return self.bind(tools=formatted_tools, **kwargs)

            # add tools to the system prompt, the old way
            system_formatted_tools = get_system_message(formatted_tools)
            self.set_system_prompt_with_tools(system_formatted_tools)
        return self


@deprecated(since="0.1.0", removal="0.2.0", alternative="ChatBedrock")
class BedrockChat(ChatBedrock):
    pass
