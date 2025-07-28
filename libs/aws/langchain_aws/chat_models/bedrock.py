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
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    is_data_content_block,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import BaseModel, ConfigDict, Field, model_validator

from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_aws.function_calling import (
    ToolsOutputParser,
    _lc_tool_calls_to_anthropic_tool_use_blocks,
    convert_to_anthropic_tool,
    get_system_message,
)
from langchain_aws.llms.bedrock import (
    BedrockBase,
    _combine_generation_info_for_llm_result,
)
from langchain_aws.utils import (
    anthropic_tokens_supported,
    get_num_tokens_anthropic,
    get_token_ids_anthropic,
)

logger = logging.getLogger(__name__)


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
    """Convert a list of messages to a prompt for Llama 3."""

    return "\n".join(
        ["<|begin_of_text|>"]
        + [_convert_one_message_to_text_llama3(message) for message in messages]
        + ["<|start_header_id|>assistant<|end_header_id|>\n\n"]
    )


def _convert_one_message_to_text_llama4(message: BaseMessage) -> str:
    if isinstance(message, ChatMessage):
        message_text = (
            f"<|header_start|>{message.role}<|header_end|>{message.content}<|eot|>"
        )
    elif isinstance(message, HumanMessage):
        message_text = (
            f"<|header_start|>user<|header_end|>{message.content}<|eot|>"
        )
    elif isinstance(message, AIMessage):
        message_text = (
            f"<|header_start|>assistant<|header_end|>{message.content}<|eot|>"
        )
    elif isinstance(message, SystemMessage):
        message_text = (
            f"<|header_start|>system<|header_end|>{message.content}<|eot|>"
        )
    elif isinstance(message, ToolMessage):
        message_text = (
            f"<|header_start|>ipython<|header_end|>{message.content}<|eom|>"
        )
    else:
        raise ValueError(f"Got unknown type {message}")

    return message_text


def convert_messages_to_prompt_llama4(messages: List[BaseMessage]) -> str:
    """Convert a list of messages to a prompt for Llama 4."""

    return "\n".join(
        ["<|begin_of_text|>"]
        + [_convert_one_message_to_text_llama4(message) for message in messages]
        + ["<|header_start|>assistant<|header_end|>"]
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
    if messages is None:
        return ""
    
    messages = messages.copy()  # don't mutate the original list
    if len(messages) > 0 and not isinstance(messages[-1], AIMessage):
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


def _convert_one_message_to_text_deepseek(message: BaseMessage) -> str:
    if isinstance(message, ChatMessage):
        message_text = (
            f"<|{message.role}|>{message.content}"
        )
    elif isinstance(message, HumanMessage):
        message_text = (
            f"<|User|>{message.content}"
        )
    elif isinstance(message, AIMessage):
        message_text = (
            f"<|Assistant|>{message.content}"
        )
    elif isinstance(message, SystemMessage):
        message_text = (
            f"<|System|>{message.content}"
        )
    else:
        raise ValueError(f"Got unknown type {message}")

    return message_text


def convert_messages_to_prompt_deepseek(messages: List[BaseMessage]) -> str:
    """Convert a list of messages to a prompt for DeepSeek-R1."""
    prompt = "\n<|begin_of_sentence|>"

    for message in messages:
        prompt += _convert_one_message_to_text_deepseek(message)

    prompt += "<|Assistant|>\n\n"

    return prompt


def _convert_one_message_to_text_writer(message: BaseMessage) -> str:
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


def convert_messages_to_prompt_writer(messages: List[BaseMessage]) -> str:
    """Convert a list of messages to a prompt for Writer."""

    return "\n".join(
        [_convert_one_message_to_text_llama(message) for message in messages]
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


def _format_data_content_block(block: dict) -> dict:
    """Format standard data content block to format expected by Converse API."""
    if block["type"] == "image":
        if block["source_type"] == "base64":
            if "mime_type" not in block:
                error_message = "mime_type key is required for base64 data."
                raise ValueError(error_message)
            formatted_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block["mime_type"],
                    "data": block["data"],
                }
            }
        else:
            error_message = "Image data only supported through in-line base64 format."
            raise ValueError(error_message)

    else:
        error_message = f"Blocks of type {block['type']} not supported."
        raise ValueError(error_message)

    return formatted_block


def _merge_messages(
    messages: Sequence[BaseMessage],
) -> List[Union[SystemMessage, AIMessage, HumanMessage]]:
    """Merge runs of human/tool messages into single human messages with content blocks."""  # noqa: E501
    merged: list = []
    for curr in messages:
        curr = curr.model_copy(deep=True)
        if isinstance(curr, ToolMessage):
            if isinstance(curr.content, list) and all(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in curr.content
            ):
                curr = HumanMessage(curr.content)  # type: ignore[misc]
            else:
                curr = HumanMessage(  # type: ignore[misc]
                    [
                        {
                            "type": "tool_result",
                            "content": curr.content,
                            "tool_use_id": curr.tool_call_id,
                        }
                    ]
                )
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
) -> Tuple[Optional[Union[str, List[Dict]]], List[Dict]]:
    """Format messages for anthropic."""
    system: Optional[Union[str, List[Dict]]] = None
    formatted_messages: List[Dict] = []

    merged_messages = _merge_messages(messages)
    for i, message in enumerate(merged_messages):
        if message.type == "system":
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            if isinstance(message.content, str):
                system = message.content
            elif isinstance(message.content, list):
                system_blocks = []
                for item in message.content:
                    if isinstance(item, str):
                        system_blocks.append({"type": "text", "text": item})
                    elif isinstance(item, dict):
                        if item.get("type") != "text":
                            raise ValueError(
                                "System message content item must be type 'text'"
                            )
                        if "text" not in item:
                            raise ValueError(
                                "System message content item must have a 'text' key"
                            )
                        content_item = {"type": "text", "text": item["text"]}
                        if item.get("cache_control"):
                            content_item["cache_control"] = {"type": "ephemeral"}
                        system_blocks.append(content_item)
                    else:
                        raise ValueError(
                            "System message content list must be a string or dict, "
                            f"instead was: {type(item)}"
                        )
                system = system_blocks
            else:
                raise ValueError(
                    "System message content must be a string or list, "
                    f"instead was: {type(message.content)}"
                )

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
            thinking_blocks = []
            native_blocks = []
            tool_blocks = []

            # First collect all blocks by type
            for item in message.content:
                if isinstance(item, str):
                    native_blocks.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError("Dict content item must have a type key")
                    elif is_data_content_block(item):
                        native_blocks.append(_format_data_content_block(item))
                    elif item["type"] == "image_url":
                        # convert format
                        source = _format_image(item["image_url"]["url"])
                        native_blocks.append({"type": "image", "source": source})
                    elif item["type"] == "image":
                        native_blocks.append(item)
                    elif item["type"] == "tool_result":
                        # Process content within tool_result
                        content_item = item["content"]
                        if isinstance(content_item, list):
                            # Handle list content inside tool_result
                            processed_list = []
                            for list_item in content_item:
                                if isinstance(list_item, dict) and list_item.get("type") == "image_url":
                                    # Process image in list
                                    source = _format_image(list_item["image_url"]["url"])
                                    processed_list.append({"type": "image", "source": source})
                                else:
                                    # Keep other items as is
                                    processed_list.append(list_item)
                            # Add processed list to tool_result
                            tool_blocks.append({
                                "type": "tool_result",
                                "tool_use_id": item.get("tool_use_id"),
                                "content": processed_list
                            })
                        else:
                            # For other content types, keep as is
                            tool_blocks.append(item)
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
                            tool_blocks.extend(
                                _lc_tool_calls_to_anthropic_tool_use_blocks(overlapping)
                            )
                        else:
                            item.pop("text", None)
                            tool_blocks.append(item)
                    elif item["type"] in ["thinking", "redacted_thinking"]:
                        # Store thinking blocks separately
                        thinking_blocks.append(item)
                    elif item["type"] == "text":
                        text = item.get("text", "")
                        # Only add non-empty strings for now as empty ones are not
                        # accepted.
                        # https://github.com/anthropics/anthropic-sdk-python/issues/461

                        if text.strip():
                            content_item = {"type": "text", "text": text}
                            if item.get("cache_control"):
                                content_item["cache_control"] = {"type": "ephemeral"}
                            native_blocks.append(content_item)
                    else:
                        tool_blocks.append(item)
                else:
                    raise ValueError(
                        f"Content items must be str or dict, instead was: {type(item)}"
                    )

            # Add tool calls if present (for AIMessage)
            if isinstance(message, AIMessage) and message.tool_calls:
                # Track which tool call IDs we've already processed
                used_tool_call_ids = {
                    block["id"]
                    for block in tool_blocks
                    if block.get("type") == "tool_use"
                }
                # Only add tool calls that haven't been processed yet
                new_tool_calls = [
                    tc
                    for tc in message.tool_calls
                    if tc["id"] not in used_tool_call_ids
                ]
                if new_tool_calls:
                    tool_blocks.extend(
                        _lc_tool_calls_to_anthropic_tool_use_blocks(new_tool_calls)
                    )

            # For assistant messages, when thinking blocks exist, ensure they come first
            if role == "assistant":
                content = native_blocks + tool_blocks
                if thinking_blocks:
                    content = thinking_blocks + content
            elif role == "user" and tool_blocks and native_blocks:
                content = tool_blocks + native_blocks  # tool result must precede text
                if thinking_blocks:
                    content = thinking_blocks + content
            else:
                # combine all blocks in standard order
                content = native_blocks + tool_blocks
                # Only include thinking blocks if they exist
                if thinking_blocks:
                    content = thinking_blocks + content

        elif isinstance(message, AIMessage):
            # For string content, create appropriate structure
            content_list = []

            # Add thinking blocks from additional_kwargs if present
            if message.additional_kwargs and "thinking" in message.additional_kwargs:
                thinking_data = message.additional_kwargs["thinking"]
                if thinking_data and isinstance(thinking_data, dict):
                    if "text" in thinking_data and "signature" in thinking_data:
                        content_list.append(
                            {
                                "type": "thinking",
                                "thinking": thinking_data["text"],
                                "signature": thinking_data["signature"],
                            }
                        )

            # Add base content as text block
            if message.content:
                content_list.append({"type": "text", "text": message.content})

            # Add tool calls if present
            if message.tool_calls:
                content_list.extend(
                    _lc_tool_calls_to_anthropic_tool_use_blocks(message.tool_calls)
                )

            # For assistant messages with thinking blocks, ensure they come first
            if role == "assistant" and any(
                block.get("type") in ["thinking", "redacted_thinking"]
                for block in content_list
                if isinstance(block, dict)
            ):
                # Separate thinking blocks and non-thinking blocks
                thinking_blocks = [
                    block
                    for block in content_list
                    if isinstance(block, dict)
                    and block.get("type") in ["thinking", "redacted_thinking"]
                ]
                other_blocks = [
                    block
                    for block in content_list
                    if not (
                        isinstance(block, dict)
                        and block.get("type") in ["thinking", "redacted_thinking"]
                    )
                ]
                # Combine with thinking first
                content = thinking_blocks + other_blocks
            else:
                # No thinking blocks or not an assistant message
                content = content_list
        else:
            # Simple string content
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
        elif provider == "deepseek":
            prompt = convert_messages_to_prompt_deepseek(messages=messages)
        elif provider == "meta":
            if "llama4" in model:
                prompt = convert_messages_to_prompt_llama4(messages=messages)
            elif "llama3" in model:
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
        elif provider == "writer":
            prompt = convert_messages_to_prompt_writer(messages=messages)
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


_message_type_lookups = {
    "human": "user",
    "ai": "assistant",
    "AIMessageChunk": "assistant",
    "HumanMessageChunk": "user",
}


class ChatBedrock(BaseChatModel, BedrockBase):
    """A chat model that uses the Bedrock API."""

    system_prompt_with_tools: str = ""
    beta_use_converse_api: bool = False
    """Use the new Bedrock ``converse`` API which provides a standardized interface to
    all Bedrock models. Support still in beta. See ChatBedrockConverse docs for more."""

    stop_sequences: Optional[List[str]] = Field(default=None, alias="stop")
    """Stop sequence inference parameter from new Bedrock ``converse`` API providing
    a sequence of characters that causes a model to stop generating a response. See
    https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_InferenceConfiguration.html
    for more."""

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

    @model_validator(mode="before")
    @classmethod
    def set_beta_use_converse_api(cls, values: Dict) -> Any:
        model_id = values.get("model_id", values.get("model"))

        if model_id and "beta_use_converse_api" not in values:
            values["beta_use_converse_api"] = "nova" in model_id
        return values

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)

        # For backwards compatibility, we don't transfer known parameters out of
        # model_kwargs
        model_kwargs = values.pop("model_kwargs", {})
        values = _build_model_kwargs(values, all_required_field_names)
        if model_kwargs or values.get("model_kwargs", {}):
            values["model_kwargs"] = {
                **values.get("model_kwargs", {}),
                **model_kwargs,
            }
        return values

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
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="amazon_bedrock",
            ls_model_name=self.model_id,
            ls_model_type="chat",
        )
        if ls_temperature := params.get("temperature", self.temperature):
            ls_params["ls_temperature"] = ls_temperature
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None):
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.beta_use_converse_api:
            yield from self._as_converse._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return
        provider = self._get_provider()
        prompt, system, formatted_messages = None, None, None

        if provider == "anthropic":
            system, formatted_messages = ChatPromptAdapter.format_messages(
                provider, messages
            )
            if self.system_prompt_with_tools:
                if system:
                    system = self.system_prompt_with_tools + f"\n{system}"
                else:
                    system = self.system_prompt_with_tools
        else:
            prompt = ChatPromptAdapter.convert_messages_to_prompt(
                provider=provider, messages=messages, model=self._get_base_model()
            )

        added_model_name = False
        for chunk in self._prepare_input_and_invoke_stream(
            prompt=prompt,
            system=system,
            messages=formatted_messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            if isinstance(chunk, AIMessageChunk):
                generation_chunk = ChatGenerationChunk(message=chunk)
                if run_manager:
                    run_manager.on_llm_new_token(
                        generation_chunk.text, chunk=generation_chunk
                    )
                yield generation_chunk
            else:
                delta = chunk.text
                response_metadata = None
                if generation_info := chunk.generation_info:
                    usage_metadata = generation_info.pop("usage_metadata", None)
                    response_metadata = generation_info
                    if not added_model_name:
                        response_metadata["model_name"] = self.model_id
                        added_model_name = True
                else:
                    usage_metadata = None
                generation_chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=delta,
                        response_metadata=response_metadata,
                        usage_metadata=usage_metadata,
                    )
                    if response_metadata is not None
                    else AIMessageChunk(content=delta)
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        generation_chunk.text, chunk=generation_chunk
                    )
                yield generation_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.beta_use_converse_api:
            if not self.streaming:
                return self._as_converse._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                stream_iter = self._as_converse._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
                return generate_from_stream(stream_iter)
        completion = ""
        llm_output: Dict[str, Any] = {}
        tool_calls: List[ToolCall] = []
        provider_stop_reason_code = self.provider_stop_reason_key_map.get(
            self._get_provider(), "stop_reason"
        )
        provider = self._get_provider()
        if self.streaming:
            if provider == "anthropic":
                stream_iter = self._stream(messages, stop, run_manager, **kwargs)
                return generate_from_stream(stream_iter)

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
                    provider=provider, messages=messages, model=self._get_base_model()
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
        # usage metadata
        if usage := llm_output.get("usage"):
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
            cache_write_input_tokens = usage.get("cache_write_input_tokens", 0)
            usage_metadata = UsageMetadata(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_token_details={
                    "cache_read": cache_read_input_tokens,
                    "cache_creation": cache_write_input_tokens,
                },
                total_tokens=usage.get("total_tokens", input_tokens + output_tokens),
            )
        else:
            usage_metadata = None
        logger.debug(f"The message received from Bedrock: {completion}")
        llm_output["model_id"] = self.model_id  # backward-compatibility
        llm_output["model_name"] = self.model_id
        msg = AIMessage(
            content=completion,
            additional_kwargs=llm_output,
            tool_calls=cast(List[ToolCall], tool_calls),
            usage_metadata=usage_metadata,
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
        if (
            self._model_is_anthropic
            and not self.custom_get_token_ids
            and anthropic_tokens_supported()
        ):
            return get_num_tokens_anthropic(text)
        return super().get_num_tokens(text)

    def get_token_ids(self, text: str) -> List[int]:
        if self._model_is_anthropic and not self.custom_get_token_ids:
            if anthropic_tokens_supported():
                return get_token_ids_anthropic(text)
            else:
                warnings.warn(
                    "Falling back to default token method due to missing or "
                    "incompatible `anthropic` installation "
                    "(needs <=0.38.0).\n\nIf using `anthropic>0.38.0`, "
                    "it is recommended to provide the model class with a "
                    "custom_get_token_ids method implementing a more accurate "
                    "tokenizer for Anthropic. For get_num_tokens, as another "
                    "alternative, you can implement your own token counter method "
                    "using the ChatAnthropic or AnthropicLLM classes."
                )
        return super().get_token_ids(text)

    def set_system_prompt_with_tools(self, xml_tools_system_prompt: str) -> None:
        """Workaround to bind. Sets the system prompt with tools"""
        self.system_prompt_with_tools = xml_tools_system_prompt

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
        if self.beta_use_converse_api:
            if isinstance(tool_choice, bool):
                tool_choice = "any" if tool_choice else None
            return self._as_converse.bind_tools(
                tools, tool_choice=tool_choice, **kwargs
            )
        if self._get_provider() == "anthropic":
            formatted_tools = [convert_to_anthropic_tool(tool) for tool in tools]

            # true if the model is a claude 3 model
            if "claude-" in self._get_base_model():
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
            else:
                # add tools to the system prompt, the old way
                system_formatted_tools = get_system_message(formatted_tools)
                self.set_system_prompt_with_tools(system_formatted_tools)
        return self

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
        if self.beta_use_converse_api:
            return self._as_converse.with_structured_output(
                schema, include_raw=include_raw, **kwargs
            )
        if "claude-" not in self._get_base_model():
            raise ValueError(
                f"Structured output is not supported for model {self._get_base_model()}"
            )

        tool_name = convert_to_anthropic_tool(schema)["name"]
        llm = self.bind_tools(
            [schema],
            tool_choice=tool_name,
            ls_structured_output_format={
                "kwargs": {"method": "function_calling"},
                "schema": convert_to_openai_tool(schema),
            },
        )
        if isinstance(schema, type) and is_basemodel_subclass(schema):
            output_parser = ToolsOutputParser(
                first_tool_only=True, pydantic_schemas=[schema]
            )
        else:
            output_parser = ToolsOutputParser(first_tool_only=True, args_only=True)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    @property
    def _as_converse(self) -> ChatBedrockConverse:
        kwargs = {
            k: v
            for k, v in (self.model_kwargs or {}).items()
            if k
            in (
                "stop",
                "stop_sequences",
                "max_tokens",
                "temperature",
                "top_p",
                "additional_model_request_fields",
                "additional_model_response_field_paths",
                "performance_config",
                "request_metadata",
            )
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences

        return ChatBedrockConverse(
            client=self.client,
            model=self.model_id,
            region_name=self.region_name,
            credentials_profile_name=self.credentials_profile_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            config=self.config,
            provider=self.provider or "",
            base_url=self.endpoint_url,
            guardrail_config=(self.guardrails if self._guardrails_enabled else None),  # type: ignore[call-arg]
            **kwargs,
        )
