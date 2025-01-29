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
class BedrockClaudeAdapter(ModelAdapter):

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
        system, formatted_messages = self._format_anthropic_messages(messages=messages)

        return {"system": system, "messages":formatted_messages}


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

    def _format_image(self, image_url: str) -> Dict:
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
        self,
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
        self,
        messages: List[BaseMessage],
    ) -> Tuple[Optional[str], List[Dict]]:
        """Format messages for anthropic."""
        system: Optional[str] = None
        formatted_messages: List[Dict] = []

        merged_messages = self._merge_messages(messages)
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

            role = self._message_type_lookups[message.type]
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
                            source = self._format_image(item["image_url"]["url"])
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

    def _add_newlines_before_ha(self, input_text: str) -> str:
        new_text = input_text
        for word in ["Human:", "Assistant:"]:
            new_text = new_text.replace(word, "\n\n" + word)
            for i in range(2):
                new_text = new_text.replace("\n\n\n" + word, "\n\n" + word)
        return new_text

    def _human_assistant_format(self, input_text: str) -> str:
        if input_text.count("Human:") == 0 or (
            input_text.find("Human:") > input_text.find("Assistant:")
            and "Assistant:" in input_text
        ):
            input_text = HUMAN_PROMPT + " " + input_text  # SILENT CORRECTION
        if input_text.count("Assistant:") == 0:
            input_text = input_text + ASSISTANT_PROMPT  # SILENT CORRECTION
        if input_text[: len("Human:")] == "Human:":
            input_text = "\n\n" + input_text
        input_text = self._add_newlines_before_ha(input_text)
        count = 0
        # track alternation
        for i in range(len(input_text)):
            if input_text[i : i + len(HUMAN_PROMPT)] == HUMAN_PROMPT:
                if count % 2 == 0:
                    count += 1
                else:
                    warnings.warn(ALTERNATION_ERROR + f" Received {input_text}")
            if input_text[i : i + len(ASSISTANT_PROMPT)] == ASSISTANT_PROMPT:
                if count % 2 == 1:
                    count += 1
                else:
                    warnings.warn(ALTERNATION_ERROR + f" Received {input_text}")

        if count % 2 == 1:  # Only saw Human, no Assistant
            input_text = input_text + ASSISTANT_PROMPT  # SILENT CORRECTION

        return input_text

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
        if messages:
            if tools:
                input_body["tools"] = tools
            input_body["anthropic_version"] = "bedrock-2023-05-31"
            input_body["messages"] = messages
            if system:
                input_body["system"] = system
            if max_tokens:
                input_body["max_tokens"] = max_tokens
            elif "max_tokens" not in input_body:
                input_body["max_tokens"] = 1024

        if prompt:
            input_body["prompt"] = self._human_assistant_format(prompt)
            if max_tokens:
                input_body["max_tokens_to_sample"] = max_tokens
            elif "max_tokens_to_sample" not in input_body:
                input_body["max_tokens_to_sample"] = 1024

        if temperature is not None:
            input_body["temperature"] = temperature
        return input_body
    
    def _extract_tool_calls(self, content: List[dict]) -> List[ToolCall]:
        tool_calls = []
        for block in content:
            if block["type"] != "tool_use":
                continue
            tool_calls.append(
                tool_call(name=block["name"], args=block["input"], id=block["id"])
            )
        return tool_calls

    def _prepare_output(self, response: Any) -> dict:
        text = ""
        tool_calls = []
        response_body = json.loads(response.get("body").read().decode())

        if "completion" in response_body:
            text = response_body.get("completion")
        elif "content" in response_body:
            content = response_body.get("content")
            if len(content) == 1 and content[0]["type"] == "text":
                text = content[0]["text"]
            elif any(block["type"] == "tool_use" for block in content):
                tool_calls = self._extract_tool_calls(content)

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

        tools = None
        if "claude-3" in model_id and _tools_in_params(params):
            tools = params["tools"]
        
        input_body = self._prepare_input(
            model_kwargs=params,
            prompt=input_params["prompt"],
            system=input_params["system"],
            messages=input_params["messages"],
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        body = json.dumps(input_body)
        request_options["body"] = body

        try:
            print("anthropic adapter used for invoking response")
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