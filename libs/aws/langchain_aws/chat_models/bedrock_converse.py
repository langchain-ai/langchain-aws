import json
import re
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
    Type,
    TypeVar,
    Union,
    cast,
)

import boto3
from langchain_core._api import beta
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)
from langchain_core.messages.ai import AIMessageChunk, UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_aws.function_calling import ToolsOutputParser


@beta()
class ChatBedrockConverse(BaseChatModel):
    """Chat model that uses the Bedrock converse API."""

    client: Any = Field(exclude=True)  #: :meta private:

    model_id: str = Field(alias="model")
    """Id of the model to call
    
    e.g., ``"amazon.titan-text-express-v1"``. This is equivalent to the modelID property
    in the list-foundation-models api. For custom and provisioned models, an ARN value 
    is expected.
    """

    max_tokens: Optional[int] = None
    """Max tokens to generate."""

    stop_sequences: Optional[List[str]] = Field(None, alias="stop")
    """Stop generation if any of these substrings occurs."""

    temperature: Optional[float] = None
    """Sampling temperature. Must be 0 to 1."""

    top_p: Optional[float] = None
    """The percentage of most-likely candidates that are considered for the next token.
    
    Must be 0 to 1.
    
    For example, if you choose a value of 0.8 for topP, the model selects from 
    the top 80% of the probability distribution of tokens that could be next in the 
    sequence."""

    region_name: Optional[str] = None
    """The aws region, e.g., `us-west-2`. 
    
    Falls back to AWS_DEFAULT_REGION env variable or region specified in ~/.aws/config 
    in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = Field(default=None, exclude=True)
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files.
    
    Profile should either have access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used. See: 
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    provider: str = ""
    """The model provider, e.g., amazon, cohere, ai21, etc. 
    
    When not supplied, provider is extracted from the first part of the model_id e.g. 
    'amazon' in 'amazon.titan-text-express-v1'. This value should be provided for model 
    ids that do not have the provider in them, e.g., custom and provisioned models that 
    have an ARN associated with them.
    """

    endpoint_url: Optional[str] = Field(None, alias="base_url")
    """Needed if you don't want to default to us-east-1 endpoint"""

    config: Any = None
    """An optional botocore.config.Config instance to pass to the client."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        allow_population_by_field_name = True

    @root_validator(pre=False, skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that AWS credentials to and python package exists in environment."""
        values["provider"] = values["provider"] or values["model_id"].split(".")[0]

        if values["client"] is not None:
            return values

        try:
            if values["credentials_profile_name"] is not None:
                session = boto3.Session(profile_name=values["credentials_profile_name"])
            else:
                session = boto3.Session()
        except ValueError as e:
            raise ValueError(f"Error raised by bedrock service: {e}")
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                f"profile name are valid. Bedrock error: {e}"
            ) from e

        values["region_name"] = get_from_dict_or_env(
            values,
            "region_name",
            "AWS_DEFAULT_REGION",
            default=session.region_name,
        )

        client_params = {}
        if values["region_name"]:
            client_params["region_name"] = values["region_name"]
        if values["endpoint_url"]:
            client_params["endpoint_url"] = values["endpoint_url"]
        if values["config"]:
            client_params["config"] = values["config"]

        try:
            values["client"] = session.client("bedrock-runtime", **client_params)
        except ValueError as e:
            raise ValueError(f"Error raised by bedrock service: {e}")
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                f"profile name are valid. Bedrock error: {e}"
            ) from e

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        bedrock_messages, system = _messages_to_bedrock(messages)
        params = self._converse_params(stop=stop, **_snake_to_camel_keys(kwargs))
        response = self.client.converse(
            messages=bedrock_messages, system=system, **params
        )
        response_message = _parse_response(response)
        return ChatResult(generations=[ChatGeneration(message=response_message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        bedrock_messages, system = _messages_to_bedrock(messages)
        params = self._converse_params(stop=stop, **_snake_to_camel_keys(kwargs))
        response = self.client.converse_stream(
            messages=bedrock_messages, system=system, **params
        )
        for event in response["stream"]:
            if message_chunk := _parse_stream_event(event):
                yield ChatGenerationChunk(message=message_chunk)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "any"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        if tool_choice:
            kwargs["tool_choice"] = _format_tool_choice(tool_choice)
        return self.bind(tools=_format_tools(tools), **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        llm = self.bind_tools([schema], tool_choice="any")
        if isinstance(schema, type) and issubclass(schema, BaseModel):
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

    def _converse_params(
        self,
        *,
        stop: Optional[List[str]] = None,
        stopSequences: Optional[List[str]] = None,
        maxTokens: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        topP: Optional[float] = None,
        tools: Optional[List] = None,
        toolChoice: Optional[dict] = None,
        modelId: Optional[str] = None,
        inferenceConfig: Optional[dict] = None,
        toolConfig: Optional[dict] = None,
        additionalModelRequestFields: Optional[dict] = None,
        additionalModelResponseFieldPaths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not inferenceConfig:
            inferenceConfig = {
                "maxTokens": maxTokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "topP": self.top_p or topP,
                "stopSequences": stop or stopSequences or self.stop_sequences,
            }
        if not toolConfig and tools:
            toolChoice = _format_tool_choice(toolChoice) if toolChoice else None
            toolConfig = {"tools": _format_tools(tools), "toolChoice": toolChoice}

        return _drop_none(
            {
                "modelId": modelId or self.model_id,
                "inferenceConfig": inferenceConfig,
                "toolConfig": toolConfig,
                "additionalModelRequestFields": additionalModelRequestFields,
                "additionalModelResponseFieldPaths": additionalModelResponseFieldPaths,
            }
        )

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "amazon_bedrock_converse_chat"


def _messages_to_bedrock(
    messages: List[BaseMessage],
) -> Tuple[List[Dict[str, Any]], List[Dict[Literal["text"], str]]]:
    """Handle Bedrock converse and Anthropic style content blocks"""
    bedrock_messages: List[Dict[str, Any]] = []
    bedrock_system: List[Dict[Literal["text"], str]] = []
    for msg in messages:
        content = _anthropic_to_bedrock(msg.content)
        if isinstance(msg, HumanMessage):
            bedrock_messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            bedrock_messages.append({"role": "assistant", "content": content})
        elif isinstance(msg, SystemMessage):
            if isinstance(msg.content, str):
                bedrock_system.append({"text": msg.content})
            else:
                bedrock_system.extend(
                    [
                        {
                            "text": block["text"] if isinstance(block, dict) else block
                            for block in msg.content
                        }
                    ]
                )
        elif isinstance(msg, ToolMessage):
            if bedrock_messages and bedrock_messages[-1]["role"] == "user":
                curr = bedrock_messages.pop()
            else:
                curr = {"role": "user", "content": []}

            # TODO: Add ToolMessage.status support.
            curr["content"].append(
                {"toolResult": {"content": content, "toolUseID": msg.tool_call_id}}
            )
            bedrock_messages.append(curr)
        else:
            raise ValueError()
    return bedrock_messages, bedrock_system


def _parse_response(response: Dict[str, Any]) -> AIMessage:
    anthropic_content = _bedrock_to_anthropic(
        response.pop("output")["message"]["content"]
    )
    tool_calls = _extract_tool_calls(anthropic_content)
    usage = UsageMetadata(_camel_to_snake_keys(response.pop("usage")))  # type: ignore[misc]
    return AIMessage(
        content=anthropic_content,  # type: ignore[arg-type]
        usage_metadata=usage,
        response_metadata=response,
        tool_calls=tool_calls,
    )


def _parse_stream_event(event: Dict[str, Any]) -> Optional[BaseMessageChunk]:
    if "messageStart" in event:
        # TODO: needed?
        return (
            AIMessageChunk(content=[])
            if event["messageStart"]["role"] == "assistant"
            else HumanMessageChunk(content=[])
        )
    elif "contentBlockStart" in event:
        block = {
            **_bedrock_to_anthropic([event["contentBlockStart"]["start"]])[0],
            "index": event["contentBlockStart"]["contentBlockIndex"],
        }
        tool_call_chunks = []
        if block["type"] == "tool_use":
            tool_call_chunks.append(
                ToolCallChunk(
                    name=block.get("name"),
                    id=block.get("tool_use_id"),
                    args=block.get("input"),
                    index=event["contentBlockStart"]["contentBlockIndex"],
                )
            )
        return AIMessageChunk(content=[block], tool_call_chunks=tool_call_chunks)
    elif "contentBlockDelta" in event:
        block = {
            **_bedrock_to_anthropic([event["contentBlockDelta"]["delta"]])[0],
            "index": event["contentBlockDelta"]["contentBlockIndex"],
        }
        tool_call_chunks = []
        if block["type"] == "tool_use":
            tool_call_chunks.append(
                ToolCallChunk(
                    name=block.get("name"),
                    id=block.get("tool_use_id"),
                    args=block.get("input"),
                    index=event["contentBlockDelta"]["contentBlockIndex"],
                )
            )
        return AIMessageChunk(content=[block], tool_call_chunks=tool_call_chunks)
    elif "contentBlockStop" in event:
        # TODO: needed?
        return AIMessageChunk(
            content=[{"index": event["contentBlockStop"]["contentBlockIndex"]}]
        )
    elif "messageStop" in event:
        # TODO: snake case response metadata?
        return AIMessageChunk(content=[], response_metadata=event["messageStop"])
    elif "metadata" in event:
        usage = UsageMetadata(_camel_to_snake_keys(event["metadata"].pop("usage")))  # type: ignore[misc]
        return AIMessageChunk(
            content=[], response_metadata=event["metadata"], usage_metadata=usage
        )
    elif "Exception" in list(event.keys())[0]:
        name, info = list(event.items())[0]
        raise ValueError(
            f"Received AWS exception {name}:\n\n{json.dumps(info, indent=2)}"
        )
    else:
        raise ValueError(f"Received unsupported stream event:\n\n{event}")


def _anthropic_to_bedrock(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    bedrock_content: List[Dict[str, Any]] = []
    for block in _snake_to_camel_keys(content):
        if isinstance(block, str):
            bedrock_content.append({"text": block})
        # assume block is already in bedrock format
        elif "type" not in block:
            bedrock_content.append(block)
        elif block["type"] == "text":
            bedrock_content.append({"text": block["text"]})
        elif block["type"] == "image":
            bedrock_content.append({"image": {}})
        elif block["type"] == "tool_use":
            bedrock_content.append(
                {
                    "toolUse": {
                        "toolUseId": block["id"],
                        "input": block["input"],
                        "name": block["name"],
                    }
                }
            )
        elif block["type"] == "tool_result":
            bedrock_content.append(
                {
                    "toolResult": {
                        "toolUseID": block["tool_use_id"],
                        "content": _anthropic_to_bedrock(content),
                    }
                }
            )
        # Only needed for tool_result content blocks.
        elif block["type"] == "json":
            bedrock_content.append({"json": block["json"]})
        else:
            raise ValueError(f"Unsupported content block type:\n{block}")
    return bedrock_content


def _bedrock_to_anthropic(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anthropic_content = []
    for block in _camel_to_snake_keys(content):
        if "text" in block:
            anthropic_content.append({"type": "text", "text": block["text"]})
        elif "tool_use" in block:
            block["tool_use"]["id"] = block["tool_use"].pop("tool_use_id", None)
            anthropic_content.append({"type": "tool_use", **block["tool_use"]})
        elif "image" in block:
            anthropic_content.append(
                {
                    "type": "image",
                    "source": {
                        "media_type": f"image/{block['image']['format']}",
                        "type": "base64",
                        # TODO: convert to b64 str
                        "data": block["image"]["source"]["bytes"],
                    },
                }
            )
        elif "tool_result" in block:
            anthropic_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["tool_result"]["tool_use_id"],
                    "is_error": block["tool_result"]["status"] == "success",
                    "content": _bedrock_to_anthropic(block["tool_result"]["content"]),
                }
            )
        # Only occurs in content blocks of a tool_result:
        elif "json" in block:
            anthropic_content.append({"type": "json", **block})
        else:
            raise ValueError(
                "Unexpected content block type in content. Expected to have one of "
                "'text', 'tool_use', 'image', or 'tool_result' keys. Received:\n\n"
                f"{block}"
            )
    return anthropic_content


def _format_tools(
    tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],],
) -> List[Dict[Literal["toolSpec"], Dict[str, Union[Dict[str, Any], str]]]]:
    formatted_tools: List = []
    for tool in tools:
        if isinstance(tool, dict) and "toolSpec" in tool:
            formatted_tools.append(tool)
        else:
            spec = convert_to_openai_function(tool)
            spec["inputSchema"] = {"json": spec.pop("parameters")}
            formatted_tools.append({"toolSpec": spec})
    return formatted_tools


def _format_tool_choice(
    tool_choice: Union[Dict[str, Dict], Literal["auto", "any"], str],
) -> Dict[str, Dict[str, str]]:
    if isinstance(tool_choice, dict):
        return tool_choice
    elif tool_choice in ("auto", "any"):
        return {tool_choice: {}}
    else:
        return {"tool": {"name": tool_choice}}


def _extract_tool_calls(anthropic_content: List[dict]) -> List[ToolCall]:
    tool_calls = []
    for block in anthropic_content:
        if block["type"] == "tool_use":
            tool_calls.append(
                ToolCall(name=block["name"], args=block["input"], id=block["id"])
            )
    return tool_calls


def _snake_to_camel(text: str) -> str:
    split = text.split("_")
    return "".join(split[:1] + [s.title() for s in split[1:]])


def _camel_to_snake(text: str) -> str:
    pattern = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
    return pattern.sub("_", text).lower()


_T = TypeVar("_T")


def _camel_to_snake_keys(obj: _T) -> _T:
    if isinstance(obj, list):
        return cast(_T, [_camel_to_snake_keys(e) for e in obj])
    elif isinstance(obj, dict):
        return cast(
            _T, {_camel_to_snake(k): _camel_to_snake_keys(v) for k, v in obj.items()}
        )
    else:
        return obj


def _snake_to_camel_keys(obj: _T) -> _T:
    if isinstance(obj, list):
        return cast(_T, [_snake_to_camel_keys(e) for e in obj])
    elif isinstance(obj, dict):
        return cast(
            _T, {_snake_to_camel(k): _snake_to_camel_keys(v) for k, v in obj.items()}
        )
    else:
        return obj


def _drop_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        new = {k: _drop_none(v) for k, v in obj.items() if _drop_none(v) is not None}
        return new or None
    else:
        return obj
