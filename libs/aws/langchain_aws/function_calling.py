"""Methods for creating function specs in the style of Bedrock Functions 
for supported model providers"""

import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Type,
    Union,
)

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from typing_extensions import TypedDict

PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}

SYSTEM_PROMPT_FORMAT = """In this environment you have access to a set of tools you can use to answer the user's question.

You may call them like this:
<function_calls>
<invoke>
<tool_name>$TOOL_NAME</tool_name>
<parameters>
<$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
...
</parameters>
</invoke>
</function_calls>

Here are the tools available:
<tools>
{formatted_tools}
</tools>"""  # noqa: E501

TOOL_FORMAT = """<tool_description>
<tool_name>{tool_name}</tool_name>
<description>{tool_description}</description>
<parameters>
{formatted_parameters}
</parameters>
</tool_description>"""

TOOL_PARAMETER_FORMAT = """<parameter>
<name>{parameter_name}</name>
<type>{parameter_type}</type>
<description>{parameter_description}</description>
</parameter>"""


class AnthropicTool(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]


def _get_type(parameter: Dict[str, Any]) -> str:
    if "type" in parameter:
        return parameter["type"]
    if "anyOf" in parameter:
        return json.dumps({"anyOf": parameter["anyOf"]})
    if "allOf" in parameter:
        return json.dumps({"allOf": parameter["allOf"]})
    return json.dumps(parameter)


def get_system_message(tools: List[AnthropicTool]) -> str:
    tools_data: List[Dict] = [
        {
            "tool_name": tool["name"],
            "tool_description": tool["description"],
            "formatted_parameters": "\n".join(
                [
                    TOOL_PARAMETER_FORMAT.format(
                        parameter_name=name,
                        parameter_type=_get_type(parameter),
                        parameter_description=parameter.get("description"),
                    )
                    for name, parameter in tool["input_schema"]["properties"].items()
                ]
            ),
        }
        for tool in tools
    ]
    tools_formatted = "\n".join(
        [
            TOOL_FORMAT.format(
                tool_name=tool["tool_name"],
                tool_description=tool["tool_description"],
                formatted_parameters=tool["formatted_parameters"],
            )
            for tool in tools_data
        ]
    )
    return SYSTEM_PROMPT_FORMAT.format(formatted_tools=tools_formatted)


class FunctionDescription(TypedDict):
    """Representation of a callable function to send to an LLM."""

    name: str
    """The name of the function."""
    description: str
    """A description of the function."""
    parameters: dict
    """The parameters of the function."""


class ToolDescription(TypedDict):
    """Representation of a callable function to the OpenAI API."""

    type: Literal["function"]
    function: FunctionDescription


def convert_to_anthropic_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> AnthropicTool:
    # already in Anthropic tool format
    if isinstance(tool, dict) and all(
        k in tool for k in ("name", "description", "input_schema")
    ):
        return AnthropicTool(tool)  # type: ignore
    else:
        formatted = convert_to_openai_tool(tool)["function"]
        return AnthropicTool(
            name=formatted["name"],
            description=formatted["description"],
            input_schema=formatted["parameters"],
        )
