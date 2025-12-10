"""Nova system tools helpers.

This module provides helper classes for working with Amazon Nova
2.0's system tools (nova_grounding and nova_code_interpreter).
"""

from typing import Any, Dict


class NovaSystemTool:
    """Base class for Nova system tools.

    System tools are built-in tools provided by Nova models that execute
    server-side. Unlike custom tools, system tools don't require client-side
    implementation.

    Args:
        name: The name of the system tool (e.g., "nova_grounding")

    Attributes:
        name: The system tool name
        type: Always "system_tool" to distinguish from custom tools
    """

    def __init__(self, name: str) -> None:
        """Initialize a Nova system tool.

        Args:
            name: The name of the system tool
        """
        self.name = name
        self.type = "system_tool"

    def to_bedrock_format(self) -> Dict[str, Any]:
        """Convert to Bedrock systemTool format.

        Returns:
            Dictionary in the format expected by the Bedrock Converse API:
            {"systemTool": {"name": "<tool_name>"}}
        """
        return {"systemTool": {"name": self.name}}


class NovaGroundingTool(NovaSystemTool):
    """Helper for Nova's web grounding system tool.

    The nova_grounding tool enables the model to search the web for current
    information. The model autonomously decides when to invoke the tool and
    processes the results.

    Note:
        Requires bedrock:InvokeTool IAM permission.

    Example:
        >>> from langchain_aws import ChatBedrockConverse
        >>> from langchain_aws.chat_models import NovaGroundingTool
        >>>
        >>> model = ChatBedrockConverse(model="amazon.nova-2-lite-v1:0")
        >>> model_with_search = model.bind_tools([NovaGroundingTool()])
        >>> response = model_with_search.invoke("What's the latest news?")
    """

    def __init__(self) -> None:
        """Initialize the Nova grounding tool."""
        super().__init__("nova_grounding")


class NovaCodeInterpreterTool(NovaSystemTool):
    """Helper for Nova's code interpreter system tool.

    The nova_code_interpreter tool enables the model to execute Python code
    in a sandboxed environment. Useful for calculations, data analysis, and
    other computational tasks.

    Note:
        Requires bedrock:InvokeTool IAM permission.
        Consider increasing timeout for complex code execution.

    Example:
        >>> from langchain_aws import ChatBedrockConverse
        >>> from langchain_aws.chat_models import NovaCodeInterpreterTool
        >>>
        >>> model = ChatBedrockConverse(model="amazon.nova-2-lite-v1:0")
        >>> model_with_code = model.bind_tools([NovaCodeInterpreterTool()])
        >>> response = model_with_code.invoke(
        ...     "Calculate the square root of 475878756857"
        ... )
    """

    def __init__(self) -> None:
        """Initialize the Nova code interpreter tool."""
        super().__init__("nova_code_interpreter")
