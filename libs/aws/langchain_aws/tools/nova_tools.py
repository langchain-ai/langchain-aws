"""Nova system tools helpers.

This module provides helper classes for working with Amazon Nova
2.0's system tools (nova_grounding and nova_code_interpreter).

Amazon Nova 2.0 models support built-in system tools that execute
server-side. These tools enable web search (nova_grounding) and code
execution (nova_code_interpreter) capabilities.

IAM Permissions Required:
    System tools require the `bedrock:InvokeTool` IAM permission in addition
    to `bedrock:InvokeModel`.

Web Grounding (nova_grounding):
    Enables the model to search the web for current information.

    .. code-block:: python

        from langchain_aws import ChatBedrockConverse
        from langchain_aws.tools import NovaGroundingTool

        model = ChatBedrockConverse(model="amazon.nova-2-lite-v1:0")
        model_with_search = model.bind_tools([NovaGroundingTool()])

        response = model_with_search.invoke(
            "What are the latest developments in quantum computing?"
        )
        print(response.content)

    You can also use the string name directly:

    .. code-block:: python

        model_with_search = model.bind_tools(["nova_grounding"])

Code Interpreter (nova_code_interpreter):
    Enables the model to execute Python code in a sandboxed environment.

    .. code-block:: python

        from langchain_aws import ChatBedrockConverse
        from langchain_aws.tools import NovaCodeInterpreterTool

        model = ChatBedrockConverse(model="amazon.nova-2-lite-v1:0")
        model_with_code = model.bind_tools([NovaCodeInterpreterTool()])

        response = model_with_code.invoke(
            "Calculate the square root of 475878756857"
        )
        print(response.content)

Reasoning with Nova:
    Nova models support reasoning configuration similar to Claude's thinking
    feature, but use `reasoningConfig` instead of `thinking`.

    .. code-block:: python

        from langchain_aws import ChatBedrockConverse

        reasoning_config = {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": "low"  # or "medium", "high"
            }
        }

        model = ChatBedrockConverse(
            model="amazon.nova-2-lite-v1:0",
            max_tokens=10000,
            additional_model_request_fields=reasoning_config
        )

        response = model.invoke("Solve this logic puzzle: ...")
        # Access reasoning content via response.content_blocks

Combining System Tools and Reasoning:

    .. code-block:: python

        from botocore.config import Config
        from langchain_aws import ChatBedrockConverse
        from langchain_aws.tools import NovaGroundingTool

        # Increase timeout for reasoning and tool execution
        config = Config(
            connect_timeout=3600,  # 60 minutes
            read_timeout=3600,     # 60 minutes
            retries={'max_attempts': 1}
        )

        model = ChatBedrockConverse(
            model="amazon.nova-2-lite-v1:0",
            max_tokens=10000,
            config=config,
            additional_model_request_fields={
                "reasoningConfig": {
                    "type": "enabled",
                    "maxReasoningEffort": "medium"  # or "low", "high"
                }
            }
        )

        model_with_tools = model.bind_tools([NovaGroundingTool()])
        response = model_with_tools.invoke(
            "Research and explain the latest AI breakthroughs"
        )

Timeout Recommendations:
    - **Low reasoning effort**: Default timeout (30 seconds) is usually sufficient
    - **Medium reasoning effort**: Consider 300-600 seconds (5-10 minutes)
    - **High reasoning effort**: Consider 1800-3600 seconds (30-60 minutes)
    - **Code interpreter**: Consider 300-600 seconds for complex computations
    - **Web grounding**: Default timeout is usually sufficient
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
        from langchain_aws import ChatBedrockConverse
        from langchain_aws.tools import NovaGroundingTool

        model = ChatBedrockConverse(model="amazon.nova-2-lite-v1:0")
        model_with_search = model.bind_tools([NovaGroundingTool()])
        response = model_with_search.invoke("What's the latest news?")
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
        from langchain_aws import ChatBedrockConverse
        from langchain_aws.tools import NovaCodeInterpreterTool

        model = ChatBedrockConverse(model="amazon.nova-2-lite-v1:0")
        model_with_code = model.bind_tools([NovaCodeInterpreterTool()])
        response = model_with_code.invoke(
            "Calculate the square root of 475878756857"
        )
    """

    def __init__(self) -> None:
        """Initialize the Nova code interpreter tool."""
        super().__init__("nova_code_interpreter")
