"""Toolkit for navigating web with AWS browser with thread support."""

import logging
from typing import Dict, List, Tuple

from langchain_core.tools import BaseTool

from .browser_session_manager import BrowserSessionManager
from .browser_tools import create_thread_aware_tools

logger = logging.getLogger(__name__)


class BrowserToolkit:
    """Toolkit for navigating web with AWS browser with thread support.

    This toolkit provides a set of tools for working with a remote browser
    and supports multiple threads by maintaining separate browser sessions
    for each thread ID. Browsers are created lazily only when needed.

    Example:
        ```python

        import asyncio
        from langgraph.prebuilt import create_react_agent
        from langchain_aws.tools import create_browser_toolkit

        async def main():
            # Create and setup the browser toolkit
            toolkit, browser_tools = create_browser_toolkit(region="us-west-2")

            # Create a ReAct agent using the browser tools
            agent = create_react_agent(
                model="bedrock_converse:us.anthropic.claude-3-5-haiku-20241022-v1:0",
                tools=browser_tools
            )

            # Create runnable config with thread ID
            config = {
                "configurable": {
                    "thread_id": "session123"
                }
            }

            # Invoke the agent with a specific task using thread ID
            result = await agent.ainvoke(
                "Navigate to https://www.example.com and tell me the main heading on the page.",
                config=config
            )

            # Clean up browser resources when done
            await toolkit.cleanup()

            return result

        # Run the example
        asyncio.run(main())
        ```
    """

    def __init__(self, region: str = "us-west-2"):
        """
        Initialize the toolkit

        Args:
            region: AWS region for the browser client
        """
        self.region = region
        self.session_manager = BrowserSessionManager(region=region)
        self.tools: List[BaseTool] = []
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Initialize tools without creating any browsers."""
        # Create thread-aware tools with the session manager
        # Browsers will be created lazily when tools are used
        tools_dict = create_thread_aware_tools(self.session_manager)
        self.tools = list(tools_dict.values())

    def get_tools(self) -> List[BaseTool]:
        """
        Get the list of thread-aware browser tools

        Returns:
            List of LangChain tools
        """
        return self.tools

    def get_tools_by_name(self) -> Dict[str, BaseTool]:
        """
        Get a dictionary of tools mapped by their names

        Returns:
            Dictionary of {tool_name: tool}
        """
        return {tool.name: tool for tool in self.tools}

    async def cleanup(self) -> None:
        """Clean up all browser sessions"""
        await self.session_manager.close_all_browsers()
        logger.info("All browser sessions cleaned up")


def create_browser_toolkit(
    region: str = "us-west-2",
) -> Tuple[BrowserToolkit, List[BaseTool]]:
    """
    Create a BrowserToolkit with thread support

    Args:
        region: AWS region for browser client

    Returns:
        Tuple of (toolkit, tools)
    """
    toolkit = BrowserToolkit(region=region)
    tools = toolkit.get_tools()
    return toolkit, tools