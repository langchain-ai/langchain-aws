"""Toolkit for navigating web with AWS browser with thread support."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from bedrock_agentcore.tools.config import (
    BrowserExtension,
    ProfileConfiguration,
    ProxyConfiguration,
)
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
        from langchain.agents import create_agent
        from langchain_aws.tools import create_browser_toolkit

        async def main():
            # Create toolkit with proxy and extensions
            toolkit, browser_tools = create_browser_toolkit(
                region="us-west-2",
                proxy_configuration={
                    "proxies": [{
                        "externalProxy": {
                            "server": "proxy.example.com",
                            "port": 8080,
                            "credentials": {
                                "basicAuth": {
                                    "secretArn": "arn:aws:secretsmanager:..."
                                }
                            },
                        }
                    }],
                },
                extensions=[{
                    "location": {
                        "s3": {"bucket": "my-bucket", "prefix": "ext.zip"}
                    }
                }],
                profile_configuration={
                    "profileIdentifier": "my-profile-id"
                },
            )

            # Create a ReAct agent using the browser tools
            agent = create_agent(
                "bedrock_converse:us.anthropic.claude-haiku-4-5-20251001-v1:0",
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
                "Navigate to https://www.example.com and tell me "
                "the main heading on the page.",
                config=config
            )

            # Clean up browser resources when done
            await toolkit.cleanup()

            return result

        # Run the example
        asyncio.run(main())
        ```

    """

    def __init__(
        self,
        region: str = "us-west-2",
        proxy_configuration: Optional[Union[ProxyConfiguration, Dict[str, Any]]] = None,
        extensions: Optional[Sequence[Union[BrowserExtension, Dict[str, Any]]]] = None,
        profile_configuration: Optional[
            Union[ProfileConfiguration, Dict[str, Any]]
        ] = None,
    ):
        """Initialize the toolkit.

        Args:
            region: AWS region for the browser client.
            proxy_configuration: Proxy routing config. Accepts a
                ``ProxyConfiguration`` dataclass or equivalent dict with
                ``proxies`` and optional ``bypass`` keys.
            extensions: Browser extensions to load. Accepts a list of
                ``BrowserExtension`` dataclasses or equivalent dicts with
                an S3 ``location``.
            profile_configuration: Profile for persisting browser state
                across sessions. Accepts a ``ProfileConfiguration``
                dataclass or dict with a ``profileIdentifier`` key.

        """
        self.region = region
        self.session_manager = BrowserSessionManager(
            region=region,
            proxy_configuration=proxy_configuration,
            extensions=extensions,
            profile_configuration=profile_configuration,
        )
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
    proxy_configuration: Optional[Union[ProxyConfiguration, Dict[str, Any]]] = None,
    extensions: Optional[Sequence[Union[BrowserExtension, Dict[str, Any]]]] = None,
    profile_configuration: Optional[Union[ProfileConfiguration, Dict[str, Any]]] = None,
) -> Tuple[BrowserToolkit, List[BaseTool]]:
    """Create a BrowserToolkit with thread support.

    Args:
        region: AWS region for browser client.
        proxy_configuration: Proxy routing config. Accepts a
            ``ProxyConfiguration`` dataclass or equivalent dict.
        extensions: Browser extensions to load. Accepts a list of
            ``BrowserExtension`` dataclasses or equivalent dicts.
        profile_configuration: Profile for persisting browser state.
            Accepts a ``ProfileConfiguration`` dataclass or dict.

    Returns:
        Tuple of (toolkit, tools).

    """
    toolkit = BrowserToolkit(
        region=region,
        proxy_configuration=proxy_configuration,
        extensions=extensions,
        profile_configuration=profile_configuration,
    )
    tools = toolkit.get_tools()
    return toolkit, tools
