"""AgentCore Web Search tool for LangChain.

Web search reaches the service through an AgentCore Gateway connector target
today, and a direct API is expected later. Which transport is used is decided
inside the AgentCore SDK client rather than here, so every gateway argument on
this toolkit is keyword-only and optional, and only the arguments actually
supplied are forwarded. A later transport that needs none of them works through
the same call, without a signature change in this package.
"""

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bedrock_agentcore.tools.web_search_client import (
    WebSearchClient,
    WebSearchResponse,
)
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

#: Web search is offered in a subset of regions, so this default is not the one
#: the other AgentCore toolkits in this package use. It is applied only when no
#: region and no gateway ARN were given, because the SDK prefers an explicit
#: region over the one carried in the ARN.
DEFAULT_REGION = "us-east-1"

_TOOL_NAME = "web_search"

_TOOL_DESCRIPTION = """Search the web for current information and return \
source-attributed results.

Use this tool for:
- Questions about recent events, releases, prices or anything time-sensitive
- Facts that need a citable source
- Topics outside your training data

Each result carries a title, a URL, a publication date when the index reports
one, and an extract. Cite the URLs you used in your answer.

Prefer one broad query over several narrow ones. Use include_domains to restrict
a search to sources you trust, such as official documentation."""


class WebSearchInput(BaseModel):
    """Input schema for the web search tool."""

    query: str = Field(
        description="What to search for, as a natural language query. "
        "200 characters or fewer."
    )
    max_results: Optional[int] = Field(
        default=None,
        description="How many results to return, between 1 and 25. "
        "Leave unset for the service default.",
    )
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="Only return results from these domains, e.g. "
        '["docs.aws.amazon.com"]. A root domain also matches its subdomains. '
        "Can only narrow the search, never widen it.",
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None,
        description="Drop results from these domains.",
    )
    published_after: Optional[str] = Field(
        default=None,
        description="Only return pages published on or after this date, "
        "as ISO-8601 UTC, e.g. 2026-01-01T00:00:00Z.",
    )
    published_before: Optional[str] = Field(
        default=None,
        description="Only return pages published on or before this date, "
        "as ISO-8601 UTC.",
    )


def _format_response(response: WebSearchResponse) -> str:
    """Render search results as text an LLM can cite from.

    Args:
        response: The results of one search.

    Returns:
        One numbered block per result, or a plain sentence when there were none.
    """
    if not response.results:
        return (
            "No results. The query may be too narrow, or a domain or date "
            "filter may have excluded everything."
        )

    blocks: List[str] = []
    for position, result in enumerate(response.results, start=1):
        lines = [f"{position}. {result.title or 'Untitled'}"]
        if result.url:
            lines.append(f"   URL: {result.url}")
        if result.published_date:
            lines.append(f"   Published: {result.published_date}")
        if result.text:
            lines.append(f"   {result.text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


class WebSearchToolkit:
    """Toolkit exposing AgentCore Web Search as a LangChain tool.

    The search runs over an AgentCore Gateway target that has the web search
    connector attached. Calls are authenticated with SigV4 from the ambient AWS
    credentials; there is no web search API key. Those credentials need
    `bedrock-agentcore:InvokeGateway` on the gateway, and the gateway's own
    service role needs `bedrock-agentcore:InvokeWebSearch` on the connector.

    A failed search raises `ToolException`, so a retry wrapper sees it as a
    failure. To let the model read the message and correct its own query instead,
    set `handle_tool_error` on the tool.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_aws.tools import create_web_search_toolkit

        # A gateway ARN carries its own region. Pass region= alongside a
        # gateway_id, or to override the region in an ARN.
        toolkit, tools = create_web_search_toolkit(
            region="us-east-1",
            gateway_id="my-gateway-abc123",
        )
        try:
            agent = create_agent(model, tools=tools)
            result = agent.invoke({"messages": [("user", "latest boto3 release")]})
        finally:
            toolkit.close()
        ```

        To have the model see the failure and retry, rather than the run ending:

        ```python
        for tool in tools:
            tool.handle_tool_error = True
        ```
    """

    def __init__(
        self,
        region: Optional[str] = None,
        *,
        gateway_id: Optional[str] = None,
        gateway_arn: Optional[str] = None,
        gateway_endpoint: Optional[str] = None,
        target_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        boto3_session: Optional[Any] = None,
        client: Optional[WebSearchClient] = None,
    ) -> None:
        """Initialize the toolkit.

        Exactly one of `gateway_id`, `gateway_arn`, `gateway_endpoint` or
        `client` says where the search goes.

        Args:
            region: AWS region to call. Defaults to `DEFAULT_REGION`, except when
                `gateway_arn` is given, in which case the ARN's region is used.
            gateway_id: ID of a gateway carrying a web search connector target.
            gateway_arn: ARN of that gateway. The ID and the region are read from it.
            gateway_endpoint: A gateway MCP endpoint URL, if one is already known.
            target_name: Name the connector target was created under. Supplying it
                avoids a tool discovery round trip on the first search.
            tool_name: Fully qualified name of the gateway tool, if already known.
            boto3_session: Session to take credentials from.
            client: An SDK client to use as is. The caller keeps ownership of it,
                so `close` leaves it open.

        Raises:
            ValueError: If no destination is identified, or more than one is.
        """
        if client is not None and any(
            value is not None for value in (gateway_id, gateway_arn, gateway_endpoint)
        ):
            msg = (
                "Pass either client or one of gateway_id, gateway_arn or "
                "gateway_endpoint, not both."
            )
            raise ValueError(msg)

        # The SDK prefers an explicit region over the one in a gateway ARN, so
        # filling the default in here would silently override the ARN's region.
        # Leaving it unset is what lets the ARN decide.
        if region is None and gateway_arn is None:
            region = DEFAULT_REGION

        self.region = region
        self._gateway_id = gateway_id
        self._gateway_arn = gateway_arn
        self._gateway_endpoint = gateway_endpoint
        self._target_name = target_name
        self._tool_name = tool_name
        self._boto3_session = boto3_session
        self._owns_client = client is None
        self._client = client if client is not None else self._create_client()
        self._closed = False
        self._lock = threading.Lock()
        self.tools: List[BaseTool] = self._create_tools()

    def _create_client(self) -> WebSearchClient:
        """Build the SDK client, forwarding only the arguments that were given.

        Returns:
            A client pointed at whichever destination was identified.
        """
        # Passing `gateway_id=None` explicitly would tie this package to the
        # gateway transport. Forwarding only what was supplied means a future
        # transport needing no gateway argument works through this same path.
        kwargs: Dict[str, Any] = {"integration_source": "langchain"}
        optional = (
            ("region", self.region),
            ("gateway_id", self._gateway_id),
            ("gateway_arn", self._gateway_arn),
            ("gateway_endpoint", self._gateway_endpoint),
            ("target_name", self._target_name),
            ("tool_name", self._tool_name),
            ("boto3_session", self._boto3_session),
        )
        for name, value in optional:
            if value is not None:
                kwargs[name] = value
        return WebSearchClient(**kwargs)

    def get_tools(self) -> List[BaseTool]:
        """Get the web search tools.

        Returns:
            The tools this toolkit exposes.
        """
        return self.tools

    def _search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
    ) -> str:
        """Run one search and render the results.

        Args:
            query: What to search for.
            max_results: How many results to return.
            include_domains: Restrict results to these domains.
            exclude_domains: Drop results from these domains.
            published_after: Earliest publication date, ISO-8601 UTC.
            published_before: Latest publication date, ISO-8601 UTC.

        Returns:
            The results as text.

        Raises:
            RuntimeError: If the toolkit has been closed.
            ToolException: If the search itself failed.
        """
        if self._closed:
            msg = "This web search toolkit has been closed."
            raise RuntimeError(msg)
        try:
            response = self._client.search(
                query,
                max_results=max_results,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                published_after=published_after,
                published_before=published_before,
            )
        except Exception as exc:
            # Raised rather than returned as text so that a failure is recorded
            # as one, which is what `with_retry` and any callback handler need to
            # see. A caller who would rather the model read the message and fix
            # its own query sets `handle_tool_error` on the tool.
            logger.warning("Web search failed: %s", exc)
            msg = f"Web search failed: {exc}"
            raise ToolException(msg) from exc
        return _format_response(response)

    async def _asearch(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
    ) -> str:
        """Run `_search` off the event loop.

        The SDK client is synchronous, so awaiting it directly would block every
        other task in an async agent for the length of the search.

        Args:
            query: What to search for.
            max_results: How many results to return.
            include_domains: Restrict results to these domains.
            exclude_domains: Drop results from these domains.
            published_after: Earliest publication date, ISO-8601 UTC.
            published_before: Latest publication date, ISO-8601 UTC.

        Returns:
            The results as text.

        Raises:
            RuntimeError: If the toolkit has been closed.
            ToolException: If the search itself failed.
        """
        return await asyncio.to_thread(
            self._search,
            query,
            max_results,
            include_domains,
            exclude_domains,
            published_after,
            published_before,
        )

    def _create_tools(self) -> List[BaseTool]:
        """Create the LangChain tools for web search.

        Returns:
            A list holding the single web search tool.
        """
        search_tool = StructuredTool.from_function(
            name=_TOOL_NAME,
            func=self._search,
            coroutine=self._asearch,
            args_schema=WebSearchInput,
            description=_TOOL_DESCRIPTION,
        )
        return [search_tool]

    def close(self) -> None:
        """Release the underlying client, if this toolkit created it."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_client:
                self._client.close()

    def __enter__(self) -> "WebSearchToolkit":
        """Enter the context manager.

        Returns:
            This toolkit.
        """
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the toolkit on exit."""
        self.close()


def create_web_search_toolkit(
    region: Optional[str] = None,
    *,
    gateway_id: Optional[str] = None,
    gateway_arn: Optional[str] = None,
    gateway_endpoint: Optional[str] = None,
    target_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    boto3_session: Optional[Any] = None,
    client: Optional[WebSearchClient] = None,
) -> Tuple[WebSearchToolkit, List[BaseTool]]:
    """Create an AgentCore web search toolkit and its tools.

    This factory is synchronous, unlike the browser and code interpreter
    factories in this package, because a search needs no session to be set up
    first.

    Args:
        region: AWS region to call. Defaults to `DEFAULT_REGION`, except when
            `gateway_arn` is given, in which case the ARN's region is used.
        gateway_id: ID of a gateway carrying a web search connector target.
        gateway_arn: ARN of that gateway. The ID and the region are read from it.
        gateway_endpoint: A gateway MCP endpoint URL, if one is already known.
        target_name: Name the connector target was created under. Supplying it
            avoids a tool discovery round trip on the first search.
        tool_name: Fully qualified name of the gateway tool, if already known.
        boto3_session: Session to take credentials from.
        client: An SDK client to use as is.

    Returns:
        Tuple of (toolkit, tools).

    Example:
        >>> toolkit, tools = create_web_search_toolkit(gateway_id="my-gw-abc123")
        >>> toolkit.close()  # When done
    """
    toolkit = WebSearchToolkit(
        region=region,
        gateway_id=gateway_id,
        gateway_arn=gateway_arn,
        gateway_endpoint=gateway_endpoint,
        target_name=target_name,
        tool_name=tool_name,
        boto3_session=boto3_session,
        client=client,
    )
    return toolkit, toolkit.get_tools()
