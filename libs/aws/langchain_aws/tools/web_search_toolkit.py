"""AgentCore Web Search tool for LangChain.

Web search reaches the service either directly, through the Web Search API, or
through an AgentCore Gateway connector target. Which transport is used is decided
inside the AgentCore SDK client rather than here: with no gateway argument, the
SDK calls the Web Search API. Every argument on this toolkit is keyword-only and
optional, and only the arguments actually supplied are forwarded, so this package
works with SDK versions that predate an argument as long as it isn't used.
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
from pydantic import BaseModel, Field, field_validator

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

    @field_validator("include_domains", "exclude_domains", mode="before")
    @classmethod
    def _one_domain_as_a_list(cls, value: Any) -> Any:
        """Accept a single domain string, which models often send instead of a list.

        Args:
            value: The argument as the model sent it.

        Returns:
            The value, with a lone string wrapped in a list.
        """
        return [value] if isinstance(value, str) else value


class WebSearchQueryInput(BaseModel):
    """Input schema for the web search tool when filters are set by code only."""

    query: str = Field(
        description="What to search for, as a natural language query. "
        "200 characters or fewer."
    )


_TOOL_DESCRIPTION_QUERY_ONLY = """Search the web for current information and return \
source-attributed results.

Use this tool for:
- Questions about recent events, releases, prices or anything time-sensitive
- Facts that need a citable source
- Topics outside your training data

Each result carries a title, a URL, a publication date when the index reports
one, and an extract. Cite the URLs you used in your answer.

Prefer one broad query over several narrow ones."""


def _source(result: Any) -> Dict[str, Any]:
    """Describe one result as plain data, for a tool artifact."""
    published_at = getattr(result, "published_at", None)
    return {
        "title": result.title,
        "url": result.url,
        "published_date": result.published_date,
        "published_at": published_at.isoformat() if published_at else None,
        "text": result.text,
    }


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

    With no gateway argument, the search calls the Web Search API, and the ambient
    AWS credentials need `bedrock-agentcore:InvokeWebSearch` on
    `arn:<partition>:bedrock-agentcore:<region>:aws:tool/web-search.v1`. With a
    gateway argument, the search runs over that gateway's web search target: the
    credentials need `bedrock-agentcore:InvokeGateway` on the gateway, and the
    gateway's own service role needs `bedrock-agentcore:InvokeWebSearch`. Calls are
    authenticated with SigV4; there is no web search API key.

    Rules the model must not change go on the toolkit, not in the prompt. Filters
    set here (`exclude_domains`, `published_after` and the others) are carried by
    every search, and a model's own filter arguments can only narrow them. Set
    `model_filters=False` to show the model a query argument only, `max_searches`
    to cap the searches per toolkit, and `return_sources=True` to get the results
    back as data (a tool artifact) alongside the text the model reads.

    A failed search raises `ToolException`, so a retry wrapper sees it as a
    failure. To let the model read the message and correct its own query instead,
    set `handle_tool_error` on the tool.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_aws.tools import create_web_search_toolkit

        # A gateway ARN carries its own region. Pass region= alongside a
        # gateway_id, or to override the region in an ARN.
        toolkit, tools = create_web_search_toolkit(region="us-east-1")
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
        web_search_id: Optional[str] = None,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
        model_filters: bool = True,
        max_searches: Optional[int] = None,
        return_sources: bool = False,
    ) -> None:
        """Initialize the toolkit.

        At most one of `gateway_id`, `gateway_arn`, `gateway_endpoint` or `client`
        says where the search goes. With none of them, the search calls the Web
        Search API.

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
            web_search_id: Web search tool the Web Search API invokes. Leave unset
                for the SDK default.
            include_domains: Every search returns only these domains.
            exclude_domains: Every search drops these domains.
            published_after: Every search returns only pages published at or after
                this time, as ISO-8601 UTC.
            published_before: Every search returns only pages published before this
                time, as ISO-8601 UTC.
            model_filters: Whether the model can pass its own result count, domain
                and date filters. When `False`, the tool takes a query only.
            max_searches: Searches this toolkit runs before its tool answers with a
                limit message instead of searching. Unset means no limit.
            return_sources: Whether the tool also returns its results as data, a
                list of dicts in the tool message's `artifact`.

        Raises:
            ValueError: If more than one destination is identified, or if filters
                are given together with `client`.
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
        fixed = {
            "include_domains": include_domains,
            "exclude_domains": exclude_domains,
            "published_after": published_after,
            "published_before": published_before,
        }
        if client is not None and (
            web_search_id is not None or any(v is not None for v in fixed.values())
        ):
            msg = (
                "Set web_search_id and filters on the client you pass in, not on "
                "the toolkit."
            )
            raise ValueError(msg)
        if max_searches is not None and max_searches < 1:
            msg = f"max_searches must be 1 or more, got {max_searches}"
            raise ValueError(msg)

        if region is None and gateway_arn is None:
            region = DEFAULT_REGION

        self.region = region
        self._gateway_id = gateway_id
        self._gateway_arn = gateway_arn
        self._gateway_endpoint = gateway_endpoint
        self._target_name = target_name
        self._tool_name = tool_name
        self._boto3_session = boto3_session
        self._web_search_id = web_search_id
        self._fixed = fixed
        self._model_filters = model_filters
        self._max_searches = max_searches
        self._return_sources = return_sources
        self._searches = 0
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
            ("web_search_id", self._web_search_id),
            *self._fixed.items(),
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
    ) -> Any:
        """Run one search and render the results.

        Args:
            query: What to search for.
            max_results: How many results to return.
            include_domains: Restrict results to these domains.
            exclude_domains: Drop results from these domains.
            published_after: Earliest publication date, ISO-8601 UTC.
            published_before: Latest publication date, ISO-8601 UTC.

        Returns:
            The results as text, or, with `return_sources`, a tuple of the text and
            a list of the results as dicts.

        Raises:
            RuntimeError: If the toolkit has been closed.
            ToolException: If the search itself failed.
        """
        if self._closed:
            msg = "This web search toolkit has been closed."
            raise RuntimeError(msg)
        if not self._take_search():
            text = (
                f"Search limit of {self._max_searches} reached. "
                "Answer with what you have."
            )
            return (text, []) if self._return_sources else text
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
        text = _format_response(response)
        if self._return_sources:
            return text, [_source(result) for result in response.results]
        return text

    def _take_search(self) -> bool:
        """Count one search against the budget.

        Returns:
            Whether the search may run.
        """
        with self._lock:
            if self._max_searches is not None and self._searches >= self._max_searches:
                return False
            self._searches += 1
            return True

    async def _asearch(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
    ) -> Any:
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
            Whatever `_search` returns.

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
            args_schema=WebSearchInput if self._model_filters else WebSearchQueryInput,
            description=(
                _TOOL_DESCRIPTION
                if self._model_filters
                else _TOOL_DESCRIPTION_QUERY_ONLY
            ),
            response_format=(
                "content_and_artifact" if self._return_sources else "content"
            ),
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
    web_search_id: Optional[str] = None,
    include_domains: Optional[Sequence[str]] = None,
    exclude_domains: Optional[Sequence[str]] = None,
    published_after: Optional[str] = None,
    published_before: Optional[str] = None,
    model_filters: bool = True,
    max_searches: Optional[int] = None,
    return_sources: bool = False,
) -> Tuple[WebSearchToolkit, List[BaseTool]]:
    """Create an AgentCore web search toolkit and its tools.

    This factory is synchronous, unlike the browser and code interpreter
    factories in this package, because a search needs no session to be set up
    first. With no gateway argument, the tool calls the Web Search API.

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
        web_search_id: Web search tool the Web Search API invokes.
        include_domains: Every search returns only these domains.
        exclude_domains: Every search drops these domains.
        published_after: Every search returns only pages published at or after
            this time, as ISO-8601 UTC.
        published_before: Every search returns only pages published before this
            time, as ISO-8601 UTC.
        model_filters: Whether the model can pass its own filters. When `False`,
            the tool takes a query only.
        max_searches: Searches the tool runs before answering with a limit message.
        return_sources: Whether the tool also returns its results as data, in the
            tool message's `artifact`.

    Returns:
        Tuple of (toolkit, tools).

    Example:
        >>> toolkit, tools = create_web_search_toolkit(
        ...     region="us-east-1",
        ...     exclude_domains=["example.com"],
        ...     published_after="2026-11-01T00:00:00Z",
        ...     model_filters=False,
        ...     max_searches=4,
        ...     return_sources=True,
        ... )
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
        web_search_id=web_search_id,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        published_after=published_after,
        published_before=published_before,
        model_filters=model_filters,
        max_searches=max_searches,
        return_sources=return_sources,
    )
    return toolkit, toolkit.get_tools()
