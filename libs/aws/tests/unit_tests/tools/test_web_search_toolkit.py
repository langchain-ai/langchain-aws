"""Unit tests for the AgentCore web search toolkit."""

import pytest

# Web search arrived in bedrock-agentcore 1.23.0. Skip rather than fail on an
# older install of the `tools` extra, matching the other toolkit test modules.
pytest.importorskip(
    "bedrock_agentcore.tools.web_search_client",
    reason="Requires langchain-aws[tools] with bedrock-agentcore>=1.23.0",
)

from typing import Any, List, Optional  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from bedrock_agentcore.tools.web_search_client import (  # noqa: E402
    WebSearchResponse,
    WebSearchResult,
)
from langchain_core.tools import BaseTool, ToolException  # noqa: E402

from langchain_aws.tools.web_search_toolkit import (  # noqa: E402
    DEFAULT_REGION,
    WebSearchToolkit,
    _format_response,
    create_web_search_toolkit,
)

GATEWAY_ID = "my-gateway-abc123"


def _response(*results: WebSearchResult) -> WebSearchResponse:
    """Build a response carrying the given results."""
    return WebSearchResponse(results=list(results), search_id="search-1")


def _make_toolkit(
    search_return: Optional[Any] = None,
    search_side_effect: Optional[Any] = None,
    **kwargs: Any,
) -> tuple[WebSearchToolkit, MagicMock]:
    """Build a toolkit whose SDK client is a mock, and return both."""
    client = MagicMock()
    if search_side_effect is not None:
        client.search.side_effect = search_side_effect
    else:
        client.search.return_value = (
            search_return if search_return is not None else _response()
        )
    toolkit = WebSearchToolkit(client=client, **kwargs)
    return toolkit, client


class TestCreateWebSearchToolkit:
    """Tests for the factory function."""

    def test_returns_toolkit_and_one_tool(self) -> None:
        """The factory returns a toolkit plus the tools it exposes."""
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            toolkit, tools = create_web_search_toolkit(gateway_id=GATEWAY_ID)

        assert isinstance(toolkit, WebSearchToolkit)
        assert isinstance(tools, list)
        assert len(tools) == 1
        assert all(isinstance(tool, BaseTool) for tool in tools)
        assert tools[0].name == "web_search"
        assert client_cls.call_count == 1

    def test_tools_match_get_tools(self) -> None:
        """The tools returned by the factory are the toolkit's own."""
        with patch("langchain_aws.tools.web_search_toolkit.WebSearchClient"):
            toolkit, tools = create_web_search_toolkit(gateway_id=GATEWAY_ID)

        assert tools == toolkit.get_tools()

    def test_defaults_to_a_region_web_search_is_offered_in(self) -> None:
        """The default region is one the connector is actually available in."""
        assert DEFAULT_REGION == "us-east-1"
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            toolkit, _ = create_web_search_toolkit(gateway_id=GATEWAY_ID)

        assert toolkit.region == "us-east-1"
        assert client_cls.call_args.kwargs["region"] == "us-east-1"


class TestClientConstruction:
    """Tests for how the toolkit builds the SDK client."""

    def test_reports_langchain_as_the_integration_source(self) -> None:
        """Traffic from this package is attributable to LangChain."""
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            WebSearchToolkit(gateway_id=GATEWAY_ID)

        assert client_cls.call_args.kwargs["integration_source"] == "langchain"

    def test_forwards_only_the_arguments_that_were_given(self) -> None:
        """Absent arguments are omitted rather than passed as None.

        This is what lets a later transport that needs no gateway argument work
        through this same call. Passing `gateway_id=None` explicitly would tie
        the package to the gateway transport.
        """
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            WebSearchToolkit(gateway_id=GATEWAY_ID)

        kwargs = client_cls.call_args.kwargs
        assert kwargs["gateway_id"] == GATEWAY_ID
        assert set(kwargs) == {"region", "integration_source", "gateway_id"}

    def test_forwards_no_destination_when_none_was_given(self) -> None:
        """With no gateway argument, nothing gateway-shaped reaches the client.

        Today the SDK raises for want of a destination. After a direct API exists
        the same call resolves to that transport, with no change here.
        """
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            WebSearchToolkit()

        assert set(client_cls.call_args.kwargs) == {"region", "integration_source"}

    def test_forwards_every_optional_argument_when_supplied(self) -> None:
        """Each optional argument reaches the client under its own name."""
        session = object()
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            WebSearchToolkit(
                region="eu-west-1",
                gateway_arn=(
                    "arn:aws:bedrock-agentcore:eu-west-1:123456789012:gateway/gw-1"
                ),
                target_name="amazon-web-search",
                tool_name="amazon-web-search___WebSearch",
                boto3_session=session,
            )

        kwargs = client_cls.call_args.kwargs
        assert kwargs["region"] == "eu-west-1"
        assert kwargs["target_name"] == "amazon-web-search"
        assert kwargs["tool_name"] == "amazon-web-search___WebSearch"
        assert kwargs["boto3_session"] is session
        assert "gateway_arn" in kwargs
        assert "gateway_id" not in kwargs

    def test_leaves_region_unset_when_a_gateway_arn_carries_one(self) -> None:
        """A gateway ARN's region is not overridden by this package's default.

        The SDK prefers an explicit region over the one in the ARN, so passing
        the default here would send every ARN-addressed gateway to us-east-1.
        """
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            toolkit = WebSearchToolkit(
                gateway_arn=(
                    "arn:aws:bedrock-agentcore:eu-west-1:123456789012:gateway/gw-1"
                ),
            )

        assert toolkit.region is None
        assert "region" not in client_cls.call_args.kwargs

    def test_an_explicit_region_still_wins_over_a_gateway_arn(self) -> None:
        """Naming a region overrides the ARN, matching the SDK's own precedence."""
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            WebSearchToolkit(
                region="ap-northeast-1",
                gateway_arn=(
                    "arn:aws:bedrock-agentcore:eu-west-1:123456789012:gateway/gw-1"
                ),
            )

        assert client_cls.call_args.kwargs["region"] == "ap-northeast-1"

    def test_uses_a_supplied_client_as_is(self) -> None:
        """A caller-supplied client is used without building another."""
        client = MagicMock()
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            toolkit = WebSearchToolkit(client=client)

        assert toolkit._client is client
        client_cls.assert_not_called()

    def test_rejects_a_client_together_with_a_gateway(self) -> None:
        """Supplying both would leave the gateway argument silently ignored."""
        with pytest.raises(ValueError, match="not both"):
            WebSearchToolkit(client=MagicMock(), gateway_id=GATEWAY_ID)


class TestSearchTool:
    """Tests for invoking the tool."""

    def test_passes_every_filter_through_to_the_client(self) -> None:
        """Filters set by the model reach the SDK unchanged."""
        toolkit, client = _make_toolkit(
            _response(WebSearchResult(text="body", url="https://example.com"))
        )
        tool = toolkit.get_tools()[0]

        tool.invoke(
            {
                "query": "boto3 release notes",
                "max_results": 5,
                "include_domains": ["aws.amazon.com"],
                "exclude_domains": ["internal.example.com"],
                "published_after": "2026-01-01T00:00:00Z",
                "published_before": "2026-09-01T00:00:00Z",
            }
        )

        assert client.search.call_args.args == ("boto3 release notes",)
        assert client.search.call_args.kwargs == {
            "max_results": 5,
            "include_domains": ["aws.amazon.com"],
            "exclude_domains": ["internal.example.com"],
            "published_after": "2026-01-01T00:00:00Z",
            "published_before": "2026-09-01T00:00:00Z",
        }

    def test_omits_filters_the_model_did_not_set(self) -> None:
        """Unset filters arrive as None, which the SDK drops."""
        toolkit, client = _make_toolkit()
        tool = toolkit.get_tools()[0]

        tool.invoke({"query": "who maintains urllib3"})

        assert client.search.call_args.kwargs == {
            "max_results": None,
            "include_domains": None,
            "exclude_domains": None,
            "published_after": None,
            "published_before": None,
        }

    def test_renders_results_for_the_model(self) -> None:
        """The tool returns citable text rather than an object."""
        toolkit, _ = _make_toolkit(
            _response(
                WebSearchResult(
                    text="urllib3 is maintained by the urllib3 team.",
                    url="https://urllib3.readthedocs.io/",
                    title="urllib3 docs",
                    published_date="2026-08-01",
                )
            )
        )
        tool = toolkit.get_tools()[0]

        output = tool.invoke({"query": "urllib3"})

        assert "1. urllib3 docs" in output
        assert "URL: https://urllib3.readthedocs.io/" in output
        assert "Published: 2026-08-01" in output
        assert "urllib3 is maintained by the urllib3 team." in output

    def test_raises_tool_exception_when_the_search_fails(self) -> None:
        """A failure is raised, so `with_retry` and callbacks see it as one."""
        toolkit, _ = _make_toolkit(
            search_side_effect=ValueError("query must be 200 characters or fewer")
        )
        tool = toolkit.get_tools()[0]

        with pytest.raises(ToolException, match="200 characters or fewer") as caught:
            tool.invoke({"query": "x" * 300})

        assert isinstance(caught.value.__cause__, ValueError)

    def test_handle_tool_error_turns_a_failure_back_into_text(self) -> None:
        """The caller can opt into the agent correcting itself next turn."""
        toolkit, _ = _make_toolkit(
            search_side_effect=ValueError("query must be 200 characters or fewer")
        )
        tool = toolkit.get_tools()[0]
        tool.handle_tool_error = True

        output = tool.invoke({"query": "x" * 300})

        assert "Web search failed" in output
        assert "200 characters or fewer" in output

    def test_a_failed_search_is_retried_by_with_retry(self) -> None:
        """The point of raising: a transient failure gets a second attempt."""
        toolkit, _ = _make_toolkit(
            search_side_effect=[
                ConnectionError("connection reset"),
                _response(
                    WebSearchResult(
                        title="Second attempt", url="https://example.com", text="body"
                    )
                ),
            ]
        )
        tool = toolkit.get_tools()[0].with_retry(
            retry_if_exception_type=(ToolException,), stop_after_attempt=2
        )

        output = tool.invoke({"query": "boto3 release"})

        assert "Second attempt" in output

    def test_raises_after_the_toolkit_is_closed(self) -> None:
        """Using a closed toolkit is a programming error, not a search failure."""
        toolkit, _ = _make_toolkit()
        toolkit.close()

        with pytest.raises(RuntimeError, match="has been closed"):
            toolkit._search("anything")

    @pytest.mark.asyncio
    async def test_async_invocation_returns_the_same_text(self) -> None:
        """The async path exists so a search does not block the event loop."""
        toolkit, client = _make_toolkit(
            _response(WebSearchResult(text="body", title="A page"))
        )
        tool = toolkit.get_tools()[0]

        output = await tool.ainvoke({"query": "anything"})

        assert "1. A page" in output
        assert client.search.call_count == 1


class TestFormatResponse:
    """Tests for rendering results as text."""

    def test_explains_an_empty_result_set(self) -> None:
        """An empty result is a normal outcome of a narrow filter."""
        output = _format_response(WebSearchResponse(results=[]))

        assert "No results" in output
        assert "filter" in output

    def test_numbers_results_in_the_order_returned(self) -> None:
        """Ordering is the service's, so it is preserved."""
        output = _format_response(
            _response(
                WebSearchResult(text="first", title="One"),
                WebSearchResult(text="second", title="Two"),
            )
        )

        assert output.index("1. One") < output.index("2. Two")

    def test_labels_a_result_with_no_title(self) -> None:
        """A missing title still needs a heading to number against."""
        output = _format_response(_response(WebSearchResult(text="body")))

        assert "1. Untitled" in output

    def test_omits_fields_the_index_did_not_report(self) -> None:
        """Absent fields are left out rather than rendered empty."""
        output = _format_response(_response(WebSearchResult(text="body")))

        assert "URL:" not in output
        assert "Published:" not in output
        assert "body" in output


class TestClose:
    """Tests for resource cleanup."""

    def test_closes_a_client_it_created(self) -> None:
        """The toolkit owns the connection pool it opened."""
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            toolkit = WebSearchToolkit(gateway_id=GATEWAY_ID)
            toolkit.close()

        client_cls.return_value.close.assert_called_once()

    def test_leaves_a_supplied_client_open(self) -> None:
        """A caller who passed a client keeps ownership of it."""
        toolkit, client = _make_toolkit()

        toolkit.close()

        client.close.assert_not_called()

    def test_close_is_idempotent(self) -> None:
        """Closing twice is not an error and does not double-release."""
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            toolkit = WebSearchToolkit(gateway_id=GATEWAY_ID)
            toolkit.close()
            toolkit.close()

        client_cls.return_value.close.assert_called_once()

    def test_context_manager_closes_on_exit(self) -> None:
        """The toolkit can be scoped with `with`."""
        with patch(
            "langchain_aws.tools.web_search_toolkit.WebSearchClient"
        ) as client_cls:
            with WebSearchToolkit(gateway_id=GATEWAY_ID) as toolkit:
                assert toolkit.get_tools()

        client_cls.return_value.close.assert_called_once()


class TestExports:
    """Tests for the package export."""

    def test_factory_is_exported_from_langchain_aws_tools(self) -> None:
        """The tool is reachable the same way the other toolkits are."""
        import langchain_aws.tools as tools_module

        assert "create_web_search_toolkit" in tools_module.__all__
        assert callable(tools_module.create_web_search_toolkit)

    def test_input_schema_documents_every_field(self) -> None:
        """The model relies on these descriptions to fill the arguments."""
        from langchain_aws.tools.web_search_toolkit import WebSearchInput

        fields: List[str] = list(WebSearchInput.model_fields)
        assert fields == [
            "query",
            "max_results",
            "include_domains",
            "exclude_domains",
            "published_after",
            "published_before",
        ]
        for name, field in WebSearchInput.model_fields.items():
            assert field.description, f"{name} has no description"
