"""Integration tests for the AgentCore web search toolkit.

These need a gateway that already carries a web search connector target, since
creating one is a control plane operation with its own IAM requirements. Point
`WEB_SEARCH_INTEG_GATEWAY_ID` at that gateway to run them.
"""

import os
from urllib.parse import urlsplit

import pytest
from langchain_core.tools import ToolException

from langchain_aws.tools.web_search_toolkit import (
    WebSearchToolkit,
    create_web_search_toolkit,
)

GATEWAY_ID = os.environ.get("WEB_SEARCH_INTEG_GATEWAY_ID")
REGION = os.environ.get("WEB_SEARCH_INTEG_REGION", "us-east-1")

pytestmark = pytest.mark.skipif(
    not GATEWAY_ID,
    reason="Set WEB_SEARCH_INTEG_GATEWAY_ID to a gateway with a web search target",
)


def _toolkit() -> WebSearchToolkit:
    """Build a toolkit against the gateway named in the environment."""
    toolkit, _ = create_web_search_toolkit(region=REGION, gateway_id=GATEWAY_ID)
    return toolkit


def test_search_returns_attributed_results() -> None:
    """A plain search comes back with results carrying URLs to cite."""
    with _toolkit() as toolkit:
        output = toolkit.get_tools()[0].invoke(
            {"query": "boto3 release notes", "max_results": 3}
        )

    assert "Web search failed" not in output
    assert "URL: http" in output


def test_include_domains_restricts_the_sources() -> None:
    """An include filter reaches the connector and narrows the results."""
    domain = "docs.aws.amazon.com"
    with _toolkit() as toolkit:
        output = toolkit.get_tools()[0].invoke(
            {
                "query": "AgentCore gateway",
                "max_results": 5,
                "include_domains": [domain],
            }
        )

    assert "Web search failed" not in output
    urls = [
        line.strip().removeprefix("URL:").strip()
        for line in output.splitlines()
        if line.strip().startswith("URL:")
    ]
    assert urls, output
    for url in urls:
        host = urlsplit(url).hostname or ""
        assert host == domain or host.endswith(f".{domain}"), url


def test_a_query_over_the_limit_raises() -> None:
    """Client-side validation reaches the caller as a ToolException."""
    with _toolkit() as toolkit:
        with pytest.raises(ToolException, match="Web search failed"):
            toolkit.get_tools()[0].invoke({"query": "x" * 201})


@pytest.mark.asyncio
async def test_async_search_returns_results() -> None:
    """The async path reaches the same service."""
    with _toolkit() as toolkit:
        output = await toolkit.get_tools()[0].ainvoke(
            {"query": "urllib3 changelog", "max_results": 3}
        )

    assert "Web search failed" not in output
