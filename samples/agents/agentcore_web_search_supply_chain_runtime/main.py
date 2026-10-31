"""Daily dependency watch on AgentCore Runtime: the LangGraph graph from
agentcore_web_search_supply_chain_watch.ipynb, served as an AgentCore Runtime entrypoint."""

import operator
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal, TypedDict
from urllib.parse import urlparse

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.tools import WebSearchClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Send
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from pydantic import BaseModel, Field

LangchainInstrumentor().instrument()

app = BedrockAgentCoreApp()
log = app.logger

REGION = os.environ.get("AWS_REGION", "us-east-1")
GATEWAY_ID = os.environ["WEB_SEARCH_GATEWAY_ID"]
TARGET_NAME = os.environ.get("WEB_SEARCH_TARGET_NAME", "security-sources")
MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-sonnet-5")
MAX_SEARCHES_PER_BRANCH = 4
# The target's administrator blocklist, used here only to count blocked-domain sources.
ADMIN_BLOCKLIST = ["amazon.com"]

DEFAULT_WATCHLIST = {
    "npm": ["express", "axios", "lodash"],
    "PyPI": ["requests", "urllib3", "cryptography"],
    "container base images": ["openssl", "glibc", "zlib"],
}

model = init_chat_model(f"bedrock_converse:{MODEL_ID}", region_name=REGION)


def domain_of(url):
    host = urlparse(url).hostname or ""
    return host.removeprefix("www.")


def matches(url, domains):
    host = domain_of(url)
    return any(host == d or host.endswith("." + d) for d in domains)


def normalize(url):
    return url.rstrip("/").split("#")[0]


def parse_published(value):
    # The index reports dates like "05:00PM, Tuesday, October 06 2026, PDT".
    if not value:
        return None
    stamp, _, zone = value.rpartition(", ")
    offsets = {"PDT": -7, "PST": -8, "UTC": 0, "GMT": 0}
    try:
        local = datetime.strptime(stamp, "%I:%M%p, %A, %B %d %Y")
    except ValueError:
        return None
    return (local - timedelta(hours=offsets.get(zone, 0))).replace(tzinfo=timezone.utc)


def make_search_tool(window_start, max_searches=MAX_SEARCHES_PER_BRANCH):
    # One client per tool: calls on a single client run one at a time.
    client = WebSearchClient(
        region=REGION, gateway_id=GATEWAY_ID, target_name=TARGET_NAME
    )
    published_after = window_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    calls = {"count": 0}

    @tool(response_format="content_and_artifact")
    def search_security_news(query: str) -> tuple:
        """Search recent security advisories, CVE records and security news. Cite the URLs you use."""
        calls["count"] += 1
        if calls["count"] > max_searches:
            return f"Search limit of {max_searches} reached. Report what you have.", []
        response = client.search(query, max_results=8, published_after=published_after)
        sources = [
            {
                "title": r.title,
                "url": r.url,
                "published": r.published_date,
                "text": r.text,
            }
            for r in response.results
            if r.url
        ]
        text = (
            "\n\n".join(
                f"{i}. {r.title or r.url}\n   URL: {r.url}\n   Published: {r.published_date}\n   {r.text[:700]}"
                for i, r in enumerate(response.results, start=1)
                if r.url
            )
            or "No results in the watch window."
        )
        return text, sources

    search_security_news.handle_tool_error = True
    return search_security_news, client


class Finding(BaseModel):
    package: str = Field(
        description="The watched package or image this affects, exactly as named in the watchlist."
    )
    title: str = Field(description="One line: what happened.")
    identifier: str | None = Field(
        default=None, description="CVE or advisory ID if the source gives one."
    )
    severity: Literal["critical", "high", "medium", "low", "unknown"] = Field(
        default="unknown",
        description="The severity the source states. Use unknown if the source doesn't state one.",
    )
    actively_exploited: bool = Field(
        description="True only if the source says it is exploited in the wild."
    )
    action: str = Field(
        description="What the team should do, in one line. Vendor-neutral: never recommend a commercial product or service."
    )
    source_url: str = Field(
        description="URL of the search result this finding came from. Must be one you were given."
    )


class EcosystemFindings(BaseModel):
    findings: list[Finding] = Field(
        description="Empty if nothing in the window affects the watched packages."
    )


BRANCH_PROMPT = """You watch one software ecosystem for a security team.
Search for anything published in the watch window that affects the watched packages:
new CVEs, security advisories, malicious or compromised packages, and vulnerabilities
reported as exploited. Run at most four searches.

Report a finding only when a search result is about one of the watched packages itself:
a vulnerability, advisory or compromise in that package. A result that only mentions the
package in passing, or is about a different package that depends on it, is not a finding.
Use the exact source URL from the search results. Take severity only from what the source
states. Keep actions vendor-neutral, such as upgrading, patching or checking configuration,
and never recommend a commercial product or service. If nothing in the results affects the
watched packages, return no findings. Never fill gaps from memory.

Report only issues first disclosed inside the watch window. A recent result that describes
an older incident as background is not a finding."""


class WatchState(TypedDict):
    watchlist: dict
    window_start: str
    findings: Annotated[list, operator.add]
    branch_stats: Annotated[list, operator.add]
    new_findings: list
    report: str


class BranchState(TypedDict):
    ecosystem: str
    packages: list
    window_start: str


def fan_out(state: WatchState):
    return [
        Send(
            "research",
            {"ecosystem": eco, "packages": pkgs, "window_start": state["window_start"]},
        )
        for eco, pkgs in state["watchlist"].items()
    ]


def research(branch: BranchState):
    started = time.perf_counter()
    window_start = datetime.fromisoformat(branch["window_start"])
    search_tool, client = make_search_tool(window_start)
    try:
        agent = create_agent(
            model,
            tools=[search_tool],
            system_prompt=BRANCH_PROMPT,
            response_format=EcosystemFindings,
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Ecosystem: {branch['ecosystem']}. Watched packages: {', '.join(branch['packages'])}. "
                        f"Watch window starts {window_start:%Y-%m-%d %H:%M} UTC.",
                    }
                ]
            }
        )
    finally:
        client.close()

    returned = {
        normalize(s["url"]): s
        for m in result["messages"]
        if m.type == "tool"
        for s in (m.artifact or [])
    }
    searches = sum(1 for m in result["messages"] if m.type == "tool" and m.artifact)
    kept, dropped = (
        [],
        {
            "url_not_returned": 0,
            "package_not_in_source": 0,
            "id_not_in_source": 0,
            "date_outside_window": 0,
        },
    )
    for finding in result["structured_response"].findings:
        source = returned.get(normalize(finding.source_url))
        evidence = (
            ((source or {}).get("title") or "")
            + " "
            + ((source or {}).get("text") or "")
        )
        published = parse_published(source["published"]) if source else None
        if source is None:
            dropped["url_not_returned"] += 1
        elif not re.search(
            rf"(?<![\w.-]){re.escape(finding.package)}(?![\w-])",
            evidence,
            re.IGNORECASE,
        ):
            dropped["package_not_in_source"] += 1
        elif finding.identifier and finding.identifier.lower() not in evidence.lower():
            dropped["id_not_in_source"] += 1
        elif published is None or published < window_start:
            # The search already asked for published-after; this makes it a hard rule.
            dropped["date_outside_window"] += 1
        else:
            kept.append(
                {
                    **finding.model_dump(),
                    "ecosystem": branch["ecosystem"],
                    "source_published": source["published"],
                }
            )
    return {
        "findings": kept,
        "branch_stats": [
            {
                "ecosystem": branch["ecosystem"],
                "searches": searches,
                "sources_returned": len(returned),
                "findings_kept": len(kept),
                "dropped": dropped,
                "blocked_domain_sources": sum(
                    matches(u, ADMIN_BLOCKLIST) for u in returned
                ),
                "seconds": round(time.perf_counter() - started, 1),
            }
        ],
    }


def finding_key(f):
    return (
        (f.get("identifier") or normalize(f["source_url"])).lower()
        + "|"
        + f["package"].lower()
    )


def keep_new(state: WatchState, *, store):
    new = []
    for f in state["findings"]:
        key = finding_key(f)
        if store.get(("reported",), key) is None:
            store.put(
                ("reported",), key, {"first_seen": datetime.now(timezone.utc).isoformat(), **f}
            )
            new.append(f)
    return {"new_findings": new}


def report(state: WatchState):
    rows = sorted(
        state["new_findings"],
        key=lambda f: (
            not f["actively_exploited"],
            ["critical", "high", "medium", "low", "unknown"].index(f["severity"]),
        ),
    )
    if not rows:
        return {"report": "No new findings in the watch window."}
    lines = [
        "| Package | What happened | ID | Severity | Exploited | Action | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in rows:
        lines.append(
            f"| {f['package']} ({f['ecosystem']}) | {f['title']} | {f.get('identifier') or ''} | "
            f"{f['severity']} | {'yes' if f['actively_exploited'] else 'no'} | {f['action']} | "
            f"[{domain_of(f['source_url'])}]({f['source_url']}) |"
        )
    return {"report": "\n".join(lines)}


builder = StateGraph(WatchState)
builder.add_node("research", research)
builder.add_node("keep_new", keep_new)
builder.add_node("report", report)
builder.add_conditional_edges(START, fan_out, ["research"])
builder.add_edge("research", "keep_new")
builder.add_edge("keep_new", "report")
builder.add_edge("report", END)

# InMemoryStore lives as long as this Runtime session. For "already reported" to survive
# across daily runs, use a durable store with exact key lookups, such as DynamoDBStore.
store = InMemoryStore()
watch = builder.compile(store=store)


@app.entrypoint
def invoke(payload, context):
    window_days = int(payload.get("window_days", 7))
    watchlist = payload.get("watchlist") or DEFAULT_WATCHLIST
    window_start = datetime.now(timezone.utc) - timedelta(days=window_days)
    started = time.perf_counter()
    state = watch.invoke(
        {
            "watchlist": watchlist,
            "window_start": window_start.isoformat(),
            "findings": [],
            "branch_stats": [],
        }
    )
    result = {
        "wall_seconds": round(time.perf_counter() - started, 1),
        "branch_stats": state["branch_stats"],
        "findings_kept": len(state["findings"]),
        "new_findings": len(state["new_findings"]),
        "report": state["report"],
    }
    log.info(
        f"watch finished: {result['findings_kept']} kept, {result['new_findings']} new in {result['wall_seconds']}s"
    )
    return result


if __name__ == "__main__":
    app.run()
