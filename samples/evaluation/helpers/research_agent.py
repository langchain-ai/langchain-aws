"""The agent under evaluation: a due-diligence Deep Agent over SEC filings.

Kept beside the notebooks rather than inside them so the notebook cells stay
readable. The prompts are the interesting part and the notebooks print them; this
file holds the dataset and the toolkit wiring.

Shape: a coordinator fans out one browser researcher per company via `task()`, each
in its own AgentCore Browser MicroVM reading that company's 10-K financial
statements, then hands the extracted figures to an analyst on an AgentCore Code
Interpreter which computes SaaS metrics and builds a chart.

Why filings rather than a summary page: each researcher reads roughly 60,000
characters of statement text and returns about 300 characters of figures. Three of
them at once would be about 110,000 tokens of raw material in a single-agent
context. That reduction is the whole reason to delegate, and it is measurable here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from dataclasses import fields as fields_of
from pathlib import Path
from typing import Any

from botocore.config import Config as BotoConfig
from deepagents import create_deep_agent
from langchain_aws import ChatBedrockConverse

__all__ = [
    "Company",
    "COMPANIES",
    "ANSWER_KEY",
    "RULE_OF_40_RANK",
    "GROWTH_RANK",
    "COORDINATOR_PROMPT",
    "RESEARCHER_PROMPT",
    "ANALYST_PROMPT",
    "LOOPING_COORDINATOR_PROMPT",
    "build_model",
    "build_research_agent",
    "cleanup",
]

_EDGAR = "https://www.sec.gov/Archives/edgar/data"


@dataclass(frozen=True)
class Company:
    """A research target pinned to one immutable SEC filing.

    An EDGAR accession number identifies a filing that can never be edited or
    withdrawn, so figures taken from it stay correct indefinitely. The browser, the
    network and the pages are all real; only the *mutability* is removed.

    Every figure comes from the SEC's XBRL `companyfacts` API rather than being
    transcribed by hand, which removes the class of error where a sample's own answer
    key is wrong.

    Amounts are whole US dollars. The statements a researcher subagent reads usually
    display thousands or millions, so the researcher is told to report the label it
    sees. Every graded metric here is a ratio, so the display unit cancels out.
    """

    name: str
    cik: int
    accession: str
    fiscal_year: int
    period_end: str
    revenue: int
    prior_revenue: int
    gross_profit: int
    operating_income: int
    net_income: int
    research_and_development: int | None
    operating_cash_flow: int
    capital_expenditure: int
    income_statement_page: str
    cash_flow_page: str

    @property
    def _dir(self) -> str:
        return f"{_EDGAR}/{self.cik}/{self.accession.replace('-', '')}"

    @property
    def income_statement_url(self) -> str:
        """Consolidated statements of operations, as a standalone table."""
        return f"{self._dir}/{self.income_statement_page}"

    @property
    def cash_flow_url(self) -> str:
        """Consolidated statements of cash flows, as a standalone table."""
        return f"{self._dir}/{self.cash_flow_page}"

    @property
    def filing_index_url(self) -> str:
        """Human-readable index for the filing, useful when a page number moves."""
        return f"{self._dir}/{self.accession}-index.htm"

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-")

    @property
    def free_cash_flow(self) -> int:
        return self.operating_cash_flow - self.capital_expenditure


def _load_companies() -> tuple[Company, ...]:
    """Build `COMPANIES` from `dataset.json`.

    That file is generated, never typed. Point the evaluation at different companies by
    regenerating it, which is one command and no code change:

        python helpers/edgar_dataset.py --tickers CRWD ZS NET

    Every figure comes from the SEC's XBRL API, so the answer key is computed rather
    than transcribed. A transcription error in an answer key is the worst bug an
    evaluation can have, because every score after it is confidently wrong.
    """
    here = Path(__file__).resolve().parent
    # Beside this file, one level up (the notebooks' directory), or the working
    # directory. The three cover a notebook run, a script run and the flat layout
    # inside an AgentCore Runtime deployment package.
    for candidate in (here / "dataset.json", here.parent / "dataset.json",
                      Path("dataset.json")):
        if candidate.exists():
            records = json.loads(candidate.read_text())
            break
    else:
        msg = ("dataset.json not found. Generate it with:\n"
               "    python helpers/edgar_dataset.py --tickers SNOW DDOG MDB")
        raise FileNotFoundError(msg)
    fields = {f.name for f in fields_of(Company)}
    return tuple(Company(**{k: v for k, v in r.items() if k in fields}) for r in records)


#: Whatever is in dataset.json. The shipped file holds Snowflake, Datadog and MongoDB.
#: Their fiscal calendars do not align, since Datadog closes in December and the other
#: two in January. That offset is useful rather than inconvenient: a check grades whether
#: the answer discloses each period end, because a comparison that hides them is
#: misleading even when every number in it is correct.
COMPANIES: tuple[Company, ...] = _load_companies()


def revenue_growth_pct(c: Company) -> float:
    """Year over year revenue growth, in percent."""
    return (c.revenue - c.prior_revenue) / c.prior_revenue * 100


def gross_margin_pct(c: Company) -> float:
    return c.gross_profit / c.revenue * 100


def operating_margin_pct(c: Company) -> float:
    return c.operating_income / c.revenue * 100


def net_margin_pct(c: Company) -> float:
    return c.net_income / c.revenue * 100


def fcf_margin_pct(c: Company) -> float:
    """Free cash flow as a percentage of revenue, where free cash flow is
    operating cash flow less purchases of property and equipment."""
    return c.free_cash_flow / c.revenue * 100


def rule_of_40(c: Company) -> float:
    """Revenue growth plus free cash flow margin, the usual software formulation.

    Reported as a bare number rather than a percentage because that is how it is
    quoted. A company above 40 is considered to be balancing growth and efficiency.
    """
    return revenue_growth_pct(c) + fcf_margin_pct(c)


def rnd_pct_of_revenue(c: Company) -> float | None:
    """None when the company reports no research and development line."""
    if c.research_and_development is None:
        return None
    return c.research_and_development / c.revenue * 100


#: Derived values. None of these appear on the source statements, so an agent has to
#: extract several figures and compute. A model answering from memory cannot get
#: inside tolerance, and neither can one that skips its tools.
ANSWER_KEY: dict[str, dict[str, float]] = {
    c.name: {
        "revenue_growth_pct": round(revenue_growth_pct(c), 2),
        "gross_margin_pct": round(gross_margin_pct(c), 2),
        "operating_margin_pct": round(operating_margin_pct(c), 2),
        "net_margin_pct": round(net_margin_pct(c), 2),
        "fcf_margin_pct": round(fcf_margin_pct(c), 2),
        "rule_of_40": round(rule_of_40(c), 2),
        **({"rnd_pct_of_revenue": round(rnd_pct_of_revenue(c), 2)}
           if rnd_pct_of_revenue(c) is not None else {}),
    }
    for c in COMPANIES
}

#: Ranked, not hardcoded, so these stay correct when the dataset changes. For the
#: shipped three this puts Datadog first even though Snowflake leads revenue and growth,
#: so an agent reasoning "biggest company wins" gets it wrong.
RULE_OF_40_RANK = [c.name for c in sorted(COMPANIES, key=lambda c: -rule_of_40(c))]

#: Snowflake first for the shipped three. Growth and efficiency disagreeing about the
#: winner is the point of asking for both.
GROWTH_RANK = [c.name for c in sorted(COMPANIES, key=lambda c: -revenue_growth_pct(c))]

COORDINATOR_PROMPT = """You are a due-diligence research coordinator.

## Your workflow:
1. Identify which companies the request covers.
2. Launch one research subagent PER company, IN PARALLEL, using the task() tool. \
Emit all task() calls in a single response.
3. When the researchers return, delegate to the data-analyst subagent to compute \
derived metrics and build a chart. Tell it to save files in its working directory, \
never to "/" or "/tmp".
4. Present the final report from what the subagents told you, and carry the \
analyst's two ranking lines through verbatim. Do NOT call read_file, glob or ls on \
paths a subagent mentions: each subagent has its own isolated filesystem and its \
files do not exist in yours.

## Important:
- Launch research tasks in parallel (multiple task() calls in one response).
- Do NOT use the take_screenshot tool.
- State each company's fiscal period end next to its figures. These companies do \
not share a fiscal calendar, so a comparison that hides the periods is misleading.
- You have a research subagent for these companies only: {company_list}. If asked \
about any other company, say plainly that you cannot research it. Never estimate \
or recall figures for a company you did not research.
- If a figure is not present in a source, say so. Do not substitute an estimate.
"""

#: Deliberately missing the two filesystem rules from steps 3 and 4 above. Used in
#: notebook 3 as the A/B control, because without them the coordinator hunts for the
#: analyst's chart in a filesystem it cannot see and loops until LangGraph's
#: recursion ceiling. Observed burning 997 seconds before failing.
LOOPING_COORDINATOR_PROMPT = """You are a due-diligence research coordinator.

## Your workflow:
1. Identify which companies the request covers.
2. Launch one research subagent PER company, IN PARALLEL, using the task() tool. \
Emit all task() calls in a single response.
3. When the researchers return, delegate to the data-analyst subagent to compute \
derived metrics and build a chart.
4. Verify the chart file exists, then present the final report, carrying the \
analyst's two ranking lines through verbatim.

## Important:
- Launch research tasks in parallel (multiple task() calls in one response).
- Do NOT use the take_screenshot tool.
- State each company's fiscal period end next to its figures. These companies do \
not share a fiscal calendar, so a comparison that hides the periods is misleading.
- You have a research subagent for these companies only: {company_list}. If asked \
about any other company, say plainly that you cannot research it. Never estimate \
or recall figures for a company you did not research.
- If a figure is not present in a source, say so. Do not substitute an estimate.
"""

RESEARCHER_PROMPT = """You are a financial research specialist covering ONE company.

You read SEC filing statements directly. Both of your pages are single financial
statements pulled from a 10-K, so the figures are authoritative and final.

## Your process:
1. navigate_browser to your income statement URL, then extract_text.
2. Report, for the MOST RECENT fiscal year column and also the year before it: \
Revenue, Cost of revenue, Gross profit, Operating income or loss, Net income or loss, \
Research and development expense.
3. navigate_browser to your cash flow URL, then extract_text.
4. Report, for the most recent fiscal year: Net cash provided by operating \
activities, and Purchases of property and equipment.

## Important:
- Report the fiscal period END DATE and the units exactly as the statement labels \
them. These statements are in thousands of dollars, and dropping that makes every \
downstream number wrong by a factor of a thousand.
- A loss is negative. Statements show losses in parentheses, so "(1,435,165)" \
means -1435165. Report the sign explicitly.
- Do NOT use the take_screenshot tool. Use extract_text instead.
- Report only figures you actually read. If one is absent, write "not stated on \
source". Never estimate, and never fill a gap from memory of the company.
"""

ANALYST_PROMPT = """You are a financial data analyst.

## Your process:
1. Use execute_code to build a pandas DataFrame from the figures given to you.
2. Compute, per company, to 2 decimals: revenue growth percent, gross margin \
percent, operating margin percent, net margin percent, free cash flow margin \
percent, and Rule of 40. Free cash flow is operating cash flow minus purchases of \
property and equipment. Rule of 40 is revenue growth percent plus free cash flow \
margin percent.
3. Build a grouped matplotlib bar chart of the margin metrics and save it in your \
working directory.
4. Confirm the file exists with execute_command 'ls -la'. Do not use list_files: it \
returns empty output regardless of what is on disk.
5. Report a markdown table. Then, on its own line and in exactly this format, using \
" > " between names:
   `Rule of 40 ranking: <first> > <second> > <third>`
6. In prose after that line, say whether the ranking by revenue growth differs from \
the ranking by Rule of 40, and name which company leads each.

Compute every metric with code. Do not estimate any of them mentally.
Carry each company's fiscal period end through into your table.
Pre-installed: pandas, matplotlib, numpy.
"""


def build_model(model_id: str | None = None, region: str | None = None) -> ChatBedrockConverse:
    """Chat model sized for concurrent, large-payload subagent calls.

    `max_pool_connections` is far above botocore's default of 10. A three-way
    fan-out holds three browser sessions, three concurrent Converse calls and an
    interpreter at once, and each researcher's request carries whole extracted
    statement pages (roughly 60 KB of text each). At the default pool size that
    combination produced `Connection reset by peer` mid-run.
    """
    return ChatBedrockConverse(
        model=model_id or os.environ.get("EVAL_MODEL_ID", "us.anthropic.claude-sonnet-4-6"),
        region_name=region or os.environ.get("AWS_REGION", "us-west-2"),
        config=BotoConfig(
            read_timeout=300,
            connect_timeout=30,
            max_pool_connections=50,
            retries={"max_attempts": 6, "mode": "adaptive"},
        ),
    )


async def build_research_agent(
    model: Any,
    *,
    region: str | None = None,
    companies: tuple[Company, ...] = COMPANIES,
    system_prompt: str | None = None,
) -> tuple[Any, list[Any]]:
    """Build the coordinator, one researcher per company, and the analyst.

    Each researcher gets its OWN browser toolkit, which means its own MicroVM. That
    isolation is what allows genuine parallel browsing; a shared toolkit serializes
    on the session.

    Returns:
        `(agent, toolkits)`. Always clean up the toolkits: they hold live MicroVM
        sessions that bill until they idle out.
    """
    from langchain_aws.tools import (
        create_browser_toolkit,
        create_code_interpreter_toolkit,
    )

    region = region or os.environ.get("AWS_REGION", "us-west-2")
    toolkits: list[Any] = []
    subagents: list[dict[str, Any]] = []

    for company in companies:
        toolkit, tools = create_browser_toolkit(region=region)
        if hasattr(toolkit, "session_manager"):
            toolkit.session_manager.session_wait_timeout = 90.0
        toolkits.append(toolkit)
        subagents.append({
            "name": f"research-{company.slug}",
            "description": (
                f"Researches {company.name} from its fiscal {company.fiscal_year} "
                f"10-K financial statements. Use for {company.name} only."
            ),
            "system_prompt": (
                f"{RESEARCHER_PROMPT}\n\n## Your assignment\n"
                f"Company: {company.name}\n"
                f"Fiscal year: {company.fiscal_year}, ending {company.period_end}\n"
                f"Income statement: {company.income_statement_url}\n"
                f"Cash flow statement: {company.cash_flow_url}\n"
            ),
            "tools": tools,
        })

    ci_toolkit, ci_tools = await create_code_interpreter_toolkit(region=region)
    toolkits.append(ci_toolkit)
    subagents.append({
        "name": "data-analyst",
        "description": "Computes derived SaaS metrics and builds comparison charts.",
        "system_prompt": ANALYST_PROMPT,
        "tools": ci_tools,
    })

    agent = create_deep_agent(
        model=model,
        subagents=subagents,
        tools=[],
        system_prompt=(system_prompt or COORDINATOR_PROMPT).format(
            company_list=", ".join(c.name for c in companies)
        ),
        name="due-diligence-coordinator",
        checkpointer=None,
    )
    return agent, toolkits


async def cleanup(toolkits: list[Any]) -> None:
    """Stop every MicroVM session, reporting rather than raising on failure."""
    for toolkit in toolkits:
        try:
            await toolkit.cleanup()
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask results
            print(f"cleanup warning: {exc}")
