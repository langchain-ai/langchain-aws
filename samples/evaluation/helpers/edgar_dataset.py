"""Build the dataset and the answer key for any US-listed companies, from tickers.

Point this at the tickers you care about and it derives everything the evaluation
needs: the CIK, the most recent 10-K accession number, the two statement pages inside
that filing, and the figures the answer key is computed from.

    python helpers/edgar_dataset.py --tickers CRWD ZS NET

That writes `dataset.json` beside the notebooks, and `research_agent.py` picks it up on
the next import. Nothing else changes: the prompts, the checks and the notebooks are all
written against whatever is in that file.

Why this exists. Hand-entering a dataset means transcribing about twelve fields per
company, and a transcription error in an answer key is the worst kind of bug in an
evaluation, because every score afterward is confidently wrong. Everything here comes
from the SEC's own APIs, so the answer key is computed rather than typed.

Only US SEC filers work this way. For anything else (a private company, a non-US
listing, an internal system) construct `Company` objects by hand in `research_agent.py`
and supply the figures yourself. The evaluation code does not care where they came from.

SEC asks automated clients to send a descriptive User-Agent with contact information,
and rate limits to about 10 requests a second. Set `SEC_USER_AGENT` or pass
`--user-agent`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

__all__ = ["build_dataset", "load_dataset", "DATASET_PATH"]

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset.json"

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}"

#: One concept can be tagged several ways depending on who prepared the filing, so each
#: entry is a preference order rather than a single tag. Anything that resolves to no
#: tag at all is reported rather than guessed.
_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold",
                        "CostOfGoodsAndServicesSoldExcludingAmortization"),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "research_and_development": (
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capital_expenditure": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
}


#: Without these the margin metrics cannot be computed, so a gap is an error rather
#: than a None. Research and development is deliberately NOT here: plenty of companies
#: report none, and no graded metric depends on it.
_REQUIRED = ("revenue", "gross_profit", "operating_income", "net_income",
             "operating_cash_flow", "capital_expenditure")

_SUFFIXES = (" INC.", " INC", " CORPORATION", " CORP.", " CORP", " CO.", " LTD.",
             " LTD", " PLC", " HOLDINGS", " GROUP", ",")


def _clean_name(entity: str) -> str:
    """SEC entity names are shouty and suffixed. Use a form a prompt reads naturally.

    "SNOWFLAKE INC." becomes "Snowflake". The name goes into subagent descriptions and
    into the answer, so it needs to look like something a person would write.
    """
    name = " ".join(entity.split()).upper().rstrip(".")
    for suffix in _SUFFIXES:
        while name.endswith(suffix):
            name = name[: -len(suffix)].strip().rstrip(",").strip()
    return name.title()


def _get(url: str, user_agent: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent,
                                               "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed SEC hosts
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip
            raw = gzip.decompress(raw)
    time.sleep(0.15)  # stay clear of SEC's ~10 requests per second limit
    return raw


def _json(url: str, user_agent: str) -> Any:
    return json.loads(_get(url, user_agent))


def _cik_for(ticker: str, table: dict, ) -> int:
    t = ticker.upper()
    for row in table.values():
        if row["ticker"].upper() == t:
            return int(row["cik_str"])
    msg = f"ticker {ticker!r} not found in SEC's ticker table"
    raise LookupError(msg)


def _latest_10k(cik: int, user_agent: str) -> tuple[str, str]:
    """Accession number and period-end date of the most recent 10-K."""
    recent = _json(_SUBMISSIONS.format(cik=cik), user_agent)["filings"]["recent"]
    for form, accn, end in zip(recent["form"], recent["accessionNumber"],
                               recent["reportDate"]):
        if form == "10-K":
            return accn, end
    msg = f"no 10-K found for CIK {cik}"
    raise LookupError(msg)


def _statement_pages(cik: int, accn: str, user_agent: str) -> tuple[str, str]:
    """The income statement and cash flow statement pages inside a filing.

    Report page numbers are assigned per filing, so they are looked up rather than
    assumed. Comprehensive income and the supplemental cash flow schedules are excluded
    because they are separate statements that would otherwise match first.
    """
    xml = _get(f"{_ARCHIVE.format(cik=cik, accn=accn.replace('-', ''))}/FilingSummary.xml",
               user_agent).decode("utf-8", "replace")
    reports = re.findall(r"<Report[^>]*>(.*?)</Report>", xml, re.S)
    income = cash = ""
    for r in reports:
        name = (re.search(r"<ShortName>(.*?)</ShortName>", r, re.S) or [None, ""])[1].upper()
        page = (re.search(r"<HtmlFileName>(.*?)</HtmlFileName>", r, re.S) or [None, ""])[1]
        if not page:
            continue
        if not income and "COMPREHENSIVE" not in name and (
                "STATEMENTS OF OPERATIONS" in name or "STATEMENTS OF INCOME" in name):
            income = page
        if not cash and "CASH FLOW" in name and not any(
                x in name for x in ("RECONCILIATION", "SUPPLEMENT", "PARENTHETICAL")):
            cash = page
    if not income or not cash:
        msg = f"could not locate both statements in {accn} (income={income!r}, cash={cash!r})"
        raise LookupError(msg)
    return income, cash


def _annual_series(facts: dict, tags: tuple[str, ...]) -> dict[str, dict]:
    """Full-year 10-K values keyed by period END DATE, for the first tag that has any.

    Keyed by end date rather than by the `fy` field on purpose. `fy` is the fiscal year
    of the *filing* a fact appeared in, not of the fact itself, so a 10-K that restates
    two prior years tags all three with the same `fy`. Keying on `fy` looked correct for
    the first three companies tested and then lost a whole year for CrowdStrike.
    """
    from datetime import date
    gaap = facts["facts"].get("us-gaap", {})
    for tag in tags:
        out: dict[str, dict] = {}
        for u in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if u.get("form") != "10-K" or "start" not in u:
                continue
            span = (date.fromisoformat(u["end"]) - date.fromisoformat(u["start"])).days
            if not 330 <= span <= 400:      # a quarter or a stub period is not a year
                continue
            prev = out.get(u["end"])
            if prev is None or u.get("filed", "") >= prev["filed"]:
                out[u["end"]] = {"val": u["val"], "filed": u.get("filed", "")}
        if out:
            return out
    return {}


def _two_latest(ends: list[str]) -> tuple[str, str]:
    """The most recent annual period and the one before it, about a year apart."""
    from datetime import date
    ordered = sorted(ends, reverse=True)
    current = ordered[0]
    for prior in ordered[1:]:
        gap = (date.fromisoformat(current) - date.fromisoformat(prior)).days
        if 300 <= gap <= 430:
            return current, prior
    msg = f"no prior annual period about a year before {current}; have {ordered[:5]}"
    raise LookupError(msg)


def build_dataset(
    tickers: list[str],
    *,
    user_agent: str | None = None,
    out: Path | None = None,
) -> list[dict]:
    """Derive a full dataset for `tickers` and write it as JSON.

    Returns the list of company records. Each one carries the identifiers, the two
    statement URLs the researcher subagent reads, and the raw figures the answer key is
    computed from, so the answer key never has to be transcribed.
    """
    ua = user_agent or os.environ.get("SEC_USER_AGENT")
    if not ua:
        msg = ("SEC requires a descriptive User-Agent with contact info. "
               "Set SEC_USER_AGENT or pass user_agent=, for example "
               "'Acme Research acme-eng@example.com'.")
        raise ValueError(msg)

    table = _json(_TICKERS_URL, ua)
    records: list[dict] = []
    for ticker in tickers:
        cik = _cik_for(ticker, table)
        accn, _ = _latest_10k(cik, ua)
        income_page, cash_page = _statement_pages(cik, accn, ua)
        facts = _json(_FACTS.format(cik=cik), ua)
        series = {k: _annual_series(facts, tags) for k, tags in _CONCEPTS.items()}

        # Gross profit is not always tagged. Where it is missing, derive it from cost of
        # revenue, which is how the statement itself presents the subtotal.
        if not series["gross_profit"] and series["revenue"] and series["cost_of_revenue"]:
            series["gross_profit"] = {
                end: {"val": v["val"] - series["cost_of_revenue"][end]["val"], "filed": ""}
                for end, v in series["revenue"].items()
                if end in series["cost_of_revenue"]
            }

        hard = [k for k in _REQUIRED if not series[k]]
        if hard:
            msg = (f"{ticker}: no usable XBRL tag for {hard}. This dataset is built for "
                   f"companies that report a software-style income statement; "
                   f"gross profit and cost of revenue are needed for margin metrics.")
            raise LookupError(msg)

        current, prior = _two_latest(list(series["revenue"]))
        absent = [k for k in _REQUIRED if current not in series[k]]
        if absent:
            msg = (f"{ticker}: period {current} missing a value for {absent}. "
                   f"The metrics here assume a software-style income statement; "
                   f"a company that reports no gross profit subtotal needs its "
                   f"own metrics and its own Company entries.")
            raise LookupError(msg)

        record = {
            "name": _clean_name(facts.get("entityName", ticker)),
            "ticker": ticker.upper(),
            "cik": cik,
            "accession": accn,
            # A January year end is called FY of that calendar year (Snowflake
            # FY2026 ends 2026-01-31), and so is a December one, so the year of
            # the end date is the fiscal year label in both cases.
            "fiscal_year": int(current[:4]),
            "period_end": current,
            "prior_revenue": series["revenue"][prior]["val"],
            "income_statement_page": income_page,
            "cash_flow_page": cash_page,
        }
        for k in _CONCEPTS:
            v = series[k].get(current)
            # Optional concepts stay absent rather than becoming a silent zero, which
            # would show up later as a real-looking 0.0% of revenue.
            record[k] = v["val"] if v else None
        record.pop("cost_of_revenue", None)
        records.append(record)
        note = "" if record["research_and_development"] is not None else "  (no R&D reported)"
        print(f"  {ticker:6} CIK {cik:<9} {accn}  FY end {current}"
              f"  pages {income_page}/{cash_page}{note}")

    target = out or DATASET_PATH
    target.write_text(json.dumps(records, indent=1) + "\n")
    print(f"\nwrote {target} with {len(records)} companies")
    return records


def load_dataset(path: Path | None = None) -> list[dict] | None:
    """Read a dataset written by `build_dataset`, or None if there is not one."""
    p = path or DATASET_PATH
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", required=True,
                    help="US-listed tickers, for example SNOW DDOG MDB")
    ap.add_argument("--user-agent", default=None,
                    help="descriptive UA with contact info, or set SEC_USER_AGENT")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    build_dataset(a.tickers, user_agent=a.user_agent, out=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
