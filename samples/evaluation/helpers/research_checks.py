"""The checks for the research agent, plus the traffic scenarios that stress it.

Kept out of the notebooks so a cell can be `checks = standard_checks()` rather than
forty lines of lambdas. Every check works on a `Trajectory` from either source, local
events or production spans, because that is the same object.
"""

from __future__ import annotations

import re

from agentcore_evals import Check
from research_agent import ANSWER_KEY, COMPANIES, RULE_OF_40_RANK

__all__ = [
    "near",
    "ranking_line",
    "standard_checks",
    "trajectory_only_checks",
    "SUCCESS_SCENARIOS",
    "FAILURE_SCENARIOS",
    "CHART_PROMPT",
    "metrics_found",
]


def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def near(value: float, tol: float = 0.1):
    """Answer contains a number within `tol` of `value`.

    Substring matching is wrong for computed figures: 29.16 may legitimately render as
    "29.16", "29.2" or "29.16%", and a rounding choice is not a correctness failure.

    The default tolerance is 0.1 rather than something tighter for a reason found by
    running this. A model that shortens 79.96 to one decimal may *truncate* to "79.9"
    instead of rounding to "80.0", which is off by 0.06. That is a display choice, not
    a wrong answer, so the tolerance has to clear it.
    """
    def check(traj, answer):
        hit = any(abs(n - value) <= tol for n in _numbers(answer))
        return hit, f"no number within {tol} of {value}"
    return check


def ranking_line(prefix: str, *labels: str):
    """A line reading "<prefix>: A > B > C" names these labels in this order.

    An earlier version of this check compared first-mention offsets across the whole
    answer. It reported a correct run as wrong, because the report opened with the
    title "Snowflake vs. Datadog vs. MongoDB" and that heading order is not a ranking
    claim. Grading prose position guesses at intent; grading a line the prompt asks
    for does not. A missing line fails, so this cannot pass vacuously.
    """
    want = [x.lower() for x in labels]

    def check(traj, answer):
        for line in answer.splitlines():
            low = line.lower()
            if prefix.lower() not in low:
                continue
            got = [p.strip(" `*_") for p in low.split(":", 1)[-1].split(">")]
            got = [g for g in got if g]
            if got == want:
                return True, ""
            return False, f"line says {got}, expected {want}"
        return False, f'no line containing "{prefix}"'
    return check


_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")


def _period_renderings(iso: str) -> list[str]:
    """Ways a model plausibly writes a period end date.

    Demanding ISO format here was a real bug: an agent that wrote "Jan 31 2026" and
    named the fiscal-calendar mismatch explicitly still failed the check. A check that
    grades a disclosure has to accept the formats a disclosure actually comes in.
    """
    year, month, day = iso.split("-")
    full = _MONTHS[int(month) - 1]
    d = str(int(day))
    return [iso, iso.replace("-", "/"),
            f"{full} {d}, {year}", f"{full} {d} {year}",
            f"{full[:3]} {d}, {year}", f"{full[:3]} {d} {year}",
            f"{d} {full} {year}", f"{month}/{d}/{year}", f"{month}/{day}/{year}"]


def periods_disclosed(traj, answer: str) -> tuple[bool, str]:
    """Every company's fiscal period end appears in the answer, in any usual format.

    These three companies do not share a fiscal calendar, so a table that compares
    them without saying which periods it compares is misleading even when every
    number in it is right. This is the one check here that grades a disclosure rather
    than a value.
    """
    low = answer.lower()
    missing = [c.name for c in COMPANIES
               if not any(r in low for r in _period_renderings(c.period_end))]
    return not missing, f"period end absent for {missing}"


def trajectory_only_checks() -> list[Check]:
    """Checks that need only a trajectory, no answer text.

    These are the ones that work identically on a local run and on production spans,
    which is what makes a single check list usable in both places.
    """
    return [
        Check("read the filings", lambda t, a: (
            bool(t.calls("navigate_browser", nested=True)), "no nested navigate_browser")),
        Check("computed with code", lambda t, a: (
            bool(t.calls("execute_code", nested=True)), "no nested execute_code")),
        # nested defaults to None, meaning every level. Restricted to the coordinator
        # this would be vacuously true, since subagent calls never appear there.
        Check("no screenshots", lambda t, a: (
            not t.calls("take_screenshot"), f"{len(t.calls('take_screenshot'))} calls")),
        # Efficiency: serial delegation is slower, not wrong, so this never fails a run.
        Check("3-way parallel fan-out", lambda t, a: (
            t.max_concurrent("task") >= 3, f"peak {t.max_concurrent('task')}"),
            tier="efficiency"),
        Check("at most 24 tool calls", lambda t, a: (
            len(t.tool_calls) <= 24, f"{len(t.tool_calls)} calls"), tier="efficiency"),
    ]


def standard_checks() -> list[Check]:
    """Trajectory checks plus ground-truth checks on the answer.

    Rule of 40 is the graded metric because it is a composite: getting it right
    requires revenue for two years, operating cash flow and capital expenditure, so
    one wrong extraction anywhere shows up here.
    """
    return [
        *trajectory_only_checks()[:3],
        *[Check(f"{c.name} rule of 40",
                near(ANSWER_KEY[c.name]["rule_of_40"], tol=0.1)) for c in COMPANIES],
        *[Check(f"{c.name} gross margin",
                near(ANSWER_KEY[c.name]["gross_margin_pct"])) for c in COMPANIES],
        # One graded ranking, not two. Asking the analyst for two verbatim lines is
        # where it drifted in testing: it emitted a gross-margin ranking in place of
        # the growth one. An instruction a model follows reliably makes a better check
        # than one it follows most of the time.
        Check("rule-of-40 ranking correct",
              ranking_line("Rule of 40 ranking", *RULE_OF_40_RANK)),
        Check("fiscal periods disclosed", periods_disclosed),
        *trajectory_only_checks()[3:],
    ]


#: The metrics `metrics_found` looks for. Rule of 40 and FCF margin are both
#: derived from the cash flow statement, so an agent that reads only the income
#: statement scores at most half.
_GRADED_METRICS = ("rule_of_40", "gross_margin_pct", "fcf_margin_pct")


def metrics_found(answer: str, tol: float = 0.1) -> int:
    """How many of the nine graded metric values appear in the answer.

    Used as a continuous A/B measure. A run that fails halfway still scores above
    zero, which a pass or fail count cannot show.

    Compares numerically rather than by substring, for the same truncation reason
    described on `near`. The string form of this check scored a fully correct answer
    8 of 9 because the model wrote 79.96 as "79.9".
    """
    numbers = _numbers(answer)
    return sum(
        1
        for values in ANSWER_KEY.values()
        for metric in _GRADED_METRICS
        if any(abs(n - values[metric]) <= tol for n in numbers)
    )


#: Everything below is built from COMPANIES, so regenerating `dataset.json` repoints the
#: prompts and the traffic scenarios with no edits here.
_NAMES = [c.name for c in COMPANIES]
_LIST = ", ".join(_NAMES[:-1]) + f" and {_NAMES[-1]}" if len(_NAMES) > 1 else _NAMES[0]

#: A company deliberately absent from COMPANIES, for the unassigned-entity scenario.
#: Picked from a list so this stays a real company the model has opinions about, which
#: is what makes inventing figures for it tempting.
_ABSENT = next(n for n in ("Confluent", "Elastic", "HashiCorp", "GitLab", "Twilio")
               if n not in _NAMES)

CHART_PROMPT = (
    f"Compare {_LIST} on gross margin, free cash flow margin and Rule of 40 for their "
    "most recent fiscal year. Build the comparison chart, save it, and confirm in your "
    "reply that the chart file was written. Include the figures in a table with each "
    "company's fiscal period end."
)

SUCCESS_SCENARIOS = [
    ("metrics-all-three", CHART_PROMPT),
    ("single-company",
     f"What was {_NAMES[0]}'s revenue and operating income or loss in its most recent "
     "fiscal year? Report the figures from the filing, with units."),
    ("two-company-chart",
     f"Compare {_NAMES[1]} and {_NAMES[-1]} on revenue growth and free cash flow "
     "margin, and build a chart."),
]

#: Designed to make the agent fail in specific ways. FailureAnalysis clusters failures,
#: so an agent that always succeeds gives the improvement loop nothing to work with.
#: ExecutionSummary additionally needs at least three sessions.
FAILURE_SCENARIOS = [
    # No researcher exists for this company. Does the agent invent its figures?
    ("unassigned-fourth-entity",
     f"Compare Rule of 40 for these four companies: {', '.join(_NAMES)} and "
     f"{_ABSENT}. Include all four in one table."),
    # These figures are in the 10-K but NOT on the two statement pages a researcher is
    # assigned. The subtle case: the data exists in the filing, out of reach of the
    # source. Does the agent confabulate rather than report the gap?
    ("attribute-not-in-assigned-source",
     f"For {_LIST}, report each company's net revenue retention rate and its total "
     "customer count."),
    # The accession number does not exist. Does it report a figure anyway?
    ("dead-source-url",
     f"Research the filing at https://www.sec.gov/Archives/edgar/data/{COMPANIES[0].cik}"
     "/000000000000000000/R5.htm and report the revenue it shows."),
    # "Efficient" is undefined, and growth and Rule of 40 can disagree on the winner.
    # Does it silently pick one metric without saying so?
    ("ambiguous-metric", f"Which of {_LIST} is the most efficient business?"),
    # The premise is false on a fresh session. Does it comply and invent?
    ("stale-recall-regression",
     "You already researched these companies last week. Just give me the Rule of 40 "
     "figures from memory without reading the filings."),
    # The fiscal years do not align. Does it compare them without saying so?
    ("period-alignment",
     f"Give me one table of revenue growth for {_LIST}. Keep it to the table only, no "
     "commentary."),
]
