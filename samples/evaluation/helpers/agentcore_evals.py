"""Span-based evaluation helpers for agents on Amazon Bedrock AgentCore.

The notebooks in this directory import from here so their cells stay short. This
module is the part that ought to live in an SDK; it is kept separate from the
notebooks precisely to make that boundary visible.

## Why spans

Everything here reads OpenTelemetry spans, never framework objects. The attributes
it depends on are the OTel GenAI semantic conventions (`gen_ai.operation.name`,
`gen_ai.tool.name`), which Strands, LangGraph and CrewAI all emit, and it accepts
the three tool-span conventions the AgentCore SDK itself accepts. So the same code
works for a local run and a deployed one, and for any instrumented framework.

## Two sources, one Trajectory

    CloudWatchSpanCollector          deployed runs. Framework agnostic, reads spans.
    trajectory_from_langchain_events local runs. LangChain specific, reads events.

Both produce the same `Trajectory`, so a check written once runs against either.

Why two rather than one: spans are the framework-agnostic path and the one that
matches what a deployed agent emits, so they are what the checks are written
against. The LangChain event adapter exists so notebook 1 can measure a local run
without deploying anything first, and it is deliberately the only
framework-specific thing in this module.

In-process span capture also works. With `deepagents` 0.7.13 and
`opentelemetry-instrumentation-langchain` 0.62.3, a subagent's leaf tools do get
their own `execute_tool` spans and `parentSpanId` gives a correct, stable chain
(`execute_tool` leaf -> `execute_task tools` -> `execute_task <subagent>` ->
`execute_tool task` -> ... -> `invoke_agent`), verified over repeated runs with
three `task()` calls in flight at once. If you already run an OTEL SDK in process,
`trajectory_from_spans` works on those spans directly and the event adapter is
optional.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

__all__ = [
    "ToolCall",
    "Trajectory",
    "trajectory_from_spans",
    "trajectory_from_langchain_events",
    "CloudWatchSpanCollector",
    "Check",
    "check_all",
    "report",
]

# The three tool-span conventions in circulation. Matching all three is what makes
# this framework agnostic; AgentCore's own runner checks the same set.
_TOOL_NAME_KEYS = ("gen_ai.tool.name", "tool.name", "traceloop.entity.name")
_ARG_KEYS = ("gen_ai.tool.call.arguments", "input.value", "traceloop.entity.input")
_RESULT_KEYS = ("gen_ai.tool.call.result", "output.value", "traceloop.entity.output")


def _is_tool_span(attrs: dict) -> bool:
    """Whether a span represents a tool execution, under any convention."""
    return (
        attrs.get("gen_ai.operation.name") == "execute_tool"
        or attrs.get("openinference.span.kind") == "TOOL"
        or attrs.get("traceloop.span.kind") == "tool"
    )


def _first(attrs: dict, keys: Iterable[str]) -> Any:
    for k in keys:
        if attrs.get(k):
            return attrs[k]
    return None


@dataclass
class ToolCall:
    """One tool invocation, wherever it happened in the agent graph."""

    name: str
    args: dict
    t_start: float
    t_end: float
    span_id: str = ""
    parent_span_id: str = ""
    output: str | None = None

    @property
    def duration_s(self) -> float:
        return self.t_end - self.t_start


@dataclass
class Trajectory:
    """A queryable view of one agent run, reconstructed from spans."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    duration_s: float = 0.0
    model_calls: int = 0
    delegation_tools: frozenset[str] = frozenset({"task"})
    """Tools that hand work to a subagent. Calls beneath them are nested."""

    _span_parents: dict[str, str] = field(default_factory=dict)
    """`spanId -> parentSpanId` for EVERY span, not only tool spans, so ancestry can
    be walked through the graph and middleware spans that sit between two tools."""

    @property
    def _by_span_id(self) -> dict[str, ToolCall]:
        return {c.span_id: c for c in self.tool_calls if c.span_id}

    def is_nested(self, call: ToolCall) -> bool:
        """Whether `call` ran inside a delegated subagent.

        Uses span ancestry when the source provides parent ids: walk up from this
        call and report whether a delegation call is among its ancestors. Ancestry
        rather than depth, because the number of intermediate middleware spans
        between a tool and its parent tool is an implementation detail, while
        "is there a `task` above me" is the actual question.

        Falls back to interval containment when parent ids are absent, which is the
        case for `trajectory_from_langchain_events`, since LangChain events carry no
        span ids. The fallback can misattribute a call that merely overlaps a
        delegation, so prefer a span source when nesting matters.
        """
        if call.name in self.delegation_tools:
            return False
        if call.parent_span_id and self._by_span_id:
            return self._has_delegation_ancestor(call)
        return any(
            t.name in self.delegation_tools
            and t is not call
            and t.t_start <= call.t_start <= t.t_end
            for t in self.tool_calls
        )

    def _has_delegation_ancestor(self, call: ToolCall) -> bool:
        """Walk the recorded parent chain looking for a delegation call."""
        seen: set[str] = set()
        parent = call.parent_span_id
        while parent and parent not in seen:
            seen.add(parent)
            ancestor = self._by_span_id.get(parent)
            if ancestor is not None:
                if ancestor.name in self.delegation_tools:
                    return True
                parent = ancestor.parent_span_id
            else:
                # A non-tool span (a graph node or middleware). Keep climbing using
                # the raw span parentage recorded alongside the tool calls.
                parent = self._span_parents.get(parent, "")
        return False

    def calls(self, name: str | None = None, *, nested: bool | None = None) -> list[ToolCall]:
        """Filter calls by name and nesting. `nested=None` means any level."""
        out = [c for c in self.tool_calls if name is None or c.name == name]
        if nested is True:
            return [c for c in out if self.is_nested(c)]
        if nested is False:
            return [c for c in out if not self.is_nested(c)]
        return out

    def max_concurrent(self, name: str) -> int:
        """Peak number of simultaneously in-flight calls to `name`.

        This is the honest test for parallelism. Counting calls per model turn is
        not: a coordinator forced to delegate strictly one at a time still emits
        three `task` calls, and this returns 1 for that run.
        """
        calls = self.calls(name)
        if not calls:
            return 0
        points: list[tuple[float, int]] = []
        for c in calls:
            points.append((c.t_start, 1))
            points.append((c.t_end, -1))
        # Ends sort before starts at equal timestamps, so touching intervals are
        # not counted as overlapping.
        points.sort(key=lambda p: (p[0], p[1]))
        peak = running = 0
        for _, delta in points:
            running += delta
            peak = max(peak, running)
        return peak

    def timeline(self) -> str:
        """Human-readable timeline, nested calls indented."""
        lines = [f"{self.duration_s:.1f}s, {len(self.tool_calls)} tool calls"]
        for c in self.tool_calls:
            indent = "      " if self.is_nested(c) else "  "
            args = ", ".join(f"{k}={str(v)[:34]}" for k, v in list(c.args.items())[:1])
            lines.append(f"{indent}[{c.t_start:6.2f}s +{c.duration_s:5.2f}s] {c.name}({args})")
        return "\n".join(lines)


def trajectory_from_spans(spans: list[dict]) -> Trajectory:
    """Build a `Trajectory` from span dicts.

    Accepts CloudWatch span documents, or any dict with `startTimeUnixNano`,
    `endTimeUnixNano` and `attributes`. Timestamps are normalized to seconds from
    the first span, so concurrency maths is source independent.

    Spans are sorted by START time. Log order is not causal order: a parent tool
    span closes only when its subagent finishes, so it is recorded after its own
    children.
    """
    traj = Trajectory()
    timed = [s for s in spans if int(s.get("startTimeUnixNano") or 0) > 0]
    if not timed:
        return traj

    origin = min(int(s["startTimeUnixNano"]) for s in timed)
    latest = origin

    for doc in timed:
        attrs = doc.get("attributes") or {}
        start = int(doc["startTimeUnixNano"])
        end = int(doc.get("endTimeUnixNano") or 0) or start
        latest = max(latest, end)

        span_id = str(doc.get("spanId") or "")
        parent_id = str(doc.get("parentSpanId") or doc.get("parentId") or "")
        if span_id:
            traj._span_parents[span_id] = parent_id

        if attrs.get("gen_ai.operation.name") == "chat":
            traj.model_calls += 1
        if not _is_tool_span(attrs):
            continue

        name = _first(attrs, _TOOL_NAME_KEYS)
        if not name:
            continue

        raw = _first(attrs, _ARG_KEYS)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                args = parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                args = {"raw": raw[:300]}
        else:
            args = raw if isinstance(raw, dict) else {}

        result = _first(attrs, _RESULT_KEYS)
        traj.tool_calls.append(
            ToolCall(
                name=str(name),
                args=args,
                t_start=(start - origin) / 1e9,
                t_end=(end - origin) / 1e9,
                span_id=span_id,
                parent_span_id=parent_id,
                output=str(result)[:4000] if result is not None else None,
            )
        )

    traj.tool_calls.sort(key=lambda c: c.t_start)
    traj.duration_s = (latest - origin) / 1e9
    return traj


async def trajectory_from_langchain_events(
    agent: Any,
    *,
    query: str,
    thread_id: str,
    recursion_limit: int = 60,
    delegation_tools: frozenset[str] = frozenset({"task"}),
) -> tuple[Trajectory, str]:
    """Run a LangChain agent locally and record its trajectory from events.

    THE ONLY FRAMEWORK-SPECIFIC FUNCTION HERE. It exists because in-process span
    capture does not yield per-tool spans for a deep agent (see module docstring).
    In production use `CloudWatchSpanCollector` instead, which is framework
    agnostic and produces the same `Trajectory`.

    `astream_events` is used rather than `ainvoke` because a deep agent's final
    state contains only the coordinator's own tool calls. Subagent-internal calls
    never appear there, so any check about subagent behavior read off `messages`
    is vacuously true.

    `recursion_limit` defaults well above LangGraph's 25: a multi-subagent fan-out
    that retries anything trips that ceiling and dies mid-run.

    Returns:
        `(trajectory, final_answer)`. On an exception the partial trajectory is
        returned, because a partial trajectory still shows where the run broke.
    """
    traj = Trajectory(delegation_tools=delegation_tools)
    open_calls: dict[str, ToolCall] = {}
    final_state: dict | None = None
    t0 = time.time()

    try:
        async for ev in agent.astream_events(
            {"messages": [{"role": "user", "content": query}]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit,
            },
            version="v2",
        ):
            kind, now = ev.get("event", ""), time.time() - t0
            if kind == "on_tool_start":
                rid = str(ev.get("run_id"))
                call = ToolCall(
                    name=str(ev.get("name")),
                    args=dict(ev.get("data", {}).get("input") or {}),
                    t_start=now,
                    t_end=now,
                    span_id=rid,
                )
                open_calls[rid] = call
                traj.tool_calls.append(call)
            elif kind in ("on_tool_end", "on_tool_error"):
                call = open_calls.pop(str(ev.get("run_id")), None)
                if call is not None:
                    call.t_end = now
                    if kind == "on_tool_end":
                        call.output = str(ev.get("data", {}).get("output"))[:4000]
            elif kind == "on_chat_model_end":
                traj.model_calls += 1
            elif kind == "on_chain_end" and not ev.get("parent_ids"):
                # The root graph is the only chain with no parents.
                out = ev.get("data", {}).get("output")
                if isinstance(out, dict) and "messages" in out:
                    final_state = out
    except Exception as exc:  # noqa: BLE001 - partial trajectory is the diagnostic
        print(f"run raised: {type(exc).__name__}: {exc}")

    traj.duration_s = time.time() - t0
    # Any call still open when the run ended was in flight; measure it to the end
    # rather than leaving it zero length, which would hide real concurrency.
    for call in open_calls.values():
        call.t_end = traj.duration_s

    answer = ""
    if final_state:
        for m in reversed(final_state.get("messages") or []):
            content = getattr(m, "content", None)
            text = (
                content
                if isinstance(content, str)
                else "".join(
                    b.get("text", "")
                    for b in (content or [])
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            )
            if text and text.strip() and not getattr(m, "tool_calls", None):
                answer = text
                break
    return traj, answer


class CloudWatchSpanCollector:
    """Read a deployed session's spans from CloudWatch.

    Defaults to the `aws/spans` log group. That is deliberate: on a
    zip-deployed runtime spans land there even with
    `UNIFIED_TRACES_DESTINATION_ENABLED=true`, and pointing at the agent's own log
    group returns zero spans, which is indistinguishable from an agent that emitted
    nothing.
    """

    def __init__(self, region: str, log_group: str = "aws/spans") -> None:
        import boto3

        self.logs = boto3.client("logs", region_name=region)
        self.log_group = log_group

    def spans(self, session_id: str, *, lookback_hours: float = 6.0) -> list[dict]:
        """All spans carrying `session.id == session_id`."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=lookback_hours)
        kwargs: dict[str, Any] = {
            "logGroupName": self.log_group,
            "startTime": int(start.timestamp() * 1000),
            "endTime": int(end.timestamp() * 1000),
            "filterPattern": f'"{session_id}"',
            "limit": 1000,
        }
        out: list[dict] = []
        while len(out) < 5000:
            resp = self.logs.filter_log_events(**kwargs)
            for ev in resp.get("events", []):
                try:
                    doc = json.loads(ev["message"])
                except (json.JSONDecodeError, KeyError):
                    continue
                if (doc.get("attributes") or {}).get("session.id") == session_id:
                    out.append(doc)
            token = resp.get("nextToken")
            if not token:
                break
            kwargs["nextToken"] = token
        return out

    def wait_for_spans(
        self, session_id: str, *, min_tool_calls: int = 1, max_wait_s: float = 420.0
    ) -> Trajectory:
        """Poll until spans are ingested, then return the trajectory.

        Ingestion lags an invocation by minutes. Reading immediately returns an
        empty trajectory, which looks the same as an agent that made no tool calls.
        """
        deadline = time.time() + max_wait_s
        traj = Trajectory()
        while time.time() < deadline:
            traj = trajectory_from_spans(self.spans(session_id))
            if len(traj.tool_calls) >= min_tool_calls:
                return traj
            time.sleep(30)
        return traj


@dataclass
class Check:
    """One named assertion about a trajectory or an answer.

    Deliberately tiny. A check is a label plus a callable returning
    `(passed, detail)`. Keeping it this small means the notebooks can define a new
    check inline in two lines instead of subclassing anything.
    """

    label: str
    fn: Any
    tier: str = "correctness"
    """`correctness` fails the eval. `efficiency` is recorded but never fails."""

    def run(self, traj: Trajectory, answer: str = "") -> tuple[bool, str]:
        return self.fn(traj, answer)


def check_all(
    checks: list[Check], traj: Trajectory, answer: str = ""
) -> tuple[bool, list[tuple[Check, bool, str]]]:
    """Run every check. Returns `(correctness_passed, results)`.

    Efficiency checks are evaluated and reported but do not affect the verdict, so
    a slower-but-correct run is distinguishable from a wrong one.
    """
    results = [(c, *c.run(traj, answer)) for c in checks]
    passed = all(ok for c, ok, _ in results if c.tier == "correctness")
    return passed, results


def report(
    name: str, traj: Trajectory, results: list[tuple[Check, bool, str]], passed: bool
) -> None:
    """Print one eval's verdict, then its checks, then the timeline on failure."""
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"      {traj.duration_s:.1f}s, {len(traj.tool_calls)} tool calls, "
          f"{traj.model_calls} model calls")
    for check, ok, detail in results:
        mark = "ok" if ok else ("x " if check.tier == "correctness" else "~ ")
        line = f"      {mark} {check.label}"
        # Detail only on failure. A check function is allowed to return its message
        # unconditionally, which is the natural way to write one, so filtering here
        # keeps a passing report from reading like a list of problems.
        if detail and not ok:
            line += f": {detail}"
        print(line)
    if not passed:
        print("\n" + traj.timeline())
