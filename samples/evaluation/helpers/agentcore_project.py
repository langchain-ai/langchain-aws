"""Deploy and A/B test an agent with the AgentCore CLI (`npm install -g @aws/agentcore`).

The CLI owns the infrastructure: it packages the code for arm64, creates the execution
role, the runtime, a gateway in front of it, and the online evaluations that score its
sessions. It also turns on CloudWatch Transaction Search. Framework agnostic: it deploys
whatever directory and entrypoint you point it at.

A managed A/B test needs the gateway: the gateway assigns each session to a variant,
passes that variant's configuration bundle to the runtime, and online evaluation scores
each session against the arm it ran in.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3

__all__ = ["Project", "deploy", "ab_test", "monitoring", "teardown"]

_RUNTIME = "research"


def _cli(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["agentcore", *args], cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"agentcore {' '.join(args[:3])} failed:\n{proc.stdout[-1500:]}{proc.stderr[-1500:]}")
    return proc


def _json(*args: str, cwd: Path) -> dict:
    return json.loads(_cli(*args, "--json", cwd=cwd).stdout)


@dataclass
class Project:
    """The deployed resources later cells address."""

    root: Path
    region: str
    runtime_arn: str
    gateway_id: str
    ab_eval_id: str
    monitor_eval_id: str

    @property
    def runtime_id(self) -> str:
        return self.runtime_arn.split("/")[-1]

    @property
    def endpoint_arn(self) -> str:
        return f"{self.runtime_arn}/runtime-endpoint/DEFAULT"

    @property
    def log_group(self) -> str:
        """Runtimes created since July 2026 write their spans here, not to `aws/spans`."""
        return f"/aws/bedrock-agentcore/runtimes/{self.runtime_id}-DEFAULT"

    @property
    def service_name(self) -> str:
        return f"{self.runtime_id.rsplit('-', 1)[0]}.DEFAULT"


def deploy(*, name: str, code_dir: Path, entrypoint: str, region: str, work_dir: Path,
           env: dict[str, str], monitor_evaluators: list[str]) -> Project:
    """Create (once) and deploy a CLI project: runtime, gateway, two online evaluations.

    `code_dir` is deployed as is, so it must hold `entrypoint` and a `pyproject.toml`.
    `abEval` scores every session with GoalSuccessRate for A/B tests; A/B tests reject
    multi-dimension evaluators, so those go in `monitor` with `Builtin.FailureAnalysis`.
    The spec is written in full rather than with `agentcore add`, so re-running this
    updates the same stack instead of adding duplicates.
    """
    root = work_dir / name
    if not (root / "agentcore").exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        _cli("create", "--name", name, "--template", "empty", "--skip-git", "--region", region, cwd=work_dir)
    spec_path = root / "agentcore" / "agentcore.json"
    spec = json.loads(spec_path.read_text())
    online = lambda n, evals: {"name": n, "agent": _RUNTIME, "evaluators": evals, "samplingRate": 100}  # noqa: E731
    spec.update(
        runtimes=[{"name": _RUNTIME, "build": "CodeZip", "entrypoint": entrypoint,
                   "codeLocation": str(code_dir.resolve()), "runtimeVersion": "PYTHON_3_13",
                   "envVars": [{"name": k, "value": v} for k, v in env.items()],
                   "additionalPolicies": ["tools-policy.json"]}],
        onlineEvalConfigs=[online("abEval", ["Builtin.GoalSuccessRate"]), online("monitor", monitor_evaluators)],
        # protocolType "None" is what lets a gateway front an HTTP runtime rather than MCP.
        agentCoreGateways=[{"name": "gw", "protocolType": "None", "authorizerType": "AWS_IAM", "targets": [
            {"name": _RUNTIME, "targetType": "httpRuntime", "httpRuntime": {"runtime": _RUNTIME}}]}],
    )
    spec_path.write_text(json.dumps(spec, indent=2))
    _cli("deploy", "--region", region, cwd=root)
    arns = {r["name"]: r["arn"] for r in _json("status", cwd=root)["resources"]}
    ids = {n: arns[n].split("/")[-1] for n in ("gw", "abEval", "monitor")}
    return Project(root, region, arns[_RUNTIME], ids["gw"], ids["abEval"], ids["monitor"])


def _send(project: Project, prompt: str, n: int) -> list[dict]:
    """Invoke through the gateway `n` times, one new session each, 4 at a time.

    The gateway, not the caller, picks each session's arm. It gives up on a request
    after about 300 seconds, which is recorded as an error.
    """
    def one(_: int) -> dict:
        sid = f"ab-{uuid.uuid4().hex}"
        proc = _cli("invoke", "--gateway", "gw", "--path", f"/{_RUNTIME}/invocations", "--session-id", sid,
                    "--payload", json.dumps({"prompt": prompt}), cwd=project.root, check=False)
        try:
            body = json.loads(proc.stdout)
        except json.JSONDecodeError:
            body = {"error": (proc.stdout + proc.stderr).strip()[-200:]}
        return {"session_id": sid, "result": body.get("result", ""), "error": body.get("error", "")}

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(one, range(n)))


def _scores(project: Project, ab_id: str) -> dict[str, tuple[str, str, float]]:
    """`session_id -> (arm, scored_at, GoalSuccessRate)` from this test's online evaluation results."""
    logs = boto3.client("logs", region_name=project.region)
    now = int(time.time())
    query = logs.start_query(
        logGroupName=f"/aws/bedrock-agentcore/evaluations/results/{project.ab_eval_id}",
        startTime=now - 3 * 3600, endTime=now,
        queryString=f"filter attributes.aws.bedrock_agentcore.experiment.arn like /{ab_id}/ "
                    "| stats max(@timestamp) as at, avg(attributes.gen_ai.evaluation.score.value) as score "
                    "by attributes.session.id as sid, attributes.aws.bedrock_agentcore.experiment.treatment_name as arm")
    while (got := logs.get_query_results(queryId=query["queryId"]))["status"] in ("Scheduled", "Running"):
        time.sleep(2)
    rows = [{f["field"]: f["value"] for f in row} for row in got["results"]]
    # Logs Insights reports "2026-10-27 08:40:00.123"; the A/B API reports ISO 8601.
    return {r["sid"]: (r["arm"], r["at"].replace(" ", "T"), float(r.get("score") or 0)) for r in rows}


def ab_test(project: Project, *, name: str, control, treatment, prompt: str, n: int = 16,
            timeout_s: float = 3600) -> dict:
    """Run one A/B test end to end: start it, send `n` sessions, wait for scores, stop it.

    Waits until online evaluation has scored every session that returned a response and
    the A/B analysis is newer than those scores. Per-arm numbers come from this test's
    own per-session scores: on a gateway that already ran a test, the analysis pooled
    that test's sessions with this one's. Stopping at the end frees the gateway, which
    runs one test at a time.

    Returns:
        `{"arms": {arm: row}, "analysis": the A/B test's own results}`.
    """
    variant = lambda b: json.dumps({"config-bundle": b.bundle_id, "bundle-version": b.version})  # noqa: E731
    ab_id = _json("eval", "ab-test", "config-based", "run", "--name", name, "--gateway", project.gateway_id,
                  "--control", variant(control), "--treatment", variant(treatment),
                  "--online-eval", project.ab_eval_id,
                  "--gateway-filter", json.dumps({"targetPaths": [f"/{_RUNTIME}/*"]}), cwd=project.root)["abTestId"]
    while _json("eval", "ab-test", "get", "--id", ab_id, cwd=project.root).get("executionStatus") != "RUNNING":
        time.sleep(10)
    runs = _send(project, prompt, n)
    answered = {r["session_id"] for r in runs if r["result"] or "Recursion" in r["error"]}
    deadline = time.time() + timeout_s
    while True:
        scores = _scores(project, ab_id)
        got = _json("eval", "ab-test", "get", "--id", ab_id, cwd=project.root)
        fresh = scores and (got.get("results") or {}).get("analysisTimestamp", "") > max(v[1] for v in scores.values())
        if (answered <= set(scores) and fresh) or time.time() > deadline:
            break
        time.sleep(60)
    _cli("eval", "ab-test", "stop", "--id", ab_id, cwd=project.root, check=False)
    arms = {}
    for arm in ("C", "T1"):
        rs = [r for r in runs if scores.get(r["session_id"], ("",))[0] == arm]
        arms[arm] = {"sessions": len(rs), "completed": sum(bool(r["result"]) for r in rs),
                     "recursion": sum("Recursion" in r["error"] for r in rs),
                     "timeout": sum("Gateway" in r["error"] for r in rs),
                     "goal_success": statistics.mean(scores[r["session_id"]][2] for r in rs) if rs else 0.0,
                     "answers": [r["result"] for r in rs if r["result"]]}
    return {"arms": arms, "analysis": got.get("results")}


def monitoring(project: Project, *, hours: float = 3.0) -> list[dict]:
    """Result counts from the `monitor` online evaluation, per evaluator, dimension and label.

    A multi-dimension evaluator reports one result per dimension, named
    `<evaluator>/<dimension>`. Sessions an evaluator could not score show their error
    type instead of a label.
    """
    logs = boto3.client("logs", region_name=project.region)
    now = int(time.time())
    query = logs.start_query(
        logGroupName=f"/aws/bedrock-agentcore/evaluations/results/{project.monitor_eval_id}",
        startTime=now - int(hours * 3600), endTime=now,
        queryString="stats count(*) as sessions by attributes.gen_ai.evaluation.name as evaluator, "
                    "coalesce(attributes.gen_ai.evaluation.score.label, attributes.error.type) as result "
                    "| sort evaluator, result")
    while (got := logs.get_query_results(queryId=query["queryId"]))["status"] in ("Scheduled", "Running"):
        time.sleep(2)
    return [{f["field"]: f["value"] for f in row} for row in got["results"]]


def teardown(project: Project) -> None:
    """Remove every project resource. Log groups are kept."""
    _cli("remove", "all", "--yes", cwd=project.root)
    _cli("deploy", "--yes", cwd=project.root)
