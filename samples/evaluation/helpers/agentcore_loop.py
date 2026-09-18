"""The AgentCore improvement loop: insights, recommendation, config bundles, A/B.

Thin wrappers over the `bedrock-agentcore` data plane so a notebook can run the loop
in a few lines each. Framework agnostic: everything operates on spans and prompts.

The loop is: cluster failures, get a prompt rewrite targeting them, apply it with a
configuration bundle, and compare arms.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from bedrock_agentcore.config_bundle import ConfigBundleClient

__all__ = [
    "INSIGHT_IDS",
    "run_insights",
    "root_causes",
    "recommend_prompt",
    "Bundle",
    "make_bundle",
]

INSIGHT_IDS = [
    "Builtin.Insight.FailureAnalysis",
    "Builtin.Insight.UserIntent",
    "Builtin.Insight.ExecutionSummary",
]

_TERMINAL = {"COMPLETED", "FAILED", "STOPPED", "COMPLETED_WITH_ERRORS"}
_GOAL_SUCCESS = "arn:aws:bedrock-agentcore:::evaluator/Builtin.GoalSuccessRate"


def _dp(region: str):
    return boto3.client("bedrock-agentcore", region_name=region)


def run_insights(
    *,
    region: str,
    service_name: str,
    log_groups: list[str],
    lookback_hours: float = 6.0,
    timeout_minutes: float = 35.0,
    insight_ids: list[str] | None = None,
) -> tuple[dict, str]:
    """Cluster failures across recent sessions.

    Two constraints the API enforces and it is easy to trip:
    `insights` and `evaluators` are mutually exclusive in one job, so diagnosis and
    scoring are separate jobs. And BOTH `aws/spans` and the runtime log group must be
    supplied, because spans reference log events and omitting the runtime group
    yields incomplete results.

    Returns:
        `(detail, arn)`. The ARN is what chains diagnosis into `recommend_prompt`.
    """
    dp = _dp(region)
    now = datetime.now(timezone.utc)
    job = dp.start_batch_evaluation(
        batchEvaluationName=f"insights_{int(time.time())}",
        insights=[{"insightId": i} for i in (insight_ids or INSIGHT_IDS)],
        dataSourceConfig={"cloudWatchLogs": {
            "serviceNames": [service_name],
            "logGroupNames": log_groups,
            "filterConfig": {"timeRange": {
                "startTime": (now - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "endTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }},
        }},
        clientToken=str(uuid.uuid4()),
    )
    job_id, arn = job["batchEvaluationId"], job["batchEvaluationArn"]
    print(f"  started {job_id}")

    deadline = time.time() + timeout_minutes * 60
    detail: dict = {}
    while time.time() < deadline:
        detail = dp.get_batch_evaluation(batchEvaluationId=job_id)
        if detail.get("status") in _TERMINAL:
            break
        time.sleep(45)
    print(f"  status  {detail.get('status')}")
    return detail, arn


def root_causes(detail: dict) -> list[dict]:
    """Flatten `failureAnalysisResult` into a list of root causes.

    Results arrive under `failureAnalysisResult`, `userIntentResult` and
    `executionSummaryResult` at the TOP level of the response, not nested under an
    `insights` list.
    """
    out = []
    for failure in (detail.get("failureAnalysisResult") or {}).get("failures", []):
        for sub in failure.get("subCategories", []):
            for rc in sub.get("rootCauses", []):
                first = (rc.get("affectedSessions") or [{}])[0]
                signals = (first.get("failureSpans") or [{}])[0].get("signals") or [{}]
                out.append({
                    "category": failure["name"],
                    "root_cause": rc["name"],
                    "sessions": rc.get("affectedSessionCount", 0),
                    "fix_type": first.get("fixType"),
                    "explanation": first.get("explanation", ""),
                    "signal": signals[0].get("category"),
                    "recommendation": rc.get("recommendation", ""),
                })
    return out


def clusters(detail: dict) -> dict[str, list[str]]:
    """UserIntent and ExecutionSummary cluster names."""
    return {
        field: [c["name"] for c in (detail.get(key) or {}).get(field, [])]
        for key, field in (("userIntentResult", "userIntents"),
                           ("executionSummaryResult", "executionSummaries"))
    }


def recommend_prompt(
    *,
    region: str,
    current_prompt: str,
    insights_arn: str,
    evaluator_arn: str = _GOAL_SUCCESS,
    timeout_minutes: float = 25.0,
) -> tuple[str | None, str]:
    """Ask AgentCore to rewrite a system prompt, targeting `evaluator_arn`.

    The result is read from exactly one path and there is deliberately no fallback
    to "any long string in the response": `recommendationConfig` echoes back the
    prompt you submitted and is ordered BEFORE `recommendationResult`, so a generous
    search returns your own input. That presents as a recommendation identical to the
    baseline and produces an A/B whose two arms run the same prompt.

    Note the choice of `evaluator_arn` does not just grade the agent, it decides which
    direction the fix points. Targeting GoalSuccessRate pushes an agent toward acting
    without confirmation, because asking scores as less complete than answering.

    Returns:
        `(recommended_prompt_or_None, explanation)`.
    """
    dp = _dp(region)
    rec = dp.start_recommendation(
        name=f"SpRec{int(time.time())}",
        type="SYSTEM_PROMPT_RECOMMENDATION",
        recommendationConfig={"systemPromptRecommendationConfig": {
            "systemPrompt": {"text": current_prompt},
            "agentTraces": {"batchEvaluation": {"batchEvaluationArn": insights_arn}},
            "evaluationConfig": {"evaluators": [{"evaluatorArn": evaluator_arn}]},
        }},
        clientToken=str(uuid.uuid4()),
    )
    rec_id = rec["recommendationId"]
    print(f"  started {rec_id}")

    deadline = time.time() + timeout_minutes * 60
    result: dict = {}
    while time.time() < deadline:
        result = dp.get_recommendation(recommendationId=rec_id)
        if result.get("status") in ("COMPLETED", "FAILED"):
            break
        time.sleep(30)
    print(f"  status  {result.get('status')}")

    sp = (result.get("recommendationResult") or {}).get("systemPromptRecommendationResult") or {}
    prompt = sp.get("recommendedSystemPrompt")
    echoed = ((result.get("recommendationConfig") or {})
              .get("systemPromptRecommendationConfig", {})
              .get("systemPrompt", {}).get("text", ""))
    if prompt and echoed and prompt.strip() == echoed.strip():
        print("  WARNING: recommendation is identical to the submitted prompt")
    return prompt, str(sp.get("explanation") or "")


@dataclass
class Bundle:
    """A configuration bundle version, ready to attach to an invocation."""

    bundle_id: str
    verified_chars: int
    baggage: str


def make_bundle(*, region: str, runtime_arn: str, name: str, prompt: str, note: str) -> Bundle:
    """Create a configuration bundle carrying `prompt`, verifying it reads back.

    The verification is not ceremony. A bundle that does not round-trip is a silent
    no-op: both A/B arms would run the baseline prompt and the experiment would
    confidently report no difference.
    """
    ctrl = ConfigBundleClient(region_name=region)
    resp = ctrl.create_configuration_bundle(
        bundleName=name,
        description=note,
        components={runtime_arn: {"configuration": {"system_prompt": prompt}}},
        commitMessage=note,
        clientToken=str(uuid.uuid4()),
    )
    read = ctrl.get_configuration_bundle(bundleId=resp["bundleId"])
    stored = read["components"][runtime_arn]["configuration"].get("system_prompt", "")
    if stored.strip() != prompt.strip():
        raise RuntimeError(
            f"{name} did not round-trip: {len(stored)} stored vs {len(prompt)} sent"
        )
    return Bundle(
        bundle_id=resp["bundleId"],
        verified_chars=len(stored),
        baggage=(f"aws.agentcore.configbundle_arn={resp['bundleArn']},"
                 f"aws.agentcore.configbundle_version={resp['versionId']}"),
    )


def delete_bundles(*, region: str, bundle_ids: list[str]) -> None:
    """Delete configuration bundles, reporting rather than raising."""
    ctrl = ConfigBundleClient(region_name=region)
    for bid in bundle_ids:
        try:
            ctrl.delete_configuration_bundle(bundleId=bid)
            print(f"  deleted bundle {bid}")
        except Exception as exc:  # noqa: BLE001
            print(f"  bundle {bid}: {exc}")
