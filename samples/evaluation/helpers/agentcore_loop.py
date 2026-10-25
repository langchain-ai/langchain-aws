"""The AgentCore improvement loop: diagnosis, recommendation, config bundles, A/B.

Thin wrappers over the `bedrock-agentcore` SDK so a notebook can run the loop in a few
lines each. Framework agnostic: everything operates on spans and prompts.

The loop is: score recent sessions and cluster their failures, get a prompt rewrite
from the same sessions, apply it with a configuration bundle, and compare arms.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from bedrock_agentcore.config_bundle import ConfigBundleClient
from bedrock_agentcore.evaluation import EvaluationClient, RecommendationClient
from bedrock_agentcore.runtime import WaitConfig

__all__ = [
    "diagnose",
    "root_causes",
    "recommend_prompt",
    "Bundle",
    "make_bundle",
]

_GOAL_SUCCESS = "arn:aws:bedrock-agentcore:::evaluator/Builtin.GoalSuccessRate"
_WAIT = WaitConfig(max_wait=30 * 60, poll_interval=30)


def diagnose(
    *,
    region: str,
    service_name: str,
    log_groups: list[str],
    session_ids: list[str],
    lookback_hours: float = 2.0,
) -> dict:
    """Score sessions and cluster their failures, in one batch evaluation.

    `Builtin.FailureAnalysis` runs beside `Builtin.GoalSuccessRate` and returns
    root causes under `insightResults`. Its score is 1.0 when it finds failures, so
    for it higher is worse. Pass both `aws/spans` and the runtime log group: spans
    reference log events, so either one alone gives incomplete traces.

    Pass `session_ids`. With only a time range, the service found 2 of 9 sessions
    that were all in CloudWatch, and the skipped ones were not reported as failed.

    Returns:
        The GetBatchEvaluation response. Its `batchEvaluationArn` is what chains
        diagnosis into `recommend_prompt`.
    """
    now = datetime.now(timezone.utc)
    return EvaluationClient(region_name=region).start_batch_evaluation_and_wait(
        wait_config=_WAIT,
        batch_evaluation_name=f"diagnose_{int(time.time())}",
        evaluators=[{"evaluatorId": "Builtin.GoalSuccessRate"},
                    {"evaluatorId": "Builtin.FailureAnalysis"}],
        data_source_config={"cloudWatchLogs": {
            "serviceNames": [service_name],
            "logGroupNames": log_groups,
            "filterConfig": {"sessionIds": session_ids, "timeRange": {
                "startTime": now - timedelta(hours=lookback_hours), "endTime": now}},
        }},
        client_token=str(uuid.uuid4()),
    )


def root_causes(detail: dict) -> list[dict]:
    """Flatten `insightResults` into one row per root cause."""
    return [
        {"insight": i["name"], "sessions": i.get("affectedSessionCount", 0),
         "root_cause": rc["name"], "suggested_fix": rc.get("suggestedFix", "")}
        for i in detail.get("insightResults") or []
        for rc in i.get("rootCauses") or []
    ]


def recommend_prompt(
    *,
    region: str,
    current_prompt: str,
    batch_evaluation_arn: str,
    evaluator_arn: str = _GOAL_SUCCESS,
) -> tuple[str | None, str]:
    """Ask AgentCore to rewrite a system prompt, targeting `evaluator_arn`.

    The service reads the sessions the batch evaluation scored, finds where the
    target evaluator scores them poorly, and returns a revised prompt with an
    explanation of what it changed.

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
    result = RecommendationClient(region_name=region).start_recommendation_and_wait(
        wait_config=_WAIT,
        name=f"SpRec{int(time.time())}",
        type="SYSTEM_PROMPT_RECOMMENDATION",
        recommendation_config={"systemPromptRecommendationConfig": {
            "systemPrompt": {"text": current_prompt},
            "agentTraces": {"batchEvaluation": {"batchEvaluationArn": batch_evaluation_arn}},
            "evaluationConfig": {"evaluators": [{"evaluatorArn": evaluator_arn}]},
        }},
        client_token=str(uuid.uuid4()),
    )
    sp = result["recommendationResult"]["systemPromptRecommendationResult"]
    prompt = sp.get("recommendedSystemPrompt")
    if prompt and prompt.strip() == current_prompt.strip():
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
