"""The AgentCore improvement loop: recommendation, config bundles, A/B.

Thin wrappers over the `bedrock-agentcore` data plane so a notebook can run the loop
in a few lines each. Framework agnostic: everything operates on spans and prompts.

The loop is: get a prompt rewrite from the agent's own traces, apply it with a
configuration bundle, and compare arms.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import boto3
from bedrock_agentcore.config_bundle import ConfigBundleClient

__all__ = [
    "recommend_prompt",
    "Bundle",
    "make_bundle",
]

_GOAL_SUCCESS = "arn:aws:bedrock-agentcore:::evaluator/Builtin.GoalSuccessRate"


def _dp(region: str):
    return boto3.client("bedrock-agentcore", region_name=region)


def _log_group_arns(region: str, log_groups: list[str]) -> list[str]:
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    return [f"arn:aws:logs:{region}:{account}:log-group:{name}" for name in log_groups]


def recommend_prompt(
    *,
    region: str,
    current_prompt: str,
    service_name: str,
    log_groups: list[str],
    lookback_hours: float = 6.0,
    evaluator_arn: str = _GOAL_SUCCESS,
    timeout_minutes: float = 25.0,
) -> tuple[str | None, str]:
    """Ask AgentCore to rewrite a system prompt, targeting `evaluator_arn`.

    The service reads the agent's recent traces straight from CloudWatch, finds where
    the target evaluator scores it poorly, and returns a revised prompt with an
    explanation of what it changed. Pass both `aws/spans` and the runtime log group:
    spans reference log events, so either one alone gives incomplete traces.

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
    now = datetime.now(timezone.utc)
    rec = dp.start_recommendation(
        name=f"SpRec{int(time.time())}",
        type="SYSTEM_PROMPT_RECOMMENDATION",
        recommendationConfig={"systemPromptRecommendationConfig": {
            "systemPrompt": {"text": current_prompt},
            "agentTraces": {"cloudwatchLogs": {
                "logGroupArns": _log_group_arns(region, log_groups),
                "serviceNames": [service_name],
                "startTime": now - timedelta(hours=lookback_hours),
                "endTime": now,
            }},
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
