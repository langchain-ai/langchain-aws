"""The AgentCore improvement loop: diagnosis, recommendation, configuration bundles.

Thin wrappers over the `bedrock-agentcore` SDK so a notebook can run the loop in a few
lines each. Framework agnostic: everything operates on spans and prompts.

The loop is: run a dataset against the agent and cluster its failures, then have AgentCore
rewrite the prompt held in a configuration bundle, which yields a new bundle version
to A/B against the old one (see `agentcore_project.py`).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from bedrock_agentcore.config_bundle import ConfigBundleClient
from bedrock_agentcore.evaluation import DatasetClient, EvaluationClient, RecommendationClient
from bedrock_agentcore.runtime import WaitConfig

__all__ = [
    "publish_dataset",
    "diagnose",
    "root_causes",
    "recommend_prompt",
    "Bundle",
    "make_bundle",
    "delete_bundles",
]

_GOAL_SUCCESS = "arn:aws:bedrock-agentcore:::evaluator/Builtin.GoalSuccessRate"
_WAIT = WaitConfig(max_wait=30 * 60, poll_interval=30)


def publish_dataset(*, region: str, name: str, scenarios: list[tuple[str, str]]) -> str:
    """Publish `(scenario_id, prompt)` pairs as a one-turn evaluation dataset. Returns its id."""
    client = DatasetClient(region_name=region)
    dataset = client.create_dataset_and_wait(
        dataset_name=name, schema_type="AGENTCORE_EVALUATION_PREDEFINED_V1",
        source={"inlineExamples": {"examples": [
            {"scenario_id": sid, "turns": [{"input": prompt}]} for sid, prompt in scenarios]}})
    client.create_dataset_version_and_wait(dataset_id=dataset["datasetId"])
    return dataset["datasetId"]


def diagnose(*, region: str, endpoint_arn: str, service_name: str, log_group: str, dataset_id: str) -> dict:
    """Run every dataset scenario against the runtime, then score and cluster the failures.

    AgentCore invokes the agent itself, one new session per scenario, and evaluates
    exactly the sessions it created, so there is no traffic loop to write and no
    session discovery to go wrong. `Builtin.FailureAnalysis` runs beside
    `Builtin.GoalSuccessRate` and returns root causes under `insightResults`. Its
    score is 1.0 when it finds failures, so for it higher is worse.

    Returns:
        The GetBatchEvaluation response. Its `batchEvaluationArn` is what chains
        diagnosis into `recommend_prompt`.
    """
    return EvaluationClient(region_name=region).start_batch_evaluation_and_wait(
        wait_config=WaitConfig(max_wait=90 * 60, poll_interval=30),
        batch_evaluation_name=f"diagnose_{int(time.time())}",
        evaluators=[{"evaluatorId": "Builtin.GoalSuccessRate"},
                    {"evaluatorId": "Builtin.FailureAnalysis"}],
        data_source_config={"cloudWatchLogs": {"serviceNames": [service_name], "logGroupNames": [log_group]}},
        dataset_execution_config={"dataset": {"datasetId": dataset_id},
                                  "agentInvocation": {"agentCoreRuntime": {"agentRuntimeEndpointArn": endpoint_arn}}},
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


@dataclass
class Bundle:
    """One version of a configuration bundle carrying a system prompt."""

    bundle_id: str
    bundle_arn: str
    version: str
    prompt: str


def _stored_prompt(ctrl: ConfigBundleClient, bundle_id: str, version: str, runtime_arn: str) -> str:
    read = ctrl.get_configuration_bundle_version(bundleId=bundle_id, versionId=version)
    return read["components"][runtime_arn]["configuration"].get("system_prompt", "")


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
    stored = _stored_prompt(ctrl, resp["bundleId"], resp["versionId"], runtime_arn)
    if stored.strip() != prompt.strip():
        raise RuntimeError(f"{name} did not round-trip: {len(stored)} stored vs {len(prompt)} sent")
    return Bundle(resp["bundleId"], resp["bundleArn"], resp["versionId"], stored)


def recommend_prompt(
    *,
    region: str,
    control: Bundle,
    runtime_arn: str,
    batch_evaluation_arn: str,
    evaluator_arn: str = _GOAL_SUCCESS,
) -> tuple[Bundle, str]:
    """Ask AgentCore to rewrite the control bundle's prompt, targeting `evaluator_arn`.

    The service reads the sessions the batch evaluation scored, finds where the
    target evaluator scores them poorly, and writes the revised prompt as a NEW
    VERSION of the control bundle, so the result is ready to A/B as the treatment.

    `systemPromptJsonPath` is `$.<runtime ARN>.configuration.system_prompt`. The
    shorter `$.configuration.system_prompt` resolves to nothing.

    Note the choice of `evaluator_arn` does not just grade the agent, it decides which
    direction the fix points. Targeting GoalSuccessRate pushes an agent toward acting
    without confirmation, because asking scores as less complete than answering.

    Returns:
        `(treatment, explanation)`.
    """
    result = RecommendationClient(region_name=region).start_recommendation_and_wait(
        wait_config=_WAIT,
        name=f"SpRec{int(time.time())}",
        type="SYSTEM_PROMPT_RECOMMENDATION",
        recommendation_config={"systemPromptRecommendationConfig": {
            "systemPrompt": {"configurationBundle": {
                "bundleArn": control.bundle_arn, "versionId": control.version,
                "systemPromptJsonPath": f"$.{runtime_arn}.configuration.system_prompt"}},
            "agentTraces": {"batchEvaluation": {"batchEvaluationArn": batch_evaluation_arn}},
            "evaluationConfig": {"evaluators": [{"evaluatorArn": evaluator_arn}]},
        }},
        client_token=str(uuid.uuid4()),
    )
    sp = result["recommendationResult"]["systemPromptRecommendationResult"]
    version = sp["configurationBundle"]["versionId"]
    stored = _stored_prompt(ConfigBundleClient(region_name=region), control.bundle_id, version, runtime_arn)
    if stored.strip() == control.prompt.strip():
        print("  WARNING: recommendation is identical to the control prompt")
    return Bundle(control.bundle_id, control.bundle_arn, version, stored), str(sp.get("explanation") or "")


def delete_bundles(*, region: str, bundle_ids: list[str]) -> None:
    """Delete configuration bundles, reporting rather than raising."""
    ctrl = ConfigBundleClient(region_name=region)
    for bid in bundle_ids:
        try:
            ctrl.delete_configuration_bundle(bundleId=bid)
            print(f"  deleted bundle {bid}")
        except Exception as exc:  # noqa: BLE001
            print(f"  bundle {bid}: {exc}")
