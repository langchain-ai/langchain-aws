"""Deploy an agent to AgentCore Runtime as a zip code package.

Absorbs the IAM role, arm64 dependency build, S3 upload and runtime create/update so
a notebook can deploy in one call. Framework agnostic: it packages whatever entrypoint
and modules you hand it.

A zip `codeConfiguration` is used rather than a container. That skips CodeBuild and
ECR entirely, so a deploy takes about a minute instead of 8 to 12, which matters when
the point is iterating on prompts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import boto3

__all__ = ["Deployment", "deploy_agent", "delete_deployment"]

_TRUST = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
}

# bedrock-agentcore is broad here because Browser and Code Interpreter need several
# actions including WebSocket streaming, and a narrower policy breaks the automation
# stream. Scope this down for production.
_EXEC_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {"Sid": "Bedrock", "Effect": "Allow", "Resource": "*",
         "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream",
                    "bedrock:Converse", "bedrock:ConverseStream"]},
        {"Sid": "AgentCoreTools", "Effect": "Allow", "Resource": "*",
         "Action": "bedrock-agentcore:*"},
        {"Sid": "Observability", "Effect": "Allow", "Resource": "*",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                    "logs:DescribeLogGroups", "logs:DescribeLogStreams",
                    "xray:PutTraceSegments", "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules", "xray:GetSamplingTargets",
                    "cloudwatch:PutMetricData"]},
        {"Sid": "ReadPackage", "Effect": "Allow", "Resource": "*",
         "Action": ["s3:GetObject", "s3:ListBucket"]},
    ],
}


@dataclass
class Deployment:
    """Everything later steps need to address a deployed runtime."""

    name: str
    runtime_id: str
    runtime_arn: str
    role_arn: str
    region: str
    account: str
    log_group: str
    service_name: str
    s3_uri: str
    bucket: str = ""
    key: str = ""
    extra: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Persist as JSON so a later notebook or session can pick it up."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.__dict__, indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> Deployment:
        """Read a previously saved deployment."""
        return cls(**json.loads(Path(path).read_text()))


def _install_deps(deps: list[str], target: Path, python_version: str) -> None:
    """Cross-install dependencies for arm64.

    Tries pip, then `uv pip`. A uv-managed virtualenv has no `pip` module, so the
    documented `python -m pip --platform ...` path fails outright there.
    """
    pip_cmd = [
        sys.executable, "-m", "pip", "install", *deps,
        "-t", str(target),
        "--platform", "manylinux2014_aarch64",
        "--only-binary=:all:",
        "--python-version", python_version,
        "--quiet",
    ]
    uv_cmd = [
        "uv", "pip", "install", *deps,
        "--target", str(target),
        "--python-platform", "aarch64-manylinux2014",
        "--python-version", python_version,
        "--only-binary", ":all:",
        "--quiet",
    ]
    result = subprocess.run(pip_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        result = subprocess.run(uv_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("dependency install failed with pip and uv: " + result.stderr[-1200:])


def deploy_agent(
    *,
    name: str,
    entrypoint_src: str,
    modules: dict[str, str],
    deps: list[str],
    region: str | None = None,
    env: dict[str, str] | None = None,
    work_dir: str | Path = "agentcore_build",
    python_version: str = "3.13",
    wait_minutes: float = 15.0,
) -> Deployment:
    """Package and deploy an agent, returning a `Deployment`.

    Args:
        name: Runtime name. Reused on redeploy.
        entrypoint_src: Source for `main.py`, containing the `@app.entrypoint`.
        modules: Extra `{module_name: source}` written beside the entrypoint.
        deps: Requirements installed for arm64.
        region: AWS region. Defaults to `AWS_REGION`.
        env: Environment variables for the runtime.
        work_dir: Local scratch directory for the build.
        python_version: Runtime Python version.
        wait_minutes: How long to wait for READY.

    Returns:
        A `Deployment` describing the live runtime.
    """
    region = region or os.environ.get("AWS_REGION", "us-west-2")
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    iam = boto3.client("iam", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    ctrl = boto3.client("bedrock-agentcore-control", region_name=region)

    # --- execution role -----------------------------------------------------
    role_name = f"AgentCoreExec-{name}"
    try:
        role_arn = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(_TRUST),
            Description=f"Execution role for AgentCore runtime {name}",
        )["Role"]["Arn"]
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="AgentCoreExecution",
        PolicyDocument=json.dumps(_EXEC_POLICY),
    )
    print(f"  role      {role_name}")

    # --- package ------------------------------------------------------------
    work = Path(work_dir)
    if work.exists():
        shutil.rmtree(work)
    pkg = work / "pkg"
    pkg.mkdir(parents=True)

    print(f"  deps      installing {len(deps)} for aarch64 (a few minutes)")
    _install_deps(deps, pkg, python_version)

    (pkg / "main.py").write_text(entrypoint_src)
    for mod_name, src in modules.items():
        (pkg / f"{mod_name}.py").write_text(src)

    # Syntax-check the sources we wrote. Deliberately NOT an import check: the
    # dependencies in this directory are manylinux aarch64 wheels, so importing them
    # on the build machine fails with an unrelated error about pydantic_core even
    # though the package is perfectly valid for the target.
    import ast

    for src_name, src_text in [("main.py", entrypoint_src),
                               *[(f"{m}.py", t) for m, t in modules.items()]]:
        try:
            ast.parse(src_text)
        except SyntaxError as exc:
            raise RuntimeError(f"{src_name} has a syntax error: {exc}") from exc
    print(f"  syntax    {1 + len(modules)} module(s) parse")

    zip_path = work / "deployment_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(pkg):
            if "__pycache__" in root:
                continue
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                full = Path(root) / f
                zf.write(full, full.relative_to(pkg))
    print(f"  package   {zip_path.stat().st_size / 1e6:.1f} MB")

    # --- upload -------------------------------------------------------------
    bucket = f"agentcore-deploy-{account}-{region}"
    key = f"{name}/deployment_package.zip"
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    except (s3.exceptions.BucketAlreadyOwnedByYou, s3.exceptions.BucketAlreadyExists):
        pass
    s3.upload_file(str(zip_path), bucket, key)
    print(f"  uploaded  s3://{bucket}/{key}")

    # --- create or update ---------------------------------------------------
    artifact = {
        "codeConfiguration": {
            "code": {"s3": {"bucket": bucket, "prefix": key}},
            "runtime": f"PYTHON_{python_version.replace('.', '_')}",
            # This is what activates ADOT. Without it the agent runs but emits no
            # gen_ai spans, and nothing downstream can evaluate it.
            "entryPoint": ["opentelemetry-instrument", "main.py"],
        }
    }
    existing = next(
        (r["agentRuntimeId"] for r in ctrl.list_agent_runtimes(maxResults=100).get("agentRuntimes", [])
         if r.get("agentRuntimeName") == name),
        None,
    )
    common = {
        "agentRuntimeArtifact": artifact,
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "roleArn": role_arn,
        "environmentVariables": env or {"AWS_REGION": region},
    }
    if existing:
        ctrl.update_agent_runtime(agentRuntimeId=existing, **common)
        runtime_id = existing
        print(f"  runtime   updated {runtime_id}")
    else:
        runtime_id = ctrl.create_agent_runtime(agentRuntimeName=name, **common)["agentRuntimeId"]
        print(f"  runtime   created {runtime_id}")

    deadline = time.time() + wait_minutes * 60
    runtime_arn = ""
    while time.time() < deadline:
        detail = ctrl.get_agent_runtime(agentRuntimeId=runtime_id)
        status = detail.get("status")
        if status in ("READY", "ACTIVE"):
            runtime_arn = detail["agentRuntimeArn"]
            break
        if "FAILED" in str(status):
            raise RuntimeError(f"deploy failed: {detail.get('failureReason')}")
        time.sleep(10)
    if not runtime_arn:
        raise RuntimeError(f"runtime did not become ready in {wait_minutes} minutes")

    print(f"  status    READY")
    return Deployment(
        name=name,
        runtime_id=runtime_id,
        runtime_arn=runtime_arn,
        role_arn=role_arn,
        region=region,
        account=account,
        log_group=f"/aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT",
        service_name=f"{name}.DEFAULT",
        s3_uri=f"s3://{bucket}/{key}",
        bucket=bucket,
        key=key,
    )


def delete_deployment(dep: Deployment, *, delete_bucket_object: bool = True) -> None:
    """Delete the runtime, its role and its package. Log groups are kept.

    Log groups hold the spans every evaluation result references, so removing them
    would invalidate any scores you have recorded.
    """
    ctrl = boto3.client("bedrock-agentcore-control", region_name=dep.region)
    iam = boto3.client("iam", region_name=dep.region)
    s3 = boto3.client("s3", region_name=dep.region)

    for label, fn in (
        ("runtime", lambda: ctrl.delete_agent_runtime(agentRuntimeId=dep.runtime_id)),
    ):
        try:
            fn()
            print(f"  deleted {label} {dep.runtime_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label}: {exc}")

    role_name = dep.role_arn.split("/")[-1]
    try:
        for p in iam.list_role_policies(RoleName=role_name).get("PolicyNames", []):
            iam.delete_role_policy(RoleName=role_name, PolicyName=p)
        iam.delete_role(RoleName=role_name)
        print(f"  deleted role {role_name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  role: {exc}")

    if delete_bucket_object and dep.bucket and dep.key:
        try:
            s3.delete_object(Bucket=dep.bucket, Key=dep.key)
            print(f"  deleted {dep.s3_uri}")
        except Exception as exc:  # noqa: BLE001
            print(f"  s3: {exc}")

    print(f"  kept log group {dep.log_group}")
