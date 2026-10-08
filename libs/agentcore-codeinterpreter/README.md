# langchain-agentcore-codeinterpreter

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-agentcore-codeinterpreter?label=%20)](https://pypi.org/project/langchain-agentcore-codeinterpreter/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-agentcore-codeinterpreter)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-agentcore-codeinterpreter)](https://pypistats.org/packages/langchain-agentcore-codeinterpreter)

Amazon Bedrock AgentCore Code Interpreter sandbox integration for [Deep Agents](https://github.com/langchain-ai/deepagents).

This package provides `AgentCoreSandbox` — a [`SandboxBackendProtocol`](https://docs.langchain.com/oss/deepagents/sandboxes) implementation that wraps AgentCore's [Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html), a secure, isolated MicroVM environment for executing code. `AgentCoreSandbox.create()` starts a session and stops it when you close the sandbox or leave a `with` block.

> **Note:** For the LangChain `BaseTool` integration (used with `create_react_agent` and LangGraph agents), see [`langchain-aws[tools]`](https://github.com/langchain-ai/langchain-aws). This package is specifically for the Deep Agents `BaseSandbox` protocol.

## Prerequisites

**1. AWS credentials** configured via one of the following methods:

```bash
# Option 1: Long-lived IAM credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-west-2"

# Option 2: Temporary credentials (IAM roles, SSO, STS AssumeRole)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_SESSION_TOKEN="your-session-token"
export AWS_REGION="us-west-2"

# Option 3: AWS CLI profile (picks up ~/.aws/credentials + ~/.aws/config)
aws configure
# or for SSO:
aws configure sso
aws sso login --profile your-profile
```

Any method supported by the [boto3 credential chain](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) works, including EC2 instance profiles, ECS task roles, and environment variables.

**2. IAM permissions** — your credentials must allow `bedrock-agentcore:InvokeCodeInterpreter` (or the equivalent action for your region). See the [AgentCore Code Interpreter docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html) for the required IAM policy.

**3. Region availability** — Code Interpreter is available in select AWS regions. `us-west-2` is a safe default. Pass the region to `AgentCoreSandbox.create(region=...)`.

## Quick Install

```bash
pip install langchain-agentcore-codeinterpreter
```

## Usage

### Standalone

```python
from langchain_agentcore_codeinterpreter import AgentCoreSandbox

with AgentCoreSandbox.create(region="us-west-2") as backend:
    result = backend.execute("echo hello")
    print(result.output, result.exit_code)  # "hello" 0
```

Sessions last 15 minutes by default. Pass `session_timeout_seconds` (up to 28,800, or 8 hours) for longer agent runs, and `identifier` to use a custom code interpreter with VPC networking or an execution role.

To configure the `CodeInterpreter` client yourself, pass a started one to the constructor instead. You then own its lifecycle:

```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

interpreter = CodeInterpreter(
    region="us-west-2", integration_source="langchain-deepagents"
)
interpreter.start()
try:
    backend = AgentCoreSandbox(interpreter=interpreter)
    ...
finally:
    interpreter.stop()
```

### With Deep Agents

```python
from deepagents import create_deep_agent

from langchain_agentcore_codeinterpreter import AgentCoreSandbox
from langchain_aws import ChatBedrockConverse

model = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-6",
    region_name="us-west-2",
)

with AgentCoreSandbox.create(
    region="us-west-2", session_timeout_seconds=3600
) as backend:
    agent = create_deep_agent(
        model=model,
        backend=backend,
        system_prompt="You are a coding assistant with sandbox access.",
    )
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Create and run a hello world script"}
            ]
        }
    )
    print(result["messages"][-1].content)
```

### File operations

```python
# Upload files
backend.upload_files(
    [
        ("data.csv", b"name,value\nalice,42\nbob,17"),
        ("analyze.py", b"import csv\nprint('ready')"),
    ]
)

# Download files
results = backend.download_files(["data.csv"])
for r in results:
    if r.content is not None:
        print(f"{r.path}: {r.content.decode()}")
    else:
        print(f"Failed: {r.path}: {r.error}")
```

## Session behavior

AgentCore sessions cannot be reconnected once stopped. Each session is a fresh, isolated MicroVM, and its files are gone when it ends. Sessions auto-expire after `session_timeout_seconds` (default 15 minutes, maximum 8 hours); calls after that return a session-expired error.

`execute()` honors `timeout` by running the command under GNU `timeout` inside the sandbox. A command stopped this way returns exit code 124.

AgentCore caps each request and response at 100 MB and gives each session 10 GB of disk. See [AgentCore quotas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/bedrock-agentcore-limits.html).

## Contributing

See the [langchain-aws contributing guide](https://github.com/langchain-ai/langchain-aws/blob/main/.github/CONTRIBUTING.md).

```bash
cd libs/agentcore-codeinterpreter

# Run unit tests (no network, no AWS credentials needed)
make tests

# Run linter
make lint

# Run integration tests (requires AWS credentials)
make integration_tests
```
