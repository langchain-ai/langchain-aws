# langchain-agentcore-codeinterpreter

[![PyPI - Version](https://img.shields.io/pypi/v/langchain-agentcore-codeinterpreter?label=%20)](https://pypi.org/project/langchain-agentcore-codeinterpreter/#history)
[![PyPI - License](https://img.shields.io/pypi/l/langchain-agentcore-codeinterpreter)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/langchain-agentcore-codeinterpreter)](https://pypistats.org/packages/langchain-agentcore-codeinterpreter)

Amazon Bedrock AgentCore Code Interpreter sandbox integration for [Deep Agents](https://github.com/langchain-ai/deepagents).

This package provides `AgentCoreSandbox` — a [`SandboxBackendProtocol`](https://docs.langchain.com/oss/deepagents/sandboxes) implementation that wraps AgentCore's [Code Interpreter](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore-code-interpreter.html), a secure, isolated MicroVM environment for executing code. The caller manages the interpreter lifecycle (`start()` / `stop()`); the sandbox backend handles command execution and file operations.

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

**2. IAM permissions** — your credentials must allow `bedrock-agentcore:InvokeCodeInterpreter` (or the equivalent action for your region). See the [AgentCore Code Interpreter docs](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore-code-interpreter.html) for the required IAM policy.

**3. Region availability** — Code Interpreter is available in select AWS regions. `us-west-2` is a safe default. Pass the region to `CodeInterpreter(region=...)`.

## Quick Install

```bash
pip install langchain-agentcore-codeinterpreter
```

## Usage

### Standalone

```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

from langchain_agentcore_codeinterpreter import AgentCoreSandbox

interpreter = CodeInterpreter(region="us-west-2")
interpreter.start()

backend = AgentCoreSandbox(interpreter=interpreter)

result = backend.execute("echo hello")
print(result.output)  # "hello"

interpreter.stop()
```

### With Deep Agents

```python
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from deepagents import create_deep_agent

from langchain_agentcore_codeinterpreter import AgentCoreSandbox

interpreter = CodeInterpreter(region="us-west-2")
interpreter.start()

model = ChatBedrockConverse(
  model="us.anthropic.claude-sonnet-4-6",
  region_name="us-west-2",
)
backend = AgentCoreSandbox(interpreter=interpreter)
agent = create_deep_agent(
    model=model,
    backend=backend,
    system_prompt="You are a coding assistant with sandbox access.",
)

try:
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Create and run a hello world script"}
            ]
        }
    )
    print(result["messages"][-1].content)
finally:
    interpreter.stop()
```

### File operations

```python
# Upload files
backend.upload_files([
    ("data.csv", b"name,value\nalice,42\nbob,17"),
    ("analyze.py", b"import csv\nprint('ready')"),
])

# Download files
results = backend.download_files(["data.csv"])
for r in results:
    if r.content is not None:
        print(f"{r.path}: {r.content.decode()}")
    else:
        print(f"Failed: {r.path}: {r.error}")
```

## Session behavior

AgentCore sessions cannot be reconnected after `interpreter.stop()` is called. Each `start()` creates a fresh, isolated MicroVM. Sessions auto-expire after a configurable timeout (default 15 minutes, maximum 8 hours).

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
