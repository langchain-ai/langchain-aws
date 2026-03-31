"""Integration tests for AgentCoreSandbox against live AgentCore sessions.

These tests require valid AWS credentials and a region where the AgentCore
Code Interpreter service is available.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from langchain_agentcore_codeinterpreter import AgentCoreSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator


_SKIP_REASON = "AWS credentials not configured (set AWS_REGION or AWS_DEFAULT_REGION)"


@pytest.fixture(scope="module")
def sandbox() -> Iterator[AgentCoreSandbox]:
    """Create a live AgentCore sandbox for the entire test module."""
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

    region = os.environ.get(
        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    )
    interpreter = CodeInterpreter(region=region, integration_source="langchain")
    interpreter.start()
    backend = AgentCoreSandbox(interpreter=interpreter)
    try:
        yield backend
    finally:
        interpreter.stop()


@pytest.mark.skipif(
    not os.environ.get("AWS_DEFAULT_REGION") and not os.environ.get("AWS_REGION"),
    reason=_SKIP_REASON,
)
class TestAgentCoreSandboxIntegration:
    """Live integration tests against the AgentCore Code Interpreter API."""

    def test_execute_echo(self, sandbox: AgentCoreSandbox) -> None:
        """A basic echo command should succeed with exit code 0."""
        result = sandbox.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.output

    def test_execute_python(self, sandbox: AgentCoreSandbox) -> None:
        """Python code execution should work."""
        result = sandbox.execute("python3 -c \"print('works')\"")
        assert result.exit_code == 0
        assert "works" in result.output

    def test_execute_error(self, sandbox: AgentCoreSandbox) -> None:
        """An invalid command should indicate failure."""
        result = sandbox.execute("nonexistent_command_xyz_12345")
        assert result.exit_code != 0 or "not found" in result.output

    def test_upload_and_download_roundtrip(self, sandbox: AgentCoreSandbox) -> None:
        """Files uploaded should be downloadable with identical content."""
        content = b"print('round trip')"
        sandbox.upload_files([("roundtrip_test.py", content)])
        results = sandbox.download_files(["roundtrip_test.py"])
        assert results[0].error is None
        assert results[0].content == content

    def test_upload_and_execute(self, sandbox: AgentCoreSandbox) -> None:
        """An uploaded script should be executable."""
        sandbox.upload_files([("exec_test.py", b"print('executed')")])
        result = sandbox.execute("python3 exec_test.py")
        assert result.exit_code == 0
        assert "executed" in result.output

    def test_download_missing_file(self, sandbox: AgentCoreSandbox) -> None:
        """Downloading a nonexistent file should return an error."""
        results = sandbox.download_files(["does_not_exist_xyz.txt"])
        assert results[0].error is not None

    def test_binary_roundtrip(self, sandbox: AgentCoreSandbox) -> None:
        """Binary content uploaded via base64 may be returned as base64 text."""
        content = b"\x00\x01\x02\xff\xfe\xfd"
        sandbox.upload_files([("binary_test.bin", content)])
        results = sandbox.download_files(["binary_test.bin"])
        assert results[0].error is None
        # AgentCore returns blob uploads as text containing the base64 string
        import base64

        assert results[0].content in (content, base64.b64encode(content))

    def test_session_id(self, sandbox: AgentCoreSandbox) -> None:
        """The sandbox should have a non-empty session ID."""
        assert sandbox.id != ""
