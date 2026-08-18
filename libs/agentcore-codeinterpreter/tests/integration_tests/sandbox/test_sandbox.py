"""Integration tests for AgentCoreSandbox against live AgentCore sessions.

These tests require valid AWS credentials and a region where the AgentCore
Code Interpreter service is available.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

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
        """Binary content uploaded as a blob should round-trip unchanged."""
        content = b"\x00\x01\x02\xff\xfe\xfd"
        sandbox.upload_files([("binary_test.bin", content)])
        results = sandbox.download_files(["binary_test.bin"])
        assert results[0].error is None
        assert results[0].content == content

    def test_session_id(self, sandbox: AgentCoreSandbox) -> None:
        """The sandbox should have a non-empty session ID."""
        assert sandbox.id != ""

    def test_read_plane_roundtrip_virtual_path(self, sandbox: AgentCoreSandbox) -> None:
        """Read plane round-trips a virtual path against a non-root cwd."""
        vpath = "/workspace/readplane_roundtrip.txt"
        vdir = "/workspace"
        content = "needle in haystack\nsecond line\n"

        write_result = sandbox.write(vpath, content)
        assert write_result.error is None, write_result.error

        read_result = sandbox.read(vpath)
        assert read_result.error is None, read_result.error
        assert read_result.file_data is not None
        assert read_result.file_data["content"].strip() == content.strip()

        ls_result = sandbox.ls(vdir)
        assert ls_result.error is None, ls_result.error
        assert ls_result.entries is not None
        assert any("readplane_roundtrip.txt" in e["path"] for e in ls_result.entries)

        grep_result = sandbox.grep("needle", vdir)
        assert grep_result.error is None, grep_result.error
        assert grep_result.matches

        glob_result = sandbox.glob("*.txt", vdir)
        assert glob_result.error is None, glob_result.error
        assert glob_result.matches

        edit_result = sandbox.edit(vpath, "needle", "pin")
        assert edit_result.error is None, edit_result.error
        after = sandbox.read(vpath)
        assert after.file_data is not None
        assert "pin in haystack" in after.file_data["content"]

    # ------------------------------------------------------------------
    # Async integration tests
    # ------------------------------------------------------------------

    def test_aexecute_echo(self, sandbox: AgentCoreSandbox) -> None:
        """aexecute() should work against a live session."""
        result = asyncio.run(sandbox.aexecute("echo async hello"))
        assert result.exit_code == 0
        assert "async hello" in result.output

    def test_aexecute_python(self, sandbox: AgentCoreSandbox) -> None:
        """aexecute() should handle Python execution."""
        result = asyncio.run(sandbox.aexecute("python3 -c \"print('async works')\""))
        assert result.exit_code == 0
        assert "async works" in result.output

    def test_aupload_and_adownload_roundtrip(self, sandbox: AgentCoreSandbox) -> None:
        """Async upload and download should produce identical content."""
        content = b"async round trip content"
        upload_result = asyncio.run(
            sandbox.aupload_files([("async_roundtrip.txt", content)])
        )
        assert upload_result[0].error is None

        download_result = asyncio.run(sandbox.adownload_files(["async_roundtrip.txt"]))
        assert download_result[0].error is None
        assert download_result[0].content == content

    def test_concurrent_aexecute(self, sandbox: AgentCoreSandbox) -> None:
        """Multiple concurrent async executions should not deadlock."""

        async def run_concurrent() -> list[Any]:
            tasks = [sandbox.aexecute(f"echo concurrent-{i}") for i in range(3)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_concurrent())
        for i, result in enumerate(results):
            assert result.exit_code == 0
            assert f"concurrent-{i}" in result.output

    def test_aread_plane_roundtrip_virtual_path(
        self, sandbox: AgentCoreSandbox
    ) -> None:
        """Async read plane round-trips a virtual path against a non-root cwd."""
        vpath = "/workspace/areadplane_roundtrip.txt"
        vdir = "/workspace"

        async def run() -> None:
            write_result = await sandbox.awrite(vpath, "async needle\n")
            assert write_result.error is None, write_result.error

            read_result = await sandbox.aread(vpath)
            assert read_result.error is None, read_result.error

            ls_result = await sandbox.als(vdir)
            assert ls_result.error is None, ls_result.error

            grep_result = await sandbox.agrep("needle", vdir)
            assert grep_result.error is None, grep_result.error

            glob_result = await sandbox.aglob("*.txt", vdir)
            assert glob_result.error is None, glob_result.error

        asyncio.run(run())
