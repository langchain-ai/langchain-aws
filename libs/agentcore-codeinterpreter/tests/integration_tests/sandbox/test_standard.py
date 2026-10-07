"""Standard deepagents sandbox conformance tests, run against live AgentCore.

Requires AWS credentials with AgentCore Code Interpreter access.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, ClassVar

import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests

from langchain_agentcore_codeinterpreter import AgentCoreSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


class TestAgentCoreSandboxStandard(SandboxIntegrationTests):
    # AgentCoreSandbox maps paths outside the session's working directory under
    # it, so the suite's shell-side checks need a root inside that directory.
    # It is read from the session rather than hardcoded.
    _root: ClassVar[str] = ""

    @property
    def sandbox_root_dir(self) -> str:
        return self._root

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        region = os.environ.get("AWS_REGION", "us-west-2")
        with AgentCoreSandbox.create(region=region) as backend:
            type(self)._root = f"{backend._get_cwd()}/test_sandbox_ops/"
            yield backend

    @pytest.fixture(autouse=True)
    def sandbox_test_root(
        self,
        request: pytest.FixtureRequest,
        sandbox_backend: SandboxBackendProtocol,
    ) -> str:
        """Same as the base fixture, but after the session's root is known."""
        node_name = request.node.name.replace("/", "_").replace(" ", "_")
        return self.sandbox_path(node_name)

    # The protocol rejects relative paths for upload and download. This package
    # has always resolved them against the sandbox cwd, and its README relies on
    # that, so rejecting them is a breaking change for its own release.
    _RELATIVE_PATHS_RESOLVED = (
        "relative upload/download paths resolve against the sandbox cwd "
        "instead of returning invalid_path"
    )

    @pytest.mark.xfail(reason=_RELATIVE_PATHS_RESOLVED, strict=True)
    def test_download_error_invalid_path_relative(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        super().test_download_error_invalid_path_relative(sandbox_backend)

    @pytest.mark.xfail(reason=_RELATIVE_PATHS_RESOLVED, strict=True)
    def test_upload_relative_path_returns_invalid_path(
        self, sandbox_backend: SandboxBackendProtocol
    ) -> None:
        super().test_upload_relative_path_returns_invalid_path(sandbox_backend)
