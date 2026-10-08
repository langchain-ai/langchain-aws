"""Standard deepagents sandbox conformance tests, run against live AgentCore.

Requires AWS credentials with AgentCore Code Interpreter access.
"""

from __future__ import annotations

import importlib.metadata
import os
from typing import TYPE_CHECKING, ClassVar

import deepagents.backends.sandbox as _base_sandbox
import pytest
from langchain_tests.integration_tests import SandboxIntegrationTests
from packaging.version import Version

from langchain_agentcore_codeinterpreter import AgentCoreSandbox

_WRITE_OVERWRITES = Version(importlib.metadata.version("deepagents")) >= Version(
    "0.7.0"
)
_GLOB_RETURNS_ABSOLUTE = hasattr(_base_sandbox, "_absolutize_glob_path")

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

    # langchain-tests 1.1.9 predates two BaseSandbox changes in deepagents 0.7.
    # These only fail on the releases that made them, so the marks are scoped.

    @pytest.mark.xfail(
        _WRITE_OVERWRITES,
        reason="deepagents>=0.7.0 write() overwrites existing files by design",
        strict=True,
    )
    def test_write_existing_file_fails(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_write_existing_file_fails(sandbox_backend, sandbox_test_root)

    _GLOB_ABSOLUTE_MARK = pytest.mark.xfail(
        _GLOB_RETURNS_ABSOLUTE,
        reason="deepagents 0.7 BaseSandbox.glob returns absolute paths",
        strict=True,
    )

    @_GLOB_ABSOLUTE_MARK
    def test_glob(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob(sandbox_backend, sandbox_test_root)

    @_GLOB_ABSOLUTE_MARK
    def test_glob_basic_pattern(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_basic_pattern(sandbox_backend, sandbox_test_root)

    @_GLOB_ABSOLUTE_MARK
    def test_glob_with_directories(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_with_directories(sandbox_backend, sandbox_test_root)

    @_GLOB_ABSOLUTE_MARK
    def test_glob_hidden_files_explicitly(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_hidden_files_explicitly(sandbox_backend, sandbox_test_root)

    @_GLOB_ABSOLUTE_MARK
    def test_glob_with_character_class(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_with_character_class(sandbox_backend, sandbox_test_root)

    @_GLOB_ABSOLUTE_MARK
    def test_glob_with_question_mark(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        super().test_glob_with_question_mark(sandbox_backend, sandbox_test_root)
