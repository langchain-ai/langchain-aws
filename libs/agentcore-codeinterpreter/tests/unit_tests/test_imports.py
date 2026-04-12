"""Verify that the package and its public API can be imported."""

from __future__ import annotations

from langchain_agentcore_codeinterpreter import AgentCoreSandbox


def test_import_package() -> None:
    """Importing the package should succeed."""
    import langchain_agentcore_codeinterpreter

    assert langchain_agentcore_codeinterpreter is not None


def test_import_sandbox_class() -> None:
    """The public AgentCoreSandbox class should be importable."""
    assert AgentCoreSandbox is not None


def test_import_session_expired_error() -> None:
    """SessionExpiredError should be importable from the sandbox module."""
    from langchain_agentcore_codeinterpreter.sandbox import SessionExpiredError

    assert SessionExpiredError is not None
