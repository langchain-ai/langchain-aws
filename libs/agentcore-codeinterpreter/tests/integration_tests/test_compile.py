"""Compile-only smoke test to verify the package can be imported.

This runs as part of the integration test suite to catch packaging or
dependency issues early.
"""

import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Verify that langchain_agentcore_codeinterpreter can be imported."""
    from langchain_agentcore_codeinterpreter import AgentCoreSandbox  # noqa: F401
