"""Pin the upstream :class:`BackendProtocol.edit` signature.

``S3Backend.edit`` keeps ``replace_all`` as a positional-with-default
parameter to match ``BackendProtocol.edit`` exactly (``noqa: FBT001/FBT002``).
A protocol-side rename to keyword-only would silently diverge: Python
does not flag positional-vs-keyword-only as a Protocol mismatch, so the
backend would continue to type-check while breaking callers that pass
``replace_all`` positionally.

Snapshotting the parameter list here detects an upstream change at
``pytest`` time rather than at the user's first failing call site.
"""

from __future__ import annotations

import inspect

from deepagents.backends.protocol import BackendProtocol

from langchain_backend_aws.s3.backend import S3Backend


def _kind_label(param: inspect.Parameter) -> str:
    """Compact human-readable name for a parameter kind."""
    return param.kind.name


def test_edit_replace_all_remains_positional_with_default() -> None:
    """``replace_all`` must stay positional-with-default on both sides.

    If this test fails because the upstream protocol moved
    ``replace_all`` to keyword-only, update ``S3Backend.edit`` to match
    (drop the noqa, mark as keyword-only) and refresh the snapshot.
    """
    protocol_sig = inspect.signature(BackendProtocol.edit)
    backend_sig = inspect.signature(S3Backend.edit)

    protocol_replace_all = protocol_sig.parameters["replace_all"]
    backend_replace_all = backend_sig.parameters["replace_all"]

    # POSITIONAL_OR_KEYWORD with a default value is the documented
    # protocol shape; KEYWORD_ONLY would break positional callers.
    assert protocol_replace_all.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
        f"BackendProtocol.edit replace_all kind changed to "
        f"{_kind_label(protocol_replace_all)}; update S3Backend.edit to match."
    )
    assert backend_replace_all.kind == protocol_replace_all.kind, (
        f"S3Backend.edit replace_all is {_kind_label(backend_replace_all)} "
        f"but protocol expects {_kind_label(protocol_replace_all)}."
    )
    assert backend_replace_all.default == protocol_replace_all.default


def test_edit_parameter_order_matches_protocol() -> None:
    """All ``edit`` parameters and their kinds must match the protocol."""
    protocol_params = [
        (name, param.kind)
        for name, param in inspect.signature(BackendProtocol.edit).parameters.items()
    ]
    backend_params = [
        (name, param.kind)
        for name, param in inspect.signature(S3Backend.edit).parameters.items()
    ]
    assert protocol_params == backend_params, (
        "S3Backend.edit signature drifted from BackendProtocol.edit. "
        f"Protocol: {protocol_params}; backend: {backend_params}."
    )
