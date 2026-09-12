"""Import-compatibility regression tests for `langgraph-checkpoint`'s floor.

`langgraph_checkpoint_aws.checkpoint.agentcore.saver` supports
`langgraph-checkpoint>=3.0.0,<5.0.0` (see `pyproject.toml`). `DeltaChannel`
support (`DeltaChannelHistory`, `BaseCheckpointSaver.get_delta_channel_history`)
was added to `langgraph-checkpoint` after 3.0.0, so `DeltaChannelHistory`
must never be imported at module level from `langgraph.checkpoint.base` --
doing so breaks importing this whole package on the declared floor version,
even though the currently installed `langgraph-checkpoint` (see
`uv.lock`) does have it.
"""

import importlib
import sys

import langgraph.checkpoint.base as checkpoint_base

SAVER_MODULE = "langgraph_checkpoint_aws.checkpoint.agentcore.saver"


def test_saver_module_imports_without_delta_channel_history(monkeypatch):
    """Simulate `langgraph-checkpoint` 3.0.0: no `DeltaChannelHistory` on
    `langgraph.checkpoint.base`. Importing the saver module must still
    succeed -- it is only ever needed here as a type annotation."""
    monkeypatch.delattr(checkpoint_base, "DeltaChannelHistory", raising=False)
    monkeypatch.delitem(sys.modules, SAVER_MODULE, raising=False)

    try:
        module = importlib.import_module(SAVER_MODULE)
    except ImportError as exc:  # pragma: no cover - failure path under test
        raise AssertionError(
            "AgentCoreMemorySaver module failed to import without "
            f"DeltaChannelHistory (langgraph-checkpoint 3.0.0 floor): {exc}"
        ) from None

    # Defining `get_delta_channel_history`/`aget_delta_channel_history` must
    # not require the base class to already define them -- on a langgraph
    # version without DeltaChannel support they are simply unused methods.
    assert hasattr(module.AgentCoreMemorySaver, "get_delta_channel_history")
    assert hasattr(module.AgentCoreMemorySaver, "aget_delta_channel_history")


def test_delta_channel_history_import_is_type_checking_only():
    """Static guard: `DeltaChannelHistory` must be imported only inside an
    `if TYPE_CHECKING:` block in `saver.py`, never at module level, so a
    reviewer re-adding a top-level import trips this before it ships."""
    import ast
    from pathlib import Path

    saver_path = Path(
        importlib.import_module(SAVER_MODULE).__file__  # type: ignore[arg-type]
    )
    tree = ast.parse(saver_path.read_text())

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "langgraph.checkpoint.base"
        ):
            names = {alias.name for alias in node.names}
            if "DeltaChannelHistory" in names:
                # This import is only acceptable directly under `if
                # TYPE_CHECKING:` at module level, never mixed into the
                # unconditional top-level import block.
                assert getattr(node, "col_offset", 0) > 0, (
                    "DeltaChannelHistory must be imported only inside an "
                    "`if TYPE_CHECKING:` block, not at module level"
                )
