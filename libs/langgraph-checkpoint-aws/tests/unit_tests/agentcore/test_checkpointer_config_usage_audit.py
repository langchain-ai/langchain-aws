"""Static audit of required-`actor_id` `CheckpointerConfig` usage.

`CheckpointerConfig` (and its `ValkeyCheckpointerConfig` subclass) model a
`RunnableConfig`-backed config that requires `actor_id`. The nested-subgraph
`actor_id` inheritance bug (and its fix) only applies to savers that build this
kind of config, because a derived subgraph `RunnableConfig` omits `actor_id`.

This test statically greps the package source for references to
`CheckpointerConfig`/`ValkeyCheckpointerConfig` and asserts the exact set of
files that define or import them. If a future change introduces this pattern
in a new file (e.g. a DynamoDB saver or a non-agentcore Valkey component), this
test fails, forcing an explicit decision about whether that new usage also
needs the `_resolve_actor_id` inheritance fix.
"""

import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "langgraph_checkpoint_aws"

# The only files expected to define or import CheckpointerConfig/
# ValkeyCheckpointerConfig, relative to the package root. Requirement 3.7:
# only the (already-fixed) AgentCore saver and the Valkey AgentCore saver
# (this spec) use the required-actor_id CheckpointerConfig pattern.
EXPECTED_FILES = {
    "checkpoint/agentcore/models.py",
    "checkpoint/agentcore/saver.py",
    "checkpoint/agentcore/helpers.py",
    "checkpoint/valkey/agentcore/models.py",
    "checkpoint/valkey/agentcore/saver.py",
}

_CHECKPOINTER_CONFIG_PATTERN = re.compile(r"CheckpointerConfig\b")


def _files_referencing_checkpointer_config() -> set[str]:
    """Find package .py files that reference CheckpointerConfig by name.

    Returns:
        POSIX-style paths (relative to the package root) of every `.py` file
        under the package containing a `CheckpointerConfig` or
        `ValkeyCheckpointerConfig` reference.
    """
    matches: set[str] = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if _CHECKPOINTER_CONFIG_PATTERN.search(text):
            matches.add(path.relative_to(PACKAGE_ROOT).as_posix())
    return matches


def test_checkpointer_config_usage_matches_expected_files() -> None:
    """Only the two known AgentCore savers reference CheckpointerConfig.

    Validates: Requirements 3.7
    """
    found = _files_referencing_checkpointer_config()

    assert found == EXPECTED_FILES, (
        "CheckpointerConfig/ValkeyCheckpointerConfig usage changed. "
        f"Added: {sorted(found - EXPECTED_FILES)}; "
        f"Removed: {sorted(EXPECTED_FILES - found)}. "
        "A new file using this required-actor_id config pattern needs "
        "explicit review for the nested-subgraph actor_id inheritance fix "
        "(see checkpoint/agentcore/saver.py's _resolve_actor_id and "
        "checkpoint/valkey/agentcore/saver.py's _resolve_actor_id)."
    )


def test_known_non_agentcore_components_do_not_use_checkpointer_config() -> None:
    """DynamoDB savers/stores and non-agentcore Valkey stores stay clear.

    These components do not build config via a `RunnableConfig`-backed
    `CheckpointerConfig` that requires `actor_id`, so they are unaffected by
    the nested-subgraph `actor_id` inheritance bug and do not need the
    `_resolve_actor_id` fix. `store/agentcore/store.py` is included here too:
    it derives `actor_id` directly from an op's namespace tuple rather than
    from a `RunnableConfig`, so it also does not use this pattern.

    Validates: Requirements 3.7
    """
    components_without_the_pattern = [
        "checkpoint/dynamodb",
        "store/dynamodb",
        "checkpoint/valkey/saver.py",
        "checkpoint/valkey/base.py",
        "store/valkey",
        "store/agentcore/store.py",
    ]

    for relative_path in components_without_the_pattern:
        target = PACKAGE_ROOT / relative_path
        assert target.exists(), f"expected path not found: {target}"

        paths = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert not _CHECKPOINTER_CONFIG_PATTERN.search(text), (
                f"{path.relative_to(PACKAGE_ROOT)} unexpectedly references "
                "CheckpointerConfig; this component was audited as not using "
                "the required-actor_id config pattern (requirement 3.7). If "
                "this is an intentional new usage, add the file to "
                "EXPECTED_FILES in test_checkpointer_config_usage_audit.py "
                "and verify it needs the _resolve_actor_id inheritance fix."
            )
