"""Regression test for the double-percent-encoded traversal contract.

:func:`langchain_backend_aws.s3._paths.path_to_key` only decodes a
single layer of percent-encoding (``%2e%2e`` is rejected as an encoded
``..``). Doubly-encoded variants such as ``%252e%252e`` must be treated
as user data — *not* recursively decoded into ``..`` — because the
upstream :func:`deepagents.backends.utils.validate_path` performs at
most one decode pass before reaching the helper. If a future upstream
change introduces a second pass, the assumption breaks and the encoded
traversal slips past the segment scan; this test pins the contract so
the regression is loud rather than silent.
"""

from __future__ import annotations

from langchain_backend_aws.s3._paths import path_to_key


class TestDoubleEncodedTraversal:
    def test_double_encoded_dot_dot_is_data_not_traversal(self) -> None:
        # ``%252e%252e`` should be treated as the literal 12-character
        # string and resolve into the key as data. It must NOT decode
        # to ``..`` and trip the traversal segment guard.
        key = path_to_key("/foo/%252e%252e/bar.txt", prefix="t/")
        assert key == "t/foo/%252e%252e/bar.txt"

    def test_double_encoded_slash_is_data_not_traversal(self) -> None:
        # Same contract for ``%252f`` (would decode to ``%2f`` then
        # ``/`` if recursively decoded).
        key = path_to_key("/foo/%252fbar.txt", prefix="t/")
        assert key == "t/foo/%252fbar.txt"

    def test_double_encoded_segment_only_dot_passes(self) -> None:
        # A segment that *would* decode to ``%2e`` (encoded dot) on a
        # single pass is data on a single pass too — the guard rejects
        # only the literal ``%2e`` segment that would itself decode to
        # ``.`` once.
        key = path_to_key("/foo/%252e/bar.txt", prefix="")
        assert key == "foo/%252e/bar.txt"
