"""Unit tests for S3Backend (split from monolithic test_backend.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from langchain_backend_aws import S3Backend

from ._helpers import _client_error, _make_backend, _s3_object_response

# ------------------------------------------------------------------
# grep()
# ------------------------------------------------------------------


class TestGrep:
    """Tests for the grep method."""

    def test_grep_finds_matches(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "src/app.py",
                        "Size": 100,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator
        mock.get_object.return_value = _s3_object_response(
            b"line one\nfoo bar\nline three\nfoo again"
        )

        result = backend.grep("foo")
        assert result.error is None
        assert result.matches is not None
        assert len(result.matches) == 2
        assert result.matches[0]["line"] == 2
        assert result.matches[0]["text"] == "foo bar"
        assert result.matches[1]["line"] == 4

    def test_grep_invalid_regex(self) -> None:
        backend, _ = _make_backend()
        result = backend.grep("[unterminated")
        assert result.error is not None
        assert "Invalid regex" in result.error

    def test_grep_pattern_too_long_rejected(self) -> None:
        """Patterns longer than ``grep_max_pattern_length`` are rejected.

        ReDoS guard: a long crafted pattern can drive catastrophic
        backtracking on ``re.compile``/``re.search``. We refuse before
        compilation rather than waiting for it to blow up at runtime.
        """
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(
            bucket="b",
            client=mock_client,
            grep_max_pattern_length=16,
        )
        result = backend.grep("a" * 17)
        assert result.error is not None
        assert "grep_max_pattern_length" in result.error
        # ``re.compile`` was never called because we bailed on length.
        mock_client.get_paginator.assert_not_called()

    def test_grep_skips_lines_over_max_line_length(self) -> None:
        """Lines longer than ``grep_max_line_length`` are skipped.

        Long lines amplify regex backtracking cost super-linearly, so
        we never feed them to ``re.search``. The match contained inside
        the long line is therefore intentionally absent from results.
        """
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(
            bucket="b",
            client=mock_client,
            grep_max_line_length=20,
        )
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "a.txt",
                        "Size": 100,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock_client.get_paginator.return_value = paginator
        long_line = "x" * 50 + "needle"
        body = f"short needle\n{long_line}\nshort needle again".encode()
        mock_client.get_object.return_value = _s3_object_response(body)

        result = backend.grep("needle")
        assert result.error is None
        assert result.matches is not None
        # Two short lines match; the long line is silently skipped.
        assert len(result.matches) == 2
        assert all("x" * 50 not in m["text"] for m in result.matches)

    def test_grep_regex_timeout_fails_closed(self) -> None:
        """``regex.search`` TimeoutError surfaces as a fail-closed grep error.

        ``grep_max_pattern_length`` and ``grep_max_line_length`` reduce
        the surface area for ReDoS but cannot bound runtime alone. We
        use the ``regex`` package's ``timeout`` parameter to set a hard
        wall-clock cap; when it fires, ``grep`` must return an error
        instead of returning a partial scan as success or letting the
        exception escape across the protocol boundary.

        We patch ``regex.compile`` to return a stub whose ``.search``
        raises ``TimeoutError`` rather than crafting an actual
        catastrophic-backtracking input. The third-party ``regex``
        engine is internally hardened against many classic ReDoS
        patterns, so a truly pathological input is unreliable — but the
        contract we care about here is the *handler* path, which the
        stub exercises deterministically.
        """
        mock_client = MagicMock()
        backend = S3Backend.from_kwargs(
            bucket="b",
            client=mock_client,
            grep_regex_timeout=0.05,
        )
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "a.txt",
                        "Size": 100,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock_client.get_paginator.return_value = paginator
        mock_client.get_object.return_value = _s3_object_response(b"any line\n")

        stub_pattern = MagicMock()
        stub_pattern.search.side_effect = TimeoutError("regex timed out")

        with patch(
            "langchain_backend_aws.s3._grep.regex_mod.compile",
            return_value=stub_pattern,
        ):
            result = backend.grep("anything")

        assert result.error is not None
        assert "grep_regex_timeout" in result.error
        assert result.matches is None
        # ``search`` was actually invoked with the configured timeout.
        stub_pattern.search.assert_called_once()
        _, kwargs = stub_pattern.search.call_args
        assert kwargs.get("timeout") == 0.05

    def test_grep_with_glob_filter(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "src/app.py",
                        "Size": 100,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "src/notes.txt",
                        "Size": 50,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator
        mock.get_object.return_value = _s3_object_response(b"hello world")

        result = backend.grep("hello", glob="*.py")
        assert result.error is None
        # Only app.py should be fetched; notes.txt is filtered out
        assert mock.get_object.call_count == 1

    def test_grep_skips_oversized_files(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "huge.txt",
                        "Size": 100 * 1024 * 1024,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator

        result = backend.grep("anything")
        assert result.error is None
        assert result.matches == []
        mock.get_object.assert_not_called()

    def test_grep_skips_binary_files(self) -> None:
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "binary.bin",
                        "Size": 10,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator
        mock.get_object.return_value = _s3_object_response(b"\x80\x81\x82")

        result = backend.grep("anything")
        assert result.error is None
        assert result.matches == []

    def test_grep_traversal_blocked(self) -> None:
        backend, _ = _make_backend()
        result = backend.grep("foo", path="/../etc")
        assert result.error is not None

    def test_grep_fails_closed_on_per_object_access_denied(self) -> None:
        """A per-object fetch failure must surface as an error, not empty matches.

        Silently swallowing AccessDenied/SlowDown/transient 5xx would let
        callers conclude a pattern is absent when the backend simply
        could not inspect part of the search space — exactly the kind of
        false negative the cap-overflow contract is designed to prevent.
        """
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "ok.txt",
                        "Size": 10,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "denied.txt",
                        "Size": 10,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator
        mock.get_object.side_effect = [
            _s3_object_response(b"hello"),
            _client_error("AccessDenied"),
        ]

        result = backend.grep("hello")
        assert result.error is not None
        assert "denied.txt" in result.error
        assert "AccessDenied" in result.error
        assert result.matches is None

    def test_grep_fails_closed_on_throttle(self) -> None:
        """Throttling (SlowDown) must fail closed, not return partial matches."""
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "throttled.txt",
                        "Size": 10,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator
        mock.get_object.side_effect = _client_error("SlowDown")

        result = backend.grep("anything")
        assert result.error is not None
        assert "SlowDown" in result.error

    def test_grep_skips_race_no_such_key(self) -> None:
        """A list/read race (NoSuchKey on a freshly-listed object) is skipped.

        Treating the missing object as a benign skip is consistent with
        the snapshot a fresh listing would produce and matches the
        documented contract.
        """
        backend, mock = _make_backend()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "vanished.txt",
                        "Size": 10,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]
        mock.get_paginator.return_value = paginator
        mock.get_object.side_effect = _client_error("NoSuchKey")

        result = backend.grep("anything")
        assert result.error is None
        assert result.matches == []

    def test_grep_max_objects_limit(self) -> None:
        from langchain_backend_aws import S3BackendConfig

        mock = MagicMock()
        config = S3BackendConfig(bucket="test-bucket", grep_max_objects=2)
        backend = S3Backend(config, client=mock)
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": f"f{i}.txt",
                        "Size": 10,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    }
                    for i in range(5)
                ]
            }
        ]
        mock.get_paginator.return_value = paginator
        mock.get_object.return_value = _s3_object_response(b"hello")

        result = backend.grep("hello")
        # Fail closed when scan cap is hit; mirrors glob/ls semantics so
        # callers do not mistake "cap reached" for "no matches".
        assert result.error is not None
        assert "grep_max_objects" in result.error
        assert mock.get_object.call_count == 2
