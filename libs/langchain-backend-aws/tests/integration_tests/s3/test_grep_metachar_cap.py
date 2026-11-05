"""Live-store check for ``grep_max_pattern_metachars`` ReDoS guard."""

from __future__ import annotations

from .conftest import make_backend, skip_without_credentials

pytestmark = skip_without_credentials


def test_grep_metachar_cap_live(prefix: str) -> None:
    backend = make_backend(prefix, grep_max_pattern_metachars=10)
    backend.write("/sample.txt", "hello world\n")

    pattern = "(" * 6 + "a" + ")" * 6 + "+"
    result = backend.grep(pattern)

    assert result.error is not None
    assert "metacharacter" in result.error
