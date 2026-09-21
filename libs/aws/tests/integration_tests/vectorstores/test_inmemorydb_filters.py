import json
import operator
import os
import subprocess
from typing import Any, Callable, Iterator
from uuid import uuid4

import pytest

from langchain_aws.vectorstores.inmemorydb.filters import InMemoryDBTag, InMemoryDBText


@pytest.fixture
def search() -> Iterator[Callable[[str], set[str]]]:
    """Use redis-cli in an explicitly supplied Redis Search Docker container."""
    container = os.environ.get("REDIS_SEARCH_TEST_CONTAINER")
    if not container:
        pytest.skip("Set REDIS_SEARCH_TEST_CONTAINER to a Redis Search container")

    def command(*args: str) -> Any:
        return json.loads(
            subprocess.check_output(
                ["docker", "exec", container, "redis-cli", "-2", "--json", *args],
                text=True,
            )
        )

    index = f"filter-test-{uuid4().hex}"
    command(
        "FT.CREATE",
        index,
        "ON",
        "HASH",
        "PREFIX",
        "1",
        f"{index}:",
        "SCHEMA",
        "job",
        "TEXT",
        "tenant",
        "TAG",
    )
    try:
        for key, job, tenant in [
            ("phrase", "software engineer", "acme"),
            ("hyphen", "test-value", "acme"),
            ("comma", "hello,world", "acme"),
            ("doctor", "medical doctor", "acme"),
            ("both", "engineer doctor", "acme"),
            ("other", "software engineer", "other"),
        ]:
            command("HSET", f"{index}:{key}", "job", job, "tenant", tenant)

        def query(expression: str) -> set[str]:
            result = command(
                "FT.SEARCH", index, expression, "NOCONTENT", "DIALECT", "2"
            )
            assert isinstance(result, list), result
            return {key.removeprefix(f"{index}:") for key in result[1:]}

        yield query
    finally:
        command("FT.DROPINDEX", index, "DD")


@pytest.mark.parametrize(
    ("value", "matching"),
    [
        ("software engineer", {"phrase"}),
        ("test-value", {"hyphen"}),
    ],
)
def test_phrase_matching(
    search: Callable[[str], set[str]], value: str, matching: set[str]
) -> None:
    scope = InMemoryDBTag("tenant") == "acme"
    assert search(str(scope & (InMemoryDBText("job") == value))) == matching
    assert (
        search(str(scope & (InMemoryDBText("job") != value)))
        == {"phrase", "hyphen", "comma", "doctor", "both"} - matching
    )


@pytest.mark.parametrize(
    ("value", "matching"),
    [
        ("engine*", {"phrase", "both"}),
        ("%%engineer%%", {"phrase", "both"}),
        ("engineer|doctor", {"phrase", "doctor", "both"}),
        ("engineer doctor", {"both"}),
    ],
)
def test_like_documented_patterns(
    search: Callable[[str], set[str]], value: str, matching: set[str]
) -> None:
    scope = InMemoryDBTag("tenant") == "acme"
    assert search(str(scope & (InMemoryDBText("job") % value))) == matching


def test_like_preserves_ordinary_punctuation(
    search: Callable[[str], set[str]],
) -> None:
    expression = InMemoryDBText("job") % "hello,world"
    assert str(expression) == "@job:(hello,world)"
    assert search(str(expression)) == {"comma"}
    hyphen = InMemoryDBText("job") % "test-value"
    assert str(hyphen) == "@job:(test-value)"
    assert search(str(hyphen)) == search("@job:(test-value)")


@pytest.mark.parametrize("value", ['x") | (@tenant:{other}', r"x\) | @tenant:{other}"])
def test_like_injection_raises(value: str) -> None:
    with pytest.raises(ValueError, match="unsupported query syntax"):
        InMemoryDBText("job") % value


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
def test_phrase_injection_cannot_escape_scope(
    search: Callable[[str], set[str]], op: Callable[..., Any]
) -> None:
    scope = InMemoryDBTag("tenant") == "acme"
    expression = scope & op(InMemoryDBText("job"), 'x") | (@tenant:{other}')
    assert "other" not in search(str(expression))
