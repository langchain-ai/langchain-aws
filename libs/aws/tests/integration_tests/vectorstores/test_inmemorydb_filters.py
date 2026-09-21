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
        == {"phrase", "hyphen"} - matching
    )


@pytest.mark.parametrize("op", [operator.eq, operator.ne, operator.mod])
@pytest.mark.parametrize(
    "value",
    ['x") | (@tenant:{other}', 'x\\") | (@tenant:{other}', "x) | (@tenant:{other}"],
)
def test_filter_cannot_escape_scope(
    search: Callable[[str], set[str]], op: Callable[..., Any], value: str
) -> None:
    expression = (InMemoryDBTag("tenant") == "acme") & op(InMemoryDBText("job"), value)
    assert "other" not in search(str(expression))
