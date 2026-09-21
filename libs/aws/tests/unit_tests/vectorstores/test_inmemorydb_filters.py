import operator
from typing import Any, Callable

import pytest

from langchain_aws.vectorstores.inmemorydb.filters import (
    InMemoryDBNum,
    InMemoryDBTag,
    InMemoryDBText,
)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (operator.eq, '@content:("foo\\") | (@ssn:(\\"*")'),
        (operator.ne, '(-@content:"foo\\") | (@ssn:(\\"*")'),
    ],
)
def test_text_phrase_escapes_query_structure(
    op: Callable[..., Any], expected: str
) -> None:
    expression = op(InMemoryDBText("content"), 'foo") | (@ssn:("*')
    assert str(expression) == expected
    assert str((InMemoryDBTag("tenant") == "acme") & expression) == (
        f"(@tenant:{{acme}} {expected})"
    )


@pytest.mark.parametrize(
    "value", ["engine*", "%%engine%%", "engineer|doctor", "engineer doctor"]
)
def test_like_preserves_documented_patterns(value: str) -> None:
    assert str(InMemoryDBText("job") % value) == f"@job:({value})"


def test_like_preserves_ordinary_punctuation() -> None:
    value = "release-1.2, C++/Python: senior_dev's! #1 $5; a=b~c&d^e<z>"
    assert str(InMemoryDBText("job") % value) == f"@job:({value})"


@pytest.mark.parametrize("value", ['x") | (@tenant:{other}', r"x\) | @tenant:{other}"])
def test_like_rejects_query_structure(value: str) -> None:
    with pytest.raises(ValueError, match="unsupported query syntax"):
        InMemoryDBText("content") % value


def test_tag_escapes_value_alternation_but_preserves_lists() -> None:
    assert str(InMemoryDBTag("tenant") == "acme|other") == r"@tenant:{acme\|other}"
    assert str(InMemoryDBTag("tenant") == ["acme|other", "third"]) == (
        r"@tenant:{acme\|other|third}"
    )


@pytest.mark.parametrize(
    "op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]
)
def test_numeric_rejects_query_strings(op: Callable[..., Any]) -> None:
    with pytest.raises(TypeError):
        op(InMemoryDBNum("price"), "0] | @tenant:{other}")


def test_numeric_preserves_negative_decimal() -> None:
    assert str(InMemoryDBNum("price") == -1.5) == "@price:[-1.5 -1.5]"
