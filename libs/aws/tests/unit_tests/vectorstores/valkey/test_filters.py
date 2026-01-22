"""Unit tests for Valkey filter expressions."""

import pytest

from langchain_aws.vectorstores.valkey.filters import (
    ValkeyFilter,
    ValkeyFilterExpression,
    ValkeyFilterOperator,
    ValkeyNum,
    ValkeyTag,
    ValkeyText,
)


class TestValkeyFilter:
    """Test ValkeyFilter factory methods."""

    def test_text(self) -> None:
        field = ValkeyFilter.text("content")
        assert isinstance(field, ValkeyText)
        assert field._field == "content"

    def test_num(self) -> None:
        field = ValkeyFilter.num("price")
        assert isinstance(field, ValkeyNum)
        assert field._field == "price"

    def test_tag(self) -> None:
        field = ValkeyFilter.tag("category")
        assert isinstance(field, ValkeyTag)
        assert field._field == "category"


class TestValkeyTag:
    """Test ValkeyTag filter field."""

    def test_eq_single_tag(self) -> None:
        expr = ValkeyFilter.tag("category") == "electronics"
        assert str(expr) == "@category:{electronics}"

    def test_eq_multiple_tags(self) -> None:
        expr = ValkeyFilter.tag("category") == ["electronics", "computers"]
        assert str(expr) == "@category:{electronics|computers}"

    def test_eq_empty_list(self) -> None:
        expr = ValkeyFilter.tag("category") == []
        assert str(expr) == "*"

    def test_eq_none(self) -> None:
        expr = ValkeyFilter.tag("category") == None
        assert str(expr) == "*"

    def test_ne_single_tag(self) -> None:
        expr = ValkeyFilter.tag("category") != "electronics"
        assert str(expr) == "(-@category:{electronics})"

    def test_ne_multiple_tags(self) -> None:
        expr = ValkeyFilter.tag("category") != ["electronics", "computers"]
        assert str(expr) == "(-@category:{electronics|computers})"

    def test_tag_escaping(self) -> None:
        expr = ValkeyFilter.tag("category") == "test-value"
        assert str(expr) == "@category:{test\\-value}"

    def test_tag_set(self) -> None:
        expr = ValkeyFilter.tag("category") == {"electronics", "computers"}
        result = str(expr)
        assert result.startswith("@category:{")
        assert "electronics" in result
        assert "computers" in result

    def test_tag_tuple(self) -> None:
        expr = ValkeyFilter.tag("category") == ("electronics", "computers")
        assert str(expr) == "@category:{electronics|computers}"

    def test_equals_method(self) -> None:
        tag1 = ValkeyFilter.tag("category")
        tag1._value = ["electronics"]
        tag2 = ValkeyFilter.tag("category")
        tag2._value = ["electronics"]
        assert tag1.equals(tag2)

    def test_equals_different_field(self) -> None:
        tag1 = ValkeyFilter.tag("category")
        tag1._value = ["electronics"]
        tag2 = ValkeyFilter.tag("type")
        tag2._value = ["electronics"]
        assert not tag1.equals(tag2)

    def test_equals_different_type(self) -> None:
        tag = ValkeyFilter.tag("category")
        num = ValkeyFilter.num("price")
        assert not tag.equals(num)

    def test_operator_misuse(self) -> None:
        tag1 = ValkeyFilter.tag("category")
        tag2 = ValkeyFilter.tag("type")
        with pytest.raises(ValueError, match="Equality operators are overridden"):
            tag1 == tag2


class TestValkeyNum:
    """Test ValkeyNum filter field."""

    def test_eq(self) -> None:
        expr = ValkeyFilter.num("price") == 100
        assert str(expr) == "@price:[100 100]"

    def test_eq_float(self) -> None:
        expr = ValkeyFilter.num("price") == 99.99
        assert str(expr) == "@price:[99.99 99.99]"

    def test_ne(self) -> None:
        expr = ValkeyFilter.num("price") != 100
        assert str(expr) == "(-@price:[100 100])"

    def test_gt(self) -> None:
        expr = ValkeyFilter.num("price") > 100
        assert str(expr) == "@price:[(100 +inf]"

    def test_lt(self) -> None:
        expr = ValkeyFilter.num("price") < 100
        assert str(expr) == "@price:[-inf (100]"

    def test_ge(self) -> None:
        expr = ValkeyFilter.num("price") >= 100
        assert str(expr) == "@price:[100 +inf]"

    def test_le(self) -> None:
        expr = ValkeyFilter.num("price") <= 100
        assert str(expr) == "@price:[-inf 100]"

    def test_none_value(self) -> None:
        num = ValkeyFilter.num("price")
        num._value = None
        assert str(num) == "*"

    def test_invalid_type(self) -> None:
        num = ValkeyFilter.num("price")
        with pytest.raises(TypeError):
            num._set_value("invalid", (int, float, type(None)), ValkeyFilterOperator.EQ)

    def test_unsupported_operator(self) -> None:
        num = ValkeyFilter.num("price")
        with pytest.raises(ValueError, match="Operator .* not supported"):
            num._set_value(100, (int, float, type(None)), ValkeyFilterOperator.LIKE)

    def test_operator_misuse(self) -> None:
        num1 = ValkeyFilter.num("price")
        num2 = ValkeyFilter.num("cost")
        with pytest.raises(ValueError, match="Equality operators are overridden"):
            num1 == num2


class TestValkeyText:
    """Test ValkeyText filter field."""

    def test_eq(self) -> None:
        expr = ValkeyFilter.text("title") == "laptop"
        assert str(expr) == '@title:("laptop")'

    def test_ne(self) -> None:
        expr = ValkeyFilter.text("title") != "laptop"
        assert str(expr) == '(-@title:"laptop")'

    def test_like(self) -> None:
        expr = ValkeyFilter.text("title") % "lap*"
        assert str(expr) == "@title:(lap*)"

    def test_text_escaping(self) -> None:
        expr = ValkeyFilter.text("title") == "test-value"
        assert str(expr) == '@title:("test\\-value")'

    def test_none_value(self) -> None:
        text = ValkeyFilter.text("title")
        text._value = None
        assert str(text) == "*"

    def test_invalid_type(self) -> None:
        text = ValkeyFilter.text("title")
        with pytest.raises(TypeError):
            text._set_value(123, (str, type(None)), ValkeyFilterOperator.EQ)

    def test_operator_misuse(self) -> None:
        text1 = ValkeyFilter.text("title")
        text2 = ValkeyFilter.text("description")
        with pytest.raises(ValueError, match="Equality operators are overridden"):
            text1 == text2


class TestValkeyFilterExpression:
    """Test ValkeyFilterExpression logical operations."""

    def test_and_operation(self) -> None:
        expr1 = ValkeyFilter.tag("category") == "electronics"
        expr2 = ValkeyFilter.num("price") > 100
        combined = expr1 & expr2
        assert str(combined) == "(@category:{electronics} @price:[(100 +inf])"

    def test_or_operation(self) -> None:
        expr1 = ValkeyFilter.tag("category") == "electronics"
        expr2 = ValkeyFilter.tag("category") == "computers"
        combined = expr1 | expr2
        assert str(combined) == "(@category:{electronics} | @category:{computers})"

    def test_complex_expression(self) -> None:
        expr1 = ValkeyFilter.tag("category") == "electronics"
        expr2 = ValkeyFilter.num("price") > 100
        expr3 = ValkeyFilter.num("price") < 500
        combined = (expr1 & expr2) & expr3
        assert (
            str(combined)
            == "((@category:{electronics} @price:[(100 +inf]) @price:[-inf (500])"
        )

    def test_multiple_args_init(self) -> None:
        expr = ValkeyFilterExpression("@category:{electronics}", "@price:[100 +inf]")
        assert str(expr) == "@category:{electronics} @price:[100 +inf]"

    def test_nested_expression(self) -> None:
        expr1 = ValkeyFilter.tag("category") == "electronics"
        expr2 = ValkeyFilter.num("price") > 100
        expr3 = ValkeyFilter.text("title") % "laptop*"
        combined = (expr1 & expr2) | expr3
        assert (
            str(combined)
            == "((@category:{electronics} @price:[(100 +inf]) | @title:(laptop*))"
        )
