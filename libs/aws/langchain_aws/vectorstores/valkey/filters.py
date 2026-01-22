from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Set, Tuple, Union

from langchain_aws.utilities.redis import TokenEscaper

# mypy: disable-error-code="override"


class ValkeyFilterOperator(Enum):
    """ValkeyFilterOperator enumerator for creating ValkeyFilterExpressions."""

    EQ = 1
    NE = 2
    LT = 3
    GT = 4
    LE = 5
    GE = 6
    OR = 7
    AND = 8
    LIKE = 9
    IN = 10


class ValkeyFilter:
    """Collection of ValkeyFilterFields."""

    @staticmethod
    def text(field: str) -> "ValkeyText":
        return ValkeyText(field)

    @staticmethod
    def num(field: str) -> "ValkeyNum":
        return ValkeyNum(field)

    @staticmethod
    def tag(field: str) -> "ValkeyTag":
        return ValkeyTag(field)


class ValkeyFilterField:
    """Base class for ValkeyFilterFields."""

    escaper: "TokenEscaper" = TokenEscaper()
    OPERATORS: Dict[ValkeyFilterOperator, str] = {}

    def __init__(self, field: str):
        self._field = field
        self._value: Any = None
        self._operator: ValkeyFilterOperator = ValkeyFilterOperator.EQ

    def equals(self, other: "ValkeyFilterField") -> bool:
        if not isinstance(other, type(self)):
            return False
        return self._field == other._field and self._value == other._value

    def _set_value(
        self, val: Any, val_type: Tuple[Any], operator: ValkeyFilterOperator
    ) -> None:
        if operator not in self.OPERATORS:
            raise ValueError(
                f"Operator {operator} not supported by {self.__class__.__name__}. "
                + f"Supported operators are {self.OPERATORS.values()}."
            )

        if not isinstance(val, val_type):
            raise TypeError(
                f"Right side argument passed to operator {self.OPERATORS[operator]} "
                f"with left side "
                f"argument {self.__class__.__name__} must be of type {val_type}, "
                f"received value {val}"
            )
        self._value = val
        self._operator = operator


def check_operator_misuse(func: Callable) -> Callable:
    """Decorator to check for misuse of equality operators."""

    @wraps(func)
    def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
        other = kwargs.get("other") if "other" in kwargs else None
        if not other:
            for arg in args:
                if isinstance(arg, type(instance)):
                    other = arg
                    break

        if isinstance(other, type(instance)):
            raise ValueError(
                "Equality operators are overridden for FilterExpression creation. Use "
                ".equals() for equality checks"
            )
        return func(instance, *args, **kwargs)

    return wrapper


class ValkeyTag(ValkeyFilterField):
    """ValkeyFilterField representing a tag in a Valkey index."""

    OPERATORS: Dict[ValkeyFilterOperator, str] = {
        ValkeyFilterOperator.EQ: "==",
        ValkeyFilterOperator.NE: "!=",
        ValkeyFilterOperator.IN: "==",
    }
    OPERATOR_MAP: Dict[ValkeyFilterOperator, str] = {
        ValkeyFilterOperator.EQ: "@%s:{%s}",
        ValkeyFilterOperator.NE: "(-@%s:{%s})",
        ValkeyFilterOperator.IN: "@%s:{%s}",
    }
    SUPPORTED_VAL_TYPES = (list, set, tuple, str, type(None))

    def _set_tag_value(
        self,
        other: Union[List[str], Set[str], Tuple[str], str],
        operator: ValkeyFilterOperator,
    ) -> None:
        if isinstance(other, (list, set, tuple)):
            try:
                other = [str(val) for val in other if val]
            except ValueError:
                raise ValueError("All tags within collection must be strings")
        elif not other:
            other = []
        elif isinstance(other, str):
            other = [other]

        self._set_value(other, self.SUPPORTED_VAL_TYPES, operator)  # type: ignore

    @check_operator_misuse
    def __eq__(
        self, other: Union[List[str], Set[str], Tuple[str], str]
    ) -> "ValkeyFilterExpression":
        """Create a ValkeyTag equality filter expression."""
        self._set_tag_value(other, ValkeyFilterOperator.EQ)
        return ValkeyFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(
        self, other: Union[List[str], Set[str], Tuple[str], str]
    ) -> "ValkeyFilterExpression":
        """Create a ValkeyTag inequality filter expression."""
        self._set_tag_value(other, ValkeyFilterOperator.NE)
        return ValkeyFilterExpression(str(self))

    @property
    def _formatted_tag_value(self) -> str:
        return "|".join([self.escaper.escape(tag) for tag in self._value])

    def __str__(self) -> str:
        if not self._value:
            return "*"

        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            self._formatted_tag_value,
        )


class ValkeyNum(ValkeyFilterField):
    """ValkeyFilterField representing a numeric field in a Valkey index."""

    OPERATORS: Dict[ValkeyFilterOperator, str] = {
        ValkeyFilterOperator.EQ: "==",
        ValkeyFilterOperator.NE: "!=",
        ValkeyFilterOperator.LT: "<",
        ValkeyFilterOperator.GT: ">",
        ValkeyFilterOperator.LE: "<=",
        ValkeyFilterOperator.GE: ">=",
    }
    OPERATOR_MAP: Dict[ValkeyFilterOperator, str] = {
        ValkeyFilterOperator.EQ: "@%s:[%s %s]",
        ValkeyFilterOperator.NE: "(-@%s:[%s %s])",
        ValkeyFilterOperator.GT: "@%s:[(%s +inf]",
        ValkeyFilterOperator.LT: "@%s:[-inf (%s]",
        ValkeyFilterOperator.GE: "@%s:[%s +inf]",
        ValkeyFilterOperator.LE: "@%s:[-inf %s]",
    }
    SUPPORTED_VAL_TYPES = (int, float, type(None))

    def __str__(self) -> str:
        if self._value is None:
            return "*"

        if (
            self._operator == ValkeyFilterOperator.EQ
            or self._operator == ValkeyFilterOperator.NE
        ):
            return self.OPERATOR_MAP[self._operator] % (
                self._field,
                self._value,
                self._value,
            )
        else:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)

    @check_operator_misuse
    def __eq__(self, other: Union[int, float]) -> "ValkeyFilterExpression":
        """Create a Numeric equality filter expression."""
        self._set_value(other, self.SUPPORTED_VAL_TYPES, ValkeyFilterOperator.EQ)  # type: ignore
        return ValkeyFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: Union[int, float]) -> "ValkeyFilterExpression":
        """Create a Numeric inequality filter expression."""
        self._set_value(other, self.SUPPORTED_VAL_TYPES, ValkeyFilterOperator.NE)  # type: ignore
        return ValkeyFilterExpression(str(self))

    def __gt__(self, other: Union[int, float]) -> "ValkeyFilterExpression":
        """Create a Numeric greater than filter expression."""
        self._set_value(other, self.SUPPORTED_VAL_TYPES, ValkeyFilterOperator.GT)  # type: ignore
        return ValkeyFilterExpression(str(self))

    def __lt__(self, other: Union[int, float]) -> "ValkeyFilterExpression":
        """Create a Numeric less than filter expression."""
        self._set_value(other, self.SUPPORTED_VAL_TYPES, ValkeyFilterOperator.LT)  # type: ignore
        return ValkeyFilterExpression(str(self))

    def __ge__(self, other: Union[int, float]) -> "ValkeyFilterExpression":
        """Create a Numeric greater than or equal to filter expression."""
        self._set_value(other, self.SUPPORTED_VAL_TYPES, ValkeyFilterOperator.GE)  # type: ignore
        return ValkeyFilterExpression(str(self))

    def __le__(self, other: Union[int, float]) -> "ValkeyFilterExpression":
        """Create a Numeric less than or equal to filter expression."""
        self._set_value(other, self.SUPPORTED_VAL_TYPES, ValkeyFilterOperator.LE)  # type: ignore
        return ValkeyFilterExpression(str(self))


class ValkeyText(ValkeyFilterField):
    """ValkeyFilterField representing a text field in a Valkey index."""

    OPERATORS: Dict[ValkeyFilterOperator, str] = {
        ValkeyFilterOperator.EQ: "==",
        ValkeyFilterOperator.NE: "!=",
        ValkeyFilterOperator.LIKE: "%",
    }
    OPERATOR_MAP: Dict[ValkeyFilterOperator, str] = {
        ValkeyFilterOperator.EQ: '@%s:("%s")',
        ValkeyFilterOperator.NE: '(-@%s:"%s")',
        ValkeyFilterOperator.LIKE: "@%s:(%s)",
    }
    SUPPORTED_VAL_TYPES = (str, type(None))

    @check_operator_misuse
    def __eq__(self, other: str) -> "ValkeyFilterExpression":
        """Create a ValkeyText equality (exact match) filter expression."""
        self._set_value(other, self.SUPPORTED_VAL_TYPES, ValkeyFilterOperator.EQ)  # type: ignore
        return ValkeyFilterExpression(str(self))

    @check_operator_misuse
    def __ne__(self, other: str) -> "ValkeyFilterExpression":
        """Create a ValkeyText inequality filter expression."""
        self._set_value(other, self.SUPPORTED_VAL_TYPES, ValkeyFilterOperator.NE)  # type: ignore
        return ValkeyFilterExpression(str(self))

    def __mod__(self, other: str) -> "ValkeyFilterExpression":
        """Create a ValkeyText "LIKE" filter expression."""
        self._set_value(other, self.SUPPORTED_VAL_TYPES, ValkeyFilterOperator.LIKE)  # type: ignore
        return ValkeyFilterExpression(str(self))

    def __str__(self) -> str:
        if self._value is None:
            return "*"

        if self._operator == ValkeyFilterOperator.LIKE:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)
        else:
            return self.OPERATOR_MAP[self._operator] % (
                self._field,
                self.escaper.escape(self._value),
            )


class ValkeyFilterExpression:
    """Logical expression of Valkey filters."""

    def __init__(self, *args: Union[str, "ValkeyFilterExpression"]):
        self._filter_expression = " ".join([str(arg) for arg in args])

    def __and__(self, other: "ValkeyFilterExpression") -> "ValkeyFilterExpression":
        return ValkeyFilterExpression(f"({self._filter_expression} {other})")

    def __or__(self, other: "ValkeyFilterExpression") -> "ValkeyFilterExpression":
        return ValkeyFilterExpression(f"({self._filter_expression} | {other})")

    def __str__(self) -> str:
        return self._filter_expression


# Type alias for filter expressions
ValkeyFilterType = Union[ValkeyFilterExpression, str]
