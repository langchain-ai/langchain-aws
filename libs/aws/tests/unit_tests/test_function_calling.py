import pytest
from langchain_core.messages import AIMessage

from langchain_aws.function_calling import (
    _get_type,
    _repair_stringified_json_args,
    _repair_stringified_tool_call_message,
)


class TestNonAsciiPreservation:
    _CJK = "日本語テスト"
    _EMOJI = "hello 🌍"

    def test_get_type_anyof(self) -> None:
        param = {"anyOf": [{"type": "string", "description": self._CJK}]}
        result = _get_type(param)
        assert self._CJK in result
        assert "\\u" not in result

    def test_get_type_allof(self) -> None:
        param = {"allOf": [{"type": "object", "title": self._CJK}]}
        result = _get_type(param)
        assert self._CJK in result
        assert "\\u" not in result

    def test_get_type_fallback(self) -> None:
        param = {"description": self._EMOJI}
        result = _get_type(param)
        assert self._EMOJI in result
        assert "\\u" not in result


class TestRepairStringifiedJsonArgs:
    """Repair of tool-call args a model re-serialized as JSON strings (#1221)."""

    _PROPS = {
        "items": {"type": "array"},
        "meta": {"type": "object"},
        "nullable_items": {
            "anyOf": [{"type": "null"}, {"type": "array"}],
        },
        "nullable_meta": {
            "anyOf": [{"type": "null"}, {"type": "object"}],
        },
        "multiple_null_meta": {
            "anyOf": [
                {"type": "null"},
                {"type": "null"},
                {"type": "object"},
            ],
        },
        "ambiguous_scalar": {
            "anyOf": [
                {"type": "array"},
                {"type": "string"},
                {"type": "null"},
            ],
        },
        "ambiguous_containers": {
            "anyOf": [
                {"type": "array"},
                {"type": "object"},
                {"type": "null"},
            ],
        },
        "non_nullable_union": {"anyOf": [{"type": "array"}]},
        "nullable_scalar": {"anyOf": [{"type": "null"}, {"type": "string"}]},
        "boolean_true": True,
        "boolean_false": False,
        "malformed_truthy": "not-a-schema",
        "note": {"type": "string"},
    }

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            # bare stringified array -> unwrapped
            (
                {"items": '[{"label": "a", "values": [1, 2]}]'},
                {"items": [{"label": "a", "values": [1, 2]}]},
            ),
            # self-wrapped stringified array -> unwrapped one level deeper
            (
                {"items": '{"items": [{"label": "a"}]}'},
                {"items": [{"label": "a"}]},
            ),
            # stringified object -> unwrapped
            ({"meta": '{"k": "v"}'}, {"meta": {"k": "v"}}),
            # nullable array -> unwrapped
            (
                {"nullable_items": '[{"label": "a"}]'},
                {"nullable_items": [{"label": "a"}]},
            ),
            # nullable object -> unwrapped
            (
                {"nullable_meta": '{"k": "v"}'},
                {"nullable_meta": {"k": "v"}},
            ),
            # multiple null branches plus one container -> unwrapped
            (
                {"multiple_null_meta": '{"k": "v"}'},
                {"multiple_null_meta": {"k": "v"}},
            ),
            # nullable self-wrapped stringified array -> unwrapped one level deeper
            (
                {"nullable_items": '{"nullable_items": [{"label": "a"}]}'},
                {"nullable_items": [{"label": "a"}]},
            ),
            # declared-string field: JSON-looking value left untouched
            (
                {"note": '["looks", "like", "json"]'},
                {"note": '["looks", "like", "json"]'},
            ),
            # native null values pass through unchanged
            ({"nullable_items": None}, {"nullable_items": None}),
            # textual null decodes to a scalar and remains untouched
            ({"nullable_items": "null"}, {"nullable_items": "null"}),
            # malformed JSON left untouched (fails loudly downstream)
            ({"items": '[{"label": broken'}, {"items": '[{"label": broken'}),
            # parses but doesn't match the declared type -> untouched
            ({"items": '{"unrelated": 1}'}, {"items": '{"unrelated": 1}'}),
            # nullable container with the wrong decoded type -> untouched
            ({"nullable_meta": "[]"}, {"nullable_meta": "[]"}),
            # multiple non-null anyOf branches are ambiguous and remain untouched
            ({"ambiguous_scalar": "[]"}, {"ambiguous_scalar": "[]"}),
            ({"ambiguous_containers": "[]"}, {"ambiguous_containers": "[]"}),
            # a non-nullable anyOf does not identify a repairable container
            ({"non_nullable_union": "[]"}, {"non_nullable_union": "[]"}),
            # nullable scalar fields are not repaired
            ({"nullable_scalar": '"value"'}, {"nullable_scalar": '"value"'}),
            # non-dict property schemas are left untouched
            ({"boolean_true": "[]"}, {"boolean_true": "[]"}),
            ({"boolean_false": "[]"}, {"boolean_false": "[]"}),
            ({"malformed_truthy": "[]"}, {"malformed_truthy": "[]"}),
            # well-formed args pass through unchanged
            (
                {"items": [{"label": "a"}], "meta": {"k": "v"}, "note": "v1"},
                {"items": [{"label": "a"}], "meta": {"k": "v"}, "note": "v1"},
            ),
            # field not in the schema properties -> untouched
            ({"extra": "[1, 2]"}, {"extra": "[1, 2]"}),
        ],
        ids=[
            "bare-array",
            "self-wrapped-array",
            "object",
            "nullable-array",
            "nullable-object",
            "multiple-null-object",
            "nullable-self-wrapped-array",
            "declared-string-untouched",
            "native-null-untouched",
            "textual-null-untouched",
            "malformed-json-untouched",
            "type-mismatch-untouched",
            "nullable-type-mismatch-untouched",
            "ambiguous-scalar-union-untouched",
            "ambiguous-container-union-untouched",
            "non-nullable-union-untouched",
            "nullable-scalar-untouched",
            "boolean-true-schema-untouched",
            "boolean-false-schema-untouched",
            "truthy-malformed-schema-untouched",
            "well-formed-passthrough",
            "unknown-field-untouched",
        ],
    )
    def test_repair(self, args: dict, expected: dict) -> None:
        assert _repair_stringified_json_args(args, self._PROPS) == expected


class TestRepairStringifiedToolCallMessage:
    _PROPS = {"items": {"type": "array"}}

    def test_repairs_tool_call_args(self) -> None:
        message = AIMessage(
            "",
            tool_calls=[
                {
                    "name": "Output",
                    "args": {"items": '[{"label": "a"}]'},
                    "id": "toolu_01",
                    "type": "tool_call",
                }
            ],
        )
        repaired = _repair_stringified_tool_call_message(message, self._PROPS)
        assert repaired.tool_calls[0]["args"]["items"] == [{"label": "a"}]

    def test_well_formed_message_is_returned_unchanged(self) -> None:
        message = AIMessage(
            "",
            tool_calls=[
                {
                    "name": "Output",
                    "args": {"items": [{"label": "a"}]},
                    "id": "toolu_01",
                    "type": "tool_call",
                }
            ],
        )
        assert _repair_stringified_tool_call_message(message, self._PROPS) is message
