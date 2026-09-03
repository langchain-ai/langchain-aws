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
            # declared-string field: JSON-looking value left untouched
            (
                {"note": '["looks", "like", "json"]'},
                {"note": '["looks", "like", "json"]'},
            ),
            # malformed JSON left untouched (fails loudly downstream)
            ({"items": '[{"label": broken'}, {"items": '[{"label": broken'}),
            # parses but doesn't match the declared type -> untouched
            ({"items": '{"unrelated": 1}'}, {"items": '{"unrelated": 1}'}),
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
            "declared-string-untouched",
            "malformed-json-untouched",
            "type-mismatch-untouched",
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
