from langchain_aws.function_calling import _get_type


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
