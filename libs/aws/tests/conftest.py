import io
from typing import Any

import pytest
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config
from vcr import VCR  # type: ignore[import-untyped]


def remove_request_headers(request: Any) -> Any:
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    return request


def _coerce_bytesio(value: Any) -> Any:
    """Recursively replace `BytesIO` objects with their raw bytes.

    Streaming Bedrock responses (notably the legacy
    `invoke_model_with_response_stream` API) surface their body as a
    `BytesIO` object, which the YAML cassette serializer cannot represent.
    `getvalue()` returns the full buffer independent of the read position and
    leaves that position unchanged, so the live `BytesIO` remains fully
    readable by the test while the cassette stores plain bytes.

    Any other stream-like object (e.g. a `botocore` `StreamingBody` exposing
    `read()`, or an `EventStream` exposing only `__iter__`) is rejected loudly
    rather than passed through: silently leaving it in place would either crash
    the serializer later or, worse, record an empty/garbage body that passes on
    replay against meaningless data.
    """
    if isinstance(value, io.BytesIO):
        return value.getvalue()
    if isinstance(value, dict):
        return {k: _coerce_bytesio(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return type(value)(_coerce_bytesio(v) for v in value)
    # Byte/text payloads are iterable but already serializable, so leave them be.
    if isinstance(value, (bytes, bytearray, str)):
        return value
    # Anything still exposing a stream/iterator interface can't be serialized
    # safely; fail loudly at record time rather than write a corrupt cassette.
    if any(hasattr(value, attr) for attr in ("read", "getvalue", "__iter__")):
        msg = (
            f"Unhandled stream-like response body of type {type(value).__name__!r}; "
            "it cannot be safely serialized to a VCR cassette without risking "
            "data loss. Add explicit coercion in _coerce_bytesio."
        )
        raise TypeError(msg)
    return value


def remove_response_headers(response: dict) -> dict:
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    return _coerce_bytesio(response)


def _debug_body_matcher(r1: Any, r2: Any) -> None:
    """Body matcher that enriches VCR's mismatch message with the actual bodies.

    TEMPORARY DEBUG AID (remove once captured). It delegates to VCR's built-in
    ``body`` matcher first, so it raises ``AssertionError`` on mismatch and
    returns ``None`` on match exactly as the default does -- it never changes
    which tests pass or fail. On a genuine mismatch it re-raises with a message
    containing both request bodies, their byte lengths, and the first differing
    offset. VCR surfaces that message in its ``body - assertion failure : ...``
    output, letting the scheduled integration run capture the exact outgoing
    request body for an intermittent, locally-unreproducible cassette mismatch
    (currently ``test_agent_loop[v1]`` on Python 3.13).
    """
    from vcr.matchers import body as _vcr_body
    from vcr.matchers import read_body

    try:
        _vcr_body(r1, r2)
    except AssertionError:
        pass
    else:
        return

    def _as_bytes(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        return repr(value).encode("utf-8")

    raw1 = _as_bytes(read_body(r1))
    raw2 = _as_bytes(read_body(r2))
    first_diff = next(
        (i for i, (x, y) in enumerate(zip(raw1, raw2)) if x != y),
        min(len(raw1), len(raw2)),
    )
    msg = (
        "\n----- VCR BODY MISMATCH (debug) -----\n"
        f"outgoing len={len(raw1)} recorded len={len(raw2)} "
        f"first_diff_offset={first_diff}\n"
        f"outgoing : {raw1!r}\n"
        f"recorded : {raw2!r}\n"
        "----- END VCR BODY MISMATCH -----"
    )
    raise AssertionError(msg)


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config().copy()
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")

    return config


def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
    # TEMPORARY DEBUG: override the built-in "body" matcher to log the actual
    # bodies on mismatch. Remove with the rest of this debug aid once the
    # intermittent test_agent_loop[v1] body diff has been captured in CI.
    vcr.register_matcher("body", _debug_body_matcher)
