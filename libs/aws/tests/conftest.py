import io
import re
from typing import Any
from urllib.parse import urlparse

import pytest
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config
from vcr import VCR  # type: ignore[import-untyped]

_AWS_REGION = re.compile(r"^[a-z]{2,3}-(gov-)?[a-z]+-\d+$")


def _uri_without_aws_region(uri: str) -> str:
    parsed = urlparse(uri)
    host = ".".join(
        p for p in (parsed.hostname or "").split(".") if not _AWS_REGION.match(p)
    )
    return f"{host}{parsed.path}"


def region_agnostic_uri_matcher(r1: Any, r2: Any) -> None:
    """Match URIs while ignoring the AWS region segment of the endpoint host."""
    assert _uri_without_aws_region(r1.uri) == _uri_without_aws_region(r2.uri), (
        f"{r1.uri} != {r2.uri}"
    )


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


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests."""
    config = base_vcr_config().copy()
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    config["match_on"] = ["method", "region_agnostic_uri", "body"]

    return config


def pytest_recording_configure(config: dict, vcr: VCR) -> None:
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())
    vcr.register_matcher("region_agnostic_uri", region_agnostic_uri_matcher)
