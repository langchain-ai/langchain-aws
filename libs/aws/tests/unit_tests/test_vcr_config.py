import inspect
import os

import pytest

from tests.conftest import vcr_config


@pytest.mark.parametrize("gateway", [None, "true", "https://gateway.example.test"])
def test_vcr_config_isolates_gateway(
    monkeypatch: pytest.MonkeyPatch, gateway: str | None
) -> None:
    keys = ("LANGSMITH_GATEWAY", "LANGSMITH_GATEWAY_API_KEY")
    for key, value in zip(keys, (gateway, "synthetic-gateway-key")):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    before = {key: os.environ.get(key) for key in keys}

    with pytest.MonkeyPatch.context() as cassette_env:
        inspect.unwrap(vcr_config)(cassette_env)
        assert all(key not in os.environ for key in keys)

    assert {key: os.environ.get(key) for key in keys} == before
