from botocore.config import Config

from langchain_aws.agents.base import BedrockAgentFinish
from langchain_aws.agents.utils import (
    SDK_USER_AGENT,
    get_boto_session,
    parse_agent_response,
)


def test_get_boto3_session() -> None:
    client_params, session = get_boto_session()
    assert "config" in client_params
    config = client_params["config"]
    assert SDK_USER_AGENT in config.user_agent_extra  # type: ignore[attr-defined]


def test_get_boto_session_with_config() -> None:
    # Set default client parameters
    fake_config = Config(
        connect_timeout=240,
        read_timeout=240,
        retries={"max_attempts": 1},
    )
    client_params, session = get_boto_session(config=fake_config)
    assert "config" in client_params
    config = client_params["config"]
    # (These attributes exist at runtime but are not declared in botocore-stubs)
    assert SDK_USER_AGENT in config.user_agent_extra  # type: ignore[attr-defined]
    assert config.connect_timeout == fake_config.connect_timeout  # type: ignore[attr-defined]
    assert config.read_timeout == fake_config.read_timeout  # type: ignore[attr-defined]
    assert config.retries["max_attempts"] == fake_config.retries["max_attempts"]  # type: ignore[attr-defined]


def test_get_boto_session_with_user_agent() -> None:
    # Set default client parameters
    fake_config = Config(
        connect_timeout=240,
        read_timeout=240,
        retries={"max_attempts": 1},
        user_agent_extra="MY_USER_AGENT_EXTRA",
    )
    client_params, session = get_boto_session(config=fake_config)
    assert "config" in client_params
    config = client_params["config"]
    assert SDK_USER_AGENT in config.user_agent_extra  # type: ignore[attr-defined]
    # (These attributes exist at runtime but are not declared in botocore-stubs)
    assert fake_config.user_agent_extra in config.user_agent_extra  # type: ignore[attr-defined]
    assert config.connect_timeout == fake_config.connect_timeout  # type: ignore[attr-defined]
    assert config.read_timeout == fake_config.read_timeout  # type: ignore[attr-defined]
    assert config.retries["max_attempts"] == fake_config.retries["max_attempts"]  # type: ignore[attr-defined]


class TestNonAsciiPreservation:
    """Regression tests: non-ASCII characters must survive JSON serialization."""

    _CJK = "日本語テスト"

    def test_return_control_event(self) -> None:
        event_with_cjk = {"returnControl": {"action": self._CJK}}
        response = {
            "completion": [event_with_cjk],
            "sessionId": "test-session",
        }
        result = parse_agent_response(response)
        assert isinstance(result, BedrockAgentFinish)
        output = result.return_values["output"]
        assert self._CJK in output
        assert "\\u" not in output

    def test_trace_log(self) -> None:
        response = {
            "completion": [
                {"trace": {"message": self._CJK}},
                {"chunk": {"bytes": b"done"}},
            ],
            "sessionId": "test-session",
        }
        result = parse_agent_response(response)
        assert isinstance(result, BedrockAgentFinish)
        assert isinstance(result.trace_log, str)
        assert self._CJK in result.trace_log
        assert "\\u" not in result.trace_log
