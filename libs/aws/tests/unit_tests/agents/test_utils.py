from botocore.config import Config

from langchain_aws.agents.utils import SDK_USER_AGENT, get_boto_session


def test_get_boto3_session() -> None:
    client_params, session = get_boto_session()
    assert "config" in client_params
    config = client_params["config"]
    assert SDK_USER_AGENT in config.user_agent_extra


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
    assert SDK_USER_AGENT in config.user_agent_extra
    assert config.connect_timeout == fake_config.connect_timeout
    assert config.read_timeout == fake_config.read_timeout
    assert config.retries["max_attempts"] == fake_config.retries["max_attempts"]


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
    assert SDK_USER_AGENT in config.user_agent_extra
    assert fake_config.user_agent_extra in config.user_agent_extra
    assert config.connect_timeout == fake_config.connect_timeout
    assert config.read_timeout == fake_config.read_timeout
    assert config.retries["max_attempts"] == fake_config.retries["max_attempts"]
