import logging

import pytest

from tests.utils import is_valkey_available

if is_valkey_available():
    from langgraph_checkpoint_aws.agentcore.valkey import AgentCoreValkeySaver


logger = logging.getLogger(__name__)


TTL_SECONDS: int = 600  # 10 minutes


@pytest.fixture(scope="function")
def agentcore_valkey_saver():
    """Create Valkey saver instance for integration tests."""

    uri = "valkey://localhost:6379/1"

    def _delete_keys(saver: AgentCoreValkeySaver):
        """Cleanup test keys from Valkey server."""
        try:
            session_keys = saver.client.keys("agentcore:session:test-*")
            checkpoint_keys = saver.client.keys("agentcore:checkpoint:test-*")
            writes_keys = saver.client.keys("agentcore:writes:test-*")
            channel_keys = saver.client.keys("agentcore:channel:test-*")

            all_keys = (
                list(session_keys)  # type: ignore[arg-type]
                + list(checkpoint_keys)  # type: ignore[arg-type]
                + list(writes_keys)  # type: ignore[arg-type]
                + list(channel_keys)  # type: ignore[arg-type]
            )
            if all_keys:
                saver.client.delete(*all_keys)
        except Exception:
            logger.exception("Failed to cleanup test keys from Valkey server")

    try:
        with AgentCoreValkeySaver.from_conn_string(
            uri,
            ttl_seconds=TTL_SECONDS,
        ) as saver:
            _delete_keys(saver)
            yield saver
            _delete_keys(saver)
    except Exception as e:
        pytest.skip(f"Could not connect to Valkey server: {e}")
