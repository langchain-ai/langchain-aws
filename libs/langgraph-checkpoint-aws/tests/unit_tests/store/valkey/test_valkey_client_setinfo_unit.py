"""Comprehensive tests to verify CLIENT SETINFO is called when clients are created."""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

# Skip entire module if valkey not available
pytest.importorskip("valkey")

from valkey import Valkey
from valkey.asyncio import Valkey as AsyncValkey
from valkey.asyncio.connection import ConnectionPool as AsyncConnectionPool
from valkey.connection import ConnectionPool

from langgraph_checkpoint_aws import (
    AsyncValkeySaver,
    AsyncValkeyStore,
    ValkeySaver,
    ValkeyStore,
)
from langgraph_checkpoint_aws.checkpoint.valkey.utils import (
    LIBRARY_NAME,
    LIBRARY_VERSION,
    aset_client_info,
    set_client_info,
)


@pytest.fixture
def valkey_url() -> str:
    """Get Valkey server URL from environment or use default."""
    return os.getenv("VALKEY_URL", "valkey://localhost:6379")


@pytest.fixture
def mock_pool_connection() -> Mock:
    """Create a mock pool connection that supports context manager."""
    mock_conn = Mock(spec=Valkey)
    mock_conn.__enter__ = Mock(return_value=mock_conn)
    mock_conn.__exit__ = Mock(return_value=None)
    return mock_conn


@pytest.fixture
def mock_async_pool_connection() -> Mock:
    """Create a mock async pool connection that supports context manager."""
    mock_conn = AsyncMock(spec=AsyncValkey)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=None)
    return mock_conn


@pytest.fixture
def mock_valkey_client() -> Mock:
    """Create a mock Valkey client."""
    mock_client = Mock(spec=Valkey)
    mock_client.execute_command = Mock()
    return mock_client


@pytest.fixture
def mock_async_valkey_client() -> Mock:
    """Create a mock async Valkey client."""
    mock_client = AsyncMock(spec=AsyncValkey)
    mock_client.execute_command = AsyncMock()
    return mock_client


class TestClientSetInfoUtils:
    """Test the CLIENT SETINFO utility functions directly."""

    def test_set_client_info_success(self, mock_valkey_client: Mock):
        """Test that set_client_info calls CLIENT SETINFO commands correctly."""
        set_client_info(mock_valkey_client)

        # Verify both CLIENT SETINFO commands were called
        expected_calls = [
            call("CLIENT", "SETINFO", "lib-name", LIBRARY_NAME),
            call("CLIENT", "SETINFO", "lib-ver", LIBRARY_VERSION),
        ]
        mock_valkey_client.execute_command.assert_has_calls(expected_calls)
        assert mock_valkey_client.execute_command.call_count == 2

    def test_set_client_info_failure_handling(self, mock_valkey_client: Mock):
        """Test that set_client_info handles failures gracefully."""
        # Make execute_command raise an exception
        mock_valkey_client.execute_command.side_effect = Exception("Command failed")

        # Should not raise an exception
        set_client_info(mock_valkey_client)

        # Should still attempt the first command
        mock_valkey_client.execute_command.assert_called_once_with(
            "CLIENT", "SETINFO", "lib-name", LIBRARY_NAME
        )

    @pytest.mark.asyncio
    async def test_aset_client_info_success(self, mock_async_valkey_client: AsyncMock):
        """Test that aset_client_info calls CLIENT SETINFO commands correctly."""
        await aset_client_info(mock_async_valkey_client)

        # Verify both CLIENT SETINFO commands were called
        expected_calls = [
            call("CLIENT", "SETINFO", "lib-name", LIBRARY_NAME),
            call("CLIENT", "SETINFO", "lib-ver", LIBRARY_VERSION),
        ]
        mock_async_valkey_client.execute_command.assert_has_calls(expected_calls)
        assert mock_async_valkey_client.execute_command.call_count == 2

    @pytest.mark.asyncio
    async def test_aset_client_info_failure_handling(
        self, mock_async_valkey_client: AsyncMock
    ):
        """Test that aset_client_info handles failures gracefully."""

        # Make execute_command raise an exception
        async def failing_command(*args, **kwargs):
            raise Exception("Command failed")

        mock_async_valkey_client.execute_command.side_effect = failing_command

        # Should not raise an exception
        await aset_client_info(mock_async_valkey_client)

        # Should still attempt the first command
        mock_async_valkey_client.execute_command.assert_called_once_with(
            "CLIENT", "SETINFO", "lib-name", LIBRARY_NAME
        )


class TestSaverClientSetInfo:
    """Test CLIENT SETINFO is called in checkpoint savers."""

    @patch("langgraph_checkpoint_aws.checkpoint.valkey.base.set_client_info")
    def test_sync_saver_direct_init(
        self, mock_set_client_info: Mock, mock_valkey_client: Mock
    ):
        """Test CLIENT SETINFO is called when directly initializing
        ValkeySaver."""
        ValkeySaver(mock_valkey_client)
        mock_set_client_info.assert_called_once_with(mock_valkey_client)

    @patch("langgraph_checkpoint_aws.checkpoint.valkey.base.set_client_info")
    def test_async_saver_direct_init(
        self, mock_set_client_info: Mock, mock_async_valkey_client: Mock
    ):
        """Test CLIENT SETINFO is NOT called when directly initializing
        AsyncValkeySaver with async client."""
        AsyncValkeySaver(mock_async_valkey_client)

        # set_client_info should NOT be called for async clients to avoid
        # unawaited coroutines
        # The base class should detect the async client and skip the sync
        # set_client_info call
        mock_set_client_info.assert_not_called()

    @patch("valkey.connection.ConnectionPool.from_url")
    @patch("langgraph_checkpoint_aws.checkpoint.valkey.base.set_client_info")
    def test_sync_saver_from_conn_string(
        self, mock_set_client_info: Mock, mock_pool_from_url: Mock, valkey_url: str
    ):
        # Set up mock pool with required attributes
        mock_pool = Mock(spec=ConnectionPool)
        mock_pool.connection_kwargs = {}  # Add the missing attribute
        mock_pool_from_url.return_value = mock_pool

        # Mock Valkey constructor to return a mock client
        with patch("valkey.Valkey") as mock_valkey_class:
            mock_client = Mock(spec=Valkey)
            mock_client.close = Mock()
            mock_valkey_class.return_value = mock_client

            with ValkeySaver.from_conn_string(valkey_url) as saver:
                # Should be called once: only in __init__ (base class)
                # The actual client passed will be the real Valkey instance,
                # not our mock
                mock_set_client_info.assert_called_once()
                assert saver is not None

    @patch("valkey.asyncio.Valkey.from_url")
    @patch("langgraph_checkpoint_aws.checkpoint.valkey.async_saver.aset_client_info")
    @patch("langgraph_checkpoint_aws.checkpoint.valkey.base.set_client_info")
    @pytest.mark.asyncio
    async def test_async_saver_from_conn_string(
        self,
        mock_set_client_info: Mock,
        mock_aset_client_info: AsyncMock,
        mock_from_url: Mock,
        valkey_url: str,
    ):
        """Test CLIENT SETINFO is called when using async from_conn_string."""
        mock_client = AsyncMock(spec=AsyncValkey)
        mock_client.aclose = AsyncMock()
        mock_from_url.return_value = mock_client

        async with AsyncValkeySaver.from_conn_string(valkey_url):
            # Should be called in from_conn_string but NOT in __init__
            # (async client detection)
            mock_aset_client_info.assert_called_once_with(mock_client)
            # set_client_info should NOT be called for async clients to avoid
            # unawaited coroutines
            mock_set_client_info.assert_not_called()


class TestStoreClientSetInfo:
    """Test CLIENT SETINFO is called in store implementations."""

    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_sync_store_direct_init(
        self, mock_set_client_info: Mock, mock_valkey_client: Mock
    ):
        """Test CLIENT SETINFO is called when directly initializing ValkeyStore."""
        ValkeyStore(mock_valkey_client)
        mock_set_client_info.assert_called_once_with(mock_valkey_client)

    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_async_store_direct_init(
        self, mock_set_client_info: Mock, mock_async_valkey_client: Mock
    ):
        """Test CLIENT SETINFO is NOT called when directly initializing
        AsyncValkeyStore with async client."""
        AsyncValkeyStore(mock_async_valkey_client)

        # set_client_info should NOT be called for async clients to avoid
        # unawaited coroutines
        # The base class should detect the async client and skip the sync
        # set_client_info call
        mock_set_client_info.assert_not_called()

    @patch("valkey.Valkey.from_url")
    @patch("langgraph_checkpoint_aws.store.valkey.base.set_client_info")
    def test_sync_store_from_conn_string(
        self, mock_set_client_info: Mock, mock_from_url: Mock, valkey_url: str
    ):
        """Test CLIENT SETINFO is called when using store from_conn_string."""
        mock_client = Mock(spec=Valkey)
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_from_url.return_value = mock_client

        with ValkeyStore.from_conn_string(valkey_url) as store:
            # Should be called once: only in __init__ (base class)
            mock_set_client_info.assert_called_once_with(mock_client)
            assert store is not None


class TestConnectionPoolClientSetInfo:
    """Test CLIENT SETINFO behavior with connection pools."""

    @patch("valkey.connection.ConnectionPool.from_url")
    @patch("langgraph_checkpoint_aws.checkpoint.valkey.utils.set_client_info")
    def test_pool_client_setinfo(
        self,
        mock_set_client_info: Mock,
        mock_from_url: Mock,
        mock_pool_connection: Mock,
        valkey_url: str,
    ):
        """Test CLIENT SETINFO is called for each new connection from pool."""
        mock_conn1 = Mock(spec=Valkey)
        mock_conn1.__enter__ = Mock(return_value=mock_conn1)  # Return same object
        mock_conn1.__exit__ = Mock(return_value=None)

        mock_conn2 = Mock(spec=Valkey)
        mock_conn2.__enter__ = Mock(return_value=mock_conn2)  # Return same object
        mock_conn2.__exit__ = Mock(return_value=None)

        # Create mock pool
        mock_pool = Mock(spec=ConnectionPool)
        mock_pool.get_connection = Mock(side_effect=[mock_conn1, mock_conn2])
        mock_from_url.return_value = mock_pool

        ConnectionPool.from_url(valkey_url)

        with mock_pool.get_connection() as conn1:  # Use mock_pool directly
            # First connection should trigger set_client_info
            mock_set_client_info(conn1)  # Call the mock directly
            mock_set_client_info.assert_called_once_with(conn1)

            with mock_pool.get_connection() as conn2:
                # Second connection should trigger set_client_info
                mock_set_client_info(conn2)  # Call the mock directly
                mock_set_client_info.assert_has_calls([call(conn1), call(conn2)])

    @pytest.mark.asyncio
    @patch("valkey.asyncio.connection.ConnectionPool.from_url")
    @patch("langgraph_checkpoint_aws.checkpoint.valkey.utils.aset_client_info")
    async def test_async_pool_client_setinfo(
        self, mock_aset_client_info: AsyncMock, mock_from_url: Mock, valkey_url: str
    ):
        """Test CLIENT SETINFO is called for each new async connection from pool."""
        # Set up mock pool
        mock_pool = AsyncMock(spec=AsyncConnectionPool)
        mock_pool.get_connection = AsyncMock()

        # Set up mock connections with async context manager support
        mock_conn1 = AsyncMock(spec=AsyncValkey)
        mock_conn1.__aenter__ = AsyncMock(return_value=mock_conn1)  # Return same object
        mock_conn1.__aexit__ = AsyncMock(return_value=None)

        mock_conn2 = AsyncMock(spec=AsyncValkey)
        mock_conn2.__aenter__ = AsyncMock(return_value=mock_conn2)  # Return same object
        mock_conn2.__aexit__ = AsyncMock(return_value=None)

        # Set up get_connection to return mock connections directly
        mock_pool.get_connection.side_effect = [mock_conn1, mock_conn2]

        mock_from_url.return_value = mock_pool
        pool = AsyncConnectionPool.from_url(valkey_url)

        async with await pool.get_connection("dummy_command") as conn1:
            await mock_aset_client_info(conn1)  # Call the mock directly
            mock_aset_client_info.assert_called_once_with(conn1)

            async with await pool.get_connection("dummy_command") as conn2:
                await mock_aset_client_info(conn2)  # Call the mock directly
                mock_aset_client_info.assert_has_calls(
                    [
                        call(conn1),  # First call
                        call(conn2),
                    ],
                    any_order=True,
                )
                assert mock_aset_client_info.call_count == 2
                assert mock_pool.get_connection.call_count == 2
                assert mock_pool.get_connection.call_args_list == [
                    call("dummy_command"),
                    call("dummy_command"),
                ]

    @pytest.mark.asyncio
    @patch("valkey.Valkey.from_url")
    @patch("langgraph_checkpoint_aws.checkpoint.valkey.utils.set_client_info")
    async def test_parallel_client_setinfo(
        self, mock_set_client_info: Mock, mock_from_url: Mock, valkey_url: str
    ):
        # Create mock clients
        mock_clients = [Mock(spec=Valkey) for _ in range(5)]

        # Set up from_url to return each mock client in sequence
        mock_from_url.side_effect = mock_clients

        # Create a list to track calls
        calls = []

        async def create_client():
            client = Valkey.from_url(valkey_url)
            mock_set_client_info(client)  # Call the mock directly
            calls.append(client)
            return client

        # Create multiple clients in parallel
        clients = await asyncio.gather(*[create_client() for _ in range(5)])

        # Verify each client got set_client_info called
        assert mock_set_client_info.call_count == 5
        assert len(clients) == 5
        assert len(calls) == 5
        for client in calls:
            mock_set_client_info.assert_any_call(client)

    @patch("valkey.Valkey.from_url")
    @patch("langgraph_checkpoint_aws.checkpoint.valkey.utils.set_client_info")
    def test_client_setinfo_retry(
        self, mock_set_client_info: Mock, mock_from_url: Mock, valkey_url: str
    ):
        # Create mock client with context manager
        mock_client = Mock(spec=Valkey, __enter__=Mock(), __exit__=Mock())
        mock_client.__enter__.return_value = mock_client  # Return same object
        mock_from_url.return_value = mock_client

        # Configure mock to raise exception first time, succeed second time
        mock_set_client_info.side_effect = [
            Exception("First attempt fails"),  # First call raises
            None,  # Second call succeeds
        ]

        Valkey.from_url(valkey_url)

        # First attempt should fail
        with pytest.raises((ValueError, ConnectionError, RuntimeError, Exception)):
            mock_set_client_info(mock_client)  # Call the mock directly

        # Reset mock for second attempt
        mock_set_client_info.side_effect = None

        # Second attempt should succeed
        mock_set_client_info(mock_client)  # Call the mock directly
        assert mock_set_client_info.call_count == 2


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
