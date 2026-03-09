"""Unit tests for Valkey vector store."""

import os

import pytest

pytest.importorskip("glide_sync")

from unittest.mock import MagicMock, patch

from langchain_aws.vectorstores.valkey import ValkeyVectorStore


class TestValkeyVectorStore:
    """Test ValkeyVectorStore class."""

    @property
    def valkey_url(self) -> str:
        """Build valkey URL from environment variables."""
        host = os.environ.get("VALKEY_HOST", "localhost")
        port = os.environ.get("VALKEY_PORT", "6379")
        return f"valkey://{host}:{port}"

    @patch("langchain_aws.vectorstores.valkey.base.get_client")
    def test_init(self, mock_get_client: MagicMock) -> None:
        """Test initialization of ValkeyVectorStore."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_embeddings = MagicMock()

        store = ValkeyVectorStore(
            valkey_url=self.valkey_url,
            index_name="test_index",
            embedding=mock_embeddings,
        )

        assert store.index_name == "test_index"
        assert store.client == mock_client
        assert store._embeddings == mock_embeddings
        assert store.key_prefix == "doc:test_index"

    @patch("langchain_aws.vectorstores.valkey.base.get_client")
    def test_init_with_custom_key_prefix(self, mock_get_client: MagicMock) -> None:
        """Test initialization with custom key prefix."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_embeddings = MagicMock()

        store = ValkeyVectorStore(
            valkey_url=self.valkey_url,
            index_name="test_index",
            embedding=mock_embeddings,
            key_prefix="custom_prefix",
        )

        assert store.key_prefix == "custom_prefix"

    @patch("langchain_aws.vectorstores.valkey.base.get_client")
    def test_embeddings_property(self, mock_get_client: MagicMock) -> None:
        """Test embeddings property."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_embeddings = MagicMock()

        store = ValkeyVectorStore(
            valkey_url=self.valkey_url,
            index_name="test_index",
            embedding=mock_embeddings,
        )

        assert store.embeddings == mock_embeddings

    @patch("langchain_aws.vectorstores.valkey.base.get_client")
    def test_from_texts(self, mock_get_client: MagicMock) -> None:
        """Test from_texts class method."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        texts = ["text1", "text2"]
        metadatas = [{"key": "value1"}, {"key": "value2"}]

        store = ValkeyVectorStore.from_texts(
            texts=texts,
            embedding=mock_embeddings,
            metadatas=metadatas,
            valkey_url=self.valkey_url,
            index_name="test_index",
        )

        assert store.index_name == "test_index"
        assert mock_embeddings.embed_documents.called

    @patch("langchain_aws.vectorstores.valkey.base.get_client")
    def test_from_existing_index(self, mock_get_client: MagicMock) -> None:
        """Test from_existing_index class method."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_embeddings = MagicMock()

        store = ValkeyVectorStore.from_existing_index(
            embedding=mock_embeddings,
            index_name="existing_index",
            valkey_url=self.valkey_url,
        )

        assert store.index_name == "existing_index"
        assert store._embeddings == mock_embeddings

    @patch("langchain_aws.vectorstores.valkey.base.check_index_exists")
    @patch("glide_sync.ft.create")
    @patch("langchain_aws.vectorstores.valkey.base.get_client")
    def test_create_index_if_not_exist_creates_index(
        self,
        mock_get_client: MagicMock,
        mock_ft_create: MagicMock,
        mock_check_exists: MagicMock,
    ) -> None:
        """Test that index is created when it doesn't exist."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_embeddings = MagicMock()
        mock_check_exists.return_value = False

        store = ValkeyVectorStore(
            valkey_url=self.valkey_url,
            index_name="test_index",
            embedding=mock_embeddings,
        )

        store._create_index_if_not_exist(dim=1536)

        assert mock_check_exists.called
        assert mock_ft_create.called

    @patch("langchain_aws.vectorstores.valkey.base.check_index_exists")
    @patch("glide_sync.ft.create")
    @patch("langchain_aws.vectorstores.valkey.base.get_client")
    def test_create_index_if_not_exist_skips_existing(
        self,
        mock_get_client: MagicMock,
        mock_ft_create: MagicMock,
        mock_check_exists: MagicMock,
    ) -> None:
        """Test that index creation is skipped when index exists."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_embeddings = MagicMock()
        mock_check_exists.return_value = True

        store = ValkeyVectorStore(
            valkey_url=self.valkey_url,
            index_name="test_index",
            embedding=mock_embeddings,
        )

        store._create_index_if_not_exist(dim=1536)

        assert mock_check_exists.called
        assert not mock_ft_create.called



class TestParseValkeyUrl:
    """Test URL parsing for various connection string formats."""

    def test_basic_url(self) -> None:
        """Test basic valkey URL without auth or SSL."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkey://localhost:6379"
        )
        assert host == "localhost"
        assert port == 6379
        assert use_tls is False
        assert username is None
        assert password is None

    def test_ssl_url(self) -> None:
        """Test SSL URL (valkeyss protocol)."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkeyss://localhost:6379"
        )
        assert host == "localhost"
        assert port == 6379
        assert use_tls is True
        assert username is None
        assert password is None

    def test_rediss_url(self) -> None:
        """Test rediss protocol (also SSL)."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "rediss://localhost:6379"
        )
        assert host == "localhost"
        assert port == 6379
        assert use_tls is True
        assert username is None
        assert password is None

    def test_password_only_auth(self) -> None:
        """Test URL with password only."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkey://mypassword@localhost:6379"
        )
        assert host == "localhost"
        assert port == 6379
        assert use_tls is False
        assert username is None
        assert password == "mypassword"

    def test_username_password_auth(self) -> None:
        """Test URL with username and password."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkey://myuser:mypassword@localhost:6379"
        )
        assert host == "localhost"
        assert port == 6379
        assert use_tls is False
        assert username == "myuser"
        assert password == "mypassword"

    def test_ssl_with_auth(self) -> None:
        """Test SSL URL with username and password."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkeyss://myuser:mypassword@localhost:6379"
        )
        assert host == "localhost"
        assert port == 6379
        assert use_tls is True
        assert username == "myuser"
        assert password == "mypassword"

    def test_default_port(self) -> None:
        """Test URL without explicit port (should default to 6379)."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkey://localhost"
        )
        assert host == "localhost"
        assert port == 6379
        assert use_tls is False
        assert username is None
        assert password is None

    def test_aws_elasticache_url(self) -> None:
        """Test AWS ElastiCache URL format."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkeyss://my-cluster.cache.amazonaws.com:6379"
        )
        assert host == "my-cluster.cache.amazonaws.com"
        assert port == 6379
        assert use_tls is True
        assert username is None
        assert password is None

    def test_aws_with_auth(self) -> None:
        """Test AWS URL with authentication."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkeyss://admin:secret123@my-cluster.cache.amazonaws.com:6379"
        )
        assert host == "my-cluster.cache.amazonaws.com"
        assert port == 6379
        assert use_tls is True
        assert username == "admin"
        assert password == "secret123"

    def test_url_with_path(self) -> None:
        """Test URL with path component (should be ignored)."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkey://localhost:6379/0"
        )
        assert host == "localhost"
        assert port == 6379
        assert use_tls is False

    def test_password_with_special_chars(self) -> None:
        """Test password containing special characters."""
        from langchain_aws.utilities.valkey import _parse_valkey_url

        host, port, use_tls, username, password = _parse_valkey_url(
            "valkey://user:p@ss:w0rd!@localhost:6379"
        )
        assert host == "localhost"
        assert port == 6379
        assert username == "user"
        assert password == "p@ss:w0rd!"


class TestGetClientClusterMode:
    """Test cluster_mode parameter in get_client."""

    @patch("glide_sync.GlideClusterClient.create")
    @patch("glide_sync.GlideClient.create")
    def test_cluster_mode_true(
        self, mock_standalone_create: MagicMock, mock_cluster_create: MagicMock
    ) -> None:
        """Test explicit cluster mode."""
        from langchain_aws.utilities.valkey import get_client

        mock_cluster_create.return_value = MagicMock()
        get_client("valkey://localhost:6379", cluster_mode=True)
        assert mock_cluster_create.called
        assert not mock_standalone_create.called

    @patch("glide_sync.GlideClusterClient.create")
    @patch("glide_sync.GlideClient.create")
    def test_cluster_mode_false(
        self, mock_standalone_create: MagicMock, mock_cluster_create: MagicMock
    ) -> None:
        """Test explicit standalone mode."""
        from langchain_aws.utilities.valkey import get_client

        mock_standalone_create.return_value = MagicMock()
        get_client("valkey://localhost:6379", cluster_mode=False)
        assert mock_standalone_create.called
        assert not mock_cluster_create.called
