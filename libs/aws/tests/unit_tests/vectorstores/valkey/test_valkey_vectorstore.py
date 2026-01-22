"""Unit tests for Valkey vector store."""

import os
import pytest

pytest.importorskip("valkey")

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

    @patch("langchain_aws.utilities.valkey.get_client")
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

    @patch("langchain_aws.utilities.valkey.get_client")
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

    @patch("langchain_aws.utilities.valkey.get_client")
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

    @patch("langchain_aws.utilities.valkey.get_client")
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

    @patch("langchain_aws.utilities.valkey.get_client")
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
