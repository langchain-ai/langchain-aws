"""Tests for serialization functionality."""

import gzip
from unittest.mock import MagicMock, patch

import pytest

from langgraph_checkpoint_aws.checkpoint.dynamodb.serialization import (
    CheckpointSerializer,
    CompressionType,
)


class TestSerialization:
    """Test serialization methods."""

    def test_serialize_with_and_without_compression(self):
        """Test checkpoint serialization with and without compression."""
        mock_serde = MagicMock()

        # Without compression
        mock_serde.dumps_typed.return_value = ("msgpack", b"test_data")
        serializer = CheckpointSerializer(mock_serde)
        checkpoint_type, data, compression_type = serializer.serialize({"id": "test"})
        assert checkpoint_type == "msgpack"
        assert data == b"test_data"
        assert compression_type is None

        # With compression enabled
        mock_serde.dumps_typed.return_value = ("msgpack", b"x" * 2000)
        serializer = CheckpointSerializer(
            mock_serde, enable_compression=True, compression_threshold=1000
        )
        checkpoint_type, data, compression_type = serializer.serialize({"id": "test"})
        assert checkpoint_type == "msgpack"
        assert compression_type == CompressionType.GZIP
        assert len(data) < 2000

    def test_serialize_compression_not_beneficial(self):
        """Test that compression is skipped when not beneficial."""
        mock_serde = MagicMock()
        original_data = b"x" * 2000
        mock_serde.dumps_typed.return_value = ("msgpack", original_data)

        serializer = CheckpointSerializer(
            mock_serde,
            enable_compression=True,
            compression_threshold=1000,
            min_compression_ratio=0.1,
        )

        # Mock gzip.compress to return data that only saves 5%
        with patch("gzip.compress") as mock_compress:
            # Return compressed data that's 95% of original size (only 5% savings)
            mock_compress.return_value = b"y" * 1900

            checkpoint_type, data, compression_type = serializer.serialize(
                {"id": "test_checkpoint"}
            )

            assert checkpoint_type == "msgpack"
            # Should not compress because savings (5%) < min_compression_ratio (10%)
            assert compression_type is None
            assert data == original_data

    def test_deserialize_binary_and_compressed(self):
        """Test checkpoint deserialization with binary and compressed data."""
        mock_serde = MagicMock()
        mock_serde.loads_typed.return_value = {"id": "test_checkpoint"}
        serializer = CheckpointSerializer(mock_serde)

        # Binary (uncompressed)
        result = serializer.deserialize("msgpack", b"test_data", False)
        assert result == {"id": "test_checkpoint"}
        mock_serde.loads_typed.assert_called_with(("msgpack", b"test_data"))

        # Compressed
        compressed_data = gzip.compress(b"test_data")
        result = serializer.deserialize("msgpack", compressed_data, True)
        assert result == {"id": "test_checkpoint"}

    def test_compression_threshold_boundary(self):
        """Test compression at exact threshold boundary."""
        mock_serde = MagicMock()
        # Exactly at threshold
        mock_serde.dumps_typed.return_value = ("msgpack", b"x" * 1024)

        serializer = CheckpointSerializer(
            mock_serde, enable_compression=True, compression_threshold=1024
        )
        value = {"test": "data"}

        value_type, data, compression_type = serializer.serialize(value)

        # At threshold, should attempt compression
        assert value_type == "msgpack"
        assert compression_type == CompressionType.GZIP
        assert len(data) < 1024

    def test_deserialize_corrupted_compressed_data(self):
        """Test that corrupted compressed data raises a clear error."""

        mock_serde = MagicMock()
        serializer = CheckpointSerializer(mock_serde)

        # Try to decompress invalid data
        with pytest.raises(ValueError, match="Failed to decompress"):
            serializer.deserialize(
                "msgpack", b"not_compressed_data", CompressionType.GZIP
            )


class TestCompressionFlag:
    """Test flag-based compression control."""

    def test_compression_flag_control(self):
        """Test compression flag enables/disables compression."""
        mock_serde = MagicMock()
        mock_serde.dumps_typed.return_value = ("msgpack", b"x" * 5000)
        value = {"test": "data"}

        # Disabled - should not compress even though data > threshold
        serializer = CheckpointSerializer(
            mock_serde, enable_compression=False, compression_threshold=100
        )
        value_type, data, compression_type = serializer.serialize(value)
        assert compression_type is None
        assert data == b"x" * 5000

        # Enabled - should compress when data > threshold
        serializer = CheckpointSerializer(
            mock_serde, enable_compression=True, compression_threshold=1000
        )
        value_type, data, compression_type = serializer.serialize(value)
        assert compression_type == CompressionType.GZIP
        assert len(data) < 5000
