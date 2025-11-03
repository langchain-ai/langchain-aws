"""Serialization utilities for checkpoints and metadata."""

import gzip
import logging
from enum import Enum
from typing import Any

from langgraph.checkpoint.serde.base import SerializerProtocol

logger = logging.getLogger(__name__)


class CompressionType(str, Enum):
    """Compression types for checkpoint data."""

    GZIP = "gzip"


class CheckpointSerializer:
    """
    Handles serialization/deserialization of checkpoints and metadata.
    """

    def __init__(
        self,
        serde: SerializerProtocol,
        enable_compression: bool = False,
        compression_threshold: int = 1024,
        min_compression_ratio: float = 0.1,
        compression_level: int = 6,
    ):
        """
        Initialize serializer with intelligent compression.

        Args:
            serde: LangGraph serialization instance
            enable_compression: Global flag to enable/disable compression
                (default: False)
            compression_threshold: Size in bytes to attempt compression
                (default: 1KB)
            min_compression_ratio: Ratio to use compressed data
                (default: 10% savings)
            compression_level: gzip compression level
                (default: 6, balanced speed/ratio)
        """
        self.serde = serde
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.min_compression_ratio = min_compression_ratio
        self.compression_level = compression_level

    def _compress_if_beneficial(
        self, data: bytes
    ) -> tuple[bytes, CompressionType | None]:
        """
        Compress data only if it provides meaningful size reduction.

        Args:
            data: Raw bytes to potentially compress

        Returns:
            Tuple of (data, compression_type) where compression_type is
            CompressionType.GZIP if compressed, None if not compressed
        """
        original_size = len(data)

        if original_size == 0:
            logger.debug("Original data is empty (0B), storing uncompressed")
            return data, None

        if not self.enable_compression:
            logger.debug(
                f"Compression disabled, storing {original_size} bytes uncompressed"
            )
            return data, None

        if original_size < self.compression_threshold:
            logger.debug(
                f"Data size {original_size}B below threshold "
                f"{self.compression_threshold}B, skipping compression"
            )
            return data, None

        try:
            compressed = gzip.compress(data, compresslevel=self.compression_level)
        except Exception as e:
            logger.warning(
                f"Compression failed for {original_size}B data: {e}, "
                "storing uncompressed"
            )
            return data, None

        compressed_size = len(compressed)

        size_reduction = (original_size - compressed_size) / original_size

        # Only use compressed data if it saves enough space
        if size_reduction >= self.min_compression_ratio:
            logger.debug(
                f"Compressed {original_size}B -> {compressed_size}B "
                f"({size_reduction:.1%} reduction)"
            )
            return compressed, CompressionType.GZIP

        logger.debug(
            f"Compression not beneficial: {original_size}B -> "
            f"{compressed_size}B ({size_reduction:.1%} reduction < "
            f"{self.min_compression_ratio:.1%} threshold), "
            "storing uncompressed"
        )
        return data, None

    def serialize(self, value: Any) -> tuple[str, bytes, CompressionType | None]:
        """
        Serialize any value with intelligent compression.

        Args:
            value: Any value to serialize

        Returns:
            Tuple of (value_type, serialized_bytes, compression_type)
            compression_type is CompressionType.GZIP if compressed,
            None if not compressed

        Example:
            >>> value_type, data, compression = serializer.serialize(checkpoint)
            >>> # Store in DynamoDB with value_type and compression type
        """
        value_type, data = self.serde.dumps_typed(value)
        logger.debug(f"Serialized to {value_type}, size: {len(data)}B")

        compressed_data, compression_type = self._compress_if_beneficial(data)
        return value_type, compressed_data, compression_type

    def deserialize(
        self,
        value_type: str,
        data: bytes,
        compression_type: CompressionType | None = None,
    ) -> Any:
        """
        Deserialize any value with automatic decompression.

        Args:
            value_type: Serialization type (e.g., "json", "msgpack")
            data: Serialized bytes
            compression_type: Compression type enum (CompressionType.GZIP or None)

        Returns:
            Deserialized value

        Raises:
            ValueError: If decompression fails due to corrupted or invalid data

        Example:
            >>> value = serializer.deserialize(value_type, data, CompressionType.GZIP)
        """
        compressed_size = len(data)

        # Handle decompression if needed
        if compression_type == CompressionType.GZIP:
            try:
                data = gzip.decompress(data)
            except Exception as e:
                raise ValueError(
                    f"Failed to decompress gzip data: {e}. "
                    "Data may be corrupted or not actually compressed."
                ) from e
        else:
            logger.debug(f"Deserializing {compressed_size}B uncompressed data")

        return self.serde.loads_typed((value_type, data))
