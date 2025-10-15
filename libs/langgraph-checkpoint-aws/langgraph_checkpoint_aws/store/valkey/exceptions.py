"""Custom exception classes for Valkey store operations."""

from __future__ import annotations


class ValkeyStoreError(Exception):
    """Base exception for all Valkey store errors."""

    def __init__(
        self, message: str, *, operation: str | None = None, key: str | None = None
    ) -> None:
        """Initialize ValkeyStoreError.

        Args:
            message: Error message
            operation: Operation that caused the error
            key: Key involved in the error (if applicable)
        """
        super().__init__(message)
        self.operation = operation
        self.key = key


class ValkeyConnectionError(ValkeyStoreError):
    """Exception raised when Valkey connection fails."""

    def __init__(self, message: str, *, connection_string: str | None = None) -> None:
        """Initialize ValkeyConnectionError.

        Args:
            message: Error message
            connection_string: Connection string that failed (if applicable)
        """
        super().__init__(message, operation="connection")
        self.connection_string = connection_string


class DocumentParsingError(ValkeyStoreError):
    """Exception raised when document parsing fails."""

    def __init__(
        self,
        message: str,
        *,
        document_key: str | None = None,
        parse_type: str | None = None,
    ) -> None:
        """Initialize DocumentParsingError.

        Args:
            message: Error message
            document_key: Key of document that failed to parse
            parse_type: Type of parsing that failed (json, timestamp, etc.)
        """
        super().__init__(message, operation="document_parsing", key=document_key)
        self.parse_type = parse_type


class SearchIndexError(ValkeyStoreError):
    """Exception raised when search index operations fail."""

    def __init__(
        self,
        message: str,
        *,
        index_name: str | None = None,
        index_operation: str | None = None,
    ) -> None:
        """Initialize SearchIndexError.

        Args:
            message: Error message
            index_name: Name of the index involved
            index_operation: Index operation that failed (create, search, etc.)
        """
        super().__init__(message, operation="search_index")
        self.index_name = index_name
        self.index_operation = index_operation


class EmbeddingGenerationError(ValkeyStoreError):
    """Exception raised when embedding generation fails."""

    def __init__(
        self,
        message: str,
        *,
        text_content: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize EmbeddingGenerationError.

        Args:
            message: Error message
            text_content: Text content that failed to embed (truncated for logging)
            embedding_model: Embedding model that failed
        """
        super().__init__(message, operation="embedding_generation")
        self.text_content = (
            text_content[:100] + "..."
            if text_content and len(text_content) > 100
            else text_content
        )
        self.embedding_model = embedding_model


class ValidationError(ValkeyStoreError):
    """Exception raised when input validation fails."""

    def __init__(
        self,
        message: str,
        *,
        field_name: str | None = None,
        field_value: str | None = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Error message
            field_name: Name of field that failed validation
            field_value: Value that failed validation (truncated for logging)
        """
        super().__init__(message, operation="validation")
        self.field_name = field_name
        self.field_value = (
            str(field_value)[:100] + "..."
            if field_value and len(str(field_value)) > 100
            else str(field_value)
            if field_value
            else None
        )


class TTLConfigurationError(ValkeyStoreError):
    """Exception raised when TTL configuration is invalid."""

    def __init__(self, message: str, *, ttl_value: float | None = None) -> None:
        """Initialize TTLConfigurationError.

        Args:
            message: Error message
            ttl_value: TTL value that caused the error
        """
        super().__init__(message, operation="ttl_configuration")
        self.ttl_value = ttl_value
