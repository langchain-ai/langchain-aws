"""Exceptions for DynamoDB store operations."""


class DynamoDBStoreError(Exception):
    """Base exception for DynamoDB store errors."""

    pass


class DynamoDBConnectionError(DynamoDBStoreError):
    """Exception raised when connection to DynamoDB fails."""

    pass


class ValidationError(DynamoDBStoreError):
    """Exception raised for validation errors."""

    pass


class TableCreationError(DynamoDBStoreError):
    """Exception raised when table creation fails."""

    pass
