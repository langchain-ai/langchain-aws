"""Shared fixtures and configuration for DynamoDB store integration tests."""

import os

import pytest

try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore[assignment]
    BOTO3_AVAILABLE = False

# Default configuration for DynamoDB Local
DYNAMODB_HOST = os.getenv("DYNAMODB_HOST", "dynamodb-local")
DYNAMODB_PORT = os.getenv("DYNAMODB_PORT", "8001")
DYNAMODB_ENDPOINT_URL = f"http://{DYNAMODB_HOST}:{DYNAMODB_PORT}"
DYNAMODB_REGION = "us-east-1"
DYNAMODB_AWS_ACCESS_KEY_ID = "test"
DYNAMODB_AWS_SECRET_ACCESS_KEY = "test"


@pytest.fixture(autouse=True)
def _set_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set dummy AWS credentials for DynamoDB Local testing.

    DynamoDB Local does not validate credentials, but boto3 still requires
    them to be present. This fixture ensures dummy credentials are set for
    all integration tests and cleaned up automatically after each test.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", DYNAMODB_AWS_ACCESS_KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", DYNAMODB_AWS_SECRET_ACCESS_KEY)
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "test")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test")


def _is_dynamodb_available() -> bool:
    """Check if a DynamoDB Local server is available for testing.

    Verifies availability by making a real DynamoDB API call (list_tables)
    with dummy credentials, rather than just checking if the port is open.
    """
    if not BOTO3_AVAILABLE:
        return False

    try:
        client = boto3.client(
            "dynamodb",
            region_name=DYNAMODB_REGION,
            endpoint_url=DYNAMODB_ENDPOINT_URL,
            aws_access_key_id=DYNAMODB_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=DYNAMODB_AWS_SECRET_ACCESS_KEY,
        )
        client.list_tables(Limit=1)
        return True
    except Exception:
        return False


DYNAMODB_AVAILABLE = _is_dynamodb_available()


@pytest.fixture
def dynamodb_endpoint_url() -> str:
    """Get DynamoDB Local endpoint URL from environment or use default."""
    return DYNAMODB_ENDPOINT_URL
