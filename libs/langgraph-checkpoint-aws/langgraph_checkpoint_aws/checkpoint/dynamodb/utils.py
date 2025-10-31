"""Utility functions for DynamoDB checkpoint operations."""

from importlib.metadata import version
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.client import DynamoDBClient
    from mypy_boto3_s3.client import S3Client

# Package information
LIBRARY_NAME = "langgraph_checkpoint_aws"
LIBRARY_VERSION = version(LIBRARY_NAME)
SDK_USER_AGENT = f"{LIBRARY_NAME}#dynamodb#{LIBRARY_VERSION}"


def process_aws_client_args(
    region_name: str | None = None,
    endpoint_url: str | None = None,
    boto_config: Config | None = None,
) -> tuple[dict, dict]:
    """Process AWS client arguments and return session and client kwargs.

    Args:
        region_name: AWS region name
        endpoint_url: Custom endpoint URL
        boto_config: Boto3 config object

    Returns:
        Tuple[dict, dict]: Session kwargs and client kwargs
    """
    session_kwargs = {}
    client_kwargs = {}

    # Session parameters
    if region_name is not None:
        session_kwargs["region_name"] = region_name

    # Client parameters
    if endpoint_url is not None:
        client_kwargs["endpoint_url"] = endpoint_url

    # Always add config with user agent
    client_kwargs["config"] = create_client_config(boto_config)  # type: ignore[assignment]

    return session_kwargs, client_kwargs


def create_client_config(config: Config | None = None) -> Config:
    """Create a client config with SDK user agent and proper retry configuration.

    Args:
        config: Existing Boto3 config object

    Returns:
        Config: New config object with combined user agent and retry configuration
    """
    config_kwargs = {}
    existing_user_agent = getattr(config, "user_agent_extra", "") if config else ""

    # Add our user agent to existing one
    new_user_agent = (
        f"{existing_user_agent} x-client-framework:langgraph-dynamodb "
        f"md/sdk_user_agent/{SDK_USER_AGENT}"
    ).strip()

    # Preserve retries configuration
    if config is None or not hasattr(config, "retries"):
        config_kwargs["retries"] = {
            "mode": "adaptive",
            "max_attempts": 5,
        }
    elif config:
        config_kwargs["retries"] = config.retries  # type: ignore[attr-defined]

    # Preserve max_pool_connections if provided
    if config and hasattr(config, "max_pool_connections"):
        config_kwargs["max_pool_connections"] = config.max_pool_connections

    return Config(user_agent_extra=new_user_agent, **config_kwargs)  # type: ignore[arg-type]


def create_dynamodb_client(
    session: boto3.Session | None = None,
    region_name: str | None = None,
    endpoint_url: str | None = None,
    boto_config: Config | None = None,
) -> "DynamoDBClient":
    """Create or return a DynamoDB client.

    Args:
        client: Pre-configured DynamoDB client instance
        session: Pre-configured boto3 session instance
        region_name: AWS region name
        endpoint_url: Custom endpoint URL
        config: Boto3 config object

    Returns:
        BaseClient: Configured DynamoDB client

    Raises:
        ValueError: If provided client is not a DynamoDB client
    """

    # Process arguments
    session_kwargs, client_kwargs = process_aws_client_args(
        region_name=region_name,
        endpoint_url=endpoint_url,
        boto_config=boto_config,
    )

    # Create session if not provided
    if session is None:
        session = boto3.Session(**session_kwargs)

    # Create and return client
    return session.client("dynamodb", **client_kwargs)


def create_s3_client(
    session: boto3.Session | None = None,
    region_name: str | None = None,
    endpoint_url: str | None = None,
    boto_config: Config | None = None,
) -> "S3Client":
    """Create or return an S3 client.

    Args:
        client: Pre-configured S3 client instance
        session: Pre-configured boto3 session instance
        region_name: AWS region name
        endpoint_url: Custom endpoint URL
        boto_config: Boto3 config object

    Returns:
        BaseClient: Configured S3 client
    """

    # Process arguments
    session_kwargs, client_kwargs = process_aws_client_args(
        region_name=region_name,
        endpoint_url=endpoint_url,
        boto_config=boto_config,
    )

    # Create session if not provided
    if session is None:
        session = boto3.Session(**session_kwargs)

    # Create and return client
    return session.client("s3", **client_kwargs)
