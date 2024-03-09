import os
from typing import Any, Optional

import boto3
from botocore.config import Config as BotocoreConfig
from langchain_core.pydantic_v1 import BaseModel, Field


def get_client(
    service_name: str,
    *,
    region_name: Optional[str],
    endpoint_url: Optional[str],
    config: Optional[BotocoreConfig],
    credentials_profile_name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Gets the client for the given service name."""
    try:

        if credentials_profile_name is not None:
            session = boto3.Session(profile_name=credentials_profile_name)
        else:
            # use default credentials
            session = boto3.Session()

        region_name = (
            region_name or os.environ.get("AWS_DEFAULT_REGION") or session.region_name
        )

        client_params = {}
        if region_name:
            client_params["region_name"] = region_name
        if endpoint_url:
            client_params["endpoint_url"] = endpoint_url
        if config:
            client_params["config"] = config

        return session.client(service_name, **client_params)
    except Exception as e:
        raise ValueError(
            "Could not load credentials to authenticate with AWS client. "
            "Please check that credentials in the specified "
            "profile name are valid."
        ) from e


class BotoAuthMixin(BaseModel):
    """
    Defines the fields available for authentication with AWS.
    """

    region_name: Optional[str] = None
    """The aws region e.g., `us-west-2`. Fallsback to AWS_DEFAULT_REGION env variable
    or region specified in ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = Field(default=None, exclude=True)
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    config: Optional[BotocoreConfig] = None
    """An optional botocore.config.Config instance to pass to the client."""

    endpoint_url: Optional[str] = None
    """Needed if you don't want to default to us-east-1 endpoint"""
