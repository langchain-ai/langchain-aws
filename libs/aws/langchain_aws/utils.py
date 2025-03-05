import os
import re
from typing import Any, List

from botocore.exceptions import UnknownServiceError
from packaging import version
from pydantic import SecretStr


def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), text, maxsplit=1)[0]


def anthropic_tokens_supported() -> bool:
    """Check if all requirements for Anthropic count_tokens() are met."""
    try:
        import anthropic
    except ImportError:
        return False

    if version.parse(anthropic.__version__) > version.parse("0.38.0"):
        return False

    try:
        import httpx

        if version.parse(httpx.__version__) > version.parse("0.27.2"):
            raise ImportError()
    except ImportError:
        raise ImportError("httpx<=0.27.2 is required.")

    return True


def _get_anthropic_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


def get_num_tokens_anthropic(text: str) -> int:
    """Get the number of tokens in a string of text."""
    client = _get_anthropic_client()
    return client.count_tokens(text=text)


def get_token_ids_anthropic(text: str) -> List[int]:
    """Get the token ids for a string of text."""
    client = _get_anthropic_client()
    tokenizer = client.get_tokenizer()
    encoded_text = tokenizer.encode(text)
    return encoded_text.ids


def get_aws_client(
        region_name: str = None,
        credentials_profile_name: str = None,
        aws_access_key_id: SecretStr = None,
        aws_secret_access_key: SecretStr = None,
        aws_session_token: SecretStr = None,
        endpoint_url: str = None,
        config: Any = None,
        service_name: str = None,
    ):
    """Helper function to validate AWS credentials and create an AWS client."""

    creds = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_session_token": aws_session_token,
    }

    if creds["aws_access_key_id"] and creds["aws_secret_access_key"]:
        session_params = {
            k: v.get_secret_value() for k, v in creds.items() if v
        }
    elif any(creds.values()):
        raise ValueError(
            f"If any of aws_access_key_id, aws_secret_access_key, or aws_session_token "
            f"are specified, then both aws_access_key_id and aws_secret_access_key "
            f"must be specified. Only received "
            f"{[(k, v) for k, v in creds.items() if v]}."
        )
    elif credentials_profile_name:
        session_params = {"profile_name": credentials_profile_name}
    else:
        # Use default credentials
        session_params = {}

    try:
        import boto3

        session = boto3.Session(**session_params)

        region_name = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or session.region_name
        )

        client_params = {
            "region_name": region_name,
            "endpoint_url": endpoint_url,
            "config": config,
        }
        client_params = {
            k: v for k, v in client_params.items() if v
        }

        return session.client(
            service_name, **client_params
        )

    except ImportError:
        raise ModuleNotFoundError(
            "Could not import boto3 python package. "
            "Please install it with `pip install boto3`."
        )
    except UnknownServiceError as e:
        raise ModuleNotFoundError(
            f"Ensure that you have installed the latest boto3 package "
            f"that contains the API for `{service_name}`."
        ) from e
    except ValueError as e:
        raise ValueError(f"Error raised by service:\n\n{e}") from e
    except Exception as e:
        raise ValueError(
            "Could not load credentials to authenticate with AWS client. "
            "Please check that credentials in the specified profile name are valid. "
            f"Service error: {e}"
        ) from e

        
def thinking_in_params(params: dict) -> bool:
    """Check if the thinking parameter is enabled in the request."""
    return params.get("thinking", {}).get("type") == "enabled"
