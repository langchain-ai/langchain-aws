import os
import re
from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Literal, Optional, TypeVar, Union

from botocore.exceptions import BotoCoreError, UnknownServiceError
from packaging import version
from pydantic import SecretStr

MESSAGE_ROLES = Literal["system", "user", "assistant"]
MESSAGE_FORMAT = Dict[Literal["role", "content"], Union[MESSAGE_ROLES, str]]

INPUT_TYPE = TypeVar(
    "INPUT_TYPE", bound=Union[str, List[str], MESSAGE_FORMAT, List[MESSAGE_FORMAT]]
)
OUTPUT_TYPE = TypeVar(
    "OUTPUT_TYPE",
    bound=Union[str, List[List[float]], MESSAGE_FORMAT, List[MESSAGE_FORMAT], Iterator],
)


class ContentHandlerBase(Generic[INPUT_TYPE, OUTPUT_TYPE]):
    """A handler class to transform input from LLM & BaseChatModel to a
    format that SageMaker endpoint expects.

    Similarly, the class handles transforming output from the
    SageMaker endpoint to a format that LLM & BaseChatModel class expects.
    """

    """
    Example:
        .. code-block:: python

            class ContentHandler(ContentHandlerBase):
                content_type = "application/json"
                accepts = "application/json"

                def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                    input_str = json.dumps({prompt: prompt, **model_kwargs})
                    return input_str.encode('utf-8')
                
                def transform_output(self, output: bytes) -> str:
                    response_json = json.loads(output.read().decode("utf-8"))
                    return response_json[0]["generated_text"]
    """

    content_type: Optional[str] = "text/plain"
    """The MIME type of the input data passed to endpoint"""

    accepts: Optional[str] = "text/plain"
    """The MIME type of the response data returned from endpoint"""

    @abstractmethod
    def transform_input(self, prompt: INPUT_TYPE, model_kwargs: Dict) -> bytes:
        """Transforms the input to a format that model can accept
        as the request Body. Should return bytes or seekable file
        like object in the format specified in the content_type
        request header.
        """

    @abstractmethod
    def transform_output(self, output: bytes) -> OUTPUT_TYPE:
        """Transforms the output from the model to string that
        the LLM class expects.
        """


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
        service_name: str,
        region_name: str = None,
        credentials_profile_name: str = None,
        aws_access_key_id: SecretStr = None,
        aws_secret_access_key: SecretStr = None,
        aws_session_token: SecretStr = None,
        endpoint_url: str = None,
        config: Any = None,
    ):
    """Helper function to validate AWS credentials and create an AWS client."""

    try:
        import boto3

        region_name = (
            region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
        )

        client_params = {
            "region_name": region_name,
            "endpoint_url": endpoint_url,
            "config": config,
        }
        client_params = {
            k: v for k, v in client_params.items() if v
        }

        cred_params_provided = credentials_profile_name is not None or any([
            aws_access_key_id, aws_secret_access_key, aws_session_token
        ])

        if cred_params_provided:
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

            session = boto3.Session(**session_params)

            if not client_params.get("region_name") and session.region_name:
                client_params["region_name"] = session.region_name

            return session.client(service_name, **client_params)

        return boto3.client(service_name, **client_params)

    except UnknownServiceError as e:
        raise ModuleNotFoundError(
            f"Ensure that you have installed the latest boto3 package "
            f"that contains the API for `{service_name}`."
        ) from e
    except BotoCoreError as e:
        raise ValueError(
            "Could not load credentials to authenticate with AWS client. "
            "Please check that the specified profile name and/or its credentials are valid. "
            f"Service error: {e}"
        ) from e
    except Exception as e:
        raise ValueError(f"Error raised by service:\n\n{e}") from e


def thinking_in_params(params: dict) -> bool:
    """Check if the thinking parameter is enabled in the request."""
    return params.get("thinking", {}).get("type") == "enabled"
