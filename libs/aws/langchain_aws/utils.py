import logging
import os
import re
from abc import abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Literal, Optional, TypeVar, Union

from botocore.exceptions import BotoCoreError, UnknownServiceError
from langchain_core.messages import AIMessage
from packaging import version
from pydantic import SecretStr

logger = logging.getLogger(__name__)

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
    """A handler class to transform input from LLM and BaseChatModel to a

    format that SageMaker endpoint expects.

    Similarly, the class handles transforming output from the
    SageMaker endpoint to a format that LLM & BaseChatModel class expects.
    """

    """
    Example:
        ```python
        class ContentHandler(ContentHandlerBase):
            content_type = "application/json"
            accepts = "application/json"

            def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                input_str = json.dumps({prompt: prompt, **model_kwargs})
                return input_str.encode('utf-8')

            def transform_output(self, output: bytes) -> str:
                response_json = json.loads(output.read().decode("utf-8"))
                return response_json[0]["generated_text"]
        ```
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


def count_tokens_api_supported_for_model(model: str) -> bool:
    return any(
        x in model
        for x in (
            "claude-3-5-",
            "claude-3-7-",
            "claude-opus-4-",
            "claude-sonnet-4-",
            "claude-haiku-4-",
        )
    )


def _get_anthropic_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


def get_num_tokens_anthropic(text: str) -> int:
    """Get the number of tokens in a string of text."""
    client = _get_anthropic_client()
    return client.count_tokens(text=text)


def get_token_ids_anthropic(text: str) -> List[int]:
    """Get the token IDs for a string of text."""
    client = _get_anthropic_client()
    tokenizer = client.get_tokenizer()
    encoded_text = tokenizer.encode(text)
    return encoded_text.ids


def create_aws_client(
    service_name: str,
    region_name: Optional[str] = None,
    credentials_profile_name: Optional[str] = None,
    aws_access_key_id: Optional[SecretStr] = None,
    aws_secret_access_key: Optional[SecretStr] = None,
    aws_session_token: Optional[SecretStr] = None,
    endpoint_url: Optional[str] = None,
    config: Any = None,
    api_key: Optional[SecretStr] = None,
) -> Any:
    """Helper function to validate AWS credentials and create an AWS client.

    Args:
        service_name: The name of the AWS service to create a client for.
        region_name: AWS region name. If not provided, try to get from env variables.
        credentials_profile_name: The name of the AWS credentials profile to use.
        aws_access_key_id: AWS access key ID.
        aws_secret_access_key: AWS secret access key.
        aws_session_token: AWS session token.
        endpoint_url: The complete URL to use for the constructed client.
        config: Advanced client configuration options.
        api_key: Bedrock API key for bearer token authentication.
            If provided, sets AWS_BEARER_TOKEN_BEDROCK env variable.
    Returns:
        boto3.client: An AWS service client instance.

    """

    try:
        import boto3

        region_name = (
            region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        )

        has_aws_credentials = bool(
            credentials_profile_name
            or aws_access_key_id
            or aws_secret_access_key
            or aws_session_token
        )
        has_api_key = bool(api_key and api_key.get_secret_value())

        if has_api_key and has_aws_credentials:
            logger.warning(
                "Both api_key and AWS credentials were provided. "
                "Using api_key for authentication; AWS credentials "
                "will be ignored."
            )

        if has_api_key:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key.get_secret_value()  # type: ignore[union-attr]

        client_params = {
            "service_name": service_name,
            "region_name": region_name,
            "endpoint_url": endpoint_url,
            "config": config,
        }
        client_params = {k: v for k, v in client_params.items() if v}

        needs_session = not has_api_key and has_aws_credentials

        if not needs_session:
            return boto3.client(**client_params)

        if credentials_profile_name:
            session = boto3.Session(profile_name=credentials_profile_name)
        elif aws_access_key_id and aws_secret_access_key:
            session_params = {
                "aws_access_key_id": aws_access_key_id.get_secret_value(),
                "aws_secret_access_key": aws_secret_access_key.get_secret_value(),
            }
            if aws_session_token:
                session_params["aws_session_token"] = (
                    aws_session_token.get_secret_value()
                )
            # session_params contains valid boto3.Session parameters but type stubs are
            # overly restrictive
            session = boto3.Session(**session_params)  # type: ignore[arg-type]
        else:
            raise ValueError(
                "If providing credentials, both aws_access_key_id and "
                "aws_secret_access_key must be specified."
            )

        if not client_params.get("region_name") and session.region_name:
            client_params["region_name"] = session.region_name

        return session.client(**client_params)

    except UnknownServiceError as e:
        raise ModuleNotFoundError(
            f"Ensure that you have installed the latest boto3 package "
            f"that contains the API for `{service_name}`."
        ) from e
    except BotoCoreError as e:
        raise ValueError(
            "Could not load credentials to authenticate with AWS client. "
            "Please check that the specified profile name and/or its credentials are "
            f"valid. Service error: {e}"
        ) from e
    except Exception as e:
        raise ValueError(f"Error raised by service:\n\n{e}") from e


def thinking_in_params(params: dict) -> bool:
    """Check if the thinking parameter is enabled in the request."""
    return params.get("thinking", {}).get("type") == "enabled"


def trim_message_whitespace(messages: List[Any]) -> List[Any]:
    """Trim trailing whitespace from final AIMessage content."""
    if not messages or not isinstance(messages[-1], AIMessage):
        return messages

    last_message = messages[-1]

    if isinstance(last_message.content, str):
        trimmed = last_message.content.rstrip()
        if trimmed != last_message.content:
            last_message.content = trimmed
    elif isinstance(last_message.content, list):
        for j, block in enumerate(last_message.content):
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                trimmed = block["text"].rstrip()
                if trimmed != block["text"]:
                    block_dict: dict[Any, Any] = block
                    block_dict["text"] = trimmed
                    last_message.content[j] = block_dict

    return messages
