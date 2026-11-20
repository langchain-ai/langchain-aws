import asyncio
import logging
import os
import re
import threading
import time
from abc import abstractmethod
from datetime import timedelta
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    NoReturn,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    EventStreamError,
    IncompleteReadError,
    ProxyConnectionError,
    ReadTimeoutError,
    ResponseStreamingError,
    SSLError,
    UnknownServiceError,
)
from botocore.tokens import FrozenAuthToken
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.messages import AIMessage
from packaging import version
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class BedrockContextOverflowError(ClientError, ContextOverflowError):
    """ClientError raised when input exceeds the Bedrock model's context window."""

    def __reduce__(self) -> Tuple[Any, ...]:
        # ClientError.__reduce__ rebuilds a plain ClientError, losing the overflow type.
        return BedrockContextOverflowError, (self.response, self.operation_name)


# Lowercased fragments of the ValidationException messages Bedrock returns when
# the prompt exceeds the model's context window.
_CONTEXT_OVERFLOW_MARKERS = (
    # Claude Sonnet 4.5, DeepSeek R1
    "input is too long",
    # Claude Haiku 4.5 and newer Claude models; also matched by langchain-anthropic
    "prompt is too long",
    # Amazon Nova
    "input tokens exceeded",
    # Meta Llama, Mistral, and other open-weight models (gpt-oss, Qwen, DeepSeek V3)
    "maximum context length",
    # OpenAI GPT; also matched by langchain-openai
    "context_length_exceeded",
    "exceeds the context window",
    "input tokens exceed the configured limit",
    # Kimi K3
    "exceed model maximum",
    # matched by LiteLLM (Bedrock, vLLM/SGLang)
    "too many input tokens",
    "is longer than the model's context length",
)

_CONTEXT_OVERFLOW_ERROR_TYPES: Dict[
    Type[Exception], Type[BedrockContextOverflowError]
] = {}


def _context_overflow_error_type(
    error_type: Type[Exception],
) -> Type[BedrockContextOverflowError]:
    """Return a subclass of both the original botocore type and ContextOverflowError."""
    if issubclass(BedrockContextOverflowError, error_type):
        return BedrockContextOverflowError
    if error_type not in _CONTEXT_OVERFLOW_ERROR_TYPES:
        _CONTEXT_OVERFLOW_ERROR_TYPES[error_type] = cast(
            Type[BedrockContextOverflowError],
            type(error_type.__name__, (error_type, BedrockContextOverflowError), {}),
        )
    return _CONTEXT_OVERFLOW_ERROR_TYPES[error_type]


def _raise_if_context_overflow(error: Exception) -> None:
    """Raise a ContextOverflowError when Bedrock reports an oversized prompt."""
    if not isinstance(error, (ClientError, EventStreamError)):
        return

    error_info = error.response.get("Error", {})
    # Event-stream error codes are lowercase (e.g. "validationException").
    if str(error_info.get("Code", "")).lower() != "validationexception":
        return

    error_message = str(error_info.get("Message", "")).lower()
    if any(marker in error_message for marker in _CONTEXT_OVERFLOW_MARKERS):
        overflow_type = _context_overflow_error_type(type(error))
        raise overflow_type(error.response, error.operation_name) from error


_BEDROCK_ERROR_MAPPING: Dict[str, Type[Exception]] = {
    "throttlingexception": ModelRateLimitError,
    "toomanyrequestsexception": ModelRateLimitError,
    "servicequotaexceededexception": ModelRateLimitError,
    "modelnotreadyexception": ModelRateLimitError,
    # Generic AWS throttling codes (subset of botocore's retry throttling list).
    # Without these, a 400 carrying one would fall through to the status-code
    # fallback and be misreported as an invalid request.
    "throttling": ModelRateLimitError,
    "throttledexception": ModelRateLimitError,
    "requestthrottledexception": ModelRateLimitError,
    "requestthrottled": ModelRateLimitError,
    "requestlimitexceeded": ModelRateLimitError,
    "bandwidthlimitexceeded": ModelRateLimitError,
    "accessdeniedexception": ModelPermissionDeniedError,
    "unrecognizedclientexception": ModelAuthenticationError,
    "invalidclienttokenid": ModelAuthenticationError,
    "expiredtokenexception": ModelAuthenticationError,
    "invalidsignatureexception": ModelAuthenticationError,
    "incompletesignature": ModelAuthenticationError,
    "invalidaccesskeyid": ModelAuthenticationError,
    "resourcenotfoundexception": ModelNotFoundError,
    "validationexception": ModelInvalidRequestError,
    "badrequestexception": ModelInvalidRequestError,
    "modeltimeoutexception": ModelTimeoutError,
    "internalserverexception": ModelAPIError,
    "internalfailure": ModelAPIError,
    "serviceunavailableexception": ModelAPIError,
    "modelerrorexception": ModelAPIError,
    "modelstreamerrorexception": ModelAPIError,
}

_BEDROCK_MODEL_ERROR_TYPES: Dict[
    Tuple[Type[Exception], Type[Exception], Tuple[Type[Exception], ...]],
    Type[Exception],
] = {}

_BEDROCK_TRANSPORT_ERROR_MAPPING: Tuple[
    Tuple[Type[Exception], Type[Exception]], ...
] = (
    (ConnectTimeoutError, ModelTimeoutError),
    (ReadTimeoutError, ModelTimeoutError),
    (ConnectionClosedError, ModelConnectionError),
    (ResponseStreamingError, ModelConnectionError),
    (IncompleteReadError, ModelConnectionError),
    (EndpointConnectionError, ModelConnectionError),
    (ProxyConnectionError, ModelConnectionError),
    (SSLError, ModelConnectionError),
)


def _new_bedrock_model_error(
    mapped_type: Type[Exception], args: Tuple[Any, ...], state: Dict[str, Any]
) -> Exception:
    """Build a mapped exception by copying state, without calling `__init__`.

    Botocore exception constructors differ (`ClientError` takes a response,
    `BotoCoreError` takes keyword arguments), so the mapped error is rebuilt from
    the original's `args` and `__dict__` instead. This one path serves service
    errors, transport errors and unpickling alike.
    """
    mapped_error = mapped_type.__new__(mapped_type)
    mapped_error.__dict__.update(state)
    Exception.__init__(mapped_error, *args)
    return mapped_error


def _reduce_bedrock_model_error(error: Any) -> Tuple[Any, ...]:
    """Preserve the generated LangChain/Botocore exception type when pickling."""
    error_type = type(error)
    return (
        _rebuild_bedrock_model_error,
        (
            getattr(error_type, "__bedrock_botocore_error_type__"),
            getattr(error_type, "__bedrock_model_error_type__"),
            getattr(error_type, "__bedrock_extra_bases__"),
            error.args,
            error.__dict__,
        ),
    )


def _rebuild_bedrock_model_error(
    botocore_error_type: Type[Exception],
    model_error_type: Type[Exception],
    extra_bases: Tuple[Type[Exception], ...],
    args: Tuple[Any, ...],
    state: Dict[str, Any],
) -> Exception:
    """Rebuild a generated mapped exception during unpickling."""
    mapped_type = _bedrock_model_error_type(
        botocore_error_type, model_error_type, extra_bases
    )
    return _new_bedrock_model_error(mapped_type, args, state)


def _bedrock_model_error_type(
    botocore_error_type: Type[Exception],
    model_error_type: Type[Exception],
    extra_bases: Tuple[Type[Exception], ...] = (),
) -> Type[Exception]:
    """Create a cached exception type inheriting from Botocore and LangChain errors."""
    key = (botocore_error_type, model_error_type, extra_bases)
    if key not in _BEDROCK_MODEL_ERROR_TYPES:
        name = f"{botocore_error_type.__name__}{model_error_type.__name__}"
        _BEDROCK_MODEL_ERROR_TYPES[key] = type(
            name,
            (botocore_error_type, model_error_type, *extra_bases),
            {
                "__module__": __name__,
                "__bedrock_botocore_error_type__": botocore_error_type,
                "__bedrock_model_error_type__": model_error_type,
                "__bedrock_extra_bases__": extra_bases,
                "__reduce__": _reduce_bedrock_model_error,
            },
        )
    return _BEDROCK_MODEL_ERROR_TYPES[key]


def _as_bedrock_model_error(
    error: Exception,
    model_error_type: Type[Exception],
    *,
    extra_bases: Tuple[Type[Exception], ...] = (),
    message: Optional[str] = None,
) -> Exception:
    """Return `error` re-typed as both its own type and `model_error_type`.

    `message` replaces the exception text when given. The original's response,
    operation name and other attributes are preserved.
    """
    mapped_type = _bedrock_model_error_type(type(error), model_error_type, extra_bases)
    args = error.args if message is None else (message,)
    return _new_bedrock_model_error(mapped_type, args, error.__dict__)


def _model_error_type_from_status(status_code: Any) -> Optional[Type[Exception]]:
    """Map an HTTP status code to a LangChain model exception."""
    try:
        status = int(status_code)
    except (TypeError, ValueError):
        return None

    if status == 400:
        return ModelInvalidRequestError
    if status == 401:
        return ModelAuthenticationError
    if status == 403:
        return ModelPermissionDeniedError
    if status == 404:
        return ModelNotFoundError
    if status in (408, 504):
        return ModelTimeoutError
    if status == 429:
        return ModelRateLimitError
    if 500 <= status <= 599:
        return ModelAPIError
    return None


def _handle_bedrock_error(error: Exception) -> NoReturn:
    """Raise a standardized LangChain exception while preserving the Botocore type.

    Unknown exceptions, and errors that are already LangChain model exceptions,
    are re-raised unchanged. Context overflow is detected before
    the generic ValidationException mapping, and the InvokeTool permission message
    retains its existing actionable ValueError behavior.
    """
    if isinstance(error, ModelError):
        # Already standardized (e.g. handled by an inner layer): don't re-wrap it.
        raise error

    if not isinstance(error, (ClientError, EventStreamError)):
        for (
            botocore_error_type,
            transport_model_error_type,
        ) in _BEDROCK_TRANSPORT_ERROR_MAPPING:
            if isinstance(error, botocore_error_type):
                raise _as_bedrock_model_error(
                    error, transport_model_error_type
                ) from error
        raise error

    _raise_if_context_overflow(error)

    error_info = error.response.get("Error", {})
    error_code = str(error_info.get("Code", "")).lower()
    error_message = str(error)

    # Preserve the existing actionable message for system-tool IAM failures.
    if error_code == "accessdeniedexception" and "InvokeTool" in error_message:
        message = (
            "System tools require 'bedrock:InvokeTool' IAM permission. "
            "Please add this permission to your IAM role/user policy. "
            "Example policy statement:\n"
            "{\n"
            '  "Effect": "Allow",\n'
            '  "Action": ["bedrock:InvokeModel"],\n'
            '  "Resource": "arn:aws:bedrock:*::foundation-model/*"\n'
            "}\n"
            f"Original error: {error}"
        )
        # Also a ValueError, as before, so existing handlers keep working.
        raise _as_bedrock_model_error(
            error,
            ModelPermissionDeniedError,
            extra_bases=(ValueError,),
            message=message,
        ) from error

    model_error_type: Optional[Type[Exception]] = _BEDROCK_ERROR_MAPPING.get(error_code)

    # ModelErrorException and ModelStreamErrorException may wrap an upstream
    # provider error. Prefer its original status when it is available. botocore
    # exposes modeled error members at the top level of ``response`` (not inside
    # ``Error``); event-stream errors may carry them inside ``Error`` instead.
    response_metadata = error.response.get("ResponseMetadata", {})
    if error_code in ("modelerrorexception", "modelstreamerrorexception"):
        original_status = error.response.get(
            "originalStatusCode", error_info.get("originalStatusCode")
        )
        model_error_type = (
            _model_error_type_from_status(original_status)
            or _model_error_type_from_status(response_metadata.get("HTTPStatusCode"))
            or model_error_type
        )

    if model_error_type is None:
        model_error_type = _model_error_type_from_status(
            response_metadata.get("HTTPStatusCode")
        )

    if model_error_type is None:
        raise error

    raise _as_bedrock_model_error(error, model_error_type) from error


class _StaticTokenProvider:
    """Token provider that returns a static bearer token."""

    METHOD = "static"

    def __init__(self, token: str) -> None:
        self._token = token

    def load_token(self, **kwargs: Any) -> FrozenAuthToken:
        return FrozenAuthToken(self._token)


MESSAGE_ROLES = Literal["system", "user", "assistant"]
MESSAGE_FORMAT = Dict[Literal["role", "content"], Union[MESSAGE_ROLES, str]]

INPUT_TYPE = TypeVar(
    "INPUT_TYPE", bound=Union[str, List[str], MESSAGE_FORMAT, List[MESSAGE_FORMAT]]
)
OUTPUT_TYPE = TypeVar(
    "OUTPUT_TYPE",
    bound=Union[str, List[List[float]], MESSAGE_FORMAT, List[MESSAGE_FORMAT], Iterator],
)

MODEL_ID_GEO_PREFIXES = frozenset(
    {"eu", "us", "us-gov", "apac", "sa", "amer", "global", "jp", "au", "ca", "in"}
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
        api_key: Bedrock API key for bearer-token authentication.
            If provided, injects a static token provider into the botocore session
            to use HTTP Bearer authentication. This approach is compatible with
            multi-tenant deployments where different clients need different API keys.

    Returns:
        boto3.client: An AWS service client instance.

    """

    try:
        import boto3
        from botocore.config import Config as BotocoreConfig
        from botocore.session import get_session

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
            # Create a botocore session and inject the static token provider
            bc_session = get_session()
            token_provider_chain = bc_session.get_component("token_provider")
            static_provider = _StaticTokenProvider(api_key.get_secret_value())  # type: ignore[union-attr]
            # NOTE: botocore does not expose a public API for inserting a
            # provider into the token provider chain, so we mutate the private
            # ``_providers`` list. This is intentional and is the same approach
            # recommended for static-token workarounds; it may need to be
            # revisited if botocore changes this internal structure.
            try:
                token_provider_chain._providers.insert(0, static_provider)  # type: ignore[union-attr]
            except AttributeError:
                raise RuntimeError(
                    "Bearer token injection is no longer supported."
                    "Please open an issue on `langchain-aws` for help."
                ) from None

            # Configure client to prefer HTTP Bearer auth, merging with any
            # user-provided config. ``Config.merge`` is the documented way to
            # combine two Config objects; values in the argument take
            # precedence, so our bearer preference wins.
            # ``auth_scheme_preference`` is a valid botocore Config option but
            # may be missing from installed type stubs.
            bearer_config = BotocoreConfig(
                auth_scheme_preference="httpBearerAuth",  # type: ignore[call-arg]
            )
            if config is not None:
                bearer_config = config.merge(bearer_config)

            bearer_client_params = {
                "service_name": service_name,
                "region_name": region_name,
                "endpoint_url": endpoint_url,
                "config": bearer_config,
            }
            bearer_client_params = {k: v for k, v in bearer_client_params.items() if v}

            # Create session with our custom botocore session
            boto3_session = boto3.Session(botocore_session=bc_session)
            # bearer_client_params contains valid boto3 client kwargs but type
            # stubs are overly restrictive about the dict value types.
            return boto3_session.client(**bearer_client_params)  # type: ignore[call-overload]

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


_pending_transport_closes: "set[Any]" = set()


def _close_transport_soon(transport: Any) -> None:
    """Schedule an async ``close()`` of a discarded HTTP transport.

    Config plugins are synchronous but run inside the client's async setup,
    so a running event loop is available to schedule the close on.
    """
    close = getattr(transport, "close", None)
    if close is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(close())
    _pending_transport_closes.add(task)
    task.add_done_callback(_pending_transport_closes.discard)


def create_aws_bedrock_runtime_client(
    region_name: Optional[str] = None,
    credentials_profile_name: Optional[str] = None,
    aws_access_key_id: Optional[SecretStr] = None,
    aws_secret_access_key: Optional[SecretStr] = None,
    aws_session_token: Optional[SecretStr] = None,
    endpoint_url: Optional[str] = None,
    api_key: Optional[SecretStr] = None,
) -> Any:
    """Create an ``AsyncBedrockRuntimeClient`` from ``aws-sdk-bedrock-runtime``.

    This mirrors ``create_aws_client`` but targets the smithy-based
    ``aws-sdk-bedrock-runtime`` package required for bidirectional streaming
    APIs (e.g. Nova Sonic).

    Args:
        region_name: AWS region name. Falls back to ``AWS_REGION`` /
            ``AWS_DEFAULT_REGION`` environment variables.
        credentials_profile_name: Named profile in ``~/.aws/credentials``.
        aws_access_key_id: Explicit AWS access key ID.
        aws_secret_access_key: Explicit AWS secret access key.
        aws_session_token: Optional session token for temporary credentials.
        endpoint_url: Custom endpoint URL. If not provided, the SDK's
            built-in regional endpoint resolver is used.
        api_key: Bedrock API key for bearer-token authentication.
            If provided, sets the ``AWS_BEARER_TOKEN_BEDROCK`` environment
            variable at the process level. **Not compatible with multi-tenant
            deployments** where different clients in the same process need
            different API keys, as each call overwrites the previous value.

    Returns:
        A configured ``AsyncBedrockRuntimeClient`` instance.

    Raises:
        ModuleNotFoundError: If ``aws-sdk-bedrock-runtime`` >= 0.10 is not
            installed.
        ValueError: If credentials are incomplete or invalid.
    """
    try:
        from aws_sdk_bedrock_runtime.aio.client import (  # type: ignore[import-not-found,unused-ignore]
            AsyncBedrockRuntimeClient,
        )
    except ImportError:
        try:
            from aws_sdk_bedrock_runtime.client import (  # type: ignore[attr-defined,no-redef,unused-ignore]
                AsyncBedrockRuntimeClient,
            )
        except ImportError as e:
            raise ModuleNotFoundError(
                "Could not import aws-sdk-bedrock-runtime >= 0.10. "
                'Please install it via: pip install -U "langchain-aws[nova-sonic]"'
            ) from e

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

    # Resolve credentials: profile-based via boto3, explicit keys directly,
    # or fall back to the SDK's default credential chain.
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None

    if not has_api_key and has_aws_credentials:
        if credentials_profile_name:
            try:
                import boto3
            except ImportError as e:
                raise ModuleNotFoundError(
                    "boto3 is required when using credentials_profile_name."
                ) from e
            session = boto3.Session(profile_name=credentials_profile_name)
            credentials = session.get_credentials()
            if not credentials:
                raise ValueError(
                    f"Could not load credentials for profile "
                    f"'{credentials_profile_name}'."
                )
            creds = credentials.get_frozen_credentials()
            access_key = creds.access_key
            secret_key = creds.secret_key
            session_token = creds.token
            if not region_name and session.region_name:
                region_name = session.region_name
        elif aws_access_key_id and aws_secret_access_key:
            access_key = aws_access_key_id.get_secret_value()
            secret_key = aws_secret_access_key.get_secret_value()
            if aws_session_token:
                session_token = aws_session_token.get_secret_value()
        else:
            raise ValueError(
                "If providing credentials, both aws_access_key_id and "
                "aws_secret_access_key must be specified."
            )

    # If no explicit credentials were resolved, fall back to boto3's default
    # credential chain (env vars, instance metadata, etc.) so the smithy-based
    # SDK can authenticate.
    if not has_api_key and not access_key:
        try:
            import boto3
        except ImportError:
            pass
        else:
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials:
                creds = credentials.get_frozen_credentials()
                access_key = creds.access_key
                secret_key = creds.secret_key
                session_token = creds.token
                if not region_name and session.region_name:
                    region_name = session.region_name

    # Build a credentials identity resolver so the smithy-based SDK's SigV4
    # auth scheme can locate credentials.  Without this, the SDK silently
    # fails in a background task and the bidirectional stream hangs.
    credentials_resolver = None
    if access_key and secret_key:
        from smithy_aws_core.identity.components import AWSCredentialsIdentity

        class _StaticCredentialsResolver:
            """Minimal resolver that returns pre-resolved credentials."""

            def __init__(self, ak: str, sk: str, token: Optional[str]) -> None:
                self._identity = AWSCredentialsIdentity(
                    access_key_id=ak,
                    secret_access_key=sk,
                    session_token=token,
                )

            async def get_identity(self, **kwargs: Any) -> AWSCredentialsIdentity:
                return self._identity

        credentials_resolver = _StaticCredentialsResolver(
            access_key, secret_key, session_token
        )

    endpoint_uri = endpoint_url or (
        f"https://bedrock-runtime.{region_name}.amazonaws.com" if region_name else None
    )

    try:
        from smithy_http.aio.crt import AWSCRTHTTPClient
    except ImportError as e:
        raise ModuleNotFoundError(
            "The `awscrt` HTTP client is required for bidirectional streaming. "
            'Please install it via: pip install -U "langchain-aws[nova-sonic]"'
        ) from e

    def _configure(config: Any) -> None:
        if endpoint_uri:
            config.endpoint_uri = endpoint_uri
        if region_name:
            config.region = region_name
        if credentials_resolver is not None:
            config.aws_credentials_identity_resolver = credentials_resolver
        # Tag the smithy wire User-Agent with a langchain-aws source marker so
        # Nova Sonic's InvokeModelWithBidirectionalStream calls are attributable
        # to this package, readable by the Bedrock backend service. This is the
        # smithy analogue of the botocore user_agent_extra hook in __init__.py:
        # the SDK appends user_agent_extra verbatim to the wire User-Agent.
        from langchain_aws._version import FRAMEWORK_UA_TOKEN

        existing_ua_extra = getattr(config, "user_agent_extra", None)
        if not existing_ua_extra:
            config.user_agent_extra = FRAMEWORK_UA_TOKEN
        elif FRAMEWORK_UA_TOKEN not in existing_ua_extra:
            config.user_agent_extra = f"{existing_ua_extra} {FRAMEWORK_UA_TOKEN}"
        old_transport = getattr(config, "transport", None)
        config.transport = AWSCRTHTTPClient()
        _close_transport_soon(old_transport)

    client = AsyncBedrockRuntimeClient(plugins=[_configure])

    if hasattr(client, "_ensure_setup"):

        async def _setup() -> None:
            await client._ensure_setup()
            await asyncio.gather(
                *list(_pending_transport_closes), return_exceptions=True
            )

        def _run_setup() -> None:
            asyncio.run(_setup())

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            _run_setup()
        else:
            setup_thread = threading.Thread(target=_run_setup)
            setup_thread.start()
            setup_thread.join()

    return client


def parse_model_provider(model_id: str) -> str:
    """Extract the provider from a Bedrock model ID."""
    parts = model_id.split(".", maxsplit=2)
    if len(parts) > 1 and parts[0].lower() in MODEL_ID_GEO_PREFIXES:
        return parts[1]
    return parts[0]


def thinking_in_params(params: dict) -> bool:
    """Check if the thinking parameter is enabled in the request."""
    return params.get("thinking", {}).get("type") in ("enabled", "adaptive")


def thinking_disabled_in_params(params: dict) -> bool:
    """Check if thinking is explicitly disabled in the request."""
    return params.get("thinking", {}).get("type") == "disabled"


def thinking_on_by_default(model: str) -> bool:
    """Check if the model runs adaptive thinking without a thinking parameter."""
    return any(
        x in model
        for x in (
            "claude-sonnet-5",
            "claude-opus-5",
            "claude-fable-5",
            "claude-haiku-5",
        )
    )


def forced_tool_choice_unsupported(model: str) -> bool:
    """Check if the model rejects forced tool use (`any` or a named tool)."""
    return any(
        x in model
        for x in (
            "claude-fable-5-1",
            "claude-fable-5-2",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
        )
    )


def thinking_forced_tool_use_unsupported(model: str) -> bool:
    """Check if the model rejects or ignores forced tool use with thinking enabled."""
    if "claude-opus-4-8" in model:
        return False
    return any(
        x in model
        for x in (
            "claude-3-7-",
            "claude-sonnet-4",
            "claude-opus-4",
            "claude-haiku-4",
            "deepseek.v3",
        )
    )


def thinking_enabled_in_params(model: str, params: dict) -> bool:
    """Check if thinking is explicitly enabled in the request."""
    if "deepseek.v3" in model:
        effort = params.get("reasoning_effort")
        return bool(effort) and str(effort).lower() not in ("none", "low", "medium")
    return thinking_in_params(params)


def reasoning_effort_additional_fields(base_model: str, effort: str) -> Dict[str, Any]:
    """Translate a `reasoning_effort` value into provider-specific request fields.

    Returns an empty dict if the model family has no known translation; callers
    are responsible for warning the user in that case.
    """
    model = base_model.lower()
    if model.startswith("amazon.nova-2"):
        return {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}}
    if "claude" in model:
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
    if model.startswith("openai.gpt-oss"):
        return {"reasoning_effort": effort}
    if model.startswith("openai.gpt-"):
        return {"reasoning": {"effort": effort}}
    if model.startswith(("moonshot.", "moonshotai.", "zai.glm-")):
        return {"reasoning_effort": effort}
    return {}


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


_MANTLE_GUARDRAILS_ERR_MSG = (
    "Amazon Bedrock Guardrails are not supported on the bedrock-mantle "
    "endpoint. Please use ``ChatAnthropicBedrock`` or ``ChatBedrockConverse`` "
    "instead, which support guardrails via the bedrock-runtime endpoint."
)


def _check_no_mantle_guardrail_headers(headers: Optional[Dict[str, Any]]) -> None:
    """Reject Bedrock guardrail headers, which Mantle silently ignores."""
    # TODO: remove after Mantle adds guardrails support
    if not headers:
        return
    for key in headers:
        if key.lower().startswith("x-amzn-bedrock-guardrail"):
            raise ValueError(_MANTLE_GUARDRAILS_ERR_MSG)


class _StaticCredentialProvider:
    """Wraps resolved botocore credentials in the shape ``provide_token`` expects.

    ``aws_bedrock_token_generator.provide_token`` accepts an
    ``aws_credentials_provider`` that must expose a ``.load()`` method
    returning a ``botocore.credentials.Credentials`` (or subclass). This
    class adapts a pre-resolved credentials object to that contract.
    """

    METHOD = "explicit"

    def __init__(self, credentials: Any) -> None:
        self._credentials = credentials

    def load(self) -> Any:
        return self._credentials


#: Server-side maximum lifetime (in seconds) of a short-term Bedrock API
#: key issued by ``aws-bedrock-token-generator``. Bedrock rejects longer
#: expiries at the token-verification layer, so this is both the default
#: and the upper-bound of the ``ttl_seconds`` knob.
_BEDROCK_API_KEY_MAX_TTL_SECONDS = 43200  # 12 h


class _BedrockApiKeyProvider:
    """Generates and caches short-term Bedrock API keys.

    Uses ``aws-bedrock-token-generator`` to produce short-term API keys from
    AWS credentials.  Keys are cached and regenerated when near expiry.

    The instance is callable (sync) and also exposes an ``async_call`` method
    for use with async clients.

    This is the shared building block behind bearer-token authentication for
    the Bedrock Mantle chat models (``ChatOpenAIMantle`` /
    ``ChatAnthropicMantle``): rather than requiring a manually-minted static
    ``AWS_BEARER_TOKEN_BEDROCK``, it derives short-term keys from ordinary AWS
    credentials (explicit keys, a named profile, or the default chain) and
    transparently refreshes them.

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-generate.html
    """

    # Matches botocore's _DEFAULT_ADVISORY_REFRESH_TIMEOUT (900s) and
    # _DEFAULT_MANDATORY_REFRESH_TIMEOUT (600s) for credential refresh.
    _ADVISORY_REFRESH_SECONDS = 900  # try to refresh 15 min before expiry
    _MANDATORY_REFRESH_SECONDS = 600  # must refresh 10 min before expiry

    def __init__(
        self,
        region: str,
        *,
        ttl_seconds: int = _BEDROCK_API_KEY_MAX_TTL_SECONDS,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        credentials_profile_name: Optional[str] = None,
    ) -> None:
        self._region = region
        self._ttl_seconds = ttl_seconds
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._credentials_profile_name = credentials_profile_name
        self._token: str = ""
        self._expires_at: float = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> str:
        """Return a cached API key, refreshing if near expiry.

        Follows botocore's ``RefreshableCredentials._refresh`` pattern:
        - Outside advisory window: return cached token immediately.
        - Advisory window (15-10 min before expiry): try non-blocking
          refresh. If lock is held or refresh fails, return old token.
        - Mandatory window (<10 min before expiry): block until refresh
          completes or raise on failure.
        """
        if not self._refresh_needed(self._ADVISORY_REFRESH_SECONDS):
            return self._token

        if self._lock.acquire(blocking=False):
            try:
                # Re-check: another thread may have refreshed before we
                # acquired the lock.
                if not self._refresh_needed(self._ADVISORY_REFRESH_SECONDS):
                    return self._token
                is_mandatory = self._refresh_needed(self._MANDATORY_REFRESH_SECONDS)
                self._protected_refresh(is_mandatory=is_mandatory)
                return self._token
            finally:
                self._lock.release()
        elif self._refresh_needed(self._MANDATORY_REFRESH_SECONDS):
            # Token is close to expiry - must block until refreshed.
            with self._lock:
                # Re-check: another thread may have refreshed while we
                # were waiting for the lock.
                if not self._refresh_needed(self._MANDATORY_REFRESH_SECONDS):
                    return self._token
                self._protected_refresh(is_mandatory=True)
                return self._token

        # Another thread is refreshing; return current token. self._token is
        # never empty here because when it is, _refresh_needed() returns True
        # for the mandatory threshold, so the elif branch above always
        # triggers and blocks until a token exists.
        return self._token

    def _refresh_needed(self, refresh_in: int) -> bool:
        """Check if a refresh is needed within ``refresh_in`` seconds."""
        if not self._token:
            return True
        return time.monotonic() >= self._expires_at - refresh_in

    def _protected_refresh(self, *, is_mandatory: bool) -> None:
        """Attempt to refresh the token. Only raises if mandatory."""
        try:
            self._do_refresh()
        except Exception:
            if is_mandatory:
                raise
            logger.debug(
                "Advisory refresh failed, using existing token.",
                exc_info=True,
            )

    async def async_call(self) -> str:
        """Async-compatible version that returns a cached API key.

        Offloads to a thread to avoid blocking the event loop if a
        token refresh (network call) is needed.
        """
        import asyncio

        return await asyncio.to_thread(self)

    @staticmethod
    def _credentials_seconds_remaining(credentials: Any) -> Optional[int]:
        """Return remaining lifetime (s) for ``RefreshableCredentials``.

        Uses botocore's ``_seconds_remaining`` - a private method, but
        stable across botocore versions and the one botocore's own auth
        code uses internally. If botocore ever renames or removes it,
        the resulting ``AttributeError`` is swallowed and the caller
        degrades to its configured TTL.

        Args:
            credentials: A boto3 / botocore ``Credentials`` instance
                (any subclass).

        Returns:
            Remaining seconds if the credentials are a
            ``RefreshableCredentials`` and report a positive value;
            ``None`` for static credentials or on any error.
        """
        from botocore.credentials import RefreshableCredentials

        if not isinstance(credentials, RefreshableCredentials):
            return None
        try:
            remaining = credentials._seconds_remaining()  # type: ignore[attr-defined]
        except Exception:
            logger.debug(
                "RefreshableCredentials._seconds_remaining() raised; "
                "falling back to configured TTL.",
                exc_info=True,
            )
            return None
        return int(remaining) if remaining > 0 else None

    def _do_refresh(self) -> None:
        from aws_bedrock_token_generator import provide_token
        from botocore.credentials import Credentials
        from botocore.session import Session as BotocoreSession

        # Resolve credentials with precedence:
        # explicit access_key + secret_key > profile > default chain.
        credentials: Any = None
        if self._aws_access_key_id and self._aws_secret_access_key:
            credentials = Credentials(
                access_key=self._aws_access_key_id,
                secret_key=self._aws_secret_access_key,
                token=self._aws_session_token,
            )
        else:
            try:
                if self._credentials_profile_name:
                    session = BotocoreSession(profile=self._credentials_profile_name)
                else:
                    session = BotocoreSession()
                credentials = session.get_credentials()
            except Exception:
                logger.debug(
                    "Could not resolve credentials via botocore session; "
                    "provide_token will fall back to its own default chain.",
                    exc_info=True,
                )
                credentials = None

        # Cap TTL by underlying credential lifetime for refreshable creds
        # (AssumeRole, SSO, IRSA, etc.). Static creds have no expiry
        # metadata so we use the configured TTL directly.
        effective_ttl = self._ttl_seconds
        remaining = self._credentials_seconds_remaining(credentials)
        if remaining is not None:
            effective_ttl = min(self._ttl_seconds, remaining)

        # Wrap the resolved credentials in a CredentialProvider shape so
        # provide_token uses them; passing None lets provide_token fall
        # back to its own default resolution.
        provider = _StaticCredentialProvider(credentials) if credentials else None

        token = provide_token(
            region=self._region,
            expiry=timedelta(seconds=effective_ttl),
            aws_credentials_provider=provider,
        )
        self._token = token
        self._expires_at = time.monotonic() + effective_ttl
