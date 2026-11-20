"""Tests for mapping botocore errors to standard LangChain model exceptions."""

from typing import Any, Dict, cast
from unittest import mock

import pytest
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    EventStreamError,
    ReadTimeoutError,
)
from langchain_core.callbacks import BaseCallbackHandler

from langchain_aws import BedrockLLM, ChatBedrock, ChatBedrockConverse


@pytest.mark.parametrize(
    ("error_code", "status_code", "expected_type"),
    [
        ("ThrottlingException", 429, "ModelRateLimitError"),
        ("ServiceQuotaExceededException", 429, "ModelRateLimitError"),
        ("ModelNotReadyException", 429, "ModelRateLimitError"),
        # Generic AWS throttling codes must win over the 400 status-code fallback.
        ("Throttling", 400, "ModelRateLimitError"),
        ("RequestLimitExceeded", 400, "ModelRateLimitError"),
        ("AccessDeniedException", 403, "ModelPermissionDeniedError"),
        ("UnrecognizedClientException", 403, "ModelAuthenticationError"),
        ("ResourceNotFoundException", 404, "ModelNotFoundError"),
        ("ValidationException", 400, "ModelInvalidRequestError"),
        ("InternalServerException", 500, "ModelAPIError"),
        ("ServiceUnavailableException", 503, "ModelAPIError"),
        ("ModelTimeoutException", 504, "ModelTimeoutError"),
    ],
)
def test_standardized_bedrock_error_mapping(
    error_code: str, status_code: int, expected_type: str
) -> None:
    from botocore.exceptions import ClientError
    from langchain_core import exceptions as lc_exceptions

    from langchain_aws.utils import _handle_bedrock_error

    error = ClientError(
        {
            "Error": {"Code": error_code, "Message": "Bedrock request failed"},
            "ResponseMetadata": {
                "RequestId": "test-request-id",
                "HostId": "",
                "HTTPHeaders": {},
                "RetryAttempts": 0,
                "HTTPStatusCode": status_code,
            },
        },
        "Converse",
    )

    exception_type = getattr(lc_exceptions, expected_type)
    with pytest.raises(exception_type) as exc_info:
        _handle_bedrock_error(error)

    assert isinstance(exc_info.value, ClientError)
    assert exc_info.value.response == error.response
    assert exc_info.value.operation_name == error.operation_name
    assert exc_info.value.__cause__ is error

    import pickle

    restored = pickle.loads(pickle.dumps(exc_info.value))
    assert isinstance(restored, exception_type)
    assert isinstance(restored, ClientError)
    assert restored.response == error.response


def test_model_error_exception_falls_back_to_error_dict_status_code() -> None:
    """Event-stream style errors may carry the status inside the Error dict."""
    from botocore.exceptions import ClientError
    from langchain_core.exceptions import ModelRateLimitError

    from langchain_aws.utils import _handle_bedrock_error

    error = ClientError(
        {
            "Error": cast(
                Any,
                {
                    "Code": "ModelErrorException",
                    "Message": "The model provider returned an error",
                    "originalStatusCode": "429",
                    "originalMessage": "Too many requests",
                },
            ),
            "ResponseMetadata": {
                "RequestId": "test-request-id",
                "HostId": "",
                "HTTPHeaders": {},
                "RetryAttempts": 0,
                "HTTPStatusCode": 424,
            },
        },
        "Converse",
    )

    with pytest.raises(ModelRateLimitError) as exc_info:
        _handle_bedrock_error(error)

    assert isinstance(exc_info.value, ClientError)
    assert exc_info.value.__cause__ is error


def test_botocore_read_timeout_maps_to_model_timeout_error() -> None:
    from botocore.exceptions import ReadTimeoutError
    from langchain_core.exceptions import ModelTimeoutError

    from langchain_aws.utils import _handle_bedrock_error

    error = ReadTimeoutError(
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
    )

    with pytest.raises(ModelTimeoutError) as exc_info:
        _handle_bedrock_error(error)

    assert isinstance(exc_info.value, ReadTimeoutError)
    assert exc_info.value.__cause__ is error


def test_botocore_endpoint_error_maps_to_model_connection_error() -> None:
    from botocore.exceptions import EndpointConnectionError
    from langchain_core.exceptions import ModelConnectionError

    from langchain_aws.utils import _handle_bedrock_error

    error = EndpointConnectionError(
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
    )

    with pytest.raises(ModelConnectionError) as exc_info:
        _handle_bedrock_error(error)

    assert isinstance(exc_info.value, EndpointConnectionError)
    assert exc_info.value.__cause__ is error


def test_unknown_bedrock_error_is_reraised_unchanged() -> None:
    from botocore.exceptions import ClientError

    from langchain_aws.utils import _handle_bedrock_error

    error = ClientError(
        {
            "Error": {"Code": "SomeUnknownBedrockError", "Message": "unknown"},
            "ResponseMetadata": {
                "RequestId": "test-request-id",
                "HostId": "",
                "HTTPHeaders": {},
                "RetryAttempts": 0,
                "HTTPStatusCode": 418,
            },
        },
        "Converse",
    )

    with pytest.raises(ClientError) as exc_info:
        _handle_bedrock_error(error)

    assert exc_info.value is error


def _stubbed_converse_error(
    code: str,
    http_status: int,
    modeled_fields: Dict[str, Any] | None = None,
) -> ClientError:
    """Return the ClientError a real botocore client produces for a Converse failure.

    The error goes through botocore's response parser and error factory, so
    ``ClientError.response`` has the layout seen in production: modeled error
    members such as ``originalStatusCode`` sit at the top level of the response,
    not inside ``Error``.
    """
    from botocore.session import get_session
    from botocore.stub import Stubber

    client: Any = get_session().create_client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",  # noqa: S106
    )
    stubber = Stubber(client)
    stubber.add_client_error(
        "converse",
        service_error_code=code,
        service_message="Bedrock request failed",
        http_status_code=http_status,
        modeled_fields=modeled_fields,
    )
    with stubber:
        with pytest.raises(ClientError) as exc_info:
            client.converse(
                modelId="test-model",
                messages=[{"role": "user", "content": [{"text": "hi"}]}],
            )
    return exc_info.value


@pytest.mark.parametrize(
    ("modeled_fields", "expected_type"),
    [
        ({"originalStatusCode": 429}, "ModelRateLimitError"),
        ({"originalStatusCode": 503}, "ModelAPIError"),
        ({"originalStatusCode": 400}, "ModelInvalidRequestError"),
        (None, "ModelAPIError"),
    ],
)
def test_model_error_exception_uses_original_status_code_from_real_response(
    modeled_fields: Dict[str, Any] | None, expected_type: str
) -> None:
    from langchain_core import exceptions as lc_exceptions

    from langchain_aws.utils import _handle_bedrock_error

    error = _stubbed_converse_error("ModelErrorException", 424, modeled_fields)
    # Guard the premise of this test: the field is not inside ``Error``.
    assert "originalStatusCode" not in error.response["Error"]

    with pytest.raises(getattr(lc_exceptions, expected_type)) as exc_info:
        _handle_bedrock_error(error)

    assert exc_info.value.__cause__ is error


def test_handle_bedrock_error_is_idempotent() -> None:
    from langchain_core.exceptions import ModelRateLimitError

    from langchain_aws.utils import _handle_bedrock_error

    error = _stubbed_converse_error("ThrottlingException", 429)

    with pytest.raises(ModelRateLimitError) as first:
        _handle_bedrock_error(error)

    with pytest.raises(ModelRateLimitError) as second:
        _handle_bedrock_error(first.value)

    assert second.value is first.value
    assert type(second.value).__name__.count("ModelRateLimitError") == 1


@pytest.mark.parametrize(
    "error_name", ["ResponseStreamingError", "IncompleteReadError"]
)
def test_botocore_stream_transport_errors_map_to_model_connection_error(
    error_name: str,
) -> None:
    import pickle

    from botocore import exceptions as boto_exceptions
    from langchain_core.exceptions import ModelConnectionError

    from langchain_aws.utils import _handle_bedrock_error

    error: Exception
    if error_name == "ResponseStreamingError":
        error = boto_exceptions.ResponseStreamingError(error="connection reset")
    else:
        error = boto_exceptions.IncompleteReadError(actual_bytes=1, expected_bytes=10)

    with pytest.raises(ModelConnectionError) as exc_info:
        _handle_bedrock_error(error)

    assert isinstance(exc_info.value, type(error))
    assert exc_info.value.__cause__ is error
    assert isinstance(pickle.loads(pickle.dumps(exc_info.value)), ModelConnectionError)


def test_bedrock_context_overflow_error_still_importable_from_converse_module() -> None:
    from langchain_aws import utils
    from langchain_aws.chat_models import bedrock_converse

    assert (
        bedrock_converse.BedrockContextOverflowError
        is utils.BedrockContextOverflowError
    )


_ENDPOINT = "https://bedrock-runtime.us-east-1.amazonaws.com"


@pytest.mark.parametrize(
    "error_type",
    [ReadTimeoutError, ConnectTimeoutError, EndpointConnectionError],
    ids=lambda t: t.__name__,
)
def test_botocore_transport_errors_survive_pickling(error_type: Any) -> None:
    """These inherit from requests/urllib3 types too, so they are the riskiest."""
    import pickle

    from langchain_core.exceptions import ModelConnectionError, ModelTimeoutError

    from langchain_aws.utils import _handle_bedrock_error

    error = error_type(endpoint_url=_ENDPOINT)

    with pytest.raises((ModelTimeoutError, ModelConnectionError)) as exc_info:
        _handle_bedrock_error(error)

    restored = pickle.loads(pickle.dumps(exc_info.value))
    assert type(restored) is type(exc_info.value)
    assert isinstance(restored, type(error))
    assert str(restored) == str(error)


def test_invoke_tool_permission_error_is_value_error_and_survives_pickling() -> None:
    import pickle

    from langchain_core.exceptions import ModelPermissionDeniedError

    from langchain_aws.utils import _handle_bedrock_error

    error = ClientError(
        cast(
            Any,
            {
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": "not authorized to perform: bedrock:InvokeTool",
                },
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
        ),
        "Converse",
    )

    with pytest.raises(ValueError) as exc_info:
        _handle_bedrock_error(error)

    assert isinstance(exc_info.value, ModelPermissionDeniedError)
    assert isinstance(exc_info.value, ClientError)
    assert "bedrock:InvokeTool' IAM permission" in str(exc_info.value)
    assert exc_info.value.response == error.response  # type: ignore[attr-defined]
    assert exc_info.value.__cause__ is error

    restored = pickle.loads(pickle.dumps(exc_info.value))
    assert type(restored) is type(exc_info.value)
    assert str(restored) == str(exc_info.value)
    assert restored.response == error.response


class _ErrorRecorder(BaseCallbackHandler):
    """Record every ``on_llm_error`` call."""

    def __init__(self) -> None:
        self.errors: list[BaseException] = []

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.errors.append(error)


def _client_error(code: str, status: int, operation: str = "Converse") -> ClientError:
    return ClientError(
        cast(
            Any,
            {
                "Error": {"Code": code, "Message": "Bedrock request failed"},
                "ResponseMetadata": {"HTTPStatusCode": status},
            },
        ),
        operation,
    )


def _converse_model(client: Any) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0", region_name="us-east-1", client=client
    )


@pytest.mark.asyncio
async def test_converse_ainvoke_maps_model_errors() -> None:
    from langchain_core.exceptions import ModelRateLimitError

    error = _client_error("ThrottlingException", 429)
    client = mock.Mock()
    client.converse.side_effect = error

    with pytest.raises(ModelRateLimitError) as exc_info:
        await _converse_model(client).ainvoke("Hello")

    assert isinstance(exc_info.value, ClientError)
    assert exc_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_converse_astream_maps_open_errors() -> None:
    from langchain_core.exceptions import ModelAPIError

    error = _client_error("InternalServerException", 500, "ConverseStream")
    client = mock.Mock()
    client.converse_stream.side_effect = error

    with pytest.raises(ModelAPIError) as exc_info:
        async for _ in _converse_model(client).astream("Hello"):
            pass

    assert isinstance(exc_info.value, ClientError)
    assert exc_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_converse_astream_maps_mid_stream_errors() -> None:
    from langchain_core.exceptions import ModelRateLimitError

    error = EventStreamError(
        {"Error": {"Code": "throttlingException", "Message": "Too many requests"}},
        "ConverseStream",
    )

    def _events() -> Any:
        raise error
        yield {}

    client = mock.Mock()
    client.converse_stream.return_value = {"stream": _events()}

    with pytest.raises(ModelRateLimitError) as exc_info:
        async for _ in _converse_model(client).astream("Hello"):
            pass

    assert isinstance(exc_info.value, EventStreamError)
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("method", ["invoke", "stream"])
@pytest.mark.parametrize("model_cls", [ChatBedrock, BedrockLLM])
def test_service_error_is_reported_to_callbacks_once(
    model_cls: Any, method: str
) -> None:
    """Errors raised while opening the request reach callbacks exactly once."""
    from langchain_core.exceptions import ModelRateLimitError

    client = mock.MagicMock()
    client.invoke_model.side_effect = _client_error(
        "ThrottlingException", 429, "InvokeModel"
    )
    client.invoke_model_with_response_stream.side_effect = _client_error(
        "ThrottlingException", 429, "InvokeModelWithResponseStream"
    )
    model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    model: Any
    if model_cls is ChatBedrock:
        model = ChatBedrock(model=model_id, region="us-west-2", client=client)
    else:
        model = BedrockLLM(model=model_id, region="us-west-2", client=client)
    recorder = _ErrorRecorder()
    config: Any = {"callbacks": [recorder]}

    with pytest.raises(ModelRateLimitError):
        if method == "invoke":
            model.invoke("Hello", config=config)
        else:
            list(model.stream("Hello", config=config))

    assert len(recorder.errors) == 1
    assert isinstance(recorder.errors[0], ModelRateLimitError)
