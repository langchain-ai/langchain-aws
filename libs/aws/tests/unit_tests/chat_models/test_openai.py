"""ChatOpenAIMantle unit tests."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Tuple, Type, cast
from unittest.mock import MagicMock, patch

import openai
import pytest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel, SecretStr
from pytest import MonkeyPatch

from langchain_aws import ChatOpenAIMantle
from langchain_aws.chat_models.openai import (
    MantleAPIContextOverflowError,
    MantleContextOverflowError,
)
from langchain_aws.utils import _BedrockApiKeyProvider

MODEL_NAME = "openai.gpt-oss-120b"


class TestOpenAIMantleStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOpenAIMantle

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": MODEL_NAME,
            "region_name": "us-east-1",
            "bedrock_api_key": "test-bedrock-key",
        }

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Env vars, init args, and expected attrs for env-based initialization."""
        return (
            {
                "AWS_BEARER_TOKEN_BEDROCK": "env-bedrock-key",
                "AWS_REGION": "us-west-2",
            },
            {"model": MODEL_NAME},
            {"bedrock_api_key": "env-bedrock-key"},
        )


def test_initialization() -> None:
    """Explicit params are stored and the model is an OpenAI-compatible model."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.model_name == MODEL_NAME
    assert model.region_name == "us-east-1"
    assert cast("SecretStr", model.bedrock_api_key).get_secret_value() == "test-key"


def test_default_base_url_from_region() -> None:
    """base_url defaults to the region's Bedrock Mantle endpoint."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.openai_api_base == "https://bedrock-mantle.us-east-1.api.aws/v1"


@pytest.mark.parametrize(
    "model_name, expected_path",
    [
        ("openai.gpt-6-sol", "/openai/v1"),
        ("openai.gpt-6-luna", "/openai/v1"),
        ("openai.gpt-5.6-sol", "/openai/v1"),
        ("openai.gpt-oss-120b", "/v1"),
        ("qwen.qwen3-32b", "/v1"),
    ],
)
def test_default_base_url_route_by_model(model_name: str, expected_path: str) -> None:
    """GPT-5.x/GPT-6 default to Mantle's ``/openai/v1`` route, others to ``/v1``."""
    model = ChatOpenAIMantle(
        model=model_name,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert (
        model.openai_api_base
        == f"https://bedrock-mantle.us-east-1.api.aws{expected_path}"
    )


@pytest.mark.parametrize(
    "model_name, kwargs, expected",
    [
        ("openai.gpt-6-sol", {}, True),
        ("openai.gpt-5.6-luna", {}, True),
        ("openai.gpt-6-sol", {"use_responses_api": False}, False),
        ("openai.gpt-oss-120b", {}, None),
        ("qwen.qwen3-32b", {}, None),
    ],
)
def test_default_use_responses_api_by_model(
    model_name: str, kwargs: dict, expected: bool | None
) -> None:
    """GPT-5.x/GPT-6 default to the Responses API unless the caller sets it."""
    model = ChatOpenAIMantle(
        model=model_name,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
        **kwargs,
    )
    assert model.use_responses_api is expected


def test_explicit_base_url_is_respected() -> None:
    """An explicit base_url overrides the region-derived default."""
    custom = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        base_url=custom,
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.openai_api_base == custom


def test_missing_region_and_base_url_raises() -> None:
    """Fail locally instead of routing the Bedrock key to the default OpenAI host."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_REGION", raising=False)
        m.delenv("AWS_DEFAULT_REGION", raising=False)
        with pytest.raises(ValueError, match="region"):
            ChatOpenAIMantle(model=MODEL_NAME, bedrock_api_key=SecretStr("test-key"))


def test_explicit_base_url_without_region_ok() -> None:
    """An explicit base_url bypasses the region requirement."""
    custom = "https://bedrock-mantle.us-east-1.api.aws/v1"
    with MonkeyPatch().context() as m:
        m.delenv("AWS_REGION", raising=False)
        m.delenv("AWS_DEFAULT_REGION", raising=False)
        model = ChatOpenAIMantle(
            model=MODEL_NAME,
            base_url=custom,
            bedrock_api_key=SecretStr("test-key"),
        )
    assert model.openai_api_base == custom


def test_region_and_key_from_env() -> None:
    """Region and bedrock_api_key are read from the environment."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_DEFAULT_REGION", raising=False)
        m.setenv("AWS_REGION", "eu-west-1")
        m.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-key")
        model = ChatOpenAIMantle(model=MODEL_NAME)
        assert model.openai_api_base == "https://bedrock-mantle.eu-west-1.api.aws/v1"
        assert cast("SecretStr", model.bedrock_api_key).get_secret_value() == "env-key"


def test_bedrock_api_key_routed_to_openai_key() -> None:
    """The Bedrock API key is used as the OpenAI client api_key."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("bearer-abc"),
    )
    assert cast("SecretStr", model.openai_api_key).get_secret_value() == "bearer-abc"


def test_ls_params_provider() -> None:
    """Tracing provider is reported as openai-mantle."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    ls_params = model._get_ls_params()
    assert ls_params["ls_provider"] == "openai-mantle"
    assert ls_params["ls_model_name"] == MODEL_NAME


def test_lc_secrets() -> None:
    """The bedrock_api_key maps to its environment variable."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.lc_secrets["bedrock_api_key"] == "AWS_BEARER_TOKEN_BEDROCK"


def test_profile_resolved_from_model_name() -> None:
    """The model profile is populated from static profile data by default."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.profile is not None
    assert model.profile.get("tool_calling") is True


def test_explicit_profile_is_respected() -> None:
    """A caller-supplied profile is not overwritten by the default lookup."""
    custom = {"tool_calling": False}
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
        profile=custom,  # type: ignore[arg-type]
    )
    assert model.profile == custom


def test_profile_empty_for_unknown_model() -> None:
    """An unknown model resolves to an empty profile rather than raising."""
    model = ChatOpenAIMantle(
        model="openai.does-not-exist",
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    assert model.profile == {}


def test_stream_routes_to_responses_api() -> None:
    """_stream delegates to the Responses API path when it is active."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
        use_responses_api=True,
    )
    with (
        patch.object(
            BaseChatOpenAI, "_stream_responses", return_value=iter([])
        ) as responses,
        patch.object(BaseChatOpenAI, "_stream", return_value=iter([])) as completions,
    ):
        list(model._stream([HumanMessage("hi")]))
    responses.assert_called_once()
    completions.assert_not_called()


def test_stream_routes_to_chat_completions_when_disabled() -> None:
    """_stream falls back to Chat Completions when the Responses API is off."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
        use_responses_api=False,
    )
    with (
        patch.object(
            BaseChatOpenAI, "_stream_responses", return_value=iter([])
        ) as responses,
        patch.object(BaseChatOpenAI, "_stream", return_value=iter([])) as completions,
    ):
        list(model._stream([HumanMessage("hi")]))
    completions.assert_called_once()
    responses.assert_not_called()


def test_with_structured_output_defaults_to_json_schema() -> None:
    """with_structured_output defaults method to json_schema."""

    class Schema(BaseModel):
        answer: str

    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    with patch.object(BaseChatOpenAI, "with_structured_output") as parent:
        model.with_structured_output(Schema)
    assert parent.call_args.kwargs["method"] == "json_schema"


def test_inherits_openai_features() -> None:
    """Key BaseChatOpenAI methods are inherited unchanged."""
    model = ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
    )
    for attr in (
        "bind_tools",
        "with_structured_output",
        "_stream",
        "_astream",
        "_generate",
        "_agenerate",
    ):
        assert hasattr(model, attr)


# ---------------------------------------------------------------------------
# Credential-derived (short-term key) authentication path
# ---------------------------------------------------------------------------


def test_provider_installed_when_no_static_key() -> None:
    """With no static bearer key, a _BedrockApiKeyProvider is used as api_key."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.setenv("AWS_REGION", "us-east-1")
        model = ChatOpenAIMantle(
            model=MODEL_NAME,
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
            aws_session_token=SecretStr("TOKEN_TEST"),
        )

    provider = model.openai_api_key
    assert isinstance(provider, _BedrockApiKeyProvider)
    assert model.openai_api_base == "https://bedrock-mantle.us-east-1.api.aws/v1"
    # Explicitly-passed creds are forwarded to the provider.
    assert provider._region == "us-east-1"
    assert provider._aws_access_key_id == "AKIA_TEST"
    assert provider._aws_secret_access_key == "SECRET_TEST"
    assert provider._aws_session_token == "TOKEN_TEST"


def test_static_key_takes_precedence_over_creds() -> None:
    """A static bedrock_api_key wins over AWS credentials."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.setenv("AWS_REGION", "us-east-1")
        model = ChatOpenAIMantle(
            model=MODEL_NAME,
            bedrock_api_key=SecretStr("bearer-abc"),
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
        )
    assert cast("SecretStr", model.openai_api_key).get_secret_value() == "bearer-abc"


def test_explicit_api_key_not_overridden_by_provider() -> None:
    """An explicit api_key is left untouched (no provider installed)."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.setenv("AWS_REGION", "us-east-1")
        model = ChatOpenAIMantle(model=MODEL_NAME, api_key=SecretStr("explicit"))
    assert cast("SecretStr", model.openai_api_key).get_secret_value() == "explicit"


def test_provider_forwards_profile_and_ttl() -> None:
    """Profile name and requested TTL are forwarded; default chain otherwise."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.delenv("AWS_ACCESS_KEY_ID", raising=False)
        m.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        m.setenv("AWS_REGION", "us-east-1")
        model = ChatOpenAIMantle(
            model=MODEL_NAME,
            credentials_profile_name="my-profile",
            api_key_ttl_seconds=1800,
        )
    provider = model.openai_api_key
    assert isinstance(provider, _BedrockApiKeyProvider)
    assert provider._credentials_profile_name == "my-profile"
    assert provider._ttl_seconds == 1800
    assert provider._aws_access_key_id is None


def test_installed_provider_mints_token_when_called() -> None:
    """The installed provider returns a minted short-term key when invoked."""
    with MonkeyPatch().context() as m:
        m.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        m.setenv("AWS_REGION", "us-east-1")
        model = ChatOpenAIMantle(
            model=MODEL_NAME,
            aws_access_key_id=SecretStr("AKIA_TEST"),
            aws_secret_access_key=SecretStr("SECRET_TEST"),
        )
    provider = cast("_BedrockApiKeyProvider", model.openai_api_key)
    with patch("aws_bedrock_token_generator.provide_token", return_value="tok-xyz"):
        assert provider() == "tok-xyz"


def _make_model(**kwargs: object) -> ChatOpenAIMantle:
    return ChatOpenAIMantle(
        model=MODEL_NAME,
        region_name="us-east-1",
        bedrock_api_key=SecretStr("test-key"),
        **kwargs,  # type: ignore[arg-type]
    )


def test_guardrail_default_headers_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="not supported on the bedrock-mantle"):
        _make_model(
            default_headers={
                "X-Amzn-Bedrock-GuardrailIdentifier": "gr-1",
                "X-Amzn-Bedrock-GuardrailVersion": "1",
            },
        )


def test_guardrail_extra_headers_rejected_per_request() -> None:
    model = _make_model()
    with pytest.raises(ValueError, match="not supported on the bedrock-mantle"):
        model._get_request_payload(
            "hello",
            extra_headers={"X-Amzn-Bedrock-GuardrailIdentifier": "gr-1"},
        )


def test_max_tokens_sent_as_max_completion_tokens() -> None:
    """GPT-5.x/GPT-6 reject ``max_tokens``, so it is renamed like ``ChatOpenAI``."""
    payload = _make_model(max_tokens=64)._get_request_payload("hello")
    assert payload["max_completion_tokens"] == 64
    assert "max_tokens" not in payload


def test_non_guardrail_headers() -> None:
    model = _make_model(default_headers={"X-Custom-Header": "ok"})
    payload = model._get_request_payload(
        "hello", extra_headers={"X-Another-Header": "ok"}
    )
    assert payload["extra_headers"] == {"X-Another-Header": "ok"}


def test_openai_mantle_user_agent_tagged() -> None:
    """The OpenAI SDK's wire User-Agent carries the langchain-aws source tag, with
    the ``OpenAI/Python`` prefix preserved. ``default_headers`` is what
    ``BaseChatOpenAI`` forwards to the client's wire UA."""
    from langchain_aws._version import FRAMEWORK_UA_TOKEN

    model = _make_model()
    headers = model.default_headers
    assert headers is not None
    ua = headers["User-Agent"]
    assert ua.endswith(f" {FRAMEWORK_UA_TOKEN}")
    assert ua.startswith("OpenAI/Python")


def test_openai_mantle_user_agent_respects_caller_header() -> None:
    """A caller-supplied ``User-Agent`` gets the tag appended, not overwritten."""
    from langchain_aws._version import FRAMEWORK_UA_TOKEN

    model = _make_model(default_headers={"User-Agent": "MyApp/1.0"})
    headers = model.default_headers
    assert headers is not None
    assert headers["User-Agent"] == f"MyApp/1.0 {FRAMEWORK_UA_TOKEN}"


_MANTLE_OVERFLOW = (
    'ErrorEvent { error: APIError { type: "BadRequestError", code: Some(400), '
    "message: \"Input length (213426) exceeds model's maximum context length "
    '(131072).", param: None } }'
)


def _response() -> Any:
    # Stand-in for the SDK's HTTP response; the error classes only read these.
    return MagicMock(status_code=400, headers={}, request=MagicMock())


def _bad_request(message: str) -> openai.BadRequestError:
    return openai.BadRequestError(message, response=_response(), body=None)


def _api_error(message: str) -> openai.APIError:
    return openai.APIError(message, cast(Any, MagicMock()), body=None)


def _raising_stream(error: Exception) -> Iterator[Any]:
    raise error
    yield  # pragma: no cover


async def _raising_astream(error: Exception) -> AsyncIterator[Any]:
    raise error
    yield  # pragma: no cover


def test_invoke_context_overflow_raises_context_overflow_error() -> None:
    error = _bad_request(_MANTLE_OVERFLOW)
    with patch.object(BaseChatOpenAI, "_generate", side_effect=error):
        with pytest.raises(ContextOverflowError) as exc_info:
            _make_model().invoke("hello")

    assert isinstance(exc_info.value, MantleContextOverflowError)
    assert isinstance(exc_info.value, openai.BadRequestError)
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("model", [MODEL_NAME, "openai.gpt-5.6-sol"])
def test_stream_context_overflow_raises_context_overflow_error(model: str) -> None:
    # gpt-oss streams over Chat Completions; GPT-5.x over the Responses API.
    error = _api_error(_MANTLE_OVERFLOW)
    route = "_stream" if model == MODEL_NAME else "_stream_responses"
    with patch.object(
        BaseChatOpenAI, route, side_effect=lambda *a, **k: _raising_stream(error)
    ):
        with pytest.raises(ContextOverflowError) as exc_info:
            llm = ChatOpenAIMantle(
                model=model,
                region_name="us-east-1",
                bedrock_api_key=SecretStr("test-key"),
            )
            list(llm.stream("hello"))

    assert isinstance(exc_info.value, MantleAPIContextOverflowError)
    assert isinstance(exc_info.value, openai.APIError)
    assert exc_info.value.__cause__ is error


async def test_async_context_overflow_raises_context_overflow_error() -> None:
    invoke_error = _bad_request(_MANTLE_OVERFLOW)
    stream_error = _api_error(_MANTLE_OVERFLOW)
    with (
        patch.object(BaseChatOpenAI, "_agenerate", side_effect=invoke_error),
        patch.object(
            BaseChatOpenAI,
            "_astream",
            side_effect=lambda *a, **k: _raising_astream(stream_error),
        ),
    ):
        with pytest.raises(MantleContextOverflowError):
            await _make_model().ainvoke("hello")
        with pytest.raises(MantleAPIContextOverflowError):
            async for _ in _make_model().astream("hello"):
                pass


class _AlreadyOverflow(openai.BadRequestError, ContextOverflowError):
    pass


@pytest.mark.parametrize(
    "error",
    [
        _bad_request("Invalid 'max_output_tokens': integer below minimum value"),
        # e.g. GPT-5.x overflows, already classified by langchain-openai
        _AlreadyOverflow(
            "context_length_exceeded: maximum context length",
            response=_response(),
            body=None,
        ),
    ],
    ids=["other_error", "already_context_overflow"],
)
def test_errors_are_reraised_unchanged(error: openai.BadRequestError) -> None:
    with patch.object(BaseChatOpenAI, "_generate", side_effect=error):
        with pytest.raises(openai.BadRequestError) as exc_info:
            _make_model().invoke("hello")

    assert exc_info.value is error
