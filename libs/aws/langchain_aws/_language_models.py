"""Base bedrock chat model implementation"""
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

import boto3
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_core.utils import enforce_stop_tokens


class _BaseBedrockLanguageModel(BaseLanguageModel):
    """Base class for Bedrock models."""

    client: Any = Field(exclude=True)  #: :meta private:

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

    config: Any = None
    """An optional botocore.config.Config instance to pass to the client."""

    provider: Optional[str] = None
    """The model provider, e.g., amazon, cohere, ai21, etc. When not supplied, provider
    is extracted from the first part of the model_id e.g. 'amazon' in 
    'amazon.titan-text-express-v1'. This value should be provided for model ids that do
    not have the provider in them, e.g., custom and provisioned models that have an ARN
    associated with them."""

    model_id: str
    """Id of the model to call, e.g., amazon.titan-text-express-v1, this is
    equivalent to the modelId property in the list-foundation-models api. For custom and
    provisioned models, an ARN value is expected."""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    endpoint_url: Optional[str] = None
    """Needed if you don't want to default to us-east-1 endpoint"""

    streaming: bool = False
    """Whether to stream the results."""

    provider_stop_sequence_key_name_map: Mapping[str, str] = {
        "anthropic": "stop_sequences",
        "amazon": "stopSequences",
        "ai21": "stop_sequences",
        "cohere": "stop_sequences",
    }

    guardrails: Optional[Mapping[str, Any]] = {
        "id": None,
        "version": None,
        "trace": False,
    }
    """
    An optional dictionary to configure guardrails for Bedrock.

    This field 'guardrails' consists of two keys: 'id' and 'version',
    which should be strings, but are initialized to None. It's used to
    determine if specific guardrails are enabled and properly set.

    Type:
        Optional[Mapping[str, str]]: A mapping with 'id' and 'version' keys.

    Example:
    llm = Bedrock(model_id="<model_id>", client=<bedrock_client>,
                  model_kwargs={},
                  guardrails={
                        "id": "<guardrail_id>",
                        "version": "<guardrail_version>"})

    To enable tracing for guardrails, set the 'trace' key to True and pass a callback handler to the
    'run_manager' parameter of the 'generate', '_call' methods.

    Example:
    llm = Bedrock(model_id="<model_id>", client=<bedrock_client>,
                  model_kwargs={},
                  guardrails={
                        "id": "<guardrail_id>",
                        "version": "<guardrail_version>",
                        "trace": True},
                callbacks=[BedrockAsyncCallbackHandler()])

    [https://python.langchain.com/docs/modules/callbacks/] for more information on callback handlers.

    class BedrockAsyncCallbackHandler(AsyncCallbackHandler):
        async def on_llm_error(
            self,
            error: BaseException,
            **kwargs: Any,
        ) -> Any:
            reason = kwargs.get("reason")
            if reason == "GUARDRAIL_INTERVENED":
                ...Logic to handle guardrail intervention...
    """  # noqa: E501

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that AWS credentials to and python package exists in environment."""

        # Skip creating new client if passed in constructor
        if values["client"] is not None:
            return values

        try:

            if values["credentials_profile_name"] is not None:
                session = boto3.Session(profile_name=values["credentials_profile_name"])
            else:
                # use default credentials
                session = boto3.Session()

            values["region_name"] = get_from_dict_or_env(
                values,
                "region_name",
                "AWS_DEFAULT_REGION",
                default=session.region_name,
            )

            client_params = {}
            if values["region_name"]:
                client_params["region_name"] = values["region_name"]
            if values["endpoint_url"]:
                client_params["endpoint_url"] = values["endpoint_url"]
            if values["config"]:
                client_params["config"] = values["config"]

            values["client"] = session.client("bedrock-runtime", **client_params)
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    def _get_provider(self) -> str:
        if self.provider:
            return self.provider
        if self.model_id.startswith("arn"):
            raise ValueError(
                "Model provider should be supplied when passing a model ARN as "
                "model_id"
            )

        return self.model_id.split(".")[0]

    @property
    def _model_is_anthropic(self) -> bool:
        return self._get_provider() == "anthropic"

    @property
    def _guardrails_enabled(self) -> bool:
        """
        Determines if guardrails are enabled and correctly configured.
        Checks if 'guardrails' is a dictionary with non-empty 'id' and 'version' keys.
        Checks if 'guardrails.trace' is true.

        Returns:
            bool: True if guardrails are correctly configured, False otherwise.
        Raises:
            TypeError: If 'guardrails' lacks 'id' or 'version' keys.
        """
        try:
            return (
                isinstance(self.guardrails, dict)
                and bool(self.guardrails["id"])
                and bool(self.guardrails["version"])
            )

        except KeyError as e:
            raise TypeError(
                "Guardrails must be a dictionary with 'id' and 'version' keys."
            ) from e

    def _get_guardrails_canonical(self) -> Dict[str, Any]:
        """
        The canonical way to pass in guardrails to the bedrock service
        adheres to the following format:

        "amazon-bedrock-guardrailDetails": {
            "guardrailId": "string",
            "guardrailVersion": "string"
        }
        """
        return {
            "amazon-bedrock-guardrailDetails": {
                "guardrailId": self.guardrails.get("id"),  # type: ignore[union-attr]
                "guardrailVersion": self.guardrails.get("version"),  # type: ignore[union-attr]
            }
        }

    def _prepare_input_and_invoke(
        self,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        _model_kwargs = self.model_kwargs or {}

        provider = self._get_provider()
        params = {**_model_kwargs, **kwargs}
        if self._guardrails_enabled:
            params.update(self._get_guardrails_canonical())
        input_body = LLMInputOutputAdapter.prepare_input(
            provider=provider,
            model_kwargs=params,
            prompt=prompt,
            system=system,
            messages=messages,
        )
        body = json.dumps(input_body)
        accept = "application/json"
        contentType = "application/json"

        request_options = {
            "body": body,
            "modelId": self.model_id,
            "accept": accept,
            "contentType": contentType,
        }

        if self._guardrails_enabled:
            request_options["guardrail"] = "ENABLED"
            if self.guardrails.get("trace"):  # type: ignore[union-attr]
                request_options["trace"] = "ENABLED"

        try:
            response = self.client.invoke_model(**request_options)

            text, body = LLMInputOutputAdapter.prepare_output(
                provider, response
            ).values()

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        # Verify and raise a callback error if any intervention occurs or a signal is
        # sent from a Bedrock service,
        # such as when guardrails are triggered.
        services_trace = self._get_bedrock_services_signal(body)  # type: ignore[arg-type]

        if services_trace.get("signal") and run_manager is not None:
            run_manager.on_llm_error(
                Exception(
                    f"Error raised by bedrock service: {services_trace.get('reason')}"
                ),
                **services_trace,
            )

        return text

    def _get_bedrock_services_signal(self, body: dict) -> dict:
        """
        This function checks the response body for an interrupt flag or message that indicates
        whether any of the Bedrock services have intervened in the processing flow. It is
        primarily used to identify modifications or interruptions imposed by these services
        during the request-response cycle with a Large Language Model (LLM).
        """  # noqa: E501

        if (
            self._guardrails_enabled
            and self.guardrails.get("trace")  # type: ignore[union-attr]
            and self._is_guardrails_intervention(body)
        ):
            return {
                "signal": True,
                "reason": "GUARDRAIL_INTERVENED",
                "trace": body.get(AMAZON_BEDROCK_TRACE_KEY),
            }

        return {
            "signal": False,
            "reason": None,
            "trace": None,
        }

    def _is_guardrails_intervention(self, body: dict) -> bool:
        return body.get(GUARDRAILS_BODY_KEY) == "GUARDRAIL_INTERVENED"

    def _prepare_input_and_invoke_stream(
        self,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        _model_kwargs = self.model_kwargs or {}
        provider = self._get_provider()

        if stop:
            if provider not in self.provider_stop_sequence_key_name_map:
                raise ValueError(
                    f"Stop sequence key name for {provider} is not supported."
                )

            # stop sequence from _generate() overrides
            # stop sequences in the class attribute
            _model_kwargs[self.provider_stop_sequence_key_name_map.get(provider)] = stop

        if provider == "cohere":
            _model_kwargs["stream"] = True

        params = {**_model_kwargs, **kwargs}

        if self._guardrails_enabled:
            params.update(self._get_guardrails_canonical())

        input_body = LLMInputOutputAdapter.prepare_input(
            provider=provider,
            prompt=prompt,
            system=system,
            messages=messages,
            model_kwargs=params,
        )
        body = json.dumps(input_body)

        request_options = {
            "body": body,
            "modelId": self.model_id,
            "accept": "application/json",
            "contentType": "application/json",
        }

        if self._guardrails_enabled:
            request_options["guardrail"] = "ENABLED"
            if self.guardrails.get("trace"):  # type: ignore[union-attr]
                request_options["trace"] = "ENABLED"

        try:
            response = self.client.invoke_model_with_response_stream(**request_options)

        except Exception as e:
            raise ValueError(f"Error raised by bedrock service: {e}")

        for chunk in LLMInputOutputAdapter.prepare_output_stream(
            provider, response, stop, True if messages else False
        ):
            yield chunk
            # verify and raise callback error if any middleware intervened
            self._get_bedrock_services_signal(chunk.generation_info)  # type: ignore[arg-type]

            if run_manager is not None:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _aprepare_input_and_invoke_stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        _model_kwargs = self.model_kwargs or {}
        provider = self._get_provider()

        if stop:
            if provider not in self.provider_stop_sequence_key_name_map:
                raise ValueError(
                    f"Stop sequence key name for {provider} is not supported."
                )
            _model_kwargs[self.provider_stop_sequence_key_name_map.get(provider)] = stop

        if provider == "cohere":
            _model_kwargs["stream"] = True

        params = {**_model_kwargs, **kwargs}
        input_body = LLMInputOutputAdapter.prepare_input(
            provider=provider, prompt=prompt, model_kwargs=params
        )
        body = json.dumps(input_body)

        response = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.client.invoke_model_with_response_stream(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            ),
        )

        async for chunk in LLMInputOutputAdapter.aprepare_output_stream(
            provider, response, stop
        ):
            yield chunk
            if run_manager is not None and asyncio.iscoroutinefunction(
                run_manager.on_llm_new_token
            ):
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            elif run_manager is not None:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)  # type: ignore[unused-coroutine]