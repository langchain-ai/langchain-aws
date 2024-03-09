"""Base bedrock chat model implementation"""

from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field, root_validator

from langchain_aws._constants import AMAZON_BEDROCK_TRACE_KEY, GUARDRAILS_BODY_KEY
from langchain_aws._session import BotoAuthMixin, get_client


class _BaseBedrockLanguageModel(BaseLanguageModel, BotoAuthMixin):
    """Base class for Bedrock models that handles boto3."""

    client: Any = Field(exclude=True)  #: :meta private:

    model_id: str
    """Id of the model to call, e.g., amazon.titan-text-express-v1, this is
    equivalent to the modelId property in the list-foundation-models api. For custom and
    provisioned models, an ARN value is expected."""

    model_kwargs: Optional[Dict] = None
    """Keyword arguments to pass to the model."""

    streaming: bool = False
    """Whether to stream the results."""

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
        if values["client"] is None:
            values["client"] = get_client("bedrock-runtime", **values)

        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

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
            and _is_guardrails_intervention(body)
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


def _is_guardrails_intervention(body: dict) -> bool:
    return body.get(GUARDRAILS_BODY_KEY) == "GUARDRAIL_INTERVENED"
