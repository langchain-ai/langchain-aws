"""Bedrock large language models."""
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Optional,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult


class BedrockLLM(BaseLLM):
    """BedrockLLM large language models.

    Example:
        .. code-block:: python

            from langchain_aws import BedrockLLM

            model = BedrockLLM()
            model.invoke("Come up with 10 names for a song about parrots")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "aws-llm"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        raise NotImplementedError

    # TODO: Implement if __model_name__ supports async generation. Otherwise
    # delete method.
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        raise NotImplementedError

    # TODO: Implement if __model_name__ supports streaming. Otherwise delete method.
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        raise NotImplementedError

    # TODO: Implement if __model_name__ supports async streaming. Otherwise delete
    # method.
    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        raise NotImplementedError
