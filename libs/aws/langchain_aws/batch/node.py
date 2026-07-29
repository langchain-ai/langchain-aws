"""Optional LangGraph node wrapping ``BedrockBatchManager``.

``BedrockBatchNode`` submits a batch job, checkpoints the graph via LangGraph's
``interrupt``, and resumes when the job completes -- so the host process can shut
down in between. LangGraph is an optional dependency
(``pip install "langchain-aws[batch]"``); the import is guarded so the base
package stays lightweight (see patterns-and-conventions §4).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage  # noqa: F401

from langchain_aws.batch._formatting import BatchInput  # noqa: F401
from langchain_aws.batch.manager import BedrockBatchManager  # noqa: F401
from langchain_aws.chat_models import ChatBedrockConverse  # noqa: F401


class BedrockBatchNode:
    """A durable LangGraph node that runs Bedrock batch inference.

    The node is callable as a LangGraph node function. On first execution it
    submits a job and interrupts; on resume it checks status and, once complete,
    returns parsed results to the graph state.

    !!! warning
        Requires the ``batch`` extra:
        ``pip install "langchain-aws[batch]"``.
    """

    def __init__(
        self,
        *,
        model_id: str,
        s3_bucket: str,
        role_arn: str,
        input_key: str = "batch_inputs",
        output_key: str = "batch_results",
        region_name: Optional[str] = None,
        output_kms_key_id: Optional[str] = None,
    ) -> None:
        """Initialize the node.

        Args:
            model_id: Bedrock model ID to run the batch job against.
            s3_bucket: Bucket used for both input and output prefixes.
            role_arn: IAM service role ARN for the batch job.
            input_key: State key holding the list of batch inputs.
            output_key: State key to write parsed results to.
            region_name: AWS region. Falls back to standard resolution.
            output_kms_key_id: Optional KMS key ARN for output encryption.

        Raises:
            ImportError: If LangGraph is not installed.
        """
        try:
            import langgraph  # noqa: F401
        except ImportError as e:
            msg = (
                "Cannot import langgraph. Please install it with "
                '`pip install "langchain-aws[batch]"`.'
            )
            raise ImportError(msg) from e

        raise NotImplementedError

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Submit (or resume) the batch job as part of graph execution.

        Args:
            state: The current LangGraph state. Reads inputs from ``input_key``.

        Returns:
            A partial state update writing parsed results to ``output_key``.
        """
        from langgraph.types import interrupt  # noqa: F401

        raise NotImplementedError
