# type: ignore
from unittest.mock import MagicMock

from langchain_aws.retrievers.kendra import AmazonKendraRetriever


def test_construct_without_min_score_confidence() -> None:
    """min_score_confidence must be optional, as documented in the class's own
    Example (`AmazonKendraRetriever(index_id=...)`), which omits it entirely.

    The field previously had no `default=`, so `Optional[float]` alone did not
    make it optional under Pydantic v2 and construction raised a ValidationError.
    """
    retriever = AmazonKendraRetriever(index_id="test-index-id", client=MagicMock())
    assert retriever.min_score_confidence is None
