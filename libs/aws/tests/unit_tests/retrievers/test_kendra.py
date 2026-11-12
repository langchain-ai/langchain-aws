from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from langchain_aws import AmazonKendraRetriever
from langchain_aws.retrievers.kendra import DocumentAttributeValue

# boto3 returns Kendra dates as datetime objects
LAST_UPDATED_AT = datetime(2024, 10, 27, 3, 33, 20, tzinfo=timezone.utc)

DOCUMENT_ATTRIBUTES = [
    {"Key": "_source_uri", "Value": {"StringValue": "https://example.com/handbook"}},
    {"Key": "_last_updated_at", "Value": {"DateValue": LAST_UPDATED_AT}},
]


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def retriever(mock_client: MagicMock) -> AmazonKendraRetriever:
    return AmazonKendraRetriever(
        index_id="test_index_id", client=mock_client, min_score_confidence=None
    )


def test_retrieve_result_with_date_attribute(
    retriever: AmazonKendraRetriever, mock_client: MagicMock
) -> None:
    mock_client.retrieve.return_value = {
        "QueryId": "test_query_id",
        "ResultItems": [
            {
                "Id": "result1",
                "DocumentId": "doc1",
                "DocumentTitle": "Handbook",
                "Content": "Employees get 20 days of leave.",
                "DocumentAttributes": DOCUMENT_ATTRIBUTES,
            }
        ],
    }

    docs = retriever.invoke("leave policy")

    assert docs[0].metadata["document_attributes"] == {
        "_source_uri": "https://example.com/handbook",
        "_last_updated_at": "2024-10-27T03:33:20+00:00",
    }


def test_query_result_with_date_attribute(
    retriever: AmazonKendraRetriever, mock_client: MagicMock
) -> None:
    # the retriever falls back to the Query API when Retrieve finds nothing
    mock_client.retrieve.return_value = {"QueryId": "test_query_id", "ResultItems": []}
    mock_client.query.return_value = {
        "ResultItems": [
            {
                "Id": "result1",
                "DocumentId": "doc1",
                "Type": "DOCUMENT",
                "DocumentTitle": {"Text": "Handbook"},
                "DocumentExcerpt": {"Text": "Employees get 20 days of leave."},
                "DocumentAttributes": DOCUMENT_ATTRIBUTES,
            }
        ]
    }

    docs = retriever.invoke("leave policy")

    assert docs[0].metadata["document_attributes"] == {
        "_source_uri": "https://example.com/handbook",
        "_last_updated_at": "2024-10-27T03:33:20+00:00",
    }


def test_date_value_string_is_kept() -> None:
    value = DocumentAttributeValue(DateValue="2024-10-27T03:33:20Z")

    assert value.value == "2024-10-27T03:33:20Z"
