# type: ignore
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain_aws.retrievers.bedrock import (
    RetrievalConfig,
    SearchFilter,
    VectorSearchConfig,
)


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def mock_retriever_config():
    return RetrievalConfig(
        vectorSearchConfiguration=VectorSearchConfig(
            numberOfResults=5,
            filter=SearchFilter(in_={"key": "key", "value": ["value1", "value2"]}),
        ),
    )


@pytest.fixture
def mock_retriever_config_dict():
    return {
        "vectorSearchConfiguration": {
            "numberOfResults": 5,
            "filter": {"in": {"key": "key", "value": ["value1", "value2"]}},
        }
    }


@pytest.fixture
def amazon_retriever(mock_client, mock_retriever_config):
    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id="test_kb_id",
        retrieval_config=mock_retriever_config,
        client=mock_client,
    )


@pytest.fixture
def amazon_retriever_no_retrieval_config(mock_client, mock_retriever_config):
    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id="test_kb_id",
        client=mock_client,
    )


@pytest.fixture
def amazon_retriever_retrieval_config_dict(mock_client, mock_retriever_config_dict):
    return AmazonKnowledgeBasesRetriever(
        knowledge_base_id="test_kb_id",
        retrieval_config=mock_retriever_config_dict,
        client=mock_client,
    )


def test_retriever_invoke(amazon_retriever, mock_client):
    query = "test query"
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {"content": {"text": "result1"}, "metadata": {"key": "value1"}},
            {
                "content": {"text": "result2"},
                "metadata": {"key": "value2"},
                "score": 1,
                "location": "testLocation",
            },
            {"content": {"text": "result3"}},
        ]
    }
    documents = amazon_retriever.invoke(query, run_manager=None)

    mock_client.retrieve.assert_called_once_with(
        retrievalQuery={"text": "test query"},
        knowledgeBaseId="test_kb_id",
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 5,
                # Expecting to be called with correct "in" operator instead of "in_"
                "filter": {"in": {"key": "key", "value": ["value1", "value2"]}},
            }
        },
    )
    assert len(documents) == 3
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "result1"
    assert documents[0].metadata == {
        "score": 0,
        "source_metadata": {"key": "value1"},
        "type": "TEXT",
    }
    assert documents[1].page_content == "result2"
    assert documents[1].metadata == {
        "score": 1,
        "source_metadata": {"key": "value2"},
        "location": "testLocation",
        "type": "TEXT",
    }
    assert documents[2].page_content == "result3"
    assert documents[2].metadata == {"score": 0, "type": "TEXT"}


def test_retriever_invoke_with_score(amazon_retriever, mock_client):
    query = "test query"
    mock_client.retrieve.return_value = {
        "retrievalResults": [
            {"content": {"text": "result1"}, "metadata": {"key": "value1"}},
            {
                "content": {"text": "result2"},
                "metadata": {"key": "value2"},
                "score": 1,
                "location": "testLocation",
                "type": "TEXT",
            },
            {"content": {"text": "result3"}},
        ]
    }

    amazon_retriever.min_score_confidence = 0.6
    documents = amazon_retriever.invoke(query, run_manager=None)

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "result2"
    assert documents[0].metadata == {
        "score": 1,
        "source_metadata": {"key": "value2"},
        "location": "testLocation",
        "type": "TEXT",
    }


def test_retriever_retrieval_config_dict_invoke(
    amazon_retriever_retrieval_config_dict, mock_client
):
    documents = set_return_value_and_query(
        mock_client, amazon_retriever_retrieval_config_dict
    )
    validate_query_response_no_cutoff(documents)
    mock_client.retrieve.assert_called_once_with(
        retrievalQuery={"text": "test query"},
        knowledgeBaseId="test_kb_id",
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": 5,
                # Expecting to be called with correct "in" operator instead of "in_"
                "filter": {"in": {"key": "key", "value": ["value1", "value2"]}},
            }
        },
    )


def test_retriever_retrieval_config_dict_invoke_with_score(
    amazon_retriever_retrieval_config_dict, mock_client
):
    amazon_retriever_retrieval_config_dict.min_score_confidence = 0.6
    documents = set_return_value_and_query(
        mock_client, amazon_retriever_retrieval_config_dict
    )
    validate_query_response_with_cutoff(documents)


def test_retriever_no_retrieval_config_invoke(
    amazon_retriever_no_retrieval_config, mock_client
):
    documents = set_return_value_and_query(
        mock_client, amazon_retriever_no_retrieval_config
    )
    validate_query_response_no_cutoff(documents)
    mock_client.retrieve.assert_called_once_with(
        retrievalQuery={"text": "test query"}, knowledgeBaseId="test_kb_id"
    )


def test_retriever_no_retrieval_config_invoke_with_score(
    amazon_retriever_no_retrieval_config, mock_client
):
    amazon_retriever_no_retrieval_config.min_score_confidence = 0.6
    documents = set_return_value_and_query(
        mock_client, amazon_retriever_no_retrieval_config
    )
    validate_query_response_with_cutoff(documents)


@pytest.mark.parametrize(
    "search_results,expected_documents",
    [
        (
            [
                {
                    "content": {"text": "result"},
                    "metadata": {"key": "value1"},
                    "score": 1,
                    "location": "testLocation",
                },
                {
                    "content": {"text": "result"},
                    "metadata": {"key": "value1"},
                    "score": 0.5,
                    "location": "testLocation",
                },
            ],
            [
                Document(
                    page_content="result",
                    metadata={
                        "score": 1,
                        "location": "testLocation",
                        "source_metadata": {"key": "value1"},
                        "type": "TEXT",
                    },
                ),
                Document(
                    page_content="result",
                    metadata={
                        "score": 0.5,
                        "location": "testLocation",
                        "source_metadata": {"key": "value1"},
                        "type": "TEXT",
                    },
                ),
            ],
        ),
        # text type
        (
            [
                {
                    "content": {"text": "result", "type": "TEXT"},
                    "metadata": {"key": "value1"},
                    "score": 1,
                    "location": "testLocation",
                },
                {
                    "content": {"text": "result", "type": "TEXT"},
                    "metadata": {"key": "value1"},
                    "score": 0.5,
                    "location": "testLocation",
                },
            ],
            [
                Document(
                    page_content="result",
                    metadata={
                        "score": 1,
                        "location": "testLocation",
                        "source_metadata": {"key": "value1"},
                        "type": "TEXT",
                    },
                ),
                Document(
                    page_content="result",
                    metadata={
                        "score": 0.5,
                        "location": "testLocation",
                        "source_metadata": {"key": "value1"},
                        "type": "TEXT",
                    },
                ),
            ],
        ),
        # image type
        (
            [
                {
                    "content": {"byteContent": "bytecontent", "type": "IMAGE"},
                    "metadata": {"key": "value1"},
                    "score": 1,
                    "location": "testLocation",
                },
                {
                    "content": {"byteContent": "bytecontent", "type": "IMAGE"},
                    "metadata": {"key": "value1"},
                    "score": 0.5,
                    "location": "testLocation",
                },
            ],
            [
                Document(
                    page_content="bytecontent",
                    metadata={
                        "score": 1,
                        "location": "testLocation",
                        "source_metadata": {"key": "value1"},
                        "type": "IMAGE",
                    },
                ),
                Document(
                    page_content="bytecontent",
                    metadata={
                        "score": 0.5,
                        "location": "testLocation",
                        "source_metadata": {"key": "value1"},
                        "type": "IMAGE",
                    },
                ),
            ],
        ),
        # row type
        (
            [
                {
                    "content": {
                        "row": [
                            {"columnName": "someName1", "columnValue": "someValue1"},
                            {"columnName": "someName2", "columnValue": "someValue2"},
                        ],
                        "type": "ROW",
                    },
                    "score": 1,
                    "metadata": {"key": "value1"},
                    "location": "testLocation",
                },
                {
                    "content": {
                        "row": [
                            {"columnName": "someName1", "columnValue": "someValue1"},
                            {"columnName": "someName2", "columnValue": "someValue2"},
                        ],
                        "type": "ROW",
                    },
                    "score": 0.5,
                    "metadata": {"key": "value1"},
                    "location": "testLocation",
                },
            ],
            [
                Document(
                    page_content='[{"columnName": "someName1", "columnValue": "someValue1"}, '
                    '{"columnName": "someName2", "columnValue": "someValue2"}]',
                    metadata={
                        "score": 1,
                        "location": "testLocation",
                        "source_metadata": {"key": "value1"},
                        "type": "ROW",
                    },
                ),
                Document(
                    page_content='[{"columnName": "someName1", "columnValue": "someValue1"}, '
                    '{"columnName": "someName2", "columnValue": "someValue2"}]',
                    metadata={
                        "score": 0.5,
                        "location": "testLocation",
                        "source_metadata": {"key": "value1"},
                        "type": "ROW",
                    },
                ),
            ],
        ),
    ],
)
def test_retriever_with_multi_modal_types_then_get_valid_documents(
    mock_client, amazon_retriever, search_results, expected_documents
):
    query = "test query"
    mock_client.retrieve.return_value = {"retrievalResults": search_results}
    documents = amazon_retriever.invoke(query, run_manager=None)
    assert documents == expected_documents


@pytest.mark.parametrize(
    "search_result_input,expected_output",
    [
        # VALID INPUTS
        # no type
        ({"content": {"text": "result"}}, "result"),
        # text type
        ({"content": {"text": "result", "type": "TEXT"}}, "result"),
        # image type
        ({"content": {"byteContent": "bytecontent", "type": "IMAGE"}}, "bytecontent"),
        # row type
        (
            {
                "content": {
                    "row": [
                        {"columnName": "someName1", "columnValue": "someValue1"},
                        {"columnName": "someName2", "columnValue": "someValue2"},
                    ],
                    "type": "ROW",
                }
            },
            '[{"columnName": "someName1", "columnValue": "someValue1"}, '
            '{"columnName": "someName2", "columnValue": "someValue2"}]',
        ),
        # VALID INPUTS w/ metadata
        # no type
        ({"content": {"text": "result"}, "metadata": {"key": "value1"}}, "result"),
        # text type
        (
            {
                "content": {"text": "result", "type": "TEXT"},
                "metadata": {"key": "value1"},
            },
            "result",
        ),
        # image type
        (
            {
                "content": {"byteContent": "bytecontent", "type": "IMAGE"},
                "metadata": {"key": "value1"},
            },
            "bytecontent",
        ),
        # row type
        (
            {
                "content": {
                    "row": [
                        {"columnName": "someName1", "columnValue": "someValue1"},
                        {"columnName": "someName2", "columnValue": "someValue2"},
                    ],
                    "metadata": {"key": "value1"},
                    "type": "ROW",
                }
            },
            '[{"columnName": "someName1", "columnValue": "someValue1"}, '
            '{"columnName": "someName2", "columnValue": "someValue2"}]',
        ),
        # invalid type
        ({"content": {"invalid": "invalid", "type": "INVALID"}}, None),
        # EMPTY VALUES
        # no type
        ({"content": {"text": ""}}, ""),
        # text type
        ({"content": {"text": "", "type": "TEXT"}}, ""),
        # image type
        ({"content": {"byteContent": "", "type": "IMAGE"}}, ""),
        # row type
        ({"content": {"row": [], "type": "ROW"}}, "[]"),
        # NONE VALUES
        ({"content": {"text": None}}, None),
        # text type
        ({"content": {"text": None, "type": "TEXT"}}, None),
        # image type
        ({"content": {"byteContent": None, "type": "IMAGE"}}, None),
        # row type
        ({"content": {"row": None, "type": "ROW"}}, "[]"),
        # WRONG CONTENT
        # text
        ({"content": {"text": "result", "type": "IMAGE"}}, None),
        ({"content": {"text": "result", "type": "ROW"}}, "[]"),
        # byteContent
        ({"content": {"byteContent": "result"}}, None),
        ({"content": {"byteContent": "result", "type": "TEXT"}}, None),
        ({"content": {"byteContent": "result", "type": "ROW"}}, "[]"),
        # row
        (
            {
                "content": {
                    "row": [
                        {"columnName": "someName1", "columnValue": "someValue1"},
                        {"columnName": "someName2", "columnValue": "someValue2"},
                    ]
                }
            },
            None,
        ),
        (
            {
                "content": {
                    "row": [
                        {"columnName": "someName1", "columnValue": "someValue1"},
                        {"columnName": "someName2", "columnValue": "someValue2"},
                    ],
                    "type": "TEXT",
                }
            },
            None,
        ),
        (
            {
                "content": {
                    "row": [
                        {"columnName": "someName1", "columnValue": "someValue1"},
                        {"columnName": "someName2", "columnValue": "someValue2"},
                    ],
                    "type": "IMAGE",
                }
            },
            None,
        ),
    ],
)
def test_when_get_content_from_result_then_get_expected_content(
    search_result_input, expected_output
):
    assert (
        AmazonKnowledgeBasesRetriever._get_content_from_result(search_result_input)
        == expected_output
    )


@pytest.mark.parametrize(
    "search_result_input",
    [
        # empty content
        ({"content": {}}),
        # None content
        ({"content": None}),
        # empty dict
        ({}),
        # None search result
        None,
    ],
)
def test_when_get_content_from_result_with_invalid_content_then_raise_error(
    search_result_input,
):
    with pytest.raises(ValueError):
        AmazonKnowledgeBasesRetriever._get_content_from_result(search_result_input)


def set_return_value_and_query(
    client: Any, retriever: AmazonKnowledgeBasesRetriever
) -> List[Document]:
    query = "test query"
    client.retrieve.return_value = {
        "retrievalResults": [
            {"content": {"text": "result1"}, "metadata": {"key": "value1"}},
            {
                "content": {"text": "result2"},
                "metadata": {"key": "value2"},
                "score": 1,
                "location": "testLocation",
            },
            {"content": {"text": "result3"}},
        ]
    }
    return retriever.invoke(query, run_manager=None)


def validate_query_response_no_cutoff(documents: List[Document]):
    assert len(documents) == 3
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "result1"
    assert documents[0].metadata == {
        "score": 0,
        "source_metadata": {"key": "value1"},
        "type": "TEXT",
    }
    assert documents[1].page_content == "result2"
    assert documents[1].metadata == {
        "score": 1,
        "source_metadata": {"key": "value2"},
        "location": "testLocation",
        "type": "TEXT",
    }
    assert documents[2].page_content == "result3"
    assert documents[2].metadata == {"score": 0, "type": "TEXT"}


def validate_query_response_with_cutoff(documents: List[Document]):
    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == "result2"
    assert documents[0].metadata == {
        "score": 1,
        "source_metadata": {"key": "value2"},
        "location": "testLocation",
        "type": "TEXT",
    }
