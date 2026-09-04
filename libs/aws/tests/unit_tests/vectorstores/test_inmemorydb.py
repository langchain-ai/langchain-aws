from unittest.mock import Mock, patch

import pytest

from langchain_aws.vectorstores.inmemorydb.base import InMemoryVectorStore


@pytest.mark.parametrize(
    ("distance_threshold", "uses_range_query"),
    [(0.0, True), (0.5, True), (None, False)],
)
def test_prepare_query_selects_query_type_by_distance_threshold(
    distance_threshold: float | None, uses_range_query: bool
) -> None:
    store = object.__new__(InMemoryVectorStore)
    store._schema = Mock(
        content_key="content",
        metadata_keys=[],
        vector_dtype="float32",
    )
    range_query = Mock()
    vector_query = Mock()

    with (
        patch(
            "langchain_aws.vectorstores.inmemorydb.base._array_to_buffer",
            return_value=b"encoded-query",
        ),
        patch.object(store, "_prepare_range_query", return_value=range_query) as range,
        patch.object(store, "_prepare_vector_query", return_value=vector_query) as knn,
    ):
        query, params = store._prepare_query(
            [0.1, 0.2],
            k=3,
            distance_threshold=distance_threshold,
            with_metadata=False,
        )

    expected_params: dict[str, bytes | float] = {"vector": b"encoded-query"}
    if uses_range_query:
        assert distance_threshold is not None
        expected_params["distance_threshold"] = distance_threshold
        assert query is range_query
        range.assert_called_once_with(3, filter=None, return_fields=["content"])
        knn.assert_not_called()
    else:
        assert query is vector_query
        range.assert_not_called()
        knn.assert_called_once_with(3, filter=None, return_fields=["content"])
    assert params == expected_params
