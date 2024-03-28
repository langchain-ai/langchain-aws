from langchain_aws import llms
from tests.unit_tests import assert_all_importable

EXPECTED_ALL_LLMS = ["SagemakerEndpoint"]


def test_imports() -> None:
    assert sorted(llms.__all__) == sorted(EXPECTED_ALL_LLMS)
    assert_all_importable(llms)
