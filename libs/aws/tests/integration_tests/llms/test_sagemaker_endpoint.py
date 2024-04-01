from typing import Dict

from langchain_aws.llms import SagemakerEndpoint
from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler


class ContentHandler(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        return b""

    def transform_output(self, output: bytes) -> str:
        return ""


def test_sagemaker_endpoint_name_param() -> None:
    llm = SagemakerEndpoint(
        endpoint_name="foo",
        content_handler=ContentHandler(),
        region_name="us-west-2",
    )
    assert llm.endpoint_name == "foo"
