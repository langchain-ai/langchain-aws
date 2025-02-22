"""Test SageMakerEndpoint chat model."""

import json
from typing import Dict
from unittest.mock import Mock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_aws.chat_models.sagemaker_endpoint import ChatSagemakerEndpoint
from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler


class DefaultHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs: Dict) -> bytes:
        return json.dumps(prompt).encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.decode())
        return AIMessage(content=response_json[0]["generated_text"])


def test_sagemaker_endpoint_invoke() -> None:
    client = Mock()

    response = {
        "ContentType": "application/json",
        "Body": b'[{"generated_text": "SageMaker Endpoint"}]',
    }
    client.invoke_endpoint.return_value = response

    llm = ChatSagemakerEndpoint(
        endpoint_name="my-endpoint",
        region_name="us-west-2",
        content_handler=DefaultHandler(),
        model_kwargs={
            "parameters": {
                "max_new_tokens": 50,
            }
        },
        client=client,
    )
    messages = [
        SystemMessage(content="You are an AWS expert."),
        HumanMessage(content="What is Sagemaker endpoints?"),
    ]

    service_response = llm.invoke(messages)

    assert service_response.content == "SageMaker Endpoint"
    assert isinstance(service_response, AIMessage)
