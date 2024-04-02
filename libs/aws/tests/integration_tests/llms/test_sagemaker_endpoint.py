from typing import Dict

from langchain_aws.llms import SagemakerEndpoint
from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler

from botocore.stub import Stubber, ANY

from unittest.mock import Mock

import json
import io


class DefaultHandler(LLMContentHandler):
    accepts = "application/json"
    content_type = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        return prompt.encode()

    def transform_output(self, output: bytes) -> str:
        body = json.loads(output.read())
        return body[0]["generated_text"]
    
def create_mock_raw_stream(*data):
        raw_stream = Mock()
        def generator():
            yield from data

        raw_stream.stream = generator
        return raw_stream

def test_sagemaker_endpoint_invoke() -> None:

    client = Mock()
    response = {
        'ContentType': 'application/json',
        'Body': io.StringIO('[{"generated_text": "SageMaker Endpoint"}]')
    }
    client.invoke_endpoint.return_value = response

    llm = SagemakerEndpoint(
        endpoint_name="my-endpoint",
        region_name="us-west-2",
        content_handler=DefaultHandler(),
        model_kwargs={
            "parameters": {
                "max_new_tokens": 50,
            }
        },
        client=client
    )

    service_response = llm.invoke("What is Sagemaker endpoints?")

    assert service_response == "SageMaker Endpoint"
    client.invoke_endpoint.assert_called_once_with(
        EndpointName='my-endpoint', 
        Body=b'What is Sagemaker endpoints?', 
        ContentType='application/json', Accept='application/json'
    )


def test_sagemaker_endpoint_stream() -> None:
    class ContentHandler(LLMContentHandler):
        accepts = "application/json"
        content_type = "application/json"

        def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
            body = {
                'inputs': prompt,
                **model_kwargs
            }
            return body

        def transform_output(self, output: bytes) -> str:
            body = json.loads(output)
            return body.get("outputs")[0]


    body = (
        {'PayloadPart': {'Bytes': b'{"outputs": ["S"]}\n'}},
        {'PayloadPart': {'Bytes': b'{"outputs": ["age"]}\n'}},
        {'PayloadPart': {'Bytes': b'{"outputs": ["Maker"]}\n'}}
    )

    response = {
        'ContentType': 'application/json',
        'Body': body
    }

    client = Mock()
    client.invoke_endpoint_with_response_stream.return_value = response

    llm = SagemakerEndpoint(
        endpoint_name="my-endpoint",
        region_name="us-west-2",
        content_handler=ContentHandler(),
        client=client,
        model_kwargs={
            "parameters": {
                "max_new_tokens": 50
            }
        }
    )

    
    chunks = ['S', 'age', 'Maker']
    service_chunks = []

    for chunk in llm.stream("What is Sagemaker endpoints?"):
        service_chunks.append(chunk)

    assert service_chunks == chunks
    client.invoke_endpoint_with_response_stream.assert_called_once_with(
        EndpointName='my-endpoint', 
        Body={
            'inputs': 'What is Sagemaker endpoints?', 
            'parameters': {'max_new_tokens': 50}
        }, 
        ContentType='application/json'
    )
    
