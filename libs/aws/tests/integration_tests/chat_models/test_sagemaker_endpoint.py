"""Test SageMakerEndpoint chat model."""

import json
from typing import Any, Dict, Tuple, Type
from unittest.mock import Mock

import pytest
from langchain_core.language_models import (
    BaseChatModel,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import (
    BaseTool,
)
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_aws.chat_models.sagemaker_endpoint import (
    ChatModelContentHandler,
    ChatSagemakerEndpoint,
)


class DefaultHandler(ChatModelContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: Any, model_kwargs: Dict) -> bytes:
        return json.dumps(prompt).encode("utf-8")

    def transform_output(self, output: bytes) -> BaseMessage:
        response_json = json.loads(output.decode())
        return AIMessage(content=response_json[0]["generated_text"])


class ContentHandlerStream(ChatModelContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: Any, model_kwargs: Dict) -> bytes:
        body = {"messages": prompt, "stream": True}
        return json.dumps(body).encode("utf-8")

    def transform_output(self, output: bytes) -> AIMessageChunk:
        stop_token = "[DONE]"
        try:
            response_text = output.decode("utf-8").strip()
            if not response_text:
                return AIMessageChunk(content="")
            # Remove the "data: " prefix
            response_text = response_text[6:]
            response_json = json.loads(response_text)
            if response_json["choices"][0]["delta"]["content"] != stop_token:
                content = response_json["choices"][0]["delta"]["content"]

            return AIMessageChunk(content=content)
        except Exception as e:
            return AIMessageChunk(content=f"Error processing response: {str(e)}")


class MockStreamingBody:
    def __init__(self) -> None:
        self.data = [
            b'data: {"object":"chat.completion.chunk","id":"","created":1742696407,'
            + b'"model":"Qwen/Qwen2.5-1.5B-Instruct",'
            + b'"system_fingerprint":"3.0.1-native",'
            + b'"choices":[{"index":0,"delta":{"role":"assistant","content":"An"},'
            + b'"logprobs":null,"finish_reason":null}],"usage":null}',
            b'data: {"object":"chat.completion.chunk","id":"","created":1742696407,'
            + b'"model":"Qwen/Qwen2.5-1.5B-Instruct",'
            + b'"system_fingerprint":"3.0.1-native",'
            + b'"choices":[{"index":0,"delta":{"role":"assistant","content":" L"},'
            + b'"logprobs":null,"finish_reason":null}],"usage":null}',
            b'data: {"object":"chat.completion.chunk","id":"","created":1742696407,'
            + b'"model":"Qwen/Qwen2.5-1.5B-Instruct",'
            + b'"system_fingerprint":"3.0.1-native",'
            + b'"choices":[{"index":0,"delta":{"role":"assistant","content":"LM"},'
            + b'"logprobs":null,"finish_reason":null}],"usage":null}',
            b'data: {"object":"chat.completion.chunk","id":"","created":1742696407,'
            + b'"model":"Qwen/Qwen2.5-1.5B-Instruct",'
            + b'"system_fingerprint":"3.0.1-native",'
            + b'"choices":[{"index":0,"delta":{"role":"assistant","content":" is"},'
            + b'"logprobs":null,"finish_reason":null}],"usage":null}',
            b'data: {"object":"chat.completion.chunk","id":"","created":1742696408,'
            + b'"model":"Qwen/Qwen2.5-1.5B-Instruct",'
            + b'"system_fingerprint":"3.0.1-native",'
            + b'"choices":[{"index":0,"delta":{"role":"assistant","content":" an"},'
            + b'"logprobs":null,"finish_reason":null}],"usage":null}',
            b'data: {"object":"chat.completion.chunk","id":"","created":1742696408,'
            + b'"model":"Qwen/Qwen2.5-1.5B-Instruct",'
            + b'"system_fingerprint":"3.0.1-native",'
            + b'"choices":[{"index":0,"delta":{"role":"assistant",'
            + b'"content":" acronym"},'
            + b'"logprobs":null,"finish_reason":null}],"usage":null}',
        ]
        self.index = 0

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        if self.index < len(self.data):
            chunk = {"PayloadPart": {"Bytes": self.data[self.index]}}
            self.index += 1
            return chunk
        raise StopIteration


class TestSageMakerStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatSagemakerEndpoint

    @property
    def chat_model_params(self) -> dict:
        return {
            "endpoint_name": "my-endpoint",
            "region_name": "us-west-2",
            "content_handler": DefaultHandler(),
            "model_kwargs": {},
            "endpoint_kwargs": {},
        }

    @property
    def standard_chat_model_params(self) -> dict:
        return {
            "model_kwargs": {"temperature": 0.7},
        }

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        """Return env vars, init args, and expected instance attrs for initializing from env vars."""  # noqa: E501
        return (
            {
                "AWS_ACCESS_KEY_ID": "test-key",
                "AWS_SECRET_ACCESS_KEY": "test-secret",
                "AWS_SESSION_TOKEN": "test-token",
                "AWS_DEFAULT_REGION": "us-west-2",
            },
            {
                "endpoint_name": "my-endpoint",
                "content_handler": DefaultHandler(),
            },
            {
                "region_name": "us-west-2",
                "endpoint_name": "my-endpoint",
            },
        )

    @pytest.mark.xfail(reason="Doesn't support streaming init param.")
    def test_init_streaming(self) -> None:
        super().test_init_streaming()

    @pytest.mark.xfail(reason="Doesn't support binding tool.")
    def test_bind_tool_pydantic(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_bind_tool_pydantic(model, my_adder_tool)

    @pytest.mark.xfail(reason="Doesn't support structured output.")
    def test_with_structured_output(self, model: BaseChatModel, schema: Any) -> None:
        super().test_with_structured_output(model, schema)

    @pytest.mark.xfail(reason="Doesn't support Langsmith parameters.")
    def test_standard_params(self, model: BaseChatModel) -> None:
        super().test_standard_params(model)

    @pytest.mark.xfail(reason="Doesn't support Langsmith parameters.")
    def test_init_from_env(self) -> None:
        super().test_init_from_env()


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


def test_sagemaker_endpoint_stream() -> None:
    client = Mock()

    stream_response = {"Body": MockStreamingBody()}
    client.invoke_endpoint_with_response_stream.return_value = stream_response

    chat_model = ChatSagemakerEndpoint(
        endpoint_name="test-endpoint",
        region_name="us-west-2",
        content_handler=ContentHandlerStream(),
        client=client,
    )

    messages = [HumanMessage(content="What is an LLM?")]
    chunks = list(chat_model.stream(messages))
    assert len(chunks) > 0
    assert all(isinstance(chunk, AIMessageChunk) for chunk in chunks)
