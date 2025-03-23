# type:ignore
"""Test chat model integration."""
import json
from typing import Dict
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_aws.chat_models.sagemaker_endpoint import (
    ChatModelContentHandler,
    ChatSagemakerEndpoint,
    _messages_to_sagemaker,
)


class DefaultHandler(ChatModelContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs: Dict) -> bytes:
        return json.dumps(prompt).encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.decode())
        return AIMessage(content=response_json[0]["generated_text"])


def test_format_messages_request() -> None:
    client = Mock()
    messages = [
        SystemMessage("Output everything you have."),  # type: ignore[misc]
        HumanMessage("What is an llm?"),  # type: ignore[misc]
    ]
    kwargs = {}

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
    invocation_params = llm._format_messages_request(messages=messages, **kwargs)

    expected_invocation_params = {
        "EndpointName": "my-endpoint",
        "Body": b"""[{"role": "system", "content": "Output everything you have."}, {"role": "user", "content": "What is an llm?"}]""",
        "ContentType": "application/json",
        "Accept": "application/json",
    }
    assert invocation_params == expected_invocation_params


def test__messages_to_sagemaker() -> None:
    messages = [
        SystemMessage("foo"),  # type: ignore[misc]
        HumanMessage("bar"),  # type: ignore[misc]
        AIMessage("some answer"),
        HumanMessage("follow-up question"),  # type: ignore[misc]
    ]
    expected = [
        {"role": "system", "content": "foo"},
        {"role": "user", "content": "bar"},
        {"role": "assistant", "content": "some answer"},
        {"role": "user", "content": "follow-up question"},
    ]
    actual = _messages_to_sagemaker(messages)
    assert expected == actual
