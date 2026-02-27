# type:ignore
"""Test chat model integration."""

import json
from typing import Any, Dict, List
from unittest.mock import Mock

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

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
        "Body": (
            b'[{"role": "system", "content": "Output everything you have."}, '
            b'{"role": "user", "content": "What is an llm?"}]'
        ),
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


def _make_stream_payload(lines: List[bytes]) -> List[Dict[str, Any]]:
    return [{"PayloadPart": {"Bytes": line}} for line in lines]


class StreamingHandler(ChatModelContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def __init__(self, responses: List[BaseMessage]) -> None:
        self._responses = iter(responses)

    def transform_input(self, prompt: Any, model_kwargs: Dict) -> bytes:
        return json.dumps(prompt).encode("utf-8")

    def transform_output(self, output: bytes) -> BaseMessage:
        return next(self._responses)


def _build_streaming_llm(
    responses: List[BaseMessage],
) -> ChatSagemakerEndpoint:
    handler = StreamingHandler(responses)
    body = _make_stream_payload([b'{"text":"placeholder"}\n' for _ in responses])
    client = Mock()
    client.invoke_endpoint_with_response_stream.return_value = {"Body": body}

    return ChatSagemakerEndpoint(
        endpoint_name="test-endpoint",
        region_name="us-east-1",
        content_handler=handler,
        client=client,
    )


def test_stream_yields_usage_only_chunk_with_metadata() -> None:
    resp_meta = {"finish_reason": "stop"}
    msg_id = "cmpl-final"
    usage = {
        "input_tokens": 21,
        "output_tokens": 14,
        "total_tokens": 35,
    }
    responses: List[BaseMessage] = [
        AIMessage(content="Hello"),
        AIMessage(
            content="",
            response_metadata=resp_meta,
            id=msg_id,
            usage_metadata=usage,
        ),
    ]
    llm = _build_streaming_llm(responses)

    run_manager = Mock()
    chunks = list(llm._stream([HumanMessage(content="hi")], run_manager=run_manager))

    assert len(chunks) == 2
    assert chunks[0].message.content == "Hello"
    assert chunks[0].message.usage_metadata is None  # type: ignore[union-attr]

    final = chunks[1].message
    assert isinstance(final, AIMessageChunk)
    assert final.content == ""
    assert final.usage_metadata == usage  # type: ignore[union-attr]
    assert final.response_metadata == resp_meta
    assert final.id == msg_id

    assert run_manager.on_llm_new_token.call_count == 2
    second_call = run_manager.on_llm_new_token.call_args_list[1]
    assert second_call[0][0] == ""
    assert second_call[1]["chunk"].message.usage_metadata == usage


def test_stream_preserves_metadata_with_content() -> None:
    resp_meta = {"model": "my-model", "finish_reason": "stop"}
    msg_id = "cmpl-abc123"
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    responses: List[BaseMessage] = [
        AIMessage(
            content="world",
            usage_metadata=usage,
            response_metadata=resp_meta,
            id=msg_id,
        ),
    ]
    llm = _build_streaming_llm(responses)

    chunks = list(llm._stream([HumanMessage(content="hi")]))

    assert len(chunks) == 1
    chunk_msg = chunks[0].message
    assert chunk_msg.content == "world"
    assert chunk_msg.usage_metadata == usage  # type: ignore[union-attr]
    assert chunk_msg.response_metadata == resp_meta
    assert chunk_msg.id == msg_id


def test_stream_drops_empty_chunk_without_usage() -> None:
    responses: List[BaseMessage] = [
        AIMessage(content="Hello"),
        AIMessage(content=""),
    ]
    llm = _build_streaming_llm(responses)

    chunks = list(llm._stream([HumanMessage(content="hi")]))

    assert len(chunks) == 1
    assert chunks[0].message.content == "Hello"


def test_stream_preserves_metadata_through_stop_tokens() -> None:
    resp_meta = {"model": "my-model"}
    msg_id = "cmpl-xyz"
    usage = {
        "input_tokens": 5,
        "output_tokens": 3,
        "total_tokens": 8,
    }
    responses: List[BaseMessage] = [
        AIMessage(
            content="Hello STOP world",
            response_metadata=resp_meta,
            id=msg_id,
            usage_metadata=usage,
        ),
    ]
    llm = _build_streaming_llm(responses)

    chunks = list(llm._stream([HumanMessage(content="hi")], stop=["STOP"]))

    assert len(chunks) == 1
    chunk_msg = chunks[0].message
    assert chunk_msg.content == "Hello "
    assert chunk_msg.usage_metadata == usage  # type: ignore[union-attr]
    assert chunk_msg.response_metadata == resp_meta
    assert chunk_msg.id == msg_id


def test_stream_preserves_metadata_with_list_content() -> None:
    resp_meta = {"model": "my-model"}
    msg_id = "cmpl-list"
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    responses: List[BaseMessage] = [
        AIMessage(
            content=[{"type": "text", "text": "Hello"}, {"type": "text", "text": "!"}],
            usage_metadata=usage,
            response_metadata=resp_meta,
            id=msg_id,
        ),
    ]
    llm = _build_streaming_llm(responses)

    chunks = list(llm._stream([HumanMessage(content="hi")]))

    assert len(chunks) == 1
    chunk_msg = chunks[0].message
    expected_content = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "!"},
    ]
    assert chunk_msg.content == expected_content
    assert chunk_msg.usage_metadata == usage  # type: ignore[union-attr]
    assert chunk_msg.response_metadata == resp_meta
    assert chunk_msg.id == msg_id


def test_stream_passthrough_ai_message_chunk() -> None:
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }
    resp_meta = {"model": "vllm-7b", "finish_reason": "stop"}
    msg_id = "cmpl-passthrough"
    responses: List[BaseMessage] = [
        AIMessageChunk(content="Hello"),
        AIMessageChunk(
            content="",
            usage_metadata=usage,
            response_metadata=resp_meta,
            id=msg_id,
        ),
    ]
    llm = _build_streaming_llm(responses)

    chunks = list(llm._stream([HumanMessage(content="hi")]))

    assert len(chunks) == 2
    assert chunks[0].message.content == "Hello"
    assert isinstance(chunks[0].message, AIMessageChunk)

    final = chunks[1].message
    assert isinstance(final, AIMessageChunk)
    assert final.content == ""
    assert final.usage_metadata == usage  # type: ignore[union-attr]
    assert final.response_metadata == resp_meta
    assert final.id == msg_id
