# type:ignore

"""Test chat model integration."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_aws.chat_models.sagemaker_endpoint import (
    _messages_to_sagemaker,
)


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
