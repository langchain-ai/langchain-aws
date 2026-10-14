from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_aws import ChatAnthropicBedrock

MODEL_NAME = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_invoke() -> None:
    model = ChatAnthropicBedrock(model=MODEL_NAME)  # type: ignore[call-arg]
    result = model.invoke("Hello")
    assert result


def test_stream_usage_metadata() -> None:
    model = ChatAnthropicBedrock(model=MODEL_NAME, streaming=True)  # type: ignore[call-arg]
    result = model.invoke("Hello")
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] > 0


def test_system_tool_addition() -> None:
    model = ChatAnthropicBedrock(model="global.anthropic.claude-sonnet-5-5")  # type: ignore[call-arg]
    response = model.invoke(
        [
            HumanMessage("What time is it?"),
            SystemMessage(
                [
                    {
                        "type": "tool_addition",
                        "tool": {
                            "type": "tool_definition",
                            "definition": {
                                "name": "get_time",
                                "description": "Get the current time.",
                                "input_schema": {
                                    "type": "object",
                                    "properties": {},
                                },
                            },
                        },
                    }
                ]
            ),
        ]
    )
    assert isinstance(response, AIMessage)
    assert response.tool_calls[0]["name"] == "get_time"
