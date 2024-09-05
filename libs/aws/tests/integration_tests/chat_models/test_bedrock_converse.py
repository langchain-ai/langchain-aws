"""Standard LangChain interface tests"""

from typing import Literal, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from langchain_aws import ChatBedrockConverse


class TestBedrockStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @property
    def chat_model_params(self) -> dict:
        return {"model": "anthropic.claude-3-sonnet-20240229-v1:0"}

    @property
    def standard_chat_model_params(self) -> dict:
        return {"temperature": 0, "max_tokens": 100, "stop": []}

    @property
    def supports_image_inputs(self) -> bool:
        return True


class TestBedrockMistralStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @property
    def chat_model_params(self) -> dict:
        return {"model": "mistral.mistral-large-2402-v1:0"}

    @property
    def standard_chat_model_params(self) -> dict:
        return {"temperature": 0, "max_tokens": 100, "stop": []}

    @pytest.mark.xfail(reason="Human messages following AI messages not supported.")
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_list_content(model)


def test_structured_output_snake_case() -> None:
    model = ChatBedrockConverse(
        model="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0
    )

    class ClassifyQuery(BaseModel):
        """Classify a query."""

        query_type: Literal["cat", "dog"] = Field(
            description="Classify a query as related to cats or dogs."
        )

    chat = model.with_structured_output(ClassifyQuery)
    for chunk in chat.stream("How big are cats?"):
        assert isinstance(chunk, ClassifyQuery)


def test_structured_output_streaming() -> None:
    model = ChatBedrockConverse(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0", temperature=0
    )
    query = (
        "What weighs more, a pound of bricks or a pound of feathers? "
        "Limit your response to 20 words."
    )

    # TypedDict
    class AnswerWithJustification(TypedDict):
        """An answer to the user question along with justification for the answer."""

        answer: Annotated[str, ...]
        justification: Annotated[str, ...]

    chat = model.with_structured_output(AnswerWithJustification)
    chunk_count = 0
    for chunk in chat.stream(query):
        chunk_count = chunk_count + 1
        assert isinstance(chunk, dict)
    assert chunk_count > 1

    # Pydantic
    class AnAnswerWithJustification(BaseModel):
        """An answer to the user question along with justification for the answer."""

        answer: Annotated[str, ...]
        justification: Annotated[str, ...]

    chat = model.with_structured_output(AnAnswerWithJustification)
    chunk_count = 0
    for chunk in chat.stream(query):
        chunk_count = chunk_count + 1
        assert isinstance(chunk, AnAnswerWithJustification)
    assert chunk_count > 1


@pytest.mark.skip(reason="Needs guardrails setup to run.")
def test_guardrails() -> None:
    params = {
        "region_name": "us-west-2",
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "temperature": 0,
        "max_tokens": 100,
        "stop": [],
        "guardrail_config": {
            "guardrailIdentifier": "e7esbceow153",
            "guardrailVersion": "1",
            "trace": "enabled",
        },
    }
    chat_model = ChatBedrockConverse(**params)  # type: ignore[arg-type]
    messages = [
        HumanMessage(
            content=[
                "Create a playlist of 2 heavy metal songs.",
                {
                    "guardContent": {
                        "text": {"text": "Only answer with a list of songs."}
                    }
                },
            ]
        )
    ]
    response = chat_model.invoke(messages)

    assert (
        response.content == "Sorry, I can't answer questions about heavy metal music."
    )
    assert response.response_metadata["stopReason"] == "guardrail_intervened"
    assert response.response_metadata["trace"] is not None

    stream = chat_model.stream(messages)
    response = next(stream)
    for chunk in stream:
        response += chunk

    assert (
        response.content[0]["text"]  # type: ignore[index]
        == "Sorry, I can't answer questions about heavy metal music."
    )
    assert response.response_metadata["stopReason"] == "guardrail_intervened"
    assert response.response_metadata["trace"] is not None
