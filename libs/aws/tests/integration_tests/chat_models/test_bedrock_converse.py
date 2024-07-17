"""Standard LangChain interface tests"""

from typing import Literal, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

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
        return {
            "temperature": 0,
            "max_tokens": 100,
            "stop": [],
        }

    @property
    def supports_image_inputs(self) -> bool:
        return True

    def test_structured_output_snake_case(self, model: BaseChatModel) -> None:
        class ClassifyQuery(BaseModel):
            """Classify a query."""

            query_type: Literal["cat", "dog"] = Field(
                description="Classify a query as related to cats or dogs."
            )

        chat = model.with_structured_output(ClassifyQuery)
        for chunk in chat.stream("How big are cats?"):
            assert isinstance(chunk, ClassifyQuery)


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
