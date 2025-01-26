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


class TestBedrockNovaStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @property
    def chat_model_params(self) -> dict:
        return {"model": "us.amazon.nova-pro-v1:0", "region_name": "us-east-1"}

    @property
    def standard_chat_model_params(self) -> dict:
        return {"max_tokens": 300, "stop": []}

    @property
    def tool_choice_value(self) -> str:
        return "auto"

    @pytest.mark.xfail(reason="Tool choice 'Any' not supported.")
    def test_structured_few_shot_examples(self, model: BaseChatModel) -> None:
        super().test_structured_few_shot_examples(model)

    @pytest.mark.xfail(reason="Human messages following AI messages not supported.")
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_list_content(model)


class TestBedrockCohereStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @property
    def chat_model_params(self) -> dict:
        return {"model": "cohere.command-r-plus-v1:0"}

    @property
    def standard_chat_model_params(self) -> dict:
        return {"temperature": 0, "max_tokens": 100, "stop": []}

    @pytest.mark.xfail(reason="Cohere models don't support tool_choice.")
    def test_structured_few_shot_examples(self, model: BaseChatModel) -> None:
        pass

    @pytest.mark.xfail(reason="Cohere models don't support tool_choice.")
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        pass


class TestBedrockMetaStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @property
    def chat_model_params(self) -> dict:
        return {"model": "us.meta.llama3-2-90b-instruct-v1:0"}

    @property
    def standard_chat_model_params(self) -> dict:
        return {"temperature": 0.1, "max_tokens": 100, "stop": []}

    @pytest.mark.xfail(reason="Meta models don't support tool_choice.")
    def test_structured_few_shot_examples(self, model: BaseChatModel) -> None:
        pass

    # TODO: This needs investigation, if this is a bug with Bedrock or Llama models,
    # but this test consistently seem to return single quoted input values {input: '3'}
    # instead of {input: 3} failing the test. Upon checking with tools with non-numeric
    # inputs, tool calling seems to work as expected with Bedrock and Llama models.
    # Same problem with tool_calling_async, below.
    @pytest.mark.xfail(
        reason="Bedrock Meta models tend to return string values for integer inputs ."
    )
    def test_tool_calling(self, model: BaseChatModel) -> None:
        super().test_tool_calling(model)

    @pytest.mark.xfail(
        reason="Bedrock Meta models tend to return string values for integer inputs ."
    )
    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        await super().test_tool_calling_async(model)

    @pytest.mark.xfail(reason="Meta models don't support tool_choice.")
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        pass

    @pytest.mark.xfail(
        reason="Human messages following AI messages not supported by Bedrock."
    )
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
        model="anthropic.claude-3-sonnet-20240229-v1:0", temperature=0
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


def test_chat_bedrock_converse_with_region_name() -> None:
    model = ChatBedrockConverse(
        model="us.amazon.nova-pro-v1:0", region_name="us-east-1", temperature=0
    )
    response = model.invoke("Create a list of 3 pop songs")
    assert isinstance(response, BaseModel)
    assert response.content is not None


def test_chat_bedrock_converse_with_aws_region_env() -> None:
    import os

    os.environ["AWS_REGION"] = "us-east-1"
    model = ChatBedrockConverse(model="us.amazon.nova-pro-v1:0", temperature=0)
    response = model.invoke("Create a list of 3 pop songs")
    assert isinstance(response, BaseModel)
    assert response.content is not None
