"""Standard LangChain interface tests"""

import base64
from typing import Literal, Type, Optional

import httpx
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
)
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from langchain_aws import ChatBedrockConverse


class TestBedrockStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @property
    def chat_model_params(self) -> dict:
        return {"model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"}

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

    @property
    def has_tool_choice(self) -> bool:
        return False

    @pytest.mark.xfail(reason="Human messages following AI messages not supported.")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)


class TestBedrockNovaStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatBedrockConverse

    @property
    def chat_model_params(self) -> dict:
        return {"model": "us.amazon.nova-pro-v1:0"}

    @property
    def standard_chat_model_params(self) -> dict:
        return {"max_tokens": 300, "stop": []}

    @pytest.mark.xfail(reason="Tool choice 'Any' not supported.")
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_structured_few_shot_examples(model, my_adder_tool)

    @pytest.mark.xfail(reason="Human messages following AI messages not supported.")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)


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

    @property
    def has_tool_choice(self) -> bool:
        return False

    @pytest.mark.xfail(reason="Cohere models don't support tool_choice.")
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        pass

    @pytest.mark.xfail(reason="Cohere models don't support tool_choice.")
    def test_unicode_tool_call_integration(
        self,
        model: BaseChatModel,
        *,
        tool_choice: Optional[str] = None,
        force_tool_call: bool = False,
    ) -> None:
        pass

    @pytest.mark.xfail(reason="Generates invalid tool call.")
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

    @property
    def has_tool_choice(self) -> bool:
        return False

    @pytest.mark.xfail(reason="Meta models don't support tool_choice.")
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        pass

    @pytest.mark.xfail(reason="Meta models don't support tool_choice.")
    def test_unicode_tool_call_integration(
        self,
        model: BaseChatModel,
        *,
        tool_choice: Optional[str] = None,
        force_tool_call: bool = False,
    ) -> None:
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
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_list_content(model, my_adder_tool)


class ClassifyQuery(BaseModel):
    """Classify a query."""

    query_type: Literal["cat", "dog"] = Field(
        description="Classify a query as related to cats or dogs."
    )


def test_structured_output_snake_case() -> None:
    model = ChatBedrockConverse(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0
    )

    chat = model.with_structured_output(ClassifyQuery)
    for chunk in chat.stream("How big are cats?"):
        assert isinstance(chunk, ClassifyQuery)


def test_tool_calling_snake_case() -> None:
    model = ChatBedrockConverse(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")

    def classify_query(query_type: Literal["cat", "dog"]) -> None:
        pass

    chat = model.bind_tools([classify_query], tool_choice="any")
    response = chat.invoke("How big are cats?")
    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "classify_query"
    assert tool_call["args"] == {"query_type": "cat"}

    full = None
    for chunk in chat.stream("How big are cats?"):
        full = chunk if full is None else full + chunk  # type: ignore[assignment]
    assert isinstance(full, AIMessageChunk)
    assert len(full.tool_calls) == 1
    tool_call = full.tool_calls[0]
    assert tool_call["name"] == "classify_query"
    assert tool_call["args"] == {"query_type": "cat"}

    # Also test for response metadata, though this is not relevant to tool-calling
    invoke_metadata = response.response_metadata
    stream_metadata = full.response_metadata
    for result in [invoke_metadata, stream_metadata]:
        for expected_key in ["RequestId", "HTTPStatusCode", "HTTPHeaders"]:
            assert result["ResponseMetadata"][expected_key]
        assert isinstance(result["ResponseMetadata"]["RetryAttempts"], int)


def test_tool_calling_camel_case() -> None:
    model = ChatBedrockConverse(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")

    def classifyQuery(queryType: Literal["cat", "dog"]) -> None:
        pass

    chat = model.bind_tools([classifyQuery], tool_choice="any")
    response = chat.invoke("How big are cats?")
    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "classifyQuery"
    assert tool_call["args"] == {"queryType": "cat"}

    full = None
    for chunk in chat.stream("How big are cats?"):
        full = chunk if full is None else full + chunk  # type: ignore[assignment]
    assert isinstance(full, AIMessageChunk)
    assert len(full.tool_calls) == 1
    tool_call = full.tool_calls[0]
    assert tool_call["name"] == "classifyQuery"
    assert tool_call["args"] == {"queryType": "cat"}
    assert full.tool_calls[0]["args"] == response.tool_calls[0]["args"]


def test_structured_output_streaming() -> None:
    model = ChatBedrockConverse(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0
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


def test_tool_use_with_cache_point() -> None:
    """Test toolUse with cachepoint to verify cache metrics are being reported.

    This test creates tools with a length exceeding 1024 tokens to ensure
    caching is triggered, and verifies the response metrics indicate cache
    activity.
    """
    # Define a large number of tools to exceed 1024 tokens
    tool_classes = []

    # Each tool is simple but we'll define many of them
    for i in range(1, 20):
        # Creating a unique class for each tool
        tool_class_name = f"CalculateTool{i}"

        # Define the class using a closure to properly scope the fields
        def create_tool_class(i: int) -> Type[BaseModel]:
            class ToolClass(BaseModel):
                number1: float = Field(description=f"First number for calculation {i}")
                number2: float = Field(description=f"Second number for calculation {i}")
                operation: Literal["add", "subtract", "multiply", "divide"] = Field(
                    description=f"Operation {i} to perform on the numbers"
                )

            ToolClass.__doc__ = f"A tool to calculate the {i}th mathematical operation"
            return ToolClass

        tool_class = create_tool_class(i)
        tool_class.__name__ = tool_class_name
        tool_classes.append(tool_class)

    # Create the model instance
    model = ChatBedrockConverse(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0", temperature=0
    )

    # Create cache point configuration
    cache_point = ChatBedrockConverse.create_cache_point()

    # Bind tools with cache point
    chat = model.bind_tools(tool_classes + [cache_point], tool_choice="any")

    # Invocation
    response = chat.invoke("What's 5 + 3?")
    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) == 1

    # Verify the response has cache metrics
    assert response.usage_metadata is not None
    input_token_details = response.usage_metadata["input_token_details"]
    cache_read_input_tokens = input_token_details["cache_read"]
    cache_write_input_tokens = input_token_details["cache_creation"]
    assert cache_read_input_tokens + cache_write_input_tokens != 0


@pytest.mark.skip(reason="Needs guardrails setup to run.")
def test_guardrails() -> None:
    params = {
        "region_name": "us-west-2",
        "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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


def test_structured_output_tool_choice_not_supported() -> None:
    llm = ChatBedrockConverse(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    with pytest.warns(None) as record:  # type: ignore[call-overload]
        structured_llm = llm.with_structured_output(ClassifyQuery)
        response = structured_llm.invoke("How big are cats?")
    assert len(record) == 0
    assert isinstance(response, ClassifyQuery)

    # Unsupported params
    llm = ChatBedrockConverse(
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        max_tokens=5000,
        additional_model_request_fields={
            "thinking": {"type": "enabled", "budget_tokens": 2000}
        },
    )
    with pytest.warns(match="structured output"):
        structured_llm = llm.with_structured_output(ClassifyQuery)
    response = structured_llm.invoke("How big are cats?")
    assert isinstance(response, ClassifyQuery)

    with pytest.raises(OutputParserException):
        structured_llm.invoke("Hello!")


def test_structured_output_thinking_force_tool_use() -> None:
    # Structured output currently relies on forced tool use, which is not supported
    # when `thinking` is enabled for Claude 3.7. When this test fails, it means that
    # the feature is supported and the workarounds in `with_structured_output` should
    # be removed.

    # Instantiate as convenience for getting client
    llm = ChatBedrockConverse(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    messages = [
        {
            "role": "user",
            "content": [{"text": "Generate a username for Sally with green hair"}],
        }
    ]
    params = {
        "modelId": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "inferenceConfig": {"maxTokens": 5000},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "ClassifyQuery",
                        "description": "Classify a query.",
                        "inputSchema": {
                            "json": {
                                "properties": {
                                    "queryType": {
                                        "description": (
                                            "Classify a query as related to cats or "
                                            "dogs."
                                        ),
                                        "enum": ["cat", "dog"],
                                        "type": "string",
                                    }
                                },
                                "required": ["query_type"],
                                "type": "object",
                            }
                        },
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "ClassifyQuery"}},
        },
        "additionalModelRequestFields": {
            "thinking": {"type": "enabled", "budget_tokens": 2000}
        },
    }
    with pytest.raises(llm.client.exceptions.ValidationException):
        llm.client.converse(messages=messages, **params)


@pytest.mark.vcr
def test_thinking() -> None:
    llm = ChatBedrockConverse(
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens=4096,
        additional_model_request_fields={
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    )

    input_message = {"role": "user", "content": "What is 3^3?"}
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)

    assert [block["type"] for block in full.content] == ["reasoning_content", "text"]  # type: ignore[index,union-attr]
    assert "text" in full.content[0]["reasoning_content"]  # type: ignore[index,union-attr]
    assert "signature" in full.content[0]["reasoning_content"]  # type: ignore[index,union-attr]

    next_message = {"role": "user", "content": "Thanks!"}
    response = llm.invoke([input_message, full, next_message])

    assert [block["type"] for block in response.content] == ["reasoning_content", "text"]  # type: ignore[index,union-attr]
    assert "text" in response.content[0]["reasoning_content"]  # type: ignore[index,union-attr]
    assert "signature" in response.content[0]["reasoning_content"]  # type: ignore[index,union-attr]


@pytest.mark.vcr
def test_citations() -> None:

    llm = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-20250514-v1:0")

    input_message = {
        "role": "user",
        "content": [
            {
                "type": "document",
                "document": {
                    "format": "txt",
                    "name": "company_policy",
                    "source": {
                        "text": (
                            "Company leave policy: Employees get 20 days annual leave. "
                            "Consult with your manager for details."
                        )
                    },
                    "context": "HR Policy Manual Section 3.2",
                    "citations": {
                        "enabled": True
                    }
                }
            },
            {"type": "text", "text": "How many days of annual leave do employees get?"},
        ]
    }

    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream([input_message]):
        assert isinstance(chunk, AIMessageChunk)
        full = chunk if full is None else full + chunk
    assert isinstance(full, AIMessageChunk)
    assert any(block.get("citations") for block in full.content)  # type: ignore[union-attr]

    next_message = {"role": "user", "content": "Who should they consult with?"}
    response = llm.invoke([input_message, full, next_message])
    assert any(block.get("citations") for block in response.content)  # type: ignore[union-attr]


def test_bedrock_pdf_inputs() -> None:
    model = ChatBedrockConverse(model="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    pdf_data = base64.b64encode(httpx.get(url).content).decode("utf-8")

    message = HumanMessage(
        [
            {"type": "text", "text": "Summarize this document:"},
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": "application/pdf",
                "data": pdf_data,
                "name": "my-pdf",  # Converse requires a filename
            },
        ]
    )
    _ = model.invoke([message])

    # Test OpenAI Chat Completions format
    message = HumanMessage(
        [
            {
                "type": "text",
                "text": "Summarize this document:",
            },
            {
                "type": "file",
                "file": {
                    "filename": "my-pdf",
                    "file_data": f"data:application/pdf;base64,{pdf_data}",
                },
            },
        ]
    )
    _ = model.invoke([message])
