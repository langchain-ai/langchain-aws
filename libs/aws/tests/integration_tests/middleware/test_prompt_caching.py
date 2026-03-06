from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware

MODEL_ANTHROPIC = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
MODEL_NOVA = "us.amazon.nova-2-lite-v1:0"

# Just over 1024 tokens to exceed the Claude/Nova min tokens per cache checkpoint
# See: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
LONG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers concisely. "
    "You have deep expertise in geography, climate science, demographics, "
    "urban planning, and world history. When answering questions about cities, "
    "provide accurate and up-to-date information. "
    + "You should always strive to give the most helpful response possible. "
    * 85
)


@tool
def get_weather(city: str) -> str:
    """Simple tool for cache tests"""
    return f"The weather in {city} is sunny and 72F."


def _get_cache_stats(response: AIMessage) -> tuple[int, int]:
    um = response.usage_metadata
    if not um:
        return 0, 0
    details = um.get("input_token_details")
    if not details:
        return 0, 0
    cache_read = details.get("cache_read", 0)
    cache_write = details.get("cache_creation", 0)
    return cache_read or 0, cache_write or 0


def _make_many_tools() -> list[type[BaseModel]]:
    tools: list[type[BaseModel]] = []
    for i in range(1, 20):

        def _make(idx: int) -> type[BaseModel]:
            class T(BaseModel):
                number1: float = Field(
                    description=f"First number for calculation {idx}"
                )
                number2: float = Field(
                    description=f"Second number for calculation {idx}"
                )
                operation: Literal["add", "subtract", "multiply", "divide"] = Field(
                    description=f"Operation {idx} to perform"
                )

            T.__doc__ = f"Calculate the {idx}th math operation"
            return T

        cls = _make(i)
        cls.__name__ = f"CalculateTool{i}"
        tools.append(cls)
    return tools


def test_middleware_converse_anthropic_system_prompt() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )


def test_middleware_converse_anthropic_tools() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=_make_many_tools(),
        middleware=[middleware],
    )

    response = agent.invoke({"messages": [HumanMessage(content="What is 5 + 3?")]})
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.usage_metadata is not None
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity from tools, got read={read} write={write}"
    )


def test_middleware_converse_anthropic_system_prompt_and_tools() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Miami?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0, f"Expected cache read on final turn, got read={read} write={write}"


def test_middleware_converse_anthropic_extended_ttl() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="1h")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Miami?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity with 1h TTL, got read={read} write={write}"
    )


def test_middleware_converse_anthropic_min_messages_skips() -> None:
    llm = ChatBedrockConverse(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m", min_messages_to_cache=100)
    agent = create_agent(
        llm,
        tools=[],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke({"messages": [HumanMessage(content="Hello!")]})
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    _, write = _get_cache_stats(last_msg)
    assert write == 0, (
        f"Expected no cache write with high min_messages_to_cache, got {write}"
    )


def test_middleware_converse_nova_system_prompt() -> None:
    llm = ChatBedrockConverse(model=MODEL_NOVA)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )


def test_middleware_converse_nova_system_prompt_and_tools() -> None:
    # Nova doesn't support tool caching, making sure this case doesn't crash.
    llm = ChatBedrockConverse(model=MODEL_NOVA)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Miami?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )


def test_middleware_invoke_anthropic_system_prompt() -> None:
    llm = ChatBedrock(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the capital of France?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )


def test_middleware_invoke_anthropic_system_prompt_and_tools() -> None:
    llm = ChatBedrock(model=MODEL_ANTHROPIC)
    middleware = BedrockPromptCachingMiddleware(ttl="5m")
    agent = create_agent(
        llm,
        tools=[get_weather],
        middleware=[middleware],
        system_prompt=LONG_SYSTEM_PROMPT,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Miami?")]}
    )
    last_msg = response["messages"][-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content
    read, write = _get_cache_stats(last_msg)
    assert read > 0 or write > 0, (
        f"Expected cache activity, got read={read} write={write}"
    )
