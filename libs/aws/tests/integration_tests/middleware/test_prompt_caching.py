from typing import Literal

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware

MODEL_ANTHROPIC = "us.anthropic.claude-sonnet-5"
MODEL_NOVA = "us.amazon.nova-2-lite-v1:0"

# Claude models whose single-checkpoint (``cache_strategy="auto"``) placement we
# validate end-to-end, spanning the family's cache-token thresholds. The minimum
# tokens per cache checkpoint is 2048 for Opus and Sonnet and 4096 for Haiku, so
# the prefix below must clear the largest (Haiku) bound to cache on every model.
CLAUDE_MODELS = [
    "global.anthropic.claude-opus-4-8",
    "global.anthropic.claude-sonnet-4-6",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
]

# ~6.5k tokens: above the largest per-checkpoint minimum (Haiku's 4096) so a
# single cache checkpoint is always created, on every model under test.
HUGE_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers concisely. "
    "You have deep expertise in geography, climate science, demographics, "
    "urban planning, and world history. When answering questions about cities, "
    "provide accurate and up-to-date information. "
    + "You should always strive to give the most helpful response possible. "
    * 450
)

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
    cache_read = details.get("cache_read", 0) or 0
    cache_write = details.get("cache_creation", 0) or (
        details.get("ephemeral_5m_input_tokens", 0)
        + details.get("ephemeral_1h_input_tokens", 0)  # type: ignore[operator]
    )
    return cache_read, cache_write


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


# A fixed instruction that always trails the conversation history, mirroring a
# deterministic LangGraph node such as an intent classifier.
CLASSIFY_INSTRUCTION = (
    "Classify the intent of the conversation above into exactly one of: "
    "balance_inquiry, transfer, complaint, product_question, other. "
    "Respond with only the label."
)


def _history_block(idx: int) -> str:
    return (
        f"[turn {idx}] The customer is asking about banking matter number {idx}, "
        f"including account details, recent transactions, and policy questions."
    )


# A small system prompt paired with large history blocks. This deliberately puts
# the bulk of the reusable tokens in the *history* rather than the system prompt,
# which is what exposes the fixed-instruction cache bug: a large system prompt
# would be cached on its own by the legacy placement and mask the lost history
# reuse. Each block clears the largest per-checkpoint minimum (Haiku's 4096) so
# the first call writes a cache on every model under test.
SMALL_SYSTEM_PROMPT = "You are a banking assistant that classifies customer intent."


def _large_history_block(idx: int) -> str:
    return f"[turn {idx}] Customer conversation excerpt {idx}. " + (
        f"Account, transaction, dispute, and policy details for matter {idx}. " * 380
    )


def _tagged_system(tag: str) -> str:
    """A large system prompt with a stable per-test marker.

    The marker keeps each test's cached prefix distinct so concurrently cached
    conversations do not cross-contaminate cache-read/write measurements, while
    staying deterministic for VCR replay (unlike a random nonce).
    """
    return f"[{tag}] {HUGE_SYSTEM_PROMPT}"


@pytest.mark.vcr
@pytest.mark.parametrize("model_id", CLAUDE_MODELS)
def test_converse_auto_cache_point_request_shape(model_id: str) -> None:
    """Default cache_strategy='auto' places system + end-of-history + last points.

    With a fixed trailing instruction after the conversation history, "auto" adds
    an end-of-history checkpoint (on the penultimate sent message) on top of the
    system and last-message points. The end-of-history point is what keeps the
    system+history prefix cache-readable as the conversation grows, while the
    last-message point covers the common forward-chat and single-message shapes.
    """
    llm = ChatBedrockConverse(model=model_id, region_name="us-west-2", max_tokens=16)
    assert llm.cache_strategy == "auto"

    captured: dict = {}
    original = llm.client.converse

    def _spy(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return original(**kwargs)

    llm.client.converse = _spy  # type: ignore[method-assign]

    messages = [
        SystemMessage(content=HUGE_SYSTEM_PROMPT),
        HumanMessage(content=_history_block(1)),
        AIMessage(content="Noted."),
        HumanMessage(content=CLASSIFY_INSTRUCTION),
    ]
    response = llm.invoke(messages, cache_control={"type": "ephemeral", "ttl": "5m"})
    assert isinstance(response, AIMessage)

    def _count(blocks: list) -> int:
        return sum(1 for b in blocks if isinstance(b, dict) and "cachePoint" in b)

    sent = captured["messages"]
    # system (1) + penultimate/end-of-history (1) + last message (1) = 3 points.
    assert _count(captured.get("system", [])) == 1
    assert _count(sent[-2]["content"]) == 1  # end of history
    assert _count(sent[-1]["content"]) == 1  # trailing instruction


@pytest.mark.vcr
@pytest.mark.parametrize("model_id", CLAUDE_MODELS)
def test_converse_single_cache_point_reuse(model_id: str) -> None:
    """End-of-history checkpoint keeps cache reads growing across turns.

    Reproduces the deterministic-node shape (system + growing history + fixed
    trailing instruction) that previously left only the system prompt cached.
    The first call writes the cache; the second, deeper call must read back more
    than zero and write less, proving the history prefix is reused.
    """
    llm = ChatBedrockConverse(model=model_id, region_name="us-west-2", max_tokens=16)
    cache_control = {"type": "ephemeral", "ttl": "5m"}

    first = llm.invoke(
        [
            SystemMessage(content=HUGE_SYSTEM_PROMPT),
            HumanMessage(content=_history_block(1)),
            AIMessage(content="Noted 1."),
            HumanMessage(content=CLASSIFY_INSTRUCTION),
        ],
        cache_control=cache_control,
    )
    assert isinstance(first, AIMessage)
    _, write = _get_cache_stats(first)
    assert write > 0, f"Expected a cache write on the first call, got {write}"

    second = llm.invoke(
        [
            SystemMessage(content=HUGE_SYSTEM_PROMPT),
            HumanMessage(content=_history_block(1)),
            AIMessage(content="Noted 1."),
            HumanMessage(content=_history_block(2)),
            AIMessage(content="Noted 2."),
            HumanMessage(content=CLASSIFY_INSTRUCTION),
        ],
        cache_control=cache_control,
    )
    assert isinstance(second, AIMessage)
    read, _ = _get_cache_stats(second)
    assert read > 0, f"Expected a cache read on the deeper call, got {read}"


@pytest.mark.vcr
@pytest.mark.parametrize("model_id", CLAUDE_MODELS)
def test_converse_single_cache_point_standard_growth(model_id: str) -> None:
    """Standard chat shape (new question last) also reuses the cached prefix.

    This is the regression guard for ordinary long-prompt caching: a large
    system prompt plus a growing back-and-forth where each turn ends with a new
    user question. The end-of-history checkpoint plus Bedrock's simplified cache
    management must reuse the ``system + prior turns`` prefix, so the deeper
    second call reads back more than zero.
    """
    llm = ChatBedrockConverse(model=model_id, region_name="us-west-2", max_tokens=16)
    cache_control = {"type": "ephemeral", "ttl": "5m"}
    system = _tagged_system(f"standard-growth-{model_id}")

    first = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content="Paris."),
            HumanMessage(content="And its population?"),
        ],
        cache_control=cache_control,
    )
    assert isinstance(first, AIMessage)
    _, write = _get_cache_stats(first)
    assert write > 0, f"Expected a cache write on the first call, got {write}"

    second = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content="What is the capital of France?"),
            AIMessage(content="Paris."),
            HumanMessage(content="And its population?"),
            AIMessage(content="Roughly 2.1 million in the city proper."),
            HumanMessage(content="What about the metro area?"),
        ],
        cache_control=cache_control,
    )
    assert isinstance(second, AIMessage)
    read, _ = _get_cache_stats(second)
    assert read > 0, f"Expected a cache read on the deeper call, got {read}"


@pytest.mark.vcr
@pytest.mark.parametrize("model_id", CLAUDE_MODELS)
def test_converse_bind_cache_control_then_bind_tools_reuse(model_id: str) -> None:
    """``bind(cache_control=...).bind_tools(...)`` still caches end-to-end.

    Guards the ``bind_tools`` ordering fix against real Bedrock: binding
    ``cache_control`` first and tools second must not drop the cache settings,
    so the second, deeper call reads back the cached prefix.
    """
    llm = ChatBedrockConverse(model=model_id, region_name="us-west-2", max_tokens=16)
    bound = llm.bind(cache_control={"type": "ephemeral", "ttl": "5m"}).bind_tools(
        [get_weather]
    )

    first = bound.invoke(
        [
            SystemMessage(content=HUGE_SYSTEM_PROMPT),
            HumanMessage(content="Remember the context above."),
            AIMessage(content="Understood."),
            HumanMessage(content="Say hello."),
        ],
    )
    assert isinstance(first, AIMessage)
    _, write = _get_cache_stats(first)
    assert write > 0, f"Expected a cache write on the first call, got {write}"

    second = bound.invoke(
        [
            SystemMessage(content=HUGE_SYSTEM_PROMPT),
            HumanMessage(content="Remember the context above."),
            AIMessage(content="Understood."),
            HumanMessage(content="Say hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Say goodbye."),
        ],
    )
    assert isinstance(second, AIMessage)
    read, _ = _get_cache_stats(second)
    assert read > 0, f"Expected a cache read on the deeper call, got {read}"


def _fixed_instruction_turns(depth: int, tag: str) -> list:
    """System + ``depth`` history exchanges + a constant trailing instruction."""
    messages: list = [SystemMessage(content=f"[{tag}] {SMALL_SYSTEM_PROMPT}")]
    for i in range(1, depth + 1):
        messages.append(HumanMessage(content=_large_history_block(i)))
        messages.append(AIMessage(content=f"Noted {i}."))
    messages.append(HumanMessage(content=CLASSIFY_INSTRUCTION))
    return messages


@pytest.mark.vcr
@pytest.mark.parametrize("model_id", CLAUDE_MODELS)
def test_converse_legacy_multi_flat_read_on_fixed_instruction(model_id: str) -> None:
    """The OLD default (``cache_strategy='multi'``) loses reuse here; ``auto`` fixes it.

    This is the empirical counterpart to the placement unit tests. Both calls use
    a small system prompt so the reusable bulk lives in the history, then a fixed
    instruction always trails that history (the intent-classifier shape).

    - ``multi`` (legacy last-message placement): the checkpoint lands on the
      trailing instruction, so the deeper second call reads back ~zero of the
      history -- the cache boundary is poisoned.
    - ``auto`` (end-of-history placement): the same deeper call reads back the
      history prefix, so its cache read is strictly greater than ``multi``'s.

    Measured live, ``multi``'s second-call read is 0 while ``auto``'s is several
    thousand tokens across Opus, Sonnet, and Haiku.
    """
    cache_control = {"type": "ephemeral", "ttl": "5m"}

    def _second_call_read(strategy: str) -> int:
        llm = ChatBedrockConverse(
            model=model_id,
            region_name="us-west-2",
            max_tokens=8,
            cache_strategy=strategy,  # type: ignore[arg-type]
        )
        tag = f"legacy-contrast-{strategy}-{model_id}"
        first = llm.invoke(
            _fixed_instruction_turns(1, tag), cache_control=cache_control
        )
        assert isinstance(first, AIMessage)
        _, write = _get_cache_stats(first)
        assert write > 0, f"Expected a cache write on the first call, got {write}"

        second = llm.invoke(
            _fixed_instruction_turns(2, tag), cache_control=cache_control
        )
        assert isinstance(second, AIMessage)
        read, _ = _get_cache_stats(second)
        return read

    legacy_read = _second_call_read("multi")
    auto_read = _second_call_read("auto")

    assert legacy_read == 0, (
        f"Legacy multi placement should not reuse the history prefix on the "
        f"fixed-instruction shape, but read {legacy_read}"
    )
    assert auto_read > legacy_read, (
        f"Default 'auto' placement should read back more than legacy 'multi' "
        f"(auto={auto_read}, multi={legacy_read})"
    )
