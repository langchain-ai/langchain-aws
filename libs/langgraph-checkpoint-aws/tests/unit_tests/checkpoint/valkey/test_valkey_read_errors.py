"""A failed read must not look like a thread with no checkpoint.

LangGraph treats ``get_tuple() -> None`` as "this thread has never run": it starts
from an empty checkpoint and saves the result as the thread's newest checkpoint.
If a connection error is reported as ``None``, one failed read silently resets
the conversation, and the reset is still there after Valkey recovers.
"""

import operator
from typing import Annotated, TypedDict
from unittest.mock import patch

import pytest

pytest.importorskip("valkey")
pytest.importorskip("orjson")
pytest.importorskip("fakeredis")

import fakeredis
from langgraph.graph import END, START, StateGraph
from valkey.exceptions import ConnectionError as ValkeyConnectionError

from langgraph_checkpoint_aws import AsyncValkeySaver, ValkeySaver


class State(TypedDict):
    messages: Annotated[list, operator.add]


def _reply(state: State) -> dict:
    return {"messages": [f"reply to {state['messages'][-1]}"]}


def _graph(saver):
    builder = StateGraph(State)
    builder.add_node("reply", _reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=saver)


CONFIG = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
BLIP = ValkeyConnectionError("Connection reset by peer")


@pytest.fixture
def client():
    return fakeredis.FakeStrictRedis(decode_responses=False)


class TestValkeySaverReadErrors:
    def test_get_tuple_raises_when_latest_lookup_fails(self, client):
        saver = ValkeySaver(client=client)
        with patch.object(client, "lrange", side_effect=BLIP):
            with pytest.raises(ValkeyConnectionError):
                saver.get_tuple(CONFIG)

    def test_get_tuple_raises_when_checkpoint_fetch_fails(self, client):
        saver = ValkeySaver(client=client)
        config = {"configurable": {**CONFIG["configurable"], "checkpoint_id": "c1"}}
        with patch.object(client, "pipeline", side_effect=BLIP):
            with pytest.raises(ValkeyConnectionError):
                saver.get_tuple(config)

    def test_list_raises(self, client):
        saver = ValkeySaver(client=client)
        with patch.object(client, "lrange", side_effect=BLIP):
            with pytest.raises(ValkeyConnectionError):
                list(saver.list(CONFIG))

    def test_thread_with_no_checkpoint_is_still_none(self, client):
        assert ValkeySaver(client=client).get_tuple(CONFIG) is None
        assert list(ValkeySaver(client=client).list(CONFIG)) == []

    def test_failed_read_does_not_reset_the_thread(self, client):
        app = _graph(ValkeySaver(client=client))
        app.invoke({"messages": ["turn 1"]}, CONFIG)
        app.invoke({"messages": ["turn 2"]}, CONFIG)

        with patch.object(client, "lrange", side_effect=BLIP):
            with pytest.raises(ValkeyConnectionError):
                app.invoke({"messages": ["turn 3"]}, CONFIG)

        # Valkey is back: the history from before the failure is intact.
        assert app.get_state(CONFIG).values["messages"] == [
            "turn 1",
            "reply to turn 1",
            "turn 2",
            "reply to turn 2",
        ]


class TestAsyncValkeySaverReadErrors:
    @pytest.fixture
    def async_client(self):
        return fakeredis.FakeAsyncRedis(decode_responses=False)

    async def test_failed_read_does_not_reset_the_thread(self, async_client):
        app = _graph(AsyncValkeySaver(client=async_client))
        await app.ainvoke({"messages": ["turn 1"]}, CONFIG)
        await app.ainvoke({"messages": ["turn 2"]}, CONFIG)

        with patch.object(async_client, "lrange", side_effect=BLIP):
            with pytest.raises(ValkeyConnectionError):
                await app.ainvoke({"messages": ["turn 3"]}, CONFIG)

        state = await app.aget_state(CONFIG)
        assert state.values["messages"] == [
            "turn 1",
            "reply to turn 1",
            "turn 2",
            "reply to turn 2",
        ]

    async def test_thread_with_no_checkpoint_is_still_none(self, async_client):
        assert await AsyncValkeySaver(client=async_client).aget_tuple(CONFIG) is None
