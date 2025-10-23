import datetime
from typing import Literal

import pytest
from langchain.agents import create_agent
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langgraph.checkpoint.base import Checkpoint, uuid6

from langgraph_checkpoint_aws.async_saver import AsyncBedrockSessionSaver
from langgraph_checkpoint_aws.models import DeleteSessionRequest, EndSessionRequest


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


class TestAsyncBedrockMemorySaver:
    @pytest.fixture
    def tools(self):
        # Setup tools
        return [get_weather]

    @pytest.fixture
    def model(self):
        # Setup model
        return ChatBedrock(
            model="anthropic.claude-3-sonnet-20240229-v1:0", region="us-west-2"
        )

    @pytest.fixture
    def session_saver(self):
        # Return the instantiated object
        return AsyncBedrockSessionSaver(region_name="us-west-2")

    @pytest.fixture
    def boto_session_client(self, session_saver):
        # Return the async client wrapper
        return session_saver.session_client

    @pytest.mark.asyncio
    async def test_weather_tool_responses(self):
        # Test weather tool directly
        assert get_weather.invoke("sf") == "It's always sunny in sf"
        assert get_weather.invoke("nyc") == "It might be cloudy in nyc"

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_retrieve(
        self, boto_session_client, session_saver
    ):
        # Create session
        session_response = await boto_session_client.create_session()
        session_id = session_response.session_id
        assert session_id, "Session ID should not be empty"

        config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}
        checkpoint = Checkpoint(
            v=1,
            id=str(uuid6(clock_seq=-2)),
            ts=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            channel_values={"key": "value"},
            channel_versions={},
            versions_seen={},
            pending_sends=[],
        )
        checkpoint_metadata = {"source": "input", "step": 1, "writes": {"key": "value"}}

        try:
            saved_config = await session_saver.aput(
                config,
                checkpoint,
                checkpoint_metadata,
                {},
            )
            assert saved_config == {
                "configurable": {
                    "checkpoint_id": checkpoint["id"],
                    "checkpoint_ns": "",
                    "thread_id": session_id,
                }
            }

            checkpoint_tuple = await session_saver.aget_tuple(saved_config)
            assert checkpoint_tuple.checkpoint == checkpoint
            assert checkpoint_tuple.metadata == checkpoint_metadata
            assert checkpoint_tuple.config == saved_config

        finally:
            # Create proper request objects
            await boto_session_client.end_session(
                EndSessionRequest(session_identifier=session_id)
            )
            await boto_session_client.delete_session(
                DeleteSessionRequest(session_identifier=session_id)
            )

    @pytest.mark.asyncio
    async def test_weather_query_and_checkpointing(
        self, boto_session_client, tools, model, session_saver
    ):
        # Create session
        session_response = await boto_session_client.create_session()
        session_id = session_response.session_id
        assert session_id, "Session ID should not be empty"
        try:
            # Create graph and config
            graph = create_agent(model, tools=tools, checkpointer=session_saver)
            config = {"configurable": {"thread_id": session_id}}

            # Test weather query
            response = await graph.ainvoke(
                {"messages": [("human", "what's the weather in sf")]}, config
            )
            assert response, "Response should not be empty"

            # Test checkpoint retrieval
            checkpoint = await session_saver.aget(config)
            assert checkpoint, "Checkpoint should not be empty"

            # Test checkpoint listing
            checkpoint_tuples = [tup async for tup in session_saver.alist(config)]
            assert checkpoint_tuples, "Checkpoint tuples should not be empty"
            assert isinstance(checkpoint_tuples, list), (
                "Checkpoint tuples should be a list"
            )
        finally:
            # Create proper request objects
            await boto_session_client.end_session(
                EndSessionRequest(session_identifier=session_id)
            )
            await boto_session_client.delete_session(
                DeleteSessionRequest(session_identifier=session_id)
            )
