import datetime
from typing import Literal

import pytest
from langchain.agents import create_agent
from langchain_aws import ChatBedrock
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.base.id import uuid6

from langgraph_checkpoint_aws.saver import BedrockSessionSaver


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


class TestBedrockMemorySaver:
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
        # Setup session saver
        return BedrockSessionSaver(region_name="us-west-2")

    @pytest.fixture
    def boto_session_client(self, session_saver):
        return session_saver.session_client.client

    def test_weather_tool_responses(self):
        # Test weather tool directly
        assert get_weather.invoke("sf") == "It's always sunny in sf"
        assert get_weather.invoke("nyc") == "It might be cloudy in nyc"

    def test_checkpoint_save_and_retrieve(self, boto_session_client, session_saver):
        # Create session
        session_id = boto_session_client.create_session()["sessionId"]
        assert session_id, "Session ID should not be empty"

        config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}
        checkpoint = {
            "v": 1,
            "id": str(uuid6(clock_seq=-2)),
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "channel_values": {"key": "value"},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        checkpoint_metadata = {"source": "input", "step": 1, "writes": {"key": "value"}}

        try:
            saved_config = session_saver.put(
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

            checkpoint_tuple = session_saver.get_tuple(saved_config)
            assert checkpoint_tuple.checkpoint == checkpoint
            assert checkpoint_tuple.metadata == checkpoint_metadata
            assert checkpoint_tuple.config == saved_config

        finally:
            boto_session_client.end_session(sessionIdentifier=session_id)
            boto_session_client.delete_session(sessionIdentifier=session_id)

    def test_weather_query_and_checkpointing(
        self, boto_session_client, tools, model, session_saver
    ):
        # Create session
        session_id = boto_session_client.create_session()["sessionId"]
        assert session_id, "Session ID should not be empty"
        try:
            # Create graph and config
            graph = create_agent(model, tools=tools, checkpointer=session_saver)
            config = {"configurable": {"thread_id": session_id}}
            # Test weather query
            response = graph.invoke(
                {"messages": [("human", "what's the weather in sf")]},
                RunnableConfig(configurable=config["configurable"]),
            )
            assert response, "Response should not be empty"

            # Test checkpoint retrieval
            checkpoint = session_saver.get(config)
            assert checkpoint, "Checkpoint should not be empty"

            # Test checkpoint listing
            checkpoint_tuples = list(session_saver.list(config))
            assert checkpoint_tuples, "Checkpoint tuples should not be empty"
            assert isinstance(checkpoint_tuples, list), (
                "Checkpoint tuples should be a list"
            )
        finally:
            boto_session_client.end_session(sessionIdentifier=session_id)
            boto_session_client.delete_session(sessionIdentifier=session_id)
