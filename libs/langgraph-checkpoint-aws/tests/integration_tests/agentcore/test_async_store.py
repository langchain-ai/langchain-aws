import asyncio
import os
import random
import string
import time
import uuid
from collections.abc import Iterator

import boto3
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.base import GetOp, PutOp, SearchOp

from langgraph_checkpoint_aws.store.agentcore.store import AgentCoreMemoryStore

AWS_REGION = os.getenv("AWS_REGION", "us-west-2")


def _random_suffix(k: int = 6) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(random.choices(chars, k=k))


class TestAgentCoreMemoryStoreAsyncIntegration:
    @pytest.fixture
    def memory_id(self) -> str:
        memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
        if not memory_id:
            pytest.skip("AGENTCORE_MEMORY_ID environment variable not set")
        return memory_id

    @pytest.fixture
    def store(self, memory_id: str) -> AgentCoreMemoryStore:
        return AgentCoreMemoryStore(memory_id=memory_id, region_name=AWS_REGION)

    @pytest.fixture
    def actor_id(self) -> str:
        return "actor" + _random_suffix()

    @pytest.fixture
    def session_id(self, memory_id: str, actor_id: str) -> Iterator[str]:
        session_id = "session" + _random_suffix()
        yield session_id

        client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
        params = {
            "memoryId": memory_id,
            "actorId": actor_id,
            "sessionId": session_id,
            "maxResults": 100,
            "includePayloads": False,
        }
        while True:
            response = client.list_events(**params)
            for event in response.get("events", []):
                client.delete_event(
                    memoryId=memory_id,
                    actorId=actor_id,
                    sessionId=session_id,
                    eventId=event["eventId"],
                )
            next_token = response.get("nextToken")
            if not next_token:
                break
            params["nextToken"] = next_token

    @staticmethod
    def _event_count(memory_id: str, actor_id: str, session_id: str) -> int:
        client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
        response = client.list_events(
            memoryId=memory_id,
            actorId=actor_id,
            sessionId=session_id,
            maxResults=100,
            includePayloads=False,
        )
        return len(response.get("events", []))

    async def test_aput_creates_event(self, store, memory_id, actor_id, session_id):
        await store.aput(
            (actor_id, session_id),
            str(uuid.uuid4()),
            {"message": HumanMessage("I prefer window seats on long flights.")},
        )

        assert self._event_count(memory_id, actor_id, session_id) == 1

    async def test_concurrent_aput(self, store, memory_id, actor_id, session_id):
        await asyncio.gather(
            *(
                store.aput(
                    (actor_id, session_id),
                    str(uuid.uuid4()),
                    {"message": AIMessage(f"Noted preference number {i}.")},
                )
                for i in range(4)
            )
        )

        assert self._event_count(memory_id, actor_id, session_id) == 4

    async def test_asearch_returns_list(self, store, actor_id, session_id):
        results = await store.asearch(
            (actor_id, session_id), query="seat preferences", limit=3
        )

        assert isinstance(results, list)

    async def test_aget_missing_record_returns_none(self, store, actor_id, session_id):
        item = await store.aget((actor_id, session_id), "mem-" + "0" * 40)

        assert item is None

    async def test_abatch_mixed_ops(self, store, memory_id, actor_id, session_id):
        namespace = (actor_id, session_id)
        results = await store.abatch(
            [
                PutOp(
                    namespace=namespace,
                    key=str(uuid.uuid4()),
                    value={"message": HumanMessage("Aisle is fine on short hops.")},
                ),
                SearchOp(namespace_prefix=namespace, query="seats", limit=2),
                GetOp(namespace=namespace, key="mem-" + "1" * 40),
            ]
        )

        assert results[0] is None
        assert isinstance(results[1], list)
        assert results[2] is None
        assert self._event_count(memory_id, actor_id, session_id) == 1

    async def test_async_error_handling(self, store, actor_id, session_id):
        """Validation errors propagate through the async path unchanged."""
        with pytest.raises(ValueError, match="'message' key"):
            await store.aput(
                (actor_id, session_id), str(uuid.uuid4()), {"message": "not a message"}
            )

        with pytest.raises(ValueError, match="tuple of"):
            await store.aput(
                (actor_id,), str(uuid.uuid4()), {"message": HumanMessage("hi")}
            )

    async def test_concurrent_aput_overlaps(
        self, store, memory_id, actor_id, session_id
    ):
        """Concurrent puts complete faster than the same puts run sequentially."""
        n_calls = 6

        def make_put(i: int):
            return store.aput(
                (actor_id, session_id),
                str(uuid.uuid4()),
                {"message": HumanMessage(f"Concurrency probe {i}.")},
            )

        start = time.perf_counter()
        await make_put(0)
        single = time.perf_counter() - start

        start = time.perf_counter()
        await asyncio.gather(*(make_put(i) for i in range(1, n_calls)))
        concurrent = time.perf_counter() - start

        assert self._event_count(memory_id, actor_id, session_id) == n_calls
        assert concurrent < (n_calls - 1) * single * 0.75

    async def test_sync_async_parity(self, store, memory_id, actor_id, session_id):
        namespace = (actor_id, session_id)
        message = HumanMessage("Parity check message.")

        store.put(namespace, str(uuid.uuid4()), {"message": message})
        await store.aput(namespace, str(uuid.uuid4()), {"message": message})
        assert self._event_count(memory_id, actor_id, session_id) == 2

        sync_search = store.search(namespace, query="parity", limit=3)
        async_search = await store.asearch(namespace, query="parity", limit=3)
        assert [item.key for item in sync_search] == [item.key for item in async_search]

        missing_key = "mem-" + "2" * 40
        assert store.get(namespace, missing_key) == await store.aget(
            namespace, missing_key
        )
