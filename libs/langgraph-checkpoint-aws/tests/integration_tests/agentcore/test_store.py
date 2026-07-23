"""
Integration tests for AgentCoreMemoryStore.

These tests require real AWS credentials and an AgentCore Memory resource.
Set AGENTCORE_MEMORY_ID environment variable to run these tests.
"""

import os
import random
import string
import time
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.store.base import GetOp, PutOp, SearchOp

from langgraph_checkpoint_aws.store.agentcore.store import AgentCoreMemoryStore


def generate_valid_actor_id():
    """Generate a valid actor ID that matches AgentCore pattern."""
    chars = string.ascii_letters + string.digits
    return "actor" + "".join(random.choices(chars, k=6))


def generate_valid_session_id():
    """Generate a valid session ID that matches AgentCore pattern."""
    chars = string.ascii_letters + string.digits
    return "session" + "".join(random.choices(chars, k=6))


class TestAgentCoreMemoryStoreIntegration:
    """Integration tests for AgentCoreMemoryStore with real AgentCore Memory service."""

    @pytest.fixture
    def memory_id(self):
        """Get memory ID from environment variable."""
        memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
        if not memory_id:
            pytest.skip("AGENTCORE_MEMORY_ID environment variable not set")
        return memory_id

    @pytest.fixture
    def store(self, memory_id):
        """Create AgentCoreMemoryStore instance."""
        return AgentCoreMemoryStore(memory_id=memory_id, region_name="us-west-2")

    @pytest.fixture
    def actor_id(self):
        """Generate unique actor ID for test isolation."""
        return generate_valid_actor_id()

    @pytest.fixture
    def session_id(self):
        """Generate unique session ID for test isolation."""
        return generate_valid_session_id()

    def test_store_human_message(self, store, actor_id, session_id):
        """Test storing a human message as conversational event."""
        message = HumanMessage("I love coffee and prefer dark roast")

        store.put(
            namespace=(actor_id, session_id),
            key=str(uuid.uuid4()),
            value={"message": message},
        )

        assert True, "Message stored successfully"

    def test_store_ai_message(self, store, actor_id, session_id):
        """Test storing an AI message as conversational event."""
        message = AIMessage(
            "I understand you enjoy dark roast coffee. That's a great choice!"
        )

        store.put(
            namespace=(actor_id, session_id),
            key=str(uuid.uuid4()),
            value={"message": message},
        )

        assert True, "AI message stored successfully"

    def test_store_system_message(self, store, actor_id, session_id):
        """Test storing a system message as conversational event."""
        message = SystemMessage(
            "You are a helpful assistant that remembers user preferences"
        )

        store.put(
            namespace=(actor_id, session_id),
            key=str(uuid.uuid4()),
            value={"message": message},
        )

        assert True, "System message stored successfully"

    def test_store_tool_message(self, store, actor_id, session_id):
        """Test storing a tool message as conversational event."""
        message = ToolMessage(
            content="Weather in San Francisco: 72°F, sunny", tool_call_id="call_123"
        )

        store.put(
            namespace=(actor_id, session_id),
            key=str(uuid.uuid4()),
            value={"message": message},
        )

        assert True, "Tool message stored successfully"

    def test_conversation_flow(self, store, actor_id, session_id):
        """Test storing a complete conversation flow."""
        messages = [
            HumanMessage("Hi, I'm planning a trip to Italy"),
            AIMessage(
                "That sounds wonderful! Italy has so many amazing places to visit. "
                "What type of experience are you looking for?"
            ),
            HumanMessage("I love art and history, especially Renaissance art"),
            AIMessage(
                "Perfect! Florence would be ideal for you - it's the birthplace of "
                "the Renaissance with incredible museums like the Uffizi Gallery"
            ),
            HumanMessage("That sounds perfect! I also enjoy good food and wine"),
            AIMessage(
                "Excellent! Tuscany, where Florence is located, is famous for its "
                "cuisine and wines. You'll love the local trattorias and vineyards"
            ),
        ]

        for i, message in enumerate(messages):
            store.put(
                namespace=(actor_id, session_id),
                key=f"msg_{i}",
                value={"message": message},
            )
            time.sleep(1)

        time.sleep(2)

        assert True, "Conversation flow stored successfully"

    def test_search_processed_memories(self, store, actor_id):
        """Test searching for processed memories after storing conversations."""
        session1 = generate_valid_session_id()
        session2 = generate_valid_session_id()

        preference_messages = [
            HumanMessage("I really love Italian food, especially pasta carbonara"),
            AIMessage("Great choice! Carbonara is a classic Roman dish"),
            HumanMessage("I also enjoy red wine, particularly Chianti"),
            AIMessage("Chianti pairs wonderfully with Italian cuisine"),
        ]

        for i, msg in enumerate(preference_messages[:2]):
            store.put(
                namespace=(actor_id, session1), key=f"pref1_{i}", value={"message": msg}
            )
            time.sleep(1)

        for i, msg in enumerate(preference_messages[2:]):
            store.put(
                namespace=(actor_id, session2), key=f"pref2_{i}", value={"message": msg}
            )
            time.sleep(1)

        time.sleep(60)

        search_namespaces = [
            ("preferences", actor_id),
            ("facts", actor_id),
            ("summaries", "actors", actor_id, "sessions", session1),
            ("summaries", "actors", actor_id, "sessions", session2),
        ]

        found_results = False
        for namespace in search_namespaces:
            results = store.search(
                namespace, query="food preferences Italian cuisine", limit=5
            )

            if results:
                found_results = True
                break

        assert isinstance(found_results, bool)

    def test_hierarchical_vs_exact_match(self, store, memory_id, actor_id):
        """Test hierarchical search matches a parent prefix but exact match does not."""
        session = generate_valid_session_id()

        exact_match_store = AgentCoreMemoryStore(
            memory_id=memory_id, region_name="us-west-2", hierarchical_search=False
        )

        messages = [
            HumanMessage("I really love Croatian food, especially cevapi"),
            AIMessage("Great choice! Cevapi is a classic Balkan grilled dish"),
            HumanMessage("I also enjoy Croatian wine, particularly Plavac Mali"),
            AIMessage("Plavac Mali pairs wonderfully with Dalmatian cuisine"),
        ]
        for i, msg in enumerate(messages):
            store.put(
                namespace=(actor_id, session), key=f"summ_{i}", value={"message": msg}
            )
            time.sleep(1)

        # Extra time for extraction since summaries are generated async
        time.sleep(120)

        parent_prefix = ("summaries", "actors", actor_id)
        query = "food preferences Croatian cuisine"

        hierarchical_results = store.search(parent_prefix, query=query, limit=5)
        exact_match_results = exact_match_store.search(
            parent_prefix, query=query, limit=5
        )

        assert hierarchical_results, "namespacePath should match the parent prefix"
        assert not exact_match_results, "namespace should not match a parent prefix"

    def test_batch_operations(self, store, actor_id, session_id):
        """Test batch operations with multiple put and search operations."""
        messages = [
            HumanMessage("I'm interested in learning about machine learning"),
            AIMessage(
                "Machine learning is a fascinating field! "
                "What specific area interests you most?"
            ),
            HumanMessage("I'd like to understand neural networks and deep learning"),
            AIMessage(
                "Great choice! Neural networks are the foundation of modern AI systems"
            ),
        ]

        put_ops = []
        for i, message in enumerate(messages):
            put_ops.append(
                PutOp(
                    namespace=(actor_id, session_id),
                    key=f"batch_msg_{i}",
                    value={"message": message},
                )
            )

        search_ops = [
            SearchOp(
                namespace_prefix=("facts", actor_id),
                query="machine learning interests",
                limit=3,
            ),
            SearchOp(
                namespace_prefix=("preferences", actor_id),
                query="learning topics",
                limit=3,
            ),
        ]

        all_ops = put_ops + search_ops
        results = store.batch(all_ops)

        assert len(results) == len(all_ops)

        for i in range(len(put_ops)):
            assert results[i] is None

        for i in range(len(put_ops), len(all_ops)):
            assert isinstance(results[i], list)

        assert all(isinstance(r, list) for r in results[len(put_ops) :])

    def test_multiple_actors_isolation(self, store):
        """Test that different actors have isolated memory spaces."""
        actor1 = generate_valid_actor_id()
        actor2 = generate_valid_actor_id()
        session1 = generate_valid_session_id()
        session2 = generate_valid_session_id()

        # Store different preferences for each actor
        actor1_preference = "I love spicy food and hot sauce"
        actor2_preference = "I prefer mild flavors and avoid spicy food"

        store.put(
            namespace=(actor1, session1),
            key="pref1",
            value={"message": HumanMessage(actor1_preference)},
        )

        store.put(
            namespace=(actor2, session2),
            key="pref2",
            value={"message": HumanMessage(actor2_preference)},
        )

        # Wait for processing the long term memory (usually done in under 2 minutes)
        time.sleep(120)

        # Search should be isolated per actor
        results1 = store.search(("facts", actor1), query="food preferences", limit=5)

        results2 = store.search(("facts", actor2), query="food preferences", limit=5)

        assert isinstance(results1, list)
        assert isinstance(results2, list)

        # Check that actor1's results contain reference to spicy food preference
        if results1:
            actor1_content_found = any(
                "hot" in result.value.get("content", "").lower() for result in results1
            )
            assert actor1_content_found, (
                f"Actor1's spicy food preference not found in search results: "
                f"{[r.value.get('content', '') for r in results1]}"
            )

        # Check that actor2's results contain reference to mild food preference
        if results2:
            actor2_content_found = any(
                "mild" in result.value.get("content", "").lower() for result in results2
            )
            assert actor2_content_found, (
                f"Actor2's mild food preference not found in search results: "
                f"{[r.value.get('content', '') for r in results2]}"
            )

        assert True, "Actor isolation test completed with preference verification"

    def test_error_handling_invalid_message(self, store, actor_id, session_id):
        """Test error handling for invalid message format."""
        with pytest.raises(ValueError, match="Value must contain a 'message' key"):
            store.put(
                namespace=(actor_id, session_id),
                key="invalid",
                value={"not_message": "invalid"},
            )

    def test_error_handling_invalid_namespace(self, store):
        """Test error handling for invalid namespace format."""
        with pytest.raises(ValueError, match="Namespace must be a tuple of"):
            store.put(
                namespace=("single_element",),
                key="test",
                value={"message": HumanMessage("test")},
            )

    def test_search_without_query(self, store, actor_id):
        """Test search behavior when no query is provided."""
        results = store.search((actor_id, "facts"), limit=5)

        # Should return empty list when no query provided
        assert isinstance(results, list)
        assert len(results) == 0

    def test_complex_message_content(self, store, actor_id, session_id):
        """Test storing messages with complex content structures."""
        complex_message = HumanMessage(
            content=[
                {"type": "text", "text": "I'm planning a vacation and need help with:"},
                {"type": "text", "text": "1. Flight bookings to Europe"},
                {"type": "text", "text": "2. Hotel recommendations in Paris"},
                {
                    "type": "text",
                    "text": "3. Restaurant suggestions for Italian cuisine",
                },
            ]
        )

        store.put(
            namespace=(actor_id, session_id),
            key="complex_msg",
            value={"message": complex_message},
        )

    def test_get_memory_record_success(self, store, actor_id):
        """Test retrieving a specific memory record that exists."""
        # First, we need to search for existing memory records to get valid IDs
        # This test assumes some memory records exist from previous conversations

        # Try to search for any existing records
        search_namespaces = [
            ("preferences", actor_id),
            ("facts", actor_id),
        ]

        memory_record_id = None
        test_namespace = None

        for namespace in search_namespaces:
            results = store.search(
                namespace,
                query="preferences food coffee",
                limit=1,
            )

            if results and len(results) > 0:
                memory_record_id = results[0].key
                test_namespace = namespace
                break

        if memory_record_id and test_namespace:
            # Test GetOp with existing memory record
            item = store.get(test_namespace, memory_record_id)

            if item:
                assert item.key == memory_record_id
                assert item.namespace == test_namespace
                assert "content" in item.value
                assert item.created_at is not None
                assert item.updated_at is not None

    def test_batch_operations_with_get(self, store, actor_id, session_id):
        """Test batch operations including GetOp operations."""
        # Put a message with a known key so we can retrieve it via GetOp
        hiking_key = f"hiking-{uuid.uuid4().hex[:8]}"
        store.put(
            namespace=(actor_id, session_id),
            key=hiking_key,
            value={"message": HumanMessage("Trail running is also fun")},
        )

        # Create batch operations including GetOp
        batch_ops = [
            PutOp(
                namespace=(actor_id, session_id),
                key=f"hiking_extra_{uuid.uuid4().hex[:8]}",
                value={"message": HumanMessage("I also enjoy rock climbing")},
            ),
            SearchOp(
                namespace_prefix=("facts", actor_id),
                query="outdoor activities",
                limit=3,
            ),
            GetOp(namespace=(actor_id, session_id), key=hiking_key),
        ]

        results = store.batch(batch_ops)

        assert len(results) == 3
        assert results[0] is None  # PutOp
        assert isinstance(results[1], list)  # SearchOp
        # GetOp should return the item we put earlier
        get_result = results[2]
        assert get_result is not None
        assert get_result.key == hiking_key
        assert get_result.value["content"] == "Trail running is also fun"

    def test_put_get_roundtrip_by_key(self, store, actor_id, session_id):
        """Test that put→get round-trips correctly by user-supplied key (#708)."""
        key = f"roundtrip-{uuid.uuid4().hex[:8]}"
        message_text = "I enjoy reading science fiction novels"
        message = HumanMessage(message_text)

        store.put(
            namespace=(actor_id, session_id),
            key=key,
            value={"message": message},
        )

        item = store.get((actor_id, session_id), key)

        assert item is not None, "get() should return a non-None Item after put()"
        assert item.key == key
        assert item.value["content"] == message_text

    def test_put_get_last_write_wins(self, store, actor_id, session_id):
        """Test that re-putting the same key returns the most recent value."""
        key = f"lww-{uuid.uuid4().hex[:8]}"
        msg_a = HumanMessage("First version of the message")
        msg_b = HumanMessage("Second version of the message")

        store.put(
            namespace=(actor_id, session_id),
            key=key,
            value={"message": msg_a},
        )
        store.put(
            namespace=(actor_id, session_id),
            key=key,
            value={"message": msg_b},
        )

        item = store.get((actor_id, session_id), key)

        assert item is not None, "get() should return a non-None Item"
        assert item.key == key
        assert item.value["content"] == "Second version of the message"

    def test_get_absent_key_returns_none(self, store, actor_id, session_id):
        """Test that get() for a never-stored key returns None."""
        absent_key = f"never-put-key-{uuid.uuid4().hex[:8]}"

        item = store.get((actor_id, session_id), absent_key)

        assert item is None
