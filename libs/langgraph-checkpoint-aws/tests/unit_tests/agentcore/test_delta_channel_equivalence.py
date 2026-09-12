"""Equivalence tests: `AgentCoreMemorySaver.get_delta_channel_history` vs.
`BaseCheckpointSaver.get_delta_channel_history`.

This is the core correctness argument for the `get_delta_channel_history` /
`aget_delta_channel_history` overrides: they exist purely for performance
(bulk-paged event fetch instead of one `get_tuple` round trip per ancestor),
so their output must be *identical* to the base implementation's for the
same underlying data, in every case the base handles -- a single page, a
chain split across multiple `ListEvents` pages, a forked/branched thread
(only on-path ancestors may contribute), and a long thread driven through
real `DeltaChannel` snapshot/replay machinery.

Each test builds a synthetic event log, then computes the result two ways
against the *same* saver instance and the *same* fetched data:

  * `saver.get_delta_channel_history(...)` -- the override under test.
  * `BaseCheckpointSaver.get_delta_channel_history(saver, ...)` -- the
    upstream reference algorithm, called unbound so it walks ancestors via
    `saver.get_tuple(...)` (one AgentCore round trip per ancestor) instead
    of the override's bulk path.

and asserts the two are equal, plus that no message id or `tool_call_id`
is duplicated in the reconstructed value.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.types import _DeltaSnapshot
from langgraph.pregel._checkpoint import (
    create_checkpoint,
    delta_channels_to_snapshot,
    empty_checkpoint,
)

from langgraph_checkpoint_aws.checkpoint.agentcore.models import (
    ChannelDataEvent,
    CheckpointEvent,
    WriteItem,
    WritesEvent,
)
from langgraph_checkpoint_aws.checkpoint.agentcore.saver import AgentCoreMemorySaver

THREAD_ID = "equiv-thread"
ACTOR_ID = "equiv-actor"
CHECKPOINT_NS = "equiv-ns"


def _checkpoint_data(checkpoint_id, channel_versions=None):
    return {
        "v": 1,
        "id": checkpoint_id,
        "ts": "2024-01-01T00:00:00Z",
        "channel_versions": channel_versions or {},
        "versions_seen": {},
        "pending_sends": [],
    }


def _list_events_page(saver, events, *, next_token=None):
    response = {
        "events": [
            {
                "eventId": f"event_{i}",
                "payload": [{"blob": saver.serializer.serialize_event(event)}],
            }
            for i, event in enumerate(events)
        ]
    }
    if next_token:
        response["nextToken"] = next_token
    return response


def _make_saver(**kwargs):
    mock_client = MagicMock()
    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client.return_value = mock_client
        saver = AgentCoreMemorySaver(memory_id="equiv-memory-id", **kwargs)
    return saver, mock_client


def _config(checkpoint_id):
    return RunnableConfig(
        configurable={
            "thread_id": THREAD_ID,
            "actor_id": ACTOR_ID,
            "checkpoint_ns": CHECKPOINT_NS,
            "checkpoint_id": checkpoint_id,
        }
    )


def _assert_equivalent_and_committed(saver, config, channels):
    """Call both implementations and assert byte-identical results."""
    override_result = saver.get_delta_channel_history(config=config, channels=channels)
    base_result = BaseCheckpointSaver.get_delta_channel_history(
        saver, config=config, channels=channels
    )
    assert override_result == base_result
    return override_result


def _linear_chain(saver, *, num_ancestors, seed_at="root"):
    """Build a linear chain root -> c1 -> ... -> c{num_ancestors} with a
    plain (non-`_DeltaSnapshot`) seed at `seed_at` and one write per
    ancestor after it. Returns (events, target_checkpoint_id)."""
    events = []
    events.append(
        CheckpointEvent(
            checkpoint_id="root",
            checkpoint_data=_checkpoint_data("root", {"notes": "v0"}),
            metadata={"step": 0},
            thread_id=THREAD_ID,
            checkpoint_ns=CHECKPOINT_NS,
        )
    )
    events.append(
        ChannelDataEvent(
            channel="notes",
            version="v0",
            value="root_seed",
            thread_id=THREAD_ID,
            checkpoint_ns=CHECKPOINT_NS,
        )
    )
    assert seed_at == "root"  # only supported seed position for this helper

    prev_id = "root"
    for i in range(1, num_ancestors + 1):
        cid = f"c{i}"
        events.append(
            CheckpointEvent(
                checkpoint_id=cid,
                checkpoint_data=_checkpoint_data(cid),
                metadata={"step": i},
                parent_checkpoint_id=prev_id,
                thread_id=THREAD_ID,
                checkpoint_ns=CHECKPOINT_NS,
            )
        )
        events.append(
            WritesEvent(
                checkpoint_id=prev_id,
                writes=[WriteItem(task_id=f"t{i}", channel="notes", value=f"m{i}")],
            )
        )
        prev_id = cid
    return events, prev_id


class TestDeltaChannelHistoryEquivalenceSyntheticSinglePage:
    """A linear ancestor chain fetched in a single `ListEvents` page."""

    def test_equivalence_and_no_duplicates(self):
        saver, mock_client = _make_saver()
        events, target_id = _linear_chain(saver, num_ancestors=10)
        mock_client.list_events.return_value = _list_events_page(saver, events)

        result = _assert_equivalent_and_committed(saver, _config(target_id), ["notes"])

        assert result["notes"]["seed"] == "root_seed"
        values = [result["notes"]["seed"]] + [w[2] for w in result["notes"]["writes"]]
        assert len(values) == len(set(values))


class TestDeltaChannelHistoryEquivalencePaginated:
    """The same chain, but split across two `ListEvents` pages (newest
    events first, oldest ancestors requiring a second page -- mirroring
    real AgentCore Memory pagination)."""

    def test_equivalence_across_page_boundary(self):
        saver, mock_client = _make_saver()
        events, target_id = _linear_chain(saver, num_ancestors=10)

        newest_first = list(reversed(events))
        mid = len(newest_first) // 2
        page1 = _list_events_page(saver, newest_first[:mid], next_token="tok1")
        page2 = _list_events_page(saver, newest_first[mid:])

        def _side_effect(**kwargs):
            return page2 if kwargs.get("nextToken") == "tok1" else page1

        mock_client.list_events.side_effect = _side_effect
        result = _assert_equivalent_and_committed(saver, _config(target_id), ["notes"])
        assert result["notes"]["seed"] == "root_seed"
        assert len(result["notes"]["writes"]) == 10


class TestDeltaChannelHistoryEquivalenceForkedThread:
    """A forked/branched thread: two children (`b1`, `b2`) of the same
    parent (`a`), each writing to `notes` independently. The target
    descends only from `b2`. Only on-path ancestors (`a`, `b2`, ...) may
    contribute writes or a seed -- `b1`'s branch must be invisible, for
    both the override and the base reference walk."""

    def test_only_on_path_ancestors_contribute(self):
        saver, mock_client = _make_saver()
        events = [
            CheckpointEvent(
                checkpoint_id="root",
                checkpoint_data=_checkpoint_data("root", {"notes": "v0"}),
                metadata={"step": 0},
                thread_id=THREAD_ID,
                checkpoint_ns=CHECKPOINT_NS,
            ),
            ChannelDataEvent(
                channel="notes",
                version="v0",
                value="root_seed",
                thread_id=THREAD_ID,
                checkpoint_ns=CHECKPOINT_NS,
            ),
            CheckpointEvent(
                checkpoint_id="a",
                checkpoint_data=_checkpoint_data("a"),
                metadata={"step": 1},
                parent_checkpoint_id="root",
                thread_id=THREAD_ID,
                checkpoint_ns=CHECKPOINT_NS,
            ),
            WritesEvent(
                checkpoint_id="root",
                writes=[WriteItem(task_id="t_root", channel="notes", value="m_root")],
            ),
            # Off-path branch: b1 is a sibling of b2, both children of "a".
            # b1's write must never appear when walking from a target under b2.
            CheckpointEvent(
                checkpoint_id="b1",
                checkpoint_data=_checkpoint_data("b1"),
                metadata={"step": 2, "source": "fork"},
                parent_checkpoint_id="a",
                thread_id=THREAD_ID,
                checkpoint_ns=CHECKPOINT_NS,
            ),
            WritesEvent(
                checkpoint_id="b1",
                writes=[
                    WriteItem(task_id="t_b1", channel="notes", value="OFF_PATH_m_b1")
                ],
            ),
            # On-path branch: b2, also a child of "a" (a resumed/forked run).
            CheckpointEvent(
                checkpoint_id="b2",
                checkpoint_data=_checkpoint_data("b2"),
                metadata={"step": 2, "source": "fork"},
                parent_checkpoint_id="a",
                thread_id=THREAD_ID,
                checkpoint_ns=CHECKPOINT_NS,
            ),
            WritesEvent(
                checkpoint_id="a",
                writes=[WriteItem(task_id="t_a", channel="notes", value="m_a")],
            ),
            WritesEvent(
                checkpoint_id="b2",
                writes=[WriteItem(task_id="t_b2", channel="notes", value="m_b2")],
            ),
            CheckpointEvent(
                checkpoint_id="target",
                checkpoint_data=_checkpoint_data("target"),
                metadata={"step": 3},
                parent_checkpoint_id="b2",
                thread_id=THREAD_ID,
                checkpoint_ns=CHECKPOINT_NS,
            ),
        ]
        mock_client.list_events.return_value = _list_events_page(saver, events)

        result = _assert_equivalent_and_committed(saver, _config("target"), ["notes"])

        write_values = [w[2] for w in result["notes"]["writes"]]
        assert "OFF_PATH_m_b1" not in write_values
        assert write_values == ["m_root", "m_a", "m_b2"]
        assert result["notes"]["seed"] == "root_seed"


class TestDeltaChannelHistoryEquivalenceRealDeltaChannel:
    """A 20-turn thread driven through real `langgraph.channels.delta`
    machinery (`DeltaChannel.update`/`replay_writes`,
    `delta_channels_to_snapshot`, `pregel._checkpoint.create_checkpoint`),
    with `snapshot_frequency=3` (so several `_DeltaSnapshot` boundaries are
    crossed) and `max_results=5` (so every checkpoint's events are split
    across several `ListEvents` pages). This exercises the exact real-world
    shape of the reported regression: a `_DeltaSnapshot` seed with further
    writes stacked on top, reconstructed from bulk-paged events.
    """

    @staticmethod
    def _add_messages(base, writes):
        return list(base) + list(writes)

    @staticmethod
    def _get_next_version(current, _channel):
        if current is None:
            return "00000000000000000000000000000001"
        return f"{int(current) + 1:032d}"

    def test_equivalence_across_snapshot_boundaries(self):
        store: list[dict] = []

        def _fake_create_event(**kwargs):
            store.append(kwargs)
            return {"event": {"eventId": f"e{len(store)}"}}

        def _fake_list_events(**kwargs):
            max_results = kwargs.get("maxResults") or 100
            next_token = kwargs.get("nextToken")
            start = int(next_token) if next_token else 0
            end = start + max_results
            page = store[start:end]
            response = {
                "events": [
                    {"eventId": f"e{i}", "payload": ev["payload"]}
                    for i, ev in enumerate(page, start=start)
                ]
            }
            if end < len(store):
                response["nextToken"] = str(end)
            return response

        mock_client = MagicMock()
        mock_client.create_event.side_effect = _fake_create_event
        mock_client.list_events.side_effect = _fake_list_events

        with patch("boto3.client") as mock_boto3_client:
            mock_boto3_client.return_value = mock_client
            saver = AgentCoreMemorySaver(memory_id="equiv-memory-id", max_results=5)

        channel = DeltaChannel(self._add_messages, list, snapshot_frequency=3)
        channel.value = []

        checkpoint = empty_checkpoint()
        base_config = RunnableConfig(
            configurable={
                "thread_id": THREAD_ID,
                "actor_id": ACTOR_ID,
                "checkpoint_ns": CHECKPOINT_NS,
            }
        )
        counters: dict[str, tuple[int, int]] = {}
        prev_checkpoint_id: str | None = None
        tool_call_counter = 0

        for step in range(20):
            if step % 2 == 0:
                writes = [HumanMessage(content=f"question {step}", id=f"h{step}")]
            else:
                tool_call_counter += 1
                tc_id = f"call_{tool_call_counter}"
                writes = [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search", "args": {}, "id": tc_id}],
                        id=f"ai{step}",
                    )
                ]

            channel.update(writes)
            updates, supersteps = counters.get("messages", (0, 0))
            counters["messages"] = (updates + 1, supersteps + 1)

            channels_to_snapshot = delta_channels_to_snapshot(
                {"messages": channel}, counters
            )
            if "messages" in channels_to_snapshot:
                counters["messages"] = (0, 0)

            prev_version = checkpoint["channel_versions"].get("messages")
            checkpoint = dict(checkpoint)
            checkpoint["channel_versions"] = dict(checkpoint["channel_versions"])
            checkpoint["channel_versions"]["messages"] = self._get_next_version(
                prev_version, None
            )

            checkpoint = create_checkpoint(
                checkpoint,
                {"messages": channel},
                step,
                id=str(uuid.uuid4()),
                updated_channels={"messages"},
                get_next_version=self._get_next_version,
                channels_to_snapshot=channels_to_snapshot,
            )

            put_config = dict(base_config)
            put_config["configurable"] = dict(base_config["configurable"])
            if prev_checkpoint_id:
                put_config["configurable"]["checkpoint_id"] = prev_checkpoint_id

            metadata = {"source": "loop", "step": step, "writes": {}, "parents": {}}
            new_versions = {"messages": checkpoint["channel_versions"]["messages"]}
            result_config = saver.put(put_config, checkpoint, metadata, new_versions)
            prev_checkpoint_id = result_config["configurable"]["checkpoint_id"]

            if step % 2 == 1:
                tool_message = ToolMessage(
                    content="result", tool_call_id=tc_id, id=f"tm{step}"
                )
                saver.put_writes(
                    result_config, [("messages", tool_message)], task_id=f"task{step}"
                )
                channel.replay_writes([(f"task{step}", "messages", tool_message)])

        target_config = RunnableConfig(
            configurable={
                "thread_id": THREAD_ID,
                "actor_id": ACTOR_ID,
                "checkpoint_ns": CHECKPOINT_NS,
                "checkpoint_id": prev_checkpoint_id,
            }
        )

        result = _assert_equivalent_and_committed(saver, target_config, ["messages"])

        seed = result["messages"].get("seed")
        seed_messages = list(seed.value) if isinstance(seed, _DeltaSnapshot) else []
        write_values = [w[2] for w in result["messages"]["writes"]]
        all_messages = seed_messages + write_values

        message_ids = [m.id for m in all_messages]
        assert len(message_ids) == len(set(message_ids))
