import logging
from contextlib import contextmanager, AbstractContextManager
from langgraph_checkpoint_aws import AgentCoreMemorySaver

logger = logging.getLogger(__name__)

@contextmanager
def clean_agentcore_memory(
    saver: AgentCoreMemorySaver,
    /,
    *,
    actor_id: str,
    thread_ids: list[str],
) -> AbstractContextManager[AgentCoreMemorySaver]:
    """Cleanup AgentCoreMemorySaver resources on exit."""
    def _delete_threads(thread_ids: list[str]):
        for thread_id in thread_ids:
            try:
                saver.delete_thread(thread_id, actor_id)
            except Exception:
                logger.exception(f"Failed to cleanup thread {thread_id}")
    
    try:
        _delete_threads(thread_ids)
        yield saver
    finally:
        _delete_threads(thread_ids)
