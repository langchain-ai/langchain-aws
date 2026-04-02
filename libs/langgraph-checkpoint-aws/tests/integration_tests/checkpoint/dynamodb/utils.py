import logging
from collections.abc import Generator
from contextlib import contextmanager

from langgraph_checkpoint_aws import DynamoDBSaver

logger = logging.getLogger(__name__)


@contextmanager
def clean_dynamodb(
    saver: DynamoDBSaver,
    /,
    *,
    thread_ids: list[str],
) -> Generator[DynamoDBSaver, None, None]:
    """Clean up DynamoDB resources after tests."""

    def _delete_threads(thread_ids: list[str]):
        for thread_id in thread_ids:
            try:
                saver.delete_thread(thread_id)
                logger.info(f"Cleaned up thread: {thread_id}")
            except Exception as e:
                logger.exception(f"Failed to cleanup thread {thread_id}: {e}")

    try:
        _delete_threads(thread_ids)
        yield saver
    finally:
        _delete_threads(thread_ids)
