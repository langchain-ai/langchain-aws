import asyncio
import base64
import hashlib
import json
import uuid
from collections.abc import Sequence
from contextvars import copy_context
from functools import partial
from typing import Any, Callable, Optional, Tuple, TypeVar, Union, cast

from botocore.config import Config
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.constants import TASKS
from pydantic import BaseModel

from langgraph_checkpoint_aws import SDK_USER_AGENT
from langgraph_checkpoint_aws.constants import CHECKPOINT_PREFIX, WRITES_PREFIX
from langgraph_checkpoint_aws.models import (
    BedrockSessionContentBlock,
    SessionCheckpoint,
    SessionPendingWrite,
)

T = TypeVar("T")


def to_boto_params(model: BaseModel) -> dict:
    """
    Convert a Pydantic model to a dictionary of parameters suitable for use with boto3.

    Args:
        model (BaseModel): Pydantic model to convert to boto3 parameters

    Returns:
        dict: Dictionary of parameters compatible with boto3 API calls
    """
    return model.model_dump(by_alias=True, exclude_none=True)


def generate_deterministic_uuid(input_string: Union[str, bytes]) -> uuid.UUID:
    """
    Generate a deterministic UUID from a string input using MD5 hashing.

    Args:
        input_string: Input string or bytes to generate UUID from

    Returns:
        UUID object generated deterministically from the input
    """
    if isinstance(input_string, str):
        input_bytes = input_string.encode("utf-8")
    else:
        input_bytes = input_string

    digest = hashlib.md5(input_bytes).digest()
    return uuid.UUID(bytes=digest)


def generate_checkpoint_id(namespace: str) -> str:
    """Generate a unique identifier for checkpoint operations.

    Args:
        namespace: Namespace for the checkpoint

    Returns:
        str: Deterministic UUID as string for the checkpoint
    """
    return str(generate_deterministic_uuid(f"{CHECKPOINT_PREFIX}#{namespace}"))


def generate_write_id(namespace: str, checkpoint_id: str) -> str:
    """Generate a unique identifier for write operations.

    Args:
        namespace: Namespace for the write operation
        checkpoint_id: Associated checkpoint identifier

    Returns:
        str: Deterministic UUID as string for the write operation
    """
    return str(
        generate_deterministic_uuid(f"{WRITES_PREFIX}#{namespace}#{checkpoint_id}")
    )


def deserialize_data(serializer: SerializerProtocol, data: str) -> Any:
    """Deserialize string data into Python objects.

    Args:
        serializer: SerializerProtocol instance
        data: JSON-formatted string data

    Returns:
        Any: Deserialized Python object
    """
    return serializer.loads(data.encode())


def serialize_data(serializer: SerializerProtocol, data: Any) -> str:
    """Serialize Python objects to string format.

    Args:
        serializer: SerializerProtocol instance
        data: Python object to serialize

    Returns:
        str: Serialized string data with null characters handled
    """
    serialized = serializer.dumps(data)
    return serialized.decode().replace("\\u0000", "")


def serialize_to_base64(serializer: SerializerProtocol, data: Any) -> Tuple[str, str]:
    """Serialize data to base64 encoded format.

    Args:
        serializer: SerializerProtocol instance
        data: Data to be serialized

    Returns:
        Tuple[str, str]: Tuple of (type, base64 encoded string)
    """
    data_type, serialized = serializer.dumps_typed(data)
    encoded = base64.b64encode(serialized).decode("utf-8")
    return data_type, encoded


def deserialize_from_base64(
    serializer: SerializerProtocol, data_type: str, encoded_data: str
) -> Any:
    """Deserialize data from base64 format.

    Args:
        serializer: SerializerProtocol instance
        data_type: Type identifier of the serialized data
        encoded_data: Base64 encoded string data

    Returns:
        Any: Deserialized data object
    """
    decoded = base64.b64decode(encoded_data.encode("utf-8"))
    return serializer.loads_typed((data_type, decoded))


def construct_checkpoint_tuple(
    thread_id: str,
    checkpoint_ns: str,
    session_checkpoint: SessionCheckpoint,
    pending_writes: list[SessionPendingWrite],
    sends: list,
    serde: SerializerProtocol,
) -> CheckpointTuple:
    """Construct checkpoint tuple from components.

    Args:
        thread_id: Session thread identifier
        checkpoint_ns: Checkpoint namespace
        session_checkpoint: Checkpoint payload data
        pending_writes: List of pending write operations
        sends: List of task sends
        serde: Protocol for serialization and deserialization of objects

    Returns:
        Constructed CheckpointTuple
    """
    return CheckpointTuple(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": session_checkpoint.checkpoint_id,
            }
        },
        cast(
            Checkpoint,
            {
                **deserialize_from_base64(serde, *session_checkpoint.checkpoint),
                "pending_sends": [serde.loads_typed(s[2]) for s in sends],
                "channel_values": deserialize_from_base64(
                    serde, *session_checkpoint.channel_values
                ),
            },
        ),
        deserialize_data(serde, session_checkpoint.metadata),
        (
            {
                "configurable": {
                    "thread_id": session_checkpoint.thread_id,
                    "checkpoint_ns": session_checkpoint.checkpoint_ns,
                    "checkpoint_id": session_checkpoint.parent_checkpoint_id,
                }
            }
            if session_checkpoint.parent_checkpoint_id
            else None
        ),
        [(write.task_id, write.channel, write.value) for write in pending_writes]
        if pending_writes
        else [],
    )


def transform_pending_task_writes(
    pending_writes: list[SessionPendingWrite],
) -> list[list[Any]]:
    """Transform pending write operations into sorted list format.

    Args:
        pending_writes: List of SessionPendingWrite objects to transform

    Returns:
        list[list[Any]]: Sorted list of write operations, where each write is represented as
            a list containing [task_id, channel, value, task_path, write_idx]. Sorted by
            task_path, task_id, and write_idx.
    """
    return sorted(
        (
            list(
                writes.model_dump(
                    include={
                        "task_id",
                        "channel",
                        "value",
                        "task_path",
                        "write_idx",
                    }
                ).values()
            )
            for writes in pending_writes
            if writes.channel == TASKS
        ),
        key=lambda w: (w[3], w[0], w[4]),
    )


def create_session_checkpoint(
    checkpoint: Checkpoint,
    config: RunnableConfig,
    metadata: CheckpointMetadata,
    serializer: SerializerProtocol,
    new_versions: ChannelVersions,
) -> SessionCheckpoint:
    """
    Create a SessionCheckpoint object from the given checkpoint and related data.

    This function processes the checkpoint, extracts necessary information from the config,
    and serializes various components to create a SessionCheckpoint object.

    Args:
        checkpoint (Checkpoint): The checkpoint to process.
        config (RunnableConfig): Configuration containing thread and checkpoint information.
        metadata (CheckpointMetadata): Metadata associated with the checkpoint.
        serializer (SerializerProtocol): Serializer for data conversion.
        new_versions (ChannelVersions): New versions of channel data.

    Returns:
        SessionCheckpoint: A SessionCheckpoint object containing the processed and serialized data.
    """
    # Create copy to avoid modifying original checkpoint
    checkpoint_copy = checkpoint.copy()

    # Remove pending sends as they are handled separately
    checkpoint_copy.pop("pending_sends", None)

    # Extract required config values
    thread_id = config["configurable"]["thread_id"]
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
    checkpoint_id = checkpoint["id"]

    session_checkpoint = SessionCheckpoint(
        step_type=CHECKPOINT_PREFIX,
        thread_id=thread_id,
        checkpoint_ns=checkpoint_ns,
        checkpoint_id=checkpoint_id,
        checkpoint=serialize_to_base64(serializer, checkpoint_copy),
        metadata=serialize_data(serializer, metadata),
        # Pop and serialize channel values separately
        channel_values=serialize_to_base64(
            serializer, checkpoint_copy.pop("channel_values")
        ),
        version=serialize_to_base64(serializer, new_versions),
    )

    return session_checkpoint


def process_writes_invocation_content_blocks(
    content_blocks: list[BedrockSessionContentBlock], serializer: SerializerProtocol
) -> list[SessionPendingWrite]:
    """Process content blocks and convert them to SessionPendingWrite objects.

    Args:
        content_blocks: List of content blocks containing JSON text
        serializer: Serializer instance for value deserialization

    Returns:
        List of SessionPendingWrite objects
    """
    # Parse JSON content from content blocks
    pending_writes = []
    for content_block in content_blocks:
        pending_writes.append(json.loads(content_block.text))

    # Convert raw dictionaries into SessionPendingWrite objects
    return [
        SessionPendingWrite(
            step_type=WRITES_PREFIX,
            thread_id=write["thread_id"],
            checkpoint_ns=write["checkpoint_ns"],
            checkpoint_id=write["checkpoint_id"],
            task_id=write["task_id"],
            channel=write["channel"],
            value=deserialize_from_base64(serializer, *write["value"]),
            task_path=write["task_path"],
            write_idx=write["write_idx"],
        )
        for write in pending_writes
    ]


def process_write_operations(
    writes: Sequence[tuple[str, Any]],
    task_id: str,
    pending_writes: list[SessionPendingWrite],
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_path: str,
    serializer,
) -> tuple[list[BedrockSessionContentBlock], bool]:
    """Process write operations and generate content blocks.

    Args:
        writes: List of (channel, value) tuples to process
        task_id: Task identifier
        pending_writes: Current pending writes
        thread_id: Thread identifier
        checkpoint_ns: Checkpoint namespace
        checkpoint_id: Checkpoint identifier
        task_path: Task path
        serializer: Serializer instance for value serialization

    Returns:
        Tuple of (list of content blocks, boolean indicating if new writes were created)
    """
    content_blocks = []
    new_writes = False

    # Convert current writes to dict for faster lookup
    current_writes_dict = {
        (write.task_id, write.write_idx): write for write in pending_writes
    }

    for idx, (channel, value) in enumerate(writes):
        write_idx: int = WRITES_IDX_MAP.get(channel, idx)
        inner_key: tuple[str, int] = (task_id, write_idx)

        # Skip if write already exists and has valid index
        if write_idx >= 0 and current_writes_dict and inner_key in current_writes_dict:
            pending_write = current_writes_dict[inner_key]
        else:
            new_writes = True
            pending_write = SessionPendingWrite(
                step_type=WRITES_PREFIX,
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
                task_id=task_id,
                task_path=task_path,
                write_idx=write_idx,
                channel=channel,
                value=serialize_to_base64(serializer, value),
            )

        content_blocks.append(
            BedrockSessionContentBlock(text=pending_write.model_dump_json())
        )

    return content_blocks, new_writes


def process_aws_client_args(
    region_name: Optional[str] = None,
    credentials_profile_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    config: Optional[Config] = None,
) -> Tuple[dict, dict]:
    """
    Process AWS client arguments and return session and client kwargs.

    Args:
        region_name: AWS region name
        credentials_profile_name: AWS credentials profile name
        aws_access_key_id: AWS access key ID
        aws_secret_access_key: AWS secret access key
        aws_session_token: AWS session token
        endpoint_url: Custom endpoint URL
        config: Boto3 config object

    Returns:
        Tuple[dict, dict]: Session kwargs and client kwargs
    """
    session_kwargs = {}
    client_kwargs = {}

    # Session parameters
    if region_name is not None:
        session_kwargs["region_name"] = region_name
    if credentials_profile_name is not None:
        session_kwargs["profile_name"] = credentials_profile_name
    if aws_access_key_id is not None:
        session_kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key is not None:
        session_kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_session_token is not None:
        session_kwargs["aws_session_token"] = aws_session_token

    # Client parameters
    if endpoint_url is not None:
        client_kwargs["endpoint_url"] = endpoint_url

    client_kwargs["config"] = create_client_config(config)

    return session_kwargs, client_kwargs


def create_client_config(config: Optional[Config] = None) -> Config:
    """
    Creates a client config with SDK user agent while preserving existing config settings.

    Args:
        config: Existing Boto3 config object

    Returns:
        Config: New config object with combined user agent
    """
    config_kwargs = {}
    existing_user_agent = getattr(config, "user_agent_extra", "") if config else ""
    new_user_agent = (
        f"{existing_user_agent} x-client-framework:langgraph-checkpoint-aws "
        f"md/sdk_user_agent/{SDK_USER_AGENT}".strip()
    )

    return Config(user_agent_extra=new_user_agent, **config_kwargs)


async def run_boto3_in_executor(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a boto3 function in an executor to prevent blocking the event loop."""

    return await asyncio.get_running_loop().run_in_executor(
        None,
        cast(
            "Callable[..., T]",
            partial(copy_context().run, lambda: func(*args, **kwargs)),
        ),
    )


def _validate_bedrock_client(client: Any) -> None:
    """Validate that the provided client is a bedrock-agent-runtime client."""
    try:
        service_name = client.meta.service_model.service_name
    except AttributeError:
        raise ValueError("Invalid client: must be a boto3 client instance")

    if service_name != "bedrock-agent-runtime":
        raise ValueError(
            f"Invalid client: expected 'bedrock-agent-runtime' client, got '{service_name}' client. "
            "Please provide a bedrock-agent-runtime client."
        )
