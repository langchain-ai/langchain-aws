from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, constr
from pydantic.alias_generators import to_camel


class BedrockSessionBaseModel(BaseModel):
    """Base model for all Bedrock models.

    All models in this package inherit from this base class which provides common
    configuration including camelCase alias generation and frozen model instances.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        frozen=True,
        extra="ignore",
    )


class SessionIdentifierRequest(BedrockSessionBaseModel):
    """Request model for retrieving session details.

    Attributes:
        session_identifier: UUID or ARN representing the session identifier
    """

    session_identifier: str = Field(..., description="Required session identifier")


class SessionIdentifierResponse(BedrockSessionBaseModel):
    """Response model for session retrieval request.

    Attributes:
        session_id: UUID of the session
        session_arn: ARN of the session
        session_status: Current status of the session
        created_at: Timestamp of session creation
        last_updated_at: Timestamp of last session update
    """

    session_id: str
    session_arn: str
    session_status: str
    created_at: datetime
    last_updated_at: datetime


class CreateSessionRequest(BedrockSessionBaseModel):
    """Request model for creating a new Bedrock session.

    Attributes:
        session_metadata: Optional metadata associated with the session
        encryption_key_arn: Optional ARN of encryption key
        tags: Optional key-value pairs for tagging the session
    """

    session_metadata: Optional[dict[str, str]] = None
    encryption_key_arn: Optional[str] = None
    tags: Optional[dict[str, str]] = None


class CreateSessionResponse(BedrockSessionBaseModel):
    """Response model for session creation request.

    Attributes:
        session_id: UUID representing the session identifier
        session_arn: ARN of the created session
        created_at: Timestamp when the session was created
        session_status: Current status of the session
    """

    session_id: str
    session_arn: str
    created_at: datetime
    session_status: str


class GetSessionRequest(SessionIdentifierRequest):
    """Request model for retrieving session details.

    Attributes:
        session_identifier: UUID or ARN representing the session identifier
    """

    pass


class GetSessionResponse(SessionIdentifierResponse):
    """Response model for session retrieval request.

    Attributes:
        session_id: UUID representing the session identifier
        session_arn: ARN of the session
        created_at: Timestamp when the session was created
        session_metadata: Optional metadata associated with the session
        encryption_key_arn: Optional ARN of the encryption key used
        session_status: Current status of the session
        last_updated_at: Timestamp when the session was last updated
    """

    session_metadata: Optional[dict[str, str]] = None
    encryption_key_arn: Optional[str] = None


class EndSessionRequest(SessionIdentifierRequest):
    """Request model for ending an existing session.

    Attributes:
        session_identifier: Unique identifier of the session to end
    """

    pass


class EndSessionResponse(BedrockSessionBaseModel):
    """Response model for ending an existing session.

    Attributes:
        session_id: Unique identifier of the session to end
        session_arn: ARN of the session
        session_status: Current status of the session
    """

    session_id: str
    session_arn: str
    session_status: str


class DeleteSessionRequest(SessionIdentifierRequest):
    """Request model for deleting an existing session.

    Attributes:
        session_identifier: Unique identifier of the session to delete
    """

    pass


class InvocationSummary(BedrockSessionBaseModel):
    """Model representing a summary of an invocation.

    Attributes:
        session_id: UUID representing the session identifier
        invocation_id: UUID representing the invocation identifier
        created_at: Timestamp when the invocation was created
    """

    session_id: str
    invocation_id: str
    created_at: datetime


class InvocationIdentifierRequest(SessionIdentifierRequest):
    """Request model for retrieving invocation details.

    Attributes:
        session_identifier: Unique identifier of the session
        invocation_identifier: Required invocation identifier
    """

    invocation_identifier: str = Field(
        ..., description="Required invocation identifier"
    )


class CreateInvocationRequest(SessionIdentifierRequest):
    """Request model for creating a new invocation.

    Attributes:
        session_identifier: UUID or ARN representing the session identifier
        invocation_id: Optional custom UUID for the invocation
        description: Optional description of the invocation
    """

    invocation_id: Optional[str] = None
    description: Optional[str] = None


class CreateInvocationResponse(BedrockSessionBaseModel):
    """Response model for invocation creation.

    Attributes:
        invocation_id: UUID representing the invocation identifier
    """

    invocation_id: str


class ListInvocationsRequest(SessionIdentifierRequest):
    """Request model for listing invocations.

    Attributes:
        session_identifier: UUID or ARN representing the session identifier
        next_token: Optional token for retrieving next page of results
        max_results: Optional maximum number of results to return (1-100)
    """

    next_token: Optional[str] = None
    max_results: Optional[int] = None


class ListInvocationsResponse(BedrockSessionBaseModel):
    """Response model for listing invocations.

    Attributes:
        invocation_summaries: List of invocation summaries
        next_token: Optional token for retrieving next page of results
    """

    invocation_summaries: list[InvocationSummary]
    next_token: Optional[str] = None


class InvocationStepIdentifierRequest(InvocationIdentifierRequest):
    """Request model for retrieving invocation step details.

    Attributes:
        session_identifier: UUID or ARN representing the session identifier
        invocation_identifier: UUID representing the invocation identifier
        invocation_step_id: UUID representing the step identifier
    """

    invocation_step_id: str = Field(
        ..., description="Unique identifier of the invocation step"
    )


class BedrockSessionContentBlock(BedrockSessionBaseModel):
    """Content block for a Bedrock Agent response.

    Attributes:
        text: Optional text content of the block with minimum length of 1
    """

    text: Optional[constr(min_length=1)] = None


class InvocationStepPayload(BedrockSessionBaseModel):
    """The payload for an invocation step.

    Attributes:
        content_blocks: List of content blocks contained in the payload
    """

    content_blocks: Optional[list[BedrockSessionContentBlock]] = Field(
        None, min_length=1
    )


class InvocationStepSummary(BedrockSessionBaseModel):
    """Response model for retrieving invocation step details.

    Attributes:
        invocation_id: UUID representing the invocation identifier
        invocation_step_id: UUID representing the step identifier
        invocation_step_time: Timestamp when the step was created
    """

    invocation_id: str
    invocation_step_id: str
    invocation_step_time: datetime


class PutInvocationStepRequest(InvocationIdentifierRequest):
    """Request model for putting an invocation step.

    Attributes:
        session_identifier: UUID or ARN representing the session identifier
        invocation_identifier: UUID representing the invocation identifier
        invocation_step_id: UUID representing the step identifier
        invocation_step_time: Timestamp when the step was created
        payload: Payload containing content blocks for the step
    """

    invocation_step_id: Optional[str] = None
    invocation_step_time: datetime
    payload: InvocationStepPayload


class PutInvocationStepResponse(BedrockSessionBaseModel):
    """Response model for putting an invocation step.

    Attributes:
        invocation_step_id: UUID representing the step identifier
    """

    invocation_step_id: str


class GetInvocationStepRequest(InvocationStepIdentifierRequest):
    """Request model for retrieving invocation step details.

    Attributes:
        session_identifier: UUID or ARN representing the session identifier
        invocation_identifier: UUID representing the invocation identifier
        invocation_step_id: UUID representing the step identifier
    """

    pass


class InvocationStep(InvocationStepSummary):
    """Response model for retrieving invocation step details.

    Attributes:
        session_id: UUID representing the session identifier
        invocation_id: UUID representing the invocation identifier
        invocation_step_id: UUID representing the step identifier
        invocation_step_time: Timestamp when the step was created
        payload: Payload containing content blocks for the step
    """

    session_id: str
    payload: InvocationStepPayload


class GetInvocationStepResponse(BedrockSessionBaseModel):
    """Response model for retrieving invocation step details.

    Attributes:
        invocation_step: Payload containing content blocks for the step
    """

    invocation_step: InvocationStep


class ListInvocationStepsRequest(SessionIdentifierRequest):
    """Request model for listing invocation steps.

    Attributes:
        session_identifier: UUID or ARN representing the session identifier
        invocation_identifier: Optional UUID representing the invocation identifier
        next_token: Optional token for retrieving next page of results
        max_results: Optional maximum number of results to return (1-100)
    """

    invocation_identifier: Optional[str] = None
    next_token: Optional[str] = None
    max_results: Optional[int] = None


class ListInvocationStepsResponse(BedrockSessionBaseModel):
    """Response model for listing invocation steps.

    Attributes:
        invocation_step_summaries: List of invocation step summaries
        next_token: Optional token for retrieving next page of results
    """

    invocation_step_summaries: list[InvocationStepSummary]
    next_token: Optional[str] = None


class SessionPendingWrite(BaseModel):
    """Model representing a pending write operation in the bedrock session store."""

    step_type: str
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    task_id: str
    channel: str
    value: Any
    task_path: str
    write_idx: int


class SessionCheckpoint(BaseModel):
    """Model representing a checkpoint operation is the bedrock session store."""

    step_type: str
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    checkpoint: Any
    metadata: Any
    parent_checkpoint_id: Optional[str] = None
    channel_values: Any
    version: Any
