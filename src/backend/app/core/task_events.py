"""Task runtime event taxonomy.

Every action in the orchestration system — user input, model reasoning,
tool call, state transition — is serialized as an immutable event conforming
to one of the schemas in this module. Events are appended to a per-task log
and never mutated; all state (UI trace, FSM, resume, audit) is derived from
the log.

Design rules:
- Payloads are strongly typed via Pydantic. Validation happens on write, not
  on read.
- Tool outputs are capped in size; large artifacts live on disk and events
  carry a reference (artifact_ref) rather than the raw bytes.
- Ordering is by (sequence) not (timestamp). Timestamp is for humans.
- causal_parents lets parallel events (a parallel_group of tool calls) be
  reconstructed into a DAG even though they're logged as a flat sequence.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class Actor(str, Enum):
    USER = "USER"
    DEEPSEEK = "DEEPSEEK"
    NEMOTRON = "NEMOTRON"
    TOOL_WRAPPER = "TOOL_WRAPPER"
    SYSTEM_FSM = "SYSTEM_FSM"
    SYSTEM_CIRCUIT_BREAKER = "SYSTEM_CIRCUIT_BREAKER"
    SYSTEM_COMPACTION = "SYSTEM_COMPACTION"
    SYSTEM_SCHEDULER = "SYSTEM_SCHEDULER"
    SYSTEM_MIGRATION = "SYSTEM_MIGRATION"


class EventType(str, Enum):
    # User
    USER_PROMPT = "USER_PROMPT"
    USER_RESPONSE = "USER_RESPONSE"
    USER_CANCELLED = "USER_CANCELLED"
    # Init flow
    INIT_QUESTION = "INIT_QUESTION"
    INIT_ANSWER = "INIT_ANSWER"
    CONTEXT_WRITTEN = "CONTEXT_WRITTEN"
    # Planning (DeepSeek)
    MANIFEST_SCOPED = "MANIFEST_SCOPED"
    SUPERVISOR_BRIEF = "SUPERVISOR_BRIEF"
    GOAL_BRIEF_REVISION = "GOAL_BRIEF_REVISION"
    # Execution
    TOOL_CALL_INTENT = "TOOL_CALL_INTENT"
    TOOL_EXECUTION_RESULT = "TOOL_EXECUTION_RESULT"
    TOOL_YIELD = "TOOL_YIELD"
    ARTIFACT_WRITTEN = "ARTIFACT_WRITTEN"
    # Review (DeepSeek)
    SUPERVISOR_REVIEW = "SUPERVISOR_REVIEW"
    FINAL_ANSWER = "FINAL_ANSWER"
    # System
    STATE_TRANSITION = "STATE_TRANSITION"
    SYSTEM_CIRCUIT_BREAKER = "SYSTEM_CIRCUIT_BREAKER"
    LOG_COMPACTED = "LOG_COMPACTED"
    SYSTEM_PLUGIN_VERSION_DRIFT = "SYSTEM_PLUGIN_VERSION_DRIFT"


class ContentType(str, Enum):
    TEXT = "text"
    FILE_REF = "file_ref"
    STREAM = "stream"
    STRUCTURED = "structured"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR_TRANSIENT = "error_transient"
    ERROR_PERMANENT = "error_permanent"
    REQUIRES_HUMAN = "requires_human"


# ---------------------------------------------------------------------------
# Payload schemas — one per EventType
# ---------------------------------------------------------------------------


class UserPromptPayload(BaseModel):
    content_type: ContentType = ContentType.TEXT
    content: str
    attachments: List[str] = Field(default_factory=list)  # artifact refs


class UserResponsePayload(BaseModel):
    in_response_to: str  # event_id of the question (INIT_QUESTION or requires_human)
    content_type: ContentType = ContentType.TEXT
    content: str
    attachments: List[str] = Field(default_factory=list)


class UserCancelledPayload(BaseModel):
    reason: Optional[str] = None


class InitQuestionPayload(BaseModel):
    question_id: str
    source_plugin: Optional[str] = None
    question: str
    expected_format: Literal["text", "choice", "file", "structured"] = "text"
    choices: Optional[List[str]] = None
    required: bool = True


class InitAnswerPayload(BaseModel):
    in_response_to: str  # event_id of the INIT_QUESTION
    content: str


class ContextWrittenPayload(BaseModel):
    path: str
    document_type: Literal["CONTEXT_MD", "GOALS_MD"]
    source_plugins: List[str] = Field(default_factory=list)


class ManifestScopedPayload(BaseModel):
    candidate_tools: List[str]
    selected_tools: List[str]
    scoping_reasoning: str


class SupervisorBriefPayload(BaseModel):
    brief_id: str
    goal_statement: str
    definition_of_done: str
    active_manifest: List[str]
    constraints: Dict[str, Any] = Field(default_factory=dict)


class GoalBriefRevisionPayload(BaseModel):
    brief_id: str
    parent_brief_id: str
    amendment: str
    reason: str
    active_manifest: Optional[List[str]] = None  # None = inherit from parent


class ToolCallIntentPayload(BaseModel):
    call_id: str
    tool_name: str
    plugin_version: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    parallel_group_id: Optional[str] = None


class ToolOutput(BaseModel):
    summary: str
    artifact_ref: Optional[str] = None  # path to full output on disk
    truncated: bool = False


class ToolExecutionResultPayload(BaseModel):
    call_id: str
    status: ToolStatus
    output: ToolOutput
    execution_time_ms: int = 0
    error_detail: Optional[str] = None  # machine-readable error info for permanent errors


class ToolYieldPayload(BaseModel):
    reason: str  # free-form — DeepSeek classifies
    last_observation_ref: Optional[str] = None  # event_id of last TOOL_EXECUTION_RESULT
    suggested_options: Optional[List[str]] = None


class ArtifactWrittenPayload(BaseModel):
    call_id: str
    path: str
    bytes: int
    mime_type: Optional[str] = None
    summary: str = ""


class SupervisorReviewPayload(BaseModel):
    verdict: Literal["approve", "revise", "block"]
    reasoning: str


class FinalAnswerPayload(BaseModel):
    answer: str
    cited_event_ids: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None


class StateTransitionPayload(BaseModel):
    from_state: str
    to_state: str
    trigger_event_id: Optional[str] = None


class CircuitBreakerReason(str, Enum):
    LOOP_LIMIT_EXCEEDED = "loop_limit_exceeded"
    ERROR_THRESHOLD_EXCEEDED = "error_threshold_exceeded"
    TOKEN_BUDGET_EXCEEDED = "token_budget_exceeded"
    FATAL_WRAPPER_CRASH = "fatal_wrapper_crash"


class SystemCircuitBreakerPayload(BaseModel):
    reason: CircuitBreakerReason
    context: Dict[str, Any] = Field(default_factory=dict)


class LogCompactedPayload(BaseModel):
    range_start_seq: int
    range_end_seq: int
    summary: str
    key_facts: List[str] = Field(default_factory=list)
    raw_archive_ref: Optional[str] = None


class SystemPluginVersionDriftPayload(BaseModel):
    plugin_name: str
    recorded_version: str
    loaded_version: str
    severity: Literal["patch", "minor", "major"]


# Lookup map: event_type → payload model
PAYLOAD_SCHEMAS: Dict[EventType, type[BaseModel]] = {
    EventType.USER_PROMPT: UserPromptPayload,
    EventType.USER_RESPONSE: UserResponsePayload,
    EventType.USER_CANCELLED: UserCancelledPayload,
    EventType.INIT_QUESTION: InitQuestionPayload,
    EventType.INIT_ANSWER: InitAnswerPayload,
    EventType.CONTEXT_WRITTEN: ContextWrittenPayload,
    EventType.MANIFEST_SCOPED: ManifestScopedPayload,
    EventType.SUPERVISOR_BRIEF: SupervisorBriefPayload,
    EventType.GOAL_BRIEF_REVISION: GoalBriefRevisionPayload,
    EventType.TOOL_CALL_INTENT: ToolCallIntentPayload,
    EventType.TOOL_EXECUTION_RESULT: ToolExecutionResultPayload,
    EventType.TOOL_YIELD: ToolYieldPayload,
    EventType.ARTIFACT_WRITTEN: ArtifactWrittenPayload,
    EventType.SUPERVISOR_REVIEW: SupervisorReviewPayload,
    EventType.FINAL_ANSWER: FinalAnswerPayload,
    EventType.STATE_TRANSITION: StateTransitionPayload,
    EventType.SYSTEM_CIRCUIT_BREAKER: SystemCircuitBreakerPayload,
    EventType.LOG_COMPACTED: LogCompactedPayload,
    EventType.SYSTEM_PLUGIN_VERSION_DRIFT: SystemPluginVersionDriftPayload,
}


# ---------------------------------------------------------------------------
# Event envelope (what actually gets written / streamed)
# ---------------------------------------------------------------------------


class TaskEventDTO(BaseModel):
    """Serialized shape of a task event — for API, SSE, replay."""
    event_id: str
    task_id: str
    sequence: int
    schema_version: int = SCHEMA_VERSION
    timestamp: datetime
    actor: Actor
    event_type: EventType
    causal_parents: List[str] = Field(default_factory=list)
    payload: Dict[str, Any]


def validate_payload(event_type: EventType, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate payload against its schema; return the validated dict form.

    Raises ValidationError if the payload doesn't match its event_type schema.
    """
    schema = PAYLOAD_SCHEMAS.get(event_type)
    if schema is None:
        raise ValueError(f"Unknown event_type: {event_type}")
    return schema.model_validate(payload).model_dump(mode="json")
