"""Finite state machine for task orchestration.

The FSM is a pure function of events: given the current state + a trigger,
it computes the next state and its guards. It does not own state — that
lives in the Task row and the event log. Callers read the current state,
ask the FSM whether a transition is legal, and if so emit a
STATE_TRANSITION event via the log; the log + Task row together are the
persisted truth.

States (9 total):
    IDLE, INITIALIZING, PLANNING, EXECUTING, REVIEWING, SUSPENDED,
    COMPLETED, CANCELLED, FAILED

The three terminals (COMPLETED / CANCELLED / FAILED) are structurally
distinct so audit + analytics can tell "user aborted" from "framework
gave up" from "task succeeded." Cancel is reachable from every non-terminal
state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


TERMINAL_STATES: Set[TaskState] = {
    TaskState.COMPLETED,
    TaskState.CANCELLED,
    TaskState.FAILED,
}


class Trigger(str, Enum):
    """Named reasons for a state change.

    Used in STATE_TRANSITION events so audit can reconstruct *why* the FSM
    moved, not just that it did.
    """
    # From IDLE
    OPEN_FRESH_TASK = "open_fresh_task"            # idle → initializing (no context yet)
    USER_PROMPT_RECEIVED = "user_prompt_received"  # idle → planning

    # Init flow
    INIT_QUESTION_ANSWERED = "init_question_answered"
    INIT_COMPLETE = "init_complete"                # initializing → idle

    # Planning
    BRIEF_READY = "brief_ready"                    # planning → executing
    NEED_USER_INFO = "need_user_info"              # planning → suspended
    PLAN_GENERATION_FAILED = "plan_generation_failed"  # planning → failed

    # Execution (intra-state)
    TOOL_CALL = "tool_call"                        # executing → executing
    # Execution exits
    FINAL_ANSWER_CANDIDATE = "final_answer_candidate"  # executing → reviewing
    TOOL_YIELD = "tool_yield"                      # executing → reviewing
    TOOLKIT_INSUFFICIENT = "toolkit_insufficient"  # executing → planning (re-scope)
    REQUIRES_HUMAN = "requires_human"              # executing → suspended
    CIRCUIT_BREAKER = "circuit_breaker"            # executing → reviewing
    EXECUTION_FATAL = "execution_fatal"            # executing → failed

    # Review
    REVIEW_APPROVE = "review_approve"              # reviewing → completed
    REVIEW_REVISE = "review_revise"                # reviewing → executing
    REVIEW_RESCOPE = "review_rescope"              # reviewing → planning
    REVIEW_ASK_USER = "review_ask_user"            # reviewing → suspended

    # Suspend exits (classifier picks one)
    USER_RESUME = "user_resume"                    # suspended → executing
    USER_REPLAN = "user_replan"                    # suspended → planning
    USER_TIMEOUT = "user_timeout"                  # suspended → cancelled

    # User-initiated cancel (reachable from all non-terminal states)
    USER_CANCEL = "user_cancel"


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

TransitionKey = Tuple[TaskState, Trigger]

_TRANSITIONS: Dict[TransitionKey, TaskState] = {
    # IDLE
    (TaskState.IDLE, Trigger.OPEN_FRESH_TASK): TaskState.INITIALIZING,
    (TaskState.IDLE, Trigger.USER_PROMPT_RECEIVED): TaskState.PLANNING,
    (TaskState.IDLE, Trigger.USER_CANCEL): TaskState.CANCELLED,

    # INITIALIZING
    (TaskState.INITIALIZING, Trigger.INIT_QUESTION_ANSWERED): TaskState.INITIALIZING,
    (TaskState.INITIALIZING, Trigger.INIT_COMPLETE): TaskState.IDLE,
    (TaskState.INITIALIZING, Trigger.USER_CANCEL): TaskState.CANCELLED,

    # PLANNING
    (TaskState.PLANNING, Trigger.BRIEF_READY): TaskState.EXECUTING,
    (TaskState.PLANNING, Trigger.NEED_USER_INFO): TaskState.SUSPENDED,
    (TaskState.PLANNING, Trigger.PLAN_GENERATION_FAILED): TaskState.FAILED,
    (TaskState.PLANNING, Trigger.USER_CANCEL): TaskState.CANCELLED,

    # EXECUTING
    (TaskState.EXECUTING, Trigger.TOOL_CALL): TaskState.EXECUTING,
    (TaskState.EXECUTING, Trigger.FINAL_ANSWER_CANDIDATE): TaskState.REVIEWING,
    (TaskState.EXECUTING, Trigger.TOOL_YIELD): TaskState.REVIEWING,
    (TaskState.EXECUTING, Trigger.TOOLKIT_INSUFFICIENT): TaskState.PLANNING,
    (TaskState.EXECUTING, Trigger.REQUIRES_HUMAN): TaskState.SUSPENDED,
    (TaskState.EXECUTING, Trigger.CIRCUIT_BREAKER): TaskState.REVIEWING,
    (TaskState.EXECUTING, Trigger.EXECUTION_FATAL): TaskState.FAILED,
    (TaskState.EXECUTING, Trigger.USER_CANCEL): TaskState.CANCELLED,

    # REVIEWING
    (TaskState.REVIEWING, Trigger.REVIEW_APPROVE): TaskState.COMPLETED,
    (TaskState.REVIEWING, Trigger.REVIEW_REVISE): TaskState.EXECUTING,
    (TaskState.REVIEWING, Trigger.REVIEW_RESCOPE): TaskState.PLANNING,
    (TaskState.REVIEWING, Trigger.REVIEW_ASK_USER): TaskState.SUSPENDED,
    (TaskState.REVIEWING, Trigger.USER_CANCEL): TaskState.CANCELLED,

    # SUSPENDED
    (TaskState.SUSPENDED, Trigger.USER_RESUME): TaskState.EXECUTING,
    (TaskState.SUSPENDED, Trigger.USER_REPLAN): TaskState.PLANNING,
    (TaskState.SUSPENDED, Trigger.USER_TIMEOUT): TaskState.CANCELLED,
    (TaskState.SUSPENDED, Trigger.USER_CANCEL): TaskState.CANCELLED,
}


# ---------------------------------------------------------------------------
# Guards — preconditions attached to specific transitions
# ---------------------------------------------------------------------------


@dataclass
class GuardContext:
    """Everything a guard needs to decide. Keeps guards pure functions."""
    task_id: str
    current_state: TaskState
    trigger: Trigger
    metadata: Dict[str, object]


GuardFn = Callable[[GuardContext], Optional[str]]
"""A guard returns None to allow, or an error message to reject."""


def _require_metadata(keys: List[str]) -> GuardFn:
    def guard(ctx: GuardContext) -> Optional[str]:
        missing = [k for k in keys if k not in ctx.metadata]
        if missing:
            return f"missing metadata keys: {missing}"
        return None
    return guard


_GUARDS: Dict[Trigger, GuardFn] = {
    Trigger.BRIEF_READY: _require_metadata(["brief_id", "active_manifest"]),
    Trigger.USER_PROMPT_RECEIVED: _require_metadata(["prompt"]),
    Trigger.TOOL_CALL: _require_metadata(["tool_name"]),
    Trigger.FINAL_ANSWER_CANDIDATE: _require_metadata(["answer"]),
    Trigger.TOOL_YIELD: _require_metadata(["reason"]),
    Trigger.REQUIRES_HUMAN: _require_metadata(["question"]),
    Trigger.REVIEW_APPROVE: _require_metadata(["answer"]),
    Trigger.REVIEW_REVISE: _require_metadata(["amendment"]),
    Trigger.CIRCUIT_BREAKER: _require_metadata(["breaker_reason"]),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TransitionError(Exception):
    """Raised when a requested transition is not legal."""


@dataclass
class TransitionResult:
    from_state: TaskState
    to_state: TaskState
    trigger: Trigger


def is_terminal(state: TaskState) -> bool:
    return state in TERMINAL_STATES


def legal_transitions(state: TaskState) -> List[Trigger]:
    """All triggers that are legal from a given state."""
    return [trigger for (src, trigger) in _TRANSITIONS.keys() if src == state]


def next_state(state: TaskState, trigger: Trigger) -> Optional[TaskState]:
    return _TRANSITIONS.get((state, trigger))


def transition(
    current: TaskState,
    trigger: Trigger,
    task_id: str,
    metadata: Optional[Dict[str, object]] = None,
) -> TransitionResult:
    """Attempt a transition; raise TransitionError if not legal."""
    metadata = metadata or {}
    dest = next_state(current, trigger)
    if dest is None:
        raise TransitionError(
            f"No legal transition from {current.value} via {trigger.value}"
        )

    guard = _GUARDS.get(trigger)
    if guard is not None:
        ctx = GuardContext(
            task_id=task_id,
            current_state=current,
            trigger=trigger,
            metadata=metadata,
        )
        err = guard(ctx)
        if err is not None:
            raise TransitionError(
                f"Guard failed for {current.value} -> {dest.value} via {trigger.value}: {err}"
            )

    return TransitionResult(from_state=current, to_state=dest, trigger=trigger)
