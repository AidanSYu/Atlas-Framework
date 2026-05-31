"""Task conductor — ties the FSM, event log, supervisor, and executor together.

Lifecycle:
    create_task()              → FSM=IDLE, writes Task row.
    start_task(prompt)         → IDLE→PLANNING→EXECUTING→(REVIEWING loop)→COMPLETED
                                  Emits every event. Runs to a terminal or SUSPENDED.
    respond_to_suspended(...)  → SUSPENDED→(classifier)→EXECUTING or PLANNING
    cancel_task()              → any non-terminal → CANCELLED

The task service owns the control loop; models own the reasoning. Every
state change emits a STATE_TRANSITION event *and* updates the Task row.
Persistence lives in the log — the Task row is a cached current-state for
fast dashboard reads.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core import registry
from app.core.database import Task, TaskEvent, get_project_session
from app.core.task_events import (
    Actor,
    CircuitBreakerReason,
    EventType,
    TaskEventDTO,
)
from app.core.task_fsm import (
    TaskState,
    TransitionError,
    Trigger,
    is_terminal,
    transition,
)
from app.core.task_log import TaskLog, get_task_log
from app.services.task_executor import ExecutorBrief, ExecutorExit, TaskExecutor, get_task_executor
from app.services.task_supervisor import (
    GoalBrief,
    TaskSupervisor,
    get_task_supervisor,
)

logger = logging.getLogger(__name__)


# In-process cancel flags, keyed by task_id. Set by cancel_task(), checked by loops.
_CANCEL_EVENTS: Dict[str, asyncio.Event] = {}
# In-process running-task handles so we can observe / cancel them.
_RUNNING: Dict[str, asyncio.Task] = {}


@dataclass
class TaskRecord:
    id: str
    project_id: str
    parent_task_id: Optional[str]
    title: Optional[str]
    initial_prompt: Optional[str]
    state: TaskState
    terminal_outcome: Optional[str]
    created_at: datetime
    updated_at: datetime


class TaskService:
    def __init__(
        self,
        log: Optional[TaskLog] = None,
        supervisor: Optional[TaskSupervisor] = None,
        executor: Optional[TaskExecutor] = None,
    ):
        self._log = log or get_task_log()
        self._supervisor = supervisor or get_task_supervisor()
        self._executor = executor or get_task_executor()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_task(
        self,
        project_id: str,
        title: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        parent_task_id: Optional[str] = None,
    ) -> TaskRecord:
        if registry.get_project(project_id) is None:
            raise ValueError(f"Project {project_id} not found")
        db: Session = get_project_session(project_id)
        try:
            task = Task(
                id=str(uuid.uuid4()),
                project_id=project_id,
                parent_task_id=parent_task_id,
                title=title,
                initial_prompt=initial_prompt,
                state=TaskState.IDLE.value,
                next_sequence=0,
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            return _row_to_record(task)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_task(self, project_id: str, task_id: str) -> Optional[TaskRecord]:
        db: Session = get_project_session(project_id)
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            return _row_to_record(task) if task else None
        finally:
            db.close()

    def find_task(self, task_id: str) -> Optional[TaskRecord]:
        """Scan every registered project for a task with this id (slow path).

        Used by routes that don't have project_id in scope (e.g. legacy URLs).
        Returns the first match — task ids are UUIDs so collisions across
        projects are statistically impossible.
        """
        for entry in registry.list_projects():
            rec = self.get_task(entry.id, task_id)
            if rec is not None:
                return rec
        return None

    def list_tasks(self, project_id: str) -> List[TaskRecord]:
        db: Session = get_project_session(project_id)
        try:
            rows = (
                db.query(Task)
                .order_by(Task.created_at.desc())
                .all()
            )
            return [_row_to_record(r) for r in rows]
        finally:
            db.close()

    def list_events(self, project_id: str, task_id: str) -> List[TaskEventDTO]:
        return self._log.list_events(project_id, task_id)

    # ------------------------------------------------------------------
    # Lifecycle entry points
    # ------------------------------------------------------------------

    async def start_task(
        self,
        project_id: str,
        task_id: str,
        user_prompt: str,
        attachments: Optional[List[str]] = None,
    ) -> TaskRecord:
        """Entry: record the prompt + kick orchestration off as a background task."""
        task = self._require_task(project_id, task_id)
        if task.state not in (TaskState.IDLE, TaskState.INITIALIZING, TaskState.COMPLETED):
            raise ValueError(f"Task {task_id} is not ready to start (state={task.state.value})")

        # Follow-up on a COMPLETED task: clear the prior terminal marker so
        # the row reads as live again. The FSM transition COMPLETED → PLANNING
        # below moves the state forward; this just drops the cached outcome.
        if task.state == TaskState.COMPLETED:
            self._clear_terminal_outcome(project_id, task_id)

        existing = _RUNNING.get(task_id)
        if existing is not None and not existing.done():
            raise ValueError(f"Task {task_id} is already running")

        attachment_paths = list(attachments or [])
        self._mark_task_prompt(project_id, task_id, user_prompt)

        self._log.append(
            project_id,
            task_id,
            Actor.USER,
            EventType.USER_PROMPT,
            {"content_type": "text", "content": user_prompt, "attachments": attachment_paths},
        )
        self._transition(project_id, task_id, task.state, Trigger.USER_PROMPT_RECEIVED, {"prompt": user_prompt})

        bg = asyncio.create_task(
            self._run_orchestration_safely(project_id, task_id, user_prompt, attachment_paths)
        )
        _RUNNING[task_id] = bg
        bg.add_done_callback(lambda _t, tid=task_id: _RUNNING.pop(tid, None))
        return self._require_task(project_id, task_id)

    async def _run_orchestration_safely(
        self, project_id: str, task_id: str, user_prompt: str, attachments: List[str]
    ) -> None:
        try:
            await self._run_planning_and_execute(
                project_id, task_id, user_prompt=user_prompt, attachments=attachments
            )
        except asyncio.CancelledError:
            logger.info("Orchestration cancelled for task %s", task_id)
            raise
        except Exception as exc:
            logger.exception("Orchestration crashed for task %s", task_id)
            try:
                self._log.append(
                    project_id,
                    task_id,
                    Actor.SYSTEM_CIRCUIT_BREAKER,
                    EventType.SYSTEM_CIRCUIT_BREAKER,
                    {
                        "reason": CircuitBreakerReason.FATAL_WRAPPER_CRASH.value,
                        "context": {"phase": "orchestration", "error": str(exc)},
                    },
                )
                self._set_terminal(project_id, task_id, "failed")
            except Exception:
                logger.exception("Failed to record orchestration crash for %s", task_id)

    async def respond_to_suspended(
        self,
        project_id: str,
        task_id: str,
        user_response: str,
        attachments: Optional[List[str]] = None,
    ) -> TaskRecord:
        task = self._require_task(project_id, task_id)
        if task.state != TaskState.SUSPENDED:
            raise ValueError(f"Task {task_id} is not SUSPENDED (state={task.state.value})")

        existing = _RUNNING.get(task_id)
        if existing is not None and not existing.done():
            raise ValueError(f"Task {task_id} is already running")

        attachment_paths = list(attachments or [])
        pending_question_event_id = _find_pending_question_event(
            self._log.list_events(project_id, task_id)
        )
        self._log.append(
            project_id,
            task_id,
            Actor.USER,
            EventType.USER_RESPONSE,
            {
                "in_response_to": pending_question_event_id or "",
                "content_type": "text",
                "content": user_response,
                "attachments": attachment_paths,
            },
        )

        bg = asyncio.create_task(
            self._handle_suspend_response_safely(project_id, task_id, user_response, attachment_paths)
        )
        _RUNNING[task_id] = bg
        bg.add_done_callback(lambda _t, tid=task_id: _RUNNING.pop(tid, None))
        return self._require_task(project_id, task_id)

    async def _handle_suspend_response_safely(
        self, project_id: str, task_id: str, user_response: str, attachments: List[str]
    ) -> None:
        try:
            brief = self._load_active_brief(project_id, task_id)
            question = _extract_question_text(self._log.list_events(project_id, task_id))
            if brief is None or question is None:
                self._transition(project_id, task_id, TaskState.SUSPENDED, Trigger.USER_REPLAN)
                await self._run_planning_and_execute(
                    project_id, task_id, user_prompt=user_response, attachments=attachments
                )
                return
            classification = await self._supervisor.classify_user_response(
                question=question, response=user_response, brief=brief
            )
            if classification.classification == "resume":
                self._transition(project_id, task_id, TaskState.SUSPENDED, Trigger.USER_RESUME)
                await self._run_executor_loop(
                    project_id,
                    task_id,
                    brief,
                    resume_note=classification.synthetic_tool_result or user_response,
                    attachments=attachments,
                )
            else:
                self._transition(project_id, task_id, TaskState.SUSPENDED, Trigger.USER_REPLAN)
                await self._run_planning_and_execute(
                    project_id, task_id, user_prompt=user_response, attachments=attachments
                )
        except asyncio.CancelledError:
            logger.info("Suspend-response handling cancelled for task %s", task_id)
            raise
        except Exception as exc:
            logger.exception("Suspend-response handling crashed for task %s", task_id)
            try:
                self._log.append(
                    project_id,
                    task_id,
                    Actor.SYSTEM_CIRCUIT_BREAKER,
                    EventType.SYSTEM_CIRCUIT_BREAKER,
                    {
                        "reason": CircuitBreakerReason.FATAL_WRAPPER_CRASH.value,
                        "context": {"phase": "respond_to_suspended", "error": str(exc)},
                    },
                )
                self._set_terminal(project_id, task_id, "failed")
            except Exception:
                logger.exception("Failed to record suspend-response crash for %s", task_id)

    def delete_task(self, project_id: str, task_id: str) -> bool:
        """Hard-delete a task: cancel if running, drop events, attachments, row.

        Order matters: stop the orchestration first so it can't append more
        events after we've purged the log; then drop disk artifacts; then
        delete DB rows (events before the task row to satisfy the FK from
        TaskEvent.task_id, and orphan any child tasks rather than recursing).
        """
        # 1. Cancel in-flight orchestration. We don't await the bg task — its
        # next checkpoint will see the cancel event / CancelledError and exit;
        # any straggler append() after the row is gone will raise ValueError
        # inside the bg task and be swallowed by _run_orchestration_safely.
        cancel_evt = _CANCEL_EVENTS.pop(task_id, None)
        if cancel_evt is not None:
            cancel_evt.set()
        running = _RUNNING.pop(task_id, None)
        if running is not None and not running.done():
            running.cancel()

        # 2. Drop in-process log state (SSE subscribers, per-task sequence lock).
        self._log.purge_task(task_id)

        # 3. Delete attachment folder on disk (best-effort; missing is fine).
        try:
            from app.services.workspace_manager import get_workspace_manager
            attachments_dir = get_workspace_manager().task_attachments_path(project_id, task_id)
            if attachments_dir.exists():
                shutil.rmtree(attachments_dir, ignore_errors=True)
        except Exception:
            logger.exception("Failed to remove attachments for task %s", task_id)

        # 4. DB rows. Orphan any child tasks (we don't cascade delete a tree).
        db: Session = get_project_session(project_id)
        try:
            db.query(Task).filter(Task.parent_task_id == task_id).update(
                {Task.parent_task_id: None}, synchronize_session=False
            )
            db.query(TaskEvent).filter(TaskEvent.task_id == task_id).delete(synchronize_session=False)
            deleted = db.query(Task).filter(Task.id == task_id).delete(synchronize_session=False)
            db.commit()
            return deleted > 0
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    async def cancel_task(
        self, project_id: str, task_id: str, reason: Optional[str] = None
    ) -> TaskRecord:
        task = self._require_task(project_id, task_id)
        if is_terminal(task.state):
            return task
        evt = _CANCEL_EVENTS.get(task_id)
        if evt is not None:
            evt.set()
        running = _RUNNING.get(task_id)
        if running is not None and not running.done():
            running.cancel()
        self._log.append(
            project_id,
            task_id,
            Actor.USER,
            EventType.USER_CANCELLED,
            {"reason": reason or "user-initiated"},
        )
        self._transition(project_id, task_id, task.state, Trigger.USER_CANCEL)
        self._set_terminal(project_id, task_id, "cancelled")
        return self._require_task(project_id, task_id)

    # ------------------------------------------------------------------
    # Internal orchestration loop
    # ------------------------------------------------------------------

    async def _run_planning_and_execute(
        self,
        project_id: str,
        task_id: str,
        user_prompt: str,
        attachments: Optional[List[str]] = None,
    ) -> None:
        """PLANNING → EXECUTING → (REVIEWING loop)."""
        try:
            scoped = await self._supervisor.scope_manifest(user_prompt)
        except Exception as exc:
            logger.exception("Planning failed for task %s", task_id)
            self._transition(project_id, task_id, TaskState.PLANNING, Trigger.PLAN_GENERATION_FAILED)
            self._set_terminal(project_id, task_id, "failed")
            self._log.append(
                project_id,
                task_id,
                Actor.SYSTEM_CIRCUIT_BREAKER,
                EventType.SYSTEM_CIRCUIT_BREAKER,
                {"reason": CircuitBreakerReason.FATAL_WRAPPER_CRASH.value, "context": {"phase": "scope_manifest", "error": str(exc)}},
            )
            return

        # Single-orchestrator: the supervisor is deterministic and would emit
        # MANIFEST_SCOPED with the full catalog and SUPERVISOR_BRIEF with the raw
        # user prompt — pure filler. We still build the brief (the executor needs
        # it), but skip writing the trivial events to keep the UI uncluttered
        # and avoid telling the user "I scoped 15 tools" before any real work
        # happens.

        try:
            brief = await self._supervisor.build_brief(user_prompt, scoped)
        except Exception:
            logger.exception("Brief generation failed for task %s", task_id)
            self._transition(project_id, task_id, TaskState.PLANNING, Trigger.PLAN_GENERATION_FAILED)
            self._set_terminal(project_id, task_id, "failed")
            return

        self._transition(
            project_id,
            task_id,
            TaskState.PLANNING,
            Trigger.BRIEF_READY,
            {"brief_id": brief.brief_id, "active_manifest": brief.active_manifest},
        )

        await self._run_executor_loop(project_id, task_id, brief, attachments=attachments)

    async def _run_executor_loop(
        self,
        project_id: str,
        task_id: str,
        brief: GoalBrief,
        resume_note: Optional[str] = None,
        attachments: Optional[List[str]] = None,
    ) -> None:
        """EXECUTING → REVIEWING cycles until terminal, SUSPENDED, or rescope."""
        max_review_cycles = 4

        all_attachments = _collect_attachments(self._log.list_events(project_id, task_id))
        if attachments:
            all_attachments = _dedupe_preserving_order(list(attachments) + all_attachments)

        for _cycle in range(max_review_cycles):
            cancel_event = _ensure_cancel_event(task_id)
            exec_brief = ExecutorBrief(
                task_id=task_id,
                project_id=project_id,
                goal_statement=brief.goal_statement,
                definition_of_done=brief.definition_of_done,
                active_manifest=brief.active_manifest,
                constraints=brief.constraints,
                attachments=all_attachments,
            )
            if resume_note:
                exec_brief.goal_statement = (
                    f"{brief.goal_statement}\n\n[User clarification: {resume_note}]"
                )
                resume_note = None

            result = await self._executor.run(exec_brief, cancel_event=cancel_event)

            if cancel_event.is_set():
                return

            if result.exit_reason == ExecutorExit.FINAL_ANSWER:
                self._log.append(
                    project_id,
                    task_id,
                    Actor.ORCHESTRATOR,
                    EventType.FINAL_ANSWER,
                    {"answer": result.final_answer or "", "cited_event_ids": []},
                )
                self._transition(
                    project_id,
                    task_id,
                    TaskState.EXECUTING,
                    Trigger.FINAL_ANSWER_CANDIDATE,
                    {"answer": result.final_answer or ""},
                )
                proceed = await self._handle_review(
                    project_id, task_id, brief, candidate_answer=result.final_answer
                )
                if not proceed:
                    return
                continue

            if result.exit_reason == ExecutorExit.YIELD:
                self._transition(
                    project_id,
                    task_id,
                    TaskState.EXECUTING,
                    Trigger.TOOL_YIELD,
                    {"reason": result.yield_reason or ""},
                )
                proceed = await self._handle_review(
                    project_id, task_id, brief, yield_reason=result.yield_reason
                )
                if not proceed:
                    return
                continue

            if result.exit_reason == ExecutorExit.REQUIRES_HUMAN:
                self._transition(
                    project_id,
                    task_id,
                    TaskState.EXECUTING,
                    Trigger.REQUIRES_HUMAN,
                    {"question": result.requires_human_question or "A tool requires human input."},
                )
                self._log.append(
                    project_id,
                    task_id,
                    Actor.SUPERVISOR,
                    EventType.SUPERVISOR_REVIEW,
                    {
                        "verdict": "block",
                        "reasoning": f"Tool flagged requires_human: {result.requires_human_question}",
                    },
                )
                return

            if result.exit_reason in (ExecutorExit.CIRCUIT_BREAKER_LOOP, ExecutorExit.CIRCUIT_BREAKER_ERRORS):
                self._transition(
                    project_id,
                    task_id,
                    TaskState.EXECUTING,
                    Trigger.CIRCUIT_BREAKER,
                    {"breaker_reason": result.exit_reason},
                )
                proceed = await self._handle_review(
                    project_id, task_id, brief, circuit_breaker_reason=result.exit_reason
                )
                if not proceed:
                    return
                continue

            if result.exit_reason == ExecutorExit.FATAL:
                self._transition(project_id, task_id, TaskState.EXECUTING, Trigger.EXECUTION_FATAL)
                self._set_terminal(project_id, task_id, "failed")
                return

        self._log.append(
            project_id,
            task_id,
            Actor.SYSTEM_CIRCUIT_BREAKER,
            EventType.SYSTEM_CIRCUIT_BREAKER,
            {"reason": "loop_limit_exceeded", "context": {"phase": "review_cycle", "cycles": max_review_cycles}},
        )
        self._set_terminal(project_id, task_id, "failed")

    async def _handle_review(
        self,
        project_id: str,
        task_id: str,
        brief: GoalBrief,
        candidate_answer: Optional[str] = None,
        yield_reason: Optional[str] = None,
        circuit_breaker_reason: Optional[str] = None,
    ) -> bool:
        """Run supervisor review. Return True to continue executing, False if terminated/suspended."""
        trace_summary = _summarize_trace(self._log.list_events(project_id, task_id))
        try:
            verdict = await self._supervisor.review(
                brief,
                trace_summary,
                candidate_answer=candidate_answer,
                yield_reason=yield_reason,
                circuit_breaker_reason=circuit_breaker_reason,
            )
        except Exception as exc:
            logger.exception("Review failed for task %s", task_id)
            self._log.append(
                project_id,
                task_id,
                Actor.SYSTEM_CIRCUIT_BREAKER,
                EventType.SYSTEM_CIRCUIT_BREAKER,
                {"reason": "fatal_wrapper_crash", "context": {"phase": "review", "error": str(exc)}},
            )
            self._set_terminal(project_id, task_id, "failed")
            return False

        self._log.append(
            project_id,
            task_id,
            Actor.SUPERVISOR,
            EventType.SUPERVISOR_REVIEW,
            {
                "verdict": verdict.verdict,
                "reasoning": verdict.reasoning,
                "user_question": verdict.user_question,
            },
        )

        if verdict.verdict == "approve" and candidate_answer:
            self._transition(
                project_id,
                task_id,
                TaskState.REVIEWING,
                Trigger.REVIEW_APPROVE,
                {"answer": candidate_answer},
            )
            self._set_terminal(project_id, task_id, "completed")
            return False

        if verdict.verdict == "revise":
            amendment = verdict.amendment or verdict.reasoning or "Continue; narrow the search."
            new_brief_id = str(uuid.uuid4())
            self._log.append(
                project_id,
                task_id,
                Actor.SUPERVISOR,
                EventType.GOAL_BRIEF_REVISION,
                {
                    "brief_id": new_brief_id,
                    "parent_brief_id": brief.brief_id,
                    "amendment": amendment,
                    "reason": verdict.reasoning or "supervisor revise",
                },
            )
            self._transition(
                project_id,
                task_id,
                TaskState.REVIEWING,
                Trigger.REVIEW_REVISE,
                {"amendment": amendment},
            )
            brief.brief_id = new_brief_id
            brief.goal_statement = f"{brief.goal_statement}\n\n[Supervisor amendment: {amendment}]"
            return True

        if verdict.verdict == "rescope":
            self._transition(project_id, task_id, TaskState.REVIEWING, Trigger.REVIEW_RESCOPE)
            last_user = _most_recent_user_prompt(self._log.list_events(project_id, task_id))
            enriched = f"{last_user or brief.goal_statement}\n\n[Supervisor rescope note: {verdict.reasoning}]"
            await self._run_planning_and_execute(project_id, task_id, user_prompt=enriched)
            return False

        # ask_user → SUSPENDED
        self._transition(project_id, task_id, TaskState.REVIEWING, Trigger.REVIEW_ASK_USER)
        return False

    # ------------------------------------------------------------------
    # FSM helpers
    # ------------------------------------------------------------------

    def _transition(
        self,
        project_id: str,
        task_id: str,
        current: TaskState,
        trigger: Trigger,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskState:
        try:
            result = transition(current, trigger, task_id, metadata or {})
        except TransitionError as exc:
            logger.error("FSM transition rejected for %s: %s", task_id, exc)
            raise

        self._log.append(
            project_id,
            task_id,
            Actor.SYSTEM_FSM,
            EventType.STATE_TRANSITION,
            {
                "from_state": result.from_state.value,
                "to_state": result.to_state.value,
                "trigger_event_id": None,
            },
        )

        db: Session = get_project_session(project_id)
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is not None:
                task.state = result.to_state.value
                if is_terminal(result.to_state):
                    task.terminal_outcome = result.to_state.value
                task.updated_at = datetime.utcnow()
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        return result.to_state

    def _clear_terminal_outcome(self, project_id: str, task_id: str) -> None:
        """Clear the cached terminal_outcome so a resurrected task reads as live."""
        db: Session = get_project_session(project_id)
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is not None and task.terminal_outcome is not None:
                task.terminal_outcome = None
                task.updated_at = datetime.utcnow()
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _set_terminal(self, project_id: str, task_id: str, outcome: str) -> None:
        terminal_state = {
            "completed": TaskState.COMPLETED,
            "cancelled": TaskState.CANCELLED,
            "failed": TaskState.FAILED,
        }.get(outcome)
        db: Session = get_project_session(project_id)
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is not None:
                if terminal_state is not None:
                    task.state = terminal_state.value
                task.terminal_outcome = outcome
                task.updated_at = datetime.utcnow()
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        _CANCEL_EVENTS.pop(task_id, None)

    def _mark_task_prompt(self, project_id: str, task_id: str, user_prompt: str) -> None:
        title = _derive_task_title(user_prompt)
        db: Session = get_project_session(project_id)
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task is not None:
                if not task.initial_prompt:
                    task.initial_prompt = user_prompt
                if not task.title:
                    task.title = title
                task.updated_at = datetime.utcnow()
                db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _require_task(self, project_id: str, task_id: str) -> TaskRecord:
        rec = self.get_task(project_id, task_id)
        if rec is None:
            raise ValueError(f"Task {task_id} not found")
        return rec

    def _load_active_brief(self, project_id: str, task_id: str) -> Optional[GoalBrief]:
        events = self._log.list_events(project_id, task_id)
        active_manifest: List[str] = []
        goal: Optional[str] = None
        dod: Optional[str] = None
        brief_id: Optional[str] = None
        constraints: Dict[str, Any] = {}
        for evt in events:
            if evt.event_type == EventType.SUPERVISOR_BRIEF:
                p = evt.payload
                goal = p.get("goal_statement")
                dod = p.get("definition_of_done")
                active_manifest = list(p.get("active_manifest") or [])
                constraints = dict(p.get("constraints") or {})
                brief_id = p.get("brief_id")
            elif evt.event_type == EventType.GOAL_BRIEF_REVISION:
                p = evt.payload
                brief_id = p.get("brief_id", brief_id)
                if p.get("active_manifest"):
                    active_manifest = list(p["active_manifest"])
                amendment = p.get("amendment", "")
                if amendment and goal:
                    goal = f"{goal}\n\n[Supervisor amendment: {amendment}]"
        if not brief_id or not goal or not dod:
            return None
        return GoalBrief(
            brief_id=brief_id,
            goal_statement=goal,
            definition_of_done=dod,
            active_manifest=active_manifest,
            constraints=constraints,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _ensure_cancel_event(task_id: str) -> asyncio.Event:
    evt = _CANCEL_EVENTS.get(task_id)
    if evt is None:
        evt = asyncio.Event()
        _CANCEL_EVENTS[task_id] = evt
    return evt


def _row_to_record(task: Task) -> TaskRecord:
    return TaskRecord(
        id=task.id,
        project_id=task.project_id,
        parent_task_id=task.parent_task_id,
        title=task.title,
        initial_prompt=task.initial_prompt,
        state=TaskState(task.state),
        terminal_outcome=task.terminal_outcome,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def _summarize_trace(events: List[TaskEventDTO]) -> str:
    """Compact trace for the supervisor. Includes key events only."""
    lines: List[str] = []
    for evt in events[-40:]:
        if evt.event_type == EventType.TOOL_CALL_INTENT:
            lines.append(f"[call] {evt.payload.get('tool_name')} args={evt.payload.get('arguments')}")
        elif evt.event_type == EventType.TOOL_EXECUTION_RESULT:
            out = evt.payload.get("output") or {}
            lines.append(f"[result] status={evt.payload.get('status')} summary={str(out.get('summary',''))[:200]}")
        elif evt.event_type == EventType.TOOL_YIELD:
            lines.append(f"[yield] {evt.payload.get('reason')}")
        elif evt.event_type == EventType.FINAL_ANSWER:
            lines.append(f"[final_answer] {str(evt.payload.get('answer',''))[:400]}")
    return "\n".join(lines) if lines else "(no trace events yet)"


def _find_pending_question_event(events: List[TaskEventDTO]) -> Optional[str]:
    for evt in reversed(events):
        if evt.event_type == EventType.SUPERVISOR_REVIEW and evt.payload.get("verdict") in {"ask_user", "block"}:
            return evt.event_id
        if evt.event_type == EventType.TOOL_EXECUTION_RESULT and evt.payload.get("status") == "requires_human":
            return evt.event_id
    return None


def _extract_question_text(events: List[TaskEventDTO]) -> Optional[str]:
    for evt in reversed(events):
        if evt.event_type == EventType.SUPERVISOR_REVIEW and evt.payload.get("verdict") in {"ask_user", "block"}:
            return str(evt.payload.get("user_question") or evt.payload.get("reasoning") or "")
        if evt.event_type == EventType.TOOL_EXECUTION_RESULT and evt.payload.get("status") == "requires_human":
            out = evt.payload.get("output") or {}
            return str(out.get("summary") or "")
    return None


def _most_recent_user_prompt(events: List[TaskEventDTO]) -> Optional[str]:
    for evt in reversed(events):
        if evt.event_type == EventType.USER_PROMPT:
            return str(evt.payload.get("content") or "")
    return None


def _collect_attachments(events: List[TaskEventDTO]) -> List[str]:
    """Gather every attachment path referenced by USER_PROMPT/USER_RESPONSE events."""
    paths: List[str] = []
    for evt in events:
        if evt.event_type in (EventType.USER_PROMPT, EventType.USER_RESPONSE):
            for p in evt.payload.get("attachments") or []:
                if isinstance(p, str) and p:
                    paths.append(p)
    return _dedupe_preserving_order(paths)


def _dedupe_preserving_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _derive_task_title(prompt: str) -> str:
    text = " ".join((prompt or "").strip().split())
    if not text:
        return "Untitled task"
    return text[:48] + ("..." if len(text) > 48 else "")


# Singleton
_task_service: Optional[TaskService] = None


def get_task_service() -> TaskService:
    global _task_service
    if _task_service is None:
        _task_service = TaskService()
    return _task_service
