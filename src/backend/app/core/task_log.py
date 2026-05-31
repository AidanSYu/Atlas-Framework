"""Append-only task event log (per-project SQLite).

The log is THE source of truth. Every method that touches the database
takes `project_id` so the right per-project DB is used; in-memory
subscription fanout is project-agnostic (keyed only by task_id).

Read paths:
- `list_events(project_id, task_id)` — full ordered log for a task.
- `list_events_after(project_id, task_id, sequence)` — for SSE streaming deltas.
- `nemotron_view(project_id, task_id)` — filtered view for the tool-loop model.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.database import Task, TaskEvent, get_project_session
from app.core.task_events import (
    SCHEMA_VERSION,
    Actor,
    EventType,
    TaskEventDTO,
    validate_payload,
)

logger = logging.getLogger(__name__)


class TaskLog:
    """Thread-safe, append-only event log with in-process subscribers."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue[TaskEventDTO]]] = {}
        self._lock = threading.Lock()
        # Per-task sequence-allocation lock so concurrent appends to the same
        # task can't both read next_sequence=N and produce duplicate (task_id,
        # sequence) rows. Tasks are independent → finer-grained than a global
        # write lock.
        self._task_locks: Dict[str, threading.Lock] = {}
        self._task_locks_guard = threading.Lock()

    def _lock_for(self, task_id: str) -> threading.Lock:
        with self._task_locks_guard:
            lock = self._task_locks.get(task_id)
            if lock is None:
                lock = threading.Lock()
                self._task_locks[task_id] = lock
            return lock

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(
        self,
        project_id: str,
        task_id: str,
        actor: Actor,
        event_type: EventType,
        payload: Dict[str, Any],
        causal_parents: Optional[List[str]] = None,
    ) -> TaskEventDTO:
        validated = validate_payload(event_type, payload)
        with self._lock_for(task_id):
            db: Session = get_project_session(project_id)
            try:
                task = db.query(Task).filter(Task.id == task_id).first()
                if task is None:
                    raise ValueError(f"Task {task_id} not found")

                seq = task.next_sequence
                task.next_sequence = seq + 1
                task.updated_at = datetime.utcnow()

                event = TaskEvent(
                    id=str(uuid.uuid4()),
                    task_id=task_id,
                    sequence=seq,
                    schema_version=SCHEMA_VERSION,
                    timestamp=datetime.utcnow(),
                    actor=actor.value,
                    event_type=event_type.value,
                    causal_parents=list(causal_parents or []),
                    payload=validated,
                )
                db.add(event)
                db.commit()
                db.refresh(event)
                dto = _row_to_dto(event)
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        self._fanout(task_id, dto)
        return dto

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def list_events(self, project_id: str, task_id: str) -> List[TaskEventDTO]:
        db: Session = get_project_session(project_id)
        try:
            rows = (
                db.query(TaskEvent)
                .filter(TaskEvent.task_id == task_id)
                .order_by(TaskEvent.sequence.asc())
                .all()
            )
            return [_row_to_dto(r) for r in rows]
        finally:
            db.close()

    def list_events_after(
        self, project_id: str, task_id: str, sequence: int
    ) -> List[TaskEventDTO]:
        db: Session = get_project_session(project_id)
        try:
            rows = (
                db.query(TaskEvent)
                .filter(TaskEvent.task_id == task_id, TaskEvent.sequence > sequence)
                .order_by(TaskEvent.sequence.asc())
                .all()
            )
            return [_row_to_dto(r) for r in rows]
        finally:
            db.close()

    def latest_of_type(
        self, project_id: str, task_id: str, event_type: EventType
    ) -> Optional[TaskEventDTO]:
        db: Session = get_project_session(project_id)
        try:
            row = (
                db.query(TaskEvent)
                .filter(
                    TaskEvent.task_id == task_id,
                    TaskEvent.event_type == event_type.value,
                )
                .order_by(TaskEvent.sequence.desc())
                .first()
            )
            return _row_to_dto(row) if row else None
        finally:
            db.close()

    def nemotron_view(self, project_id: str, task_id: str) -> List[TaskEventDTO]:
        return self.list_events(project_id, task_id)

    # ------------------------------------------------------------------
    # Subscribe path (SSE)
    # ------------------------------------------------------------------

    async def subscribe(self, task_id: str) -> asyncio.Queue[TaskEventDTO]:
        queue: asyncio.Queue[TaskEventDTO] = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(task_id, []).append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue[TaskEventDTO]) -> None:
        with self._lock:
            queues = self._subscribers.get(task_id, [])
            if queue in queues:
                queues.remove(queue)
            if not queues and task_id in self._subscribers:
                del self._subscribers[task_id]

    def _fanout(self, task_id: str, dto: TaskEventDTO) -> None:
        with self._lock:
            queues = list(self._subscribers.get(task_id, []))
        for q in queues:
            try:
                q.put_nowait(dto)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full for task %s", task_id)

    # ------------------------------------------------------------------
    # Purge
    # ------------------------------------------------------------------

    def purge_task(self, task_id: str) -> None:
        """Drop in-process state for a deleted task. Open SSE streams will
        stop receiving events (their next get() hits the keepalive timeout
        and the client detects the task is gone)."""
        with self._lock:
            self._subscribers.pop(task_id, None)
        with self._task_locks_guard:
            self._task_locks.pop(task_id, None)


def _row_to_dto(row: TaskEvent) -> TaskEventDTO:
    return TaskEventDTO(
        event_id=row.id,
        task_id=row.task_id,
        sequence=row.sequence,
        schema_version=row.schema_version,
        timestamp=row.timestamp,
        actor=Actor(row.actor),
        event_type=EventType(row.event_type),
        causal_parents=list(row.causal_parents or []),
        payload=dict(row.payload or {}),
    )


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------

_task_log: Optional[TaskLog] = None


def get_task_log() -> TaskLog:
    global _task_log
    if _task_log is None:
        _task_log = TaskLog()
    return _task_log
