"""Append-only task event log.

The log is THE source of truth. Every state the UI shows, every context
window Nemotron reads, every audit trail is derived from here. The writer
is the single point where events enter the system — validates payload,
increments the per-task sequence, inserts the row.

Read paths:
- `list_events(task_id)` — full ordered log for a task (audit, UI trace).
- `list_events_after(task_id, sequence)` — for SSE streaming deltas.
- `nemotron_view(task_id)` — filtered / compacted view for the tool-loop
  model's context window (for v1 returns the full log; compaction plugs in
  later by replacing ranges with LOG_COMPACTED summaries).

The `subscribe(task_id)` queue is an in-process fan-out so SSE handlers can
see events the moment they're written, without polling the DB.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.database import Task, TaskEvent, get_session
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
        # Per-task subscription queues for SSE streaming.
        self._subscribers: Dict[str, List[asyncio.Queue[TaskEventDTO]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(
        self,
        task_id: str,
        actor: Actor,
        event_type: EventType,
        payload: Dict[str, Any],
        causal_parents: Optional[List[str]] = None,
    ) -> TaskEventDTO:
        """Validate + insert + fan-out to subscribers.

        The database transaction is the serialization point — two concurrent
        writers on the same task race on the UNIQUE(task_id, sequence) index
        and the loser retries. SQLite's row-level locking makes this safe
        enough for embedded use.
        """
        validated = validate_payload(event_type, payload)
        db: Session = get_session()
        try:
            task = db.query(Task).filter(Task.id == task_id).with_for_update().first() \
                if db.bind.dialect.name != "sqlite" else \
                db.query(Task).filter(Task.id == task_id).first()
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

    def list_events(self, task_id: str) -> List[TaskEventDTO]:
        db: Session = get_session()
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

    def list_events_after(self, task_id: str, sequence: int) -> List[TaskEventDTO]:
        db: Session = get_session()
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

    def latest_of_type(self, task_id: str, event_type: EventType) -> Optional[TaskEventDTO]:
        db: Session = get_session()
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

    def nemotron_view(self, task_id: str) -> List[TaskEventDTO]:
        """Filtered view Nemotron sees during its tool loop.

        v1: return the full log. Compaction replaces ranges with LOG_COMPACTED
        summaries later. The contract is: callers get an ordered list they
        can feed into Nemotron's context without size management concerns.
        """
        return self.list_events(task_id)

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
        """Push a freshly-written event to every subscriber on this task."""
        with self._lock:
            queues = list(self._subscribers.get(task_id, []))
        for q in queues:
            try:
                q.put_nowait(dto)
            except asyncio.QueueFull:
                # Shouldn't happen with unbounded queues; guard anyway.
                logger.warning("Subscriber queue full for task %s", task_id)


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
