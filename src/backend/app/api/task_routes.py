"""HTTP surface for the task runtime.

Endpoints (prefix: /api/task):
    POST   /                          create a new task (returns id)
    GET    /?project_id=...           list tasks in a project
    GET    /{task_id}                 fetch task record + current state
    GET    /{task_id}/events          full event log (audit / replay)
    POST   /{task_id}/start           submit user prompt, run until terminal/suspended
    POST   /{task_id}/respond         answer a SUSPENDED task's pending question
    POST   /{task_id}/cancel          cancel the task
    GET    /{task_id}/stream          SSE: stream events for this task (live + history)

Per-project isolation: a task's `project_id` is resolved by scanning the
registry (TaskService.find_task) when not in the URL, since UUIDs are unique.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from datetime import datetime, timezone

from app.core.task_events import TaskEventDTO
from app.core.task_log import get_task_log
from app.services.task_service import TaskRecord, get_task_service
from app.services.workspace_manager import get_workspace_manager


def _to_utc_iso(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


router = APIRouter(prefix="/api/task", tags=["task"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire models
# ---------------------------------------------------------------------------


class TaskCreateRequest(BaseModel):
    project_id: str
    title: Optional[str] = None
    initial_prompt: Optional[str] = None
    parent_task_id: Optional[str] = None


class TaskStartRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    attachments: List[str] = Field(default_factory=list)


class TaskRespondRequest(BaseModel):
    response: str = Field(..., min_length=1)
    attachments: List[str] = Field(default_factory=list)


class TaskAttachmentResponse(BaseModel):
    filename: str
    path: str
    size_bytes: int
    mime_type: Optional[str] = None


class TaskCancelRequest(BaseModel):
    reason: Optional[str] = None


class TaskResponse(BaseModel):
    id: str
    project_id: str
    parent_task_id: Optional[str]
    title: Optional[str]
    initial_prompt: Optional[str]
    state: str
    terminal_outcome: Optional[str]
    created_at: str
    updated_at: str


class TaskEventResponse(BaseModel):
    event_id: str
    task_id: str
    sequence: int
    schema_version: int
    timestamp: str
    actor: str
    event_type: str
    causal_parents: List[str]
    payload: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_to_response(rec: TaskRecord) -> TaskResponse:
    return TaskResponse(
        id=rec.id,
        project_id=rec.project_id,
        parent_task_id=rec.parent_task_id,
        title=rec.title,
        initial_prompt=rec.initial_prompt,
        state=rec.state.value,
        terminal_outcome=rec.terminal_outcome,
        created_at=_to_utc_iso(rec.created_at),
        updated_at=_to_utc_iso(rec.updated_at),
    )


def _event_to_response(evt: TaskEventDTO) -> TaskEventResponse:
    return TaskEventResponse(
        event_id=evt.event_id,
        task_id=evt.task_id,
        sequence=evt.sequence,
        schema_version=evt.schema_version,
        timestamp=_to_utc_iso(evt.timestamp),
        actor=evt.actor.value,
        event_type=evt.event_type.value,
        causal_parents=evt.causal_parents,
        payload=evt.payload,
    )


def _resolve_task_or_404(task_id: str) -> TaskRecord:
    rec = get_task_service().find_task(task_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return rec


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=TaskResponse)
async def create_task(body: TaskCreateRequest) -> TaskResponse:
    try:
        rec = get_task_service().create_task(
            project_id=body.project_id,
            title=body.title,
            initial_prompt=body.initial_prompt,
            parent_task_id=body.parent_task_id,
        )
        return _record_to_response(rec)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Task creation failed")
        raise HTTPException(status_code=500, detail=f"Task creation failed: {exc}") from exc


@router.get("", response_model=List[TaskResponse])
async def list_tasks(project_id: str = Query(...)) -> List[TaskResponse]:
    return [_record_to_response(r) for r in get_task_service().list_tasks(project_id)]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    return _record_to_response(_resolve_task_or_404(task_id))


@router.get("/{task_id}/events", response_model=List[TaskEventResponse])
async def list_task_events(task_id: str) -> List[TaskEventResponse]:
    rec = _resolve_task_or_404(task_id)
    events = get_task_service().list_events(rec.project_id, task_id)
    return [_event_to_response(e) for e in events]


@router.post("/{task_id}/start", response_model=TaskResponse)
async def start_task(task_id: str, body: TaskStartRequest) -> TaskResponse:
    rec = _resolve_task_or_404(task_id)
    service = get_task_service()
    try:
        rec = await service.start_task(
            rec.project_id, task_id, body.prompt.strip(), attachments=body.attachments
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Task start failed")
        raise HTTPException(status_code=500, detail=f"Task start failed: {exc}") from exc
    return _record_to_response(rec)


@router.post("/{task_id}/respond", response_model=TaskResponse)
async def respond_to_task(task_id: str, body: TaskRespondRequest) -> TaskResponse:
    rec = _resolve_task_or_404(task_id)
    try:
        rec = await get_task_service().respond_to_suspended(
            rec.project_id, task_id, body.response.strip(), attachments=body.attachments
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Task respond failed")
        raise HTTPException(status_code=500, detail=f"Task respond failed: {exc}") from exc
    return _record_to_response(rec)


@router.post("/{task_id}/attachments", response_model=TaskAttachmentResponse)
async def upload_task_attachment(
    task_id: str, file: UploadFile = File(...)
) -> TaskAttachmentResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    rec = _resolve_task_or_404(task_id)

    manager = get_workspace_manager()
    target_dir = manager.task_attachments_path(rec.project_id, task_id)

    safe_name = Path(file.filename).name
    stored_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    target_path = target_dir / stored_name

    try:
        with open(target_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as exc:
        logger.exception("Attachment upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    return TaskAttachmentResponse(
        filename=safe_name,
        path=str(target_path.resolve()),
        size_bytes=target_path.stat().st_size,
        mime_type=file.content_type,
    )


@router.post("/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(task_id: str, body: Optional[TaskCancelRequest] = None) -> TaskResponse:
    rec = _resolve_task_or_404(task_id)
    try:
        rec = await get_task_service().cancel_task(
            rec.project_id, task_id, reason=body.reason if body else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _record_to_response(rec)


@router.get("/{task_id}/stream")
async def stream_task(task_id: str, from_sequence: int = Query(-1, ge=-1)):
    """Server-Sent Events stream of task events (live + history backfill)."""
    rec = _resolve_task_or_404(task_id)
    project_id = rec.project_id

    log = get_task_log()

    async def event_generator():
        yield ": stream open\n\n"

        queue = await log.subscribe(task_id)
        try:
            history = log.list_events_after(project_id, task_id, from_sequence)
            seen_max_seq = from_sequence
            for evt in history:
                yield _sse_event(evt)
                seen_max_seq = max(seen_max_seq, evt.sequence)

            while True:
                try:
                    evt = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if evt.sequence <= seen_max_seq:
                    continue
                seen_max_seq = evt.sequence
                yield _sse_event(evt)
        finally:
            log.unsubscribe(task_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Content-Encoding": "identity",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse_event(evt: TaskEventDTO) -> str:
    payload = _event_to_response(evt).model_dump()
    return f"event: task_event\ndata: {json.dumps(payload, default=str)}\n\n"
