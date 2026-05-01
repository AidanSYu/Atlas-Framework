"""Per-project SQLite database.

The Atlas data model is fully isolated per project. Every project folder
owns its own `project.db`; nothing is shared across projects.

There is no `projects` table — the list of known projects lives in the
JSON registry at app level (see `app.core.registry`). Per-project tables
keep a `project_id` String column as a vestigial stamp, but it carries
no foreign key (the projects table doesn't exist in this database).

The engine cache holds at most `_PROJECT_ENGINE_CACHE_MAX` open engines
at once. Lesser-used engines are disposed via LRU eviction; the SQLite
file is still on disk and reopened on next access.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from threading import RLock

from sqlalchemy import (
    Column, DateTime, ForeignKey, Index, Integer, JSON, String, Text,
    create_engine, event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from datetime import datetime
import uuid

from app.core import project_paths, registry

logger = logging.getLogger(__name__)

ProjectBase = declarative_base()
_PROJECT_ENGINE_CACHE_MAX = 8


def _generate_uuid() -> str:
    return str(uuid.uuid4())


# ============================================================
# Per-project schema
# ============================================================
# All tables live in {project_root}/project.db. project_id is kept as a
# stamp (useful when rows are exported and re-mingled) but has no FK.

class Node(ProjectBase):
    __tablename__ = "nodes"

    id = Column(String, primary_key=True, default=_generate_uuid)
    label = Column(String, nullable=False, index=True)
    properties = Column(JSON, nullable=False, default=dict)

    document_id = Column(String, ForeignKey("documents.id"), nullable=True, index=True)
    project_id = Column(String, nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    document = relationship("Document", back_populates="nodes")
    outgoing_edges = relationship(
        "Edge",
        foreign_keys="Edge.source_id",
        back_populates="source_node",
        cascade="all, delete-orphan",
    )
    incoming_edges = relationship(
        "Edge",
        foreign_keys="Edge.target_id",
        back_populates="target_node",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_nodes_label", "label"),
        Index("idx_nodes_document_id", "document_id"),
        Index("idx_nodes_project_id", "project_id"),
    )


class Edge(ProjectBase):
    __tablename__ = "edges"

    id = Column(String, primary_key=True, default=_generate_uuid)
    source_id = Column(String, ForeignKey("nodes.id"), nullable=False, index=True)
    target_id = Column(String, ForeignKey("nodes.id"), nullable=False, index=True)
    type = Column(String, nullable=False, index=True)
    properties = Column(JSON, nullable=False, default=dict)

    document_id = Column(String, ForeignKey("documents.id"), nullable=True, index=True)
    project_id = Column(String, nullable=True, index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="edges")
    source_node = relationship("Node", foreign_keys=[source_id], back_populates="outgoing_edges")
    target_node = relationship("Node", foreign_keys=[target_id], back_populates="incoming_edges")

    __table_args__ = (
        Index("idx_edges_source", "source_id"),
        Index("idx_edges_target", "target_id"),
        Index("idx_edges_type", "type"),
        Index("idx_edges_document_id", "document_id"),
        Index("idx_edges_project_id", "project_id"),
        Index("idx_edges_source_target", "source_id", "target_id"),
    )


class Document(ProjectBase):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_generate_uuid)
    filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False, index=True)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String, nullable=True)

    project_id = Column(String, nullable=True, index=True)

    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    status = Column(String, default="pending")

    total_chunks = Column(Integer, default=0, nullable=False)
    processed_chunks = Column(Integer, default=0, nullable=False)

    doc_metadata = Column(JSON, default=dict)

    nodes = relationship("Node", back_populates="document", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_documents_project_id", "project_id"),
        Index("idx_documents_file_hash", "file_hash"),
    )


class DocumentChunk(ProjectBase):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=_generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)

    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=True)

    page_number = Column(Integer, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)

    chunk_metadata = Column(JSON, default=dict)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunk_document", "document_id", "chunk_index"),
    )


class DiscoverySession(ProjectBase):
    __tablename__ = "discovery_sessions"

    id = Column(String, primary_key=True, default=_generate_uuid)
    project_id = Column(String, nullable=True, index=True)
    target_params = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Task(ProjectBase):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, default=_generate_uuid)
    project_id = Column(String, nullable=False, index=True)
    parent_task_id = Column(String, ForeignKey("tasks.id"), nullable=True, index=True)

    title = Column(String, nullable=True)
    initial_prompt = Column(Text, nullable=True)

    state = Column(String, nullable=False, default="idle")
    terminal_outcome = Column(String, nullable=True)

    next_sequence = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_tasks_project_id", "project_id"),
        Index("idx_tasks_parent", "parent_task_id"),
        Index("idx_tasks_state", "state"),
    )


class CapabilityGapRecord(ProjectBase):
    __tablename__ = "capability_gaps"

    id = Column(String, primary_key=True, default=_generate_uuid)
    run_id = Column(String, nullable=False)
    stage = Column(Integer, nullable=False)
    required_function = Column(Text, nullable=False)
    input_schema = Column(JSON, nullable=False, default=dict)
    output_schema = Column(JSON, nullable=False, default=dict)
    standard_reference = Column(Text, nullable=True)
    resolution_method = Column(String, nullable=True)
    resolution_config = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)


class TaskEvent(ProjectBase):
    __tablename__ = "task_events"

    id = Column(String, primary_key=True, default=_generate_uuid)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False, index=True)
    sequence = Column(Integer, nullable=False)
    schema_version = Column(Integer, nullable=False, default=1)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    actor = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)
    causal_parents = Column(JSON, nullable=False, default=list)
    payload = Column(JSON, nullable=False, default=dict)

    __table_args__ = (
        Index("idx_task_events_task_seq", "task_id", "sequence", unique=True),
        Index("idx_task_events_type", "event_type"),
        Index("idx_task_events_actor", "actor"),
    )


# ============================================================
# Engine + session lifecycle (per-project)
# ============================================================

def _enable_sqlite_pragmas(dbapi_connection, connection_record):
    """Apply per-connection SQLite tuning."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


_engine_cache: "OrderedDict[str, Engine]" = OrderedDict()
_sessionmaker_cache: "OrderedDict[str, sessionmaker]" = OrderedDict()
_lock = RLock()


def _resolve_project_root(project_id: str) -> Path:
    entry = registry.get_project(project_id)
    if entry is None:
        raise KeyError(f"Project {project_id} not registered")
    return entry.folder()


def _build_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    event.listen(engine, "connect", _enable_sqlite_pragmas)
    return engine


def _evict_one_locked() -> None:
    """Drop the least-recently-used engine. Caller holds _lock."""
    if not _engine_cache:
        return
    oldest_id, oldest_engine = next(iter(_engine_cache.items()))
    try:
        oldest_engine.dispose()
    except Exception as exc:
        logger.warning("Engine dispose failed for %s: %s", oldest_id, exc)
    _engine_cache.pop(oldest_id, None)
    _sessionmaker_cache.pop(oldest_id, None)


def get_project_engine(project_id: str) -> Engine:
    with _lock:
        engine = _engine_cache.get(project_id)
        if engine is not None:
            _engine_cache.move_to_end(project_id)
            return engine

        if len(_engine_cache) >= _PROJECT_ENGINE_CACHE_MAX:
            _evict_one_locked()

        root = _resolve_project_root(project_id)
        db_path = project_paths.project_db_path(root)
        engine = _build_engine(db_path)
        _engine_cache[project_id] = engine
        return engine


def get_project_session(project_id: str) -> Session:
    with _lock:
        maker = _sessionmaker_cache.get(project_id)
        if maker is None:
            engine = get_project_engine(project_id)
            maker = sessionmaker(bind=engine)
            _sessionmaker_cache[project_id] = maker
        return maker()


def init_project_db(project_id: str) -> None:
    """Create per-project tables. Idempotent."""
    engine = get_project_engine(project_id)
    ProjectBase.metadata.create_all(engine)


def close_project_engine(project_id: str) -> None:
    """Dispose the engine for `project_id` so its DB file can be deleted/moved."""
    with _lock:
        engine = _engine_cache.pop(project_id, None)
        _sessionmaker_cache.pop(project_id, None)
        if engine is not None:
            try:
                engine.dispose()
            except Exception as exc:
                logger.warning("Engine dispose failed for %s: %s", project_id, exc)


def close_all_project_engines() -> None:
    """Dispose every cached engine. Used on shutdown."""
    with _lock:
        for project_id, engine in list(_engine_cache.items()):
            try:
                engine.dispose()
            except Exception as exc:
                logger.warning("Engine dispose failed for %s: %s", project_id, exc)
        _engine_cache.clear()
        _sessionmaker_cache.clear()
