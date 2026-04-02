"""
Database models and connection management (SQLite embedded).

Schema:
- Project: workspace / research project scoping
- Document: uploaded PDFs with metadata
- DocumentChunk: text chunks for retrieval
- Node: knowledge graph entities (triple store)
- Edge: knowledge graph relationships (triple store)

All IDs are stored as String (UUID text) for SQLite compatibility.
Properties stored as JSON (not JSONB - SQLite native JSON1 extension).
"""
from sqlalchemy import (
    create_engine, Column, String, Integer, DateTime, Text, ForeignKey,
    JSON, Index, text, inspect, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid

from app.core.config import settings

Base = declarative_base()
_engine = None
_SessionLocal = None


def _generate_uuid() -> str:
    """Generate a UUID string for use as primary key."""
    return str(uuid.uuid4())


# ============================================================
# PROJECT SCOPING
# ============================================================

class Project(Base):
    """Research project for scoping documents and graph data."""
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=_generate_uuid)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    nodes = relationship("Node", back_populates="project", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="project", cascade="all, delete-orphan")


# ============================================================
# TRIPLE STORE SCHEMA
# ============================================================

class Node(Base):
    """
    Flexible node table for knowledge graph.
    All node-specific data stored in JSON properties.
    """
    __tablename__ = "nodes"

    id = Column(String, primary_key=True, default=_generate_uuid)
    label = Column(String, nullable=False, index=True)  # e.g., "chemical", "concept", "person"
    properties = Column(JSON, nullable=False, default=dict)  # Flexible JSON storage

    # Foreign keys
    document_id = Column(String, ForeignKey("documents.id"), nullable=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True, index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="nodes")
    project = relationship("Project", back_populates="nodes")
    outgoing_edges = relationship(
        "Edge",
        foreign_keys="Edge.source_id",
        back_populates="source_node",
        cascade="all, delete-orphan"
    )
    incoming_edges = relationship(
        "Edge",
        foreign_keys="Edge.target_id",
        back_populates="target_node",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_nodes_label", "label"),
        Index("idx_nodes_document_id", "document_id"),
        Index("idx_nodes_project_id", "project_id"),
    )


class Edge(Base):
    """
    Flexible edge table for knowledge graph relationships.
    All edge-specific data stored in JSON properties.
    """
    __tablename__ = "edges"

    id = Column(String, primary_key=True, default=_generate_uuid)
    source_id = Column(String, ForeignKey("nodes.id"), nullable=False, index=True)
    target_id = Column(String, ForeignKey("nodes.id"), nullable=False, index=True)
    type = Column(String, nullable=False, index=True)  # e.g., "CO_OCCURS", "MENTIONS"
    properties = Column(JSON, nullable=False, default=dict)  # Flexible JSON storage

    # Foreign keys
    document_id = Column(String, ForeignKey("documents.id"), nullable=True, index=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True, index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="edges")
    project = relationship("Project", back_populates="edges")
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


# ============================================================
# DOCUMENT STORE
# ============================================================

class Document(Base):
    """Uploaded documents with metadata."""
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_generate_uuid)
    filename = Column(String, nullable=False)
    file_hash = Column(String, nullable=False, index=True)  # SHA256 for deduplication
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String, nullable=True)

    # Project scoping
    project_id = Column(String, ForeignKey("projects.id"), nullable=True, index=True)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Status: pending, processing, completed, failed
    status = Column(String, default="pending")

    # Progress tracking
    total_chunks = Column(Integer, default=0, nullable=False)
    processed_chunks = Column(Integer, default=0, nullable=False)

    # Additional metadata
    doc_metadata = Column(JSON, default=dict)

    # Relationships
    project = relationship("Project", back_populates="documents")
    nodes = relationship("Node", back_populates="document", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    # Unique within a project (same filename can exist in different projects)
    __table_args__ = (
        Index("idx_documents_project_id", "project_id"),
        Index("idx_documents_file_hash", "file_hash"),
    )


class DocumentChunk(Base):
    """Chunked document text for retrieval."""
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=_generate_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)

    # Content
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=True)

    # Source location
    page_number = Column(Integer, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)

    # Metadata
    chunk_metadata = Column(JSON, default=dict)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunk_document", "document_id", "chunk_index"),
    )


# ============================================================
# DISCOVERY SESSION
# ============================================================

class DiscoverySession(Base):
    """Discovery session recording target parameters."""
    __tablename__ = "discovery_sessions"

    id = Column(String, primary_key=True, default=_generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True, index=True)
    target_params = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ============================================================
# DATABASE SETUP (SQLite embedded)
# ============================================================

def _enable_sqlite_fks(dbapi_connection, connection_record):
    """Enable foreign key enforcement on every SQLite connection and optimize performance."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def get_engine():
    """Create or reuse a singleton SQLAlchemy engine (SQLite)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},  # Required for SQLite + threads
            echo=False,
        )
        # Enable foreign keys on every connection
        event.listen(_engine, "connect", _enable_sqlite_fks)
    return _engine


def get_session():
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()


def _run_migrations(engine) -> None:
    """Apply lightweight additive migrations for schema evolution."""
    with engine.connect() as conn:
        # Add project_id to discovery_sessions if missing (added for workspace isolation)
        try:
            conn.execute(text(
                "ALTER TABLE discovery_sessions ADD COLUMN project_id TEXT REFERENCES projects(id)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_discovery_sessions_project_id ON discovery_sessions(project_id)"
            ))
            conn.commit()
        except Exception:
            pass  # Column already exists — SQLite raises OperationalError, safe to ignore


def init_db():
    """Initialize database and create all tables.

    For SQLite this is straightforward - just create the file and tables.
    """
    try:
        engine = get_engine()
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(engine)
        _run_migrations(engine)
        return engine
    except Exception as e:
        raise RuntimeError(f"FATAL: Database initialization failed: {e}") from e


def reset_db():
    """Drop all tables and recreate (USE WITH CAUTION)."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
