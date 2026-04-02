from typing import List, Dict, Any, Optional
import uuid
import json
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import HTTPException
from app.core.database import get_session, Document, DiscoverySession
from app.core.config import settings

import logging
logger = logging.getLogger(__name__)

class PropertyConstraint(BaseModel):
    property: str
    operator: str
    value: Any

class ProjectTargetParams(BaseModel):
    domain: str
    objective: str
    propertyConstraints: List[PropertyConstraint]
    domainSpecificConstraints: Dict[str, Any]
    corpusDocumentIds: List[str]
    projectId: Optional[str] = None

class DiscoverySessionService:
    @staticmethod
    def get_schema() -> Dict[str, Any]:
        return {
            "domain": "organic_chemistry",
            "target_schema": ["biologicalTarget", "propertyConstraints", "forbiddenSubstructures"]
        }

    @staticmethod
    def initialize_session(params: ProjectTargetParams) -> Dict[str, Any]:
        session = get_session()
        try:
            # Validate corpusDocumentIds
            if params.corpusDocumentIds:
                doc_ids = set(params.corpusDocumentIds)
                existing_docs = session.query(Document.id).filter(Document.id.in_(doc_ids)).all()
                existing_doc_ids = {doc.id for doc in existing_docs}

                missing_docs = doc_ids - existing_doc_ids
                if missing_docs:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Documents not found: {', '.join(missing_docs)}"
                    )

            # Create session row
            epoch_id = str(uuid.uuid4())
            new_ds = DiscoverySession(
                project_id=params.projectId,
                target_params=params.model_dump()
            )
            session.add(new_ds)
            session.commit()
            session.refresh(new_ds)

            # Create session folder structure
            session_id = new_ds.id
            session_name = params.objective or f"Session {datetime.now().strftime('%Y-%m-%d')}"
            base_path = Path(settings.DATA_DIR) / "discovery" / session_id
            base_path.mkdir(parents=True, exist_ok=True)

            # Create generated files directory
            generated_path = base_path / "generated"
            generated_path.mkdir(exist_ok=True)

            # Create jobs.json
            jobs_file = base_path / "jobs.json"
            jobs_file.write_text(json.dumps({}))

            return {
                "session_id": session_id,
                "session_name": session_name,
                "epoch_id": epoch_id,
                "folder_path": str(base_path),
                "status": "initialized"
            }

        except HTTPException:
            raise
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"Error initializing session: {str(e)}")
        finally:
            session.close()

    @staticmethod
    def get_sessions(project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List discovery sessions from database, optionally filtered by project."""
        session = get_session()
        try:
            query = session.query(DiscoverySession).order_by(DiscoverySession.created_at.desc())
            if project_id:
                query = query.filter(DiscoverySession.project_id == project_id)
            db_sessions = query.all()

            result = []
            for db_session in db_sessions:
                session_id = db_session.id
                target_params = db_session.target_params or {}
                session_name = target_params.get("objective", f"Session {db_session.created_at.strftime('%Y-%m-%d')}")

                # Check if folder exists
                session_path = Path(settings.DATA_DIR) / "discovery" / session_id
                status = "idle"  # Default status

                result.append({
                    "session_id": session_id,
                    "session_name": session_name,
                    "created_at": db_session.created_at.isoformat() if db_session.created_at else None,
                    "status": status,
                    "folder_exists": session_path.exists()
                })

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")
        finally:
            session.close()

    @staticmethod
    def get_session_files(session_id: str) -> List[Dict[str, Any]]:
        """List all files in a discovery session folder (root + generated/).

        Returns files sorted with root-level docs first, then generated scripts.
        """
        try:
            # Check if session exists in database
            session = get_session()
            db_session = session.query(DiscoverySession).filter(DiscoverySession.id == session_id).first()
            session.close()

            if not db_session:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

            session_path = Path(settings.DATA_DIR) / "discovery" / session_id

            if not session_path.exists():
                return []

            result = []

            # 1. Root-level files (SESSION_INIT.md, session_memory.json, jobs.json, etc.)
            for file_path in session_path.iterdir():
                if file_path.is_file():
                    result.append({
                        "filename": file_path.name,
                        "path": file_path.name,
                        "size_bytes": file_path.stat().st_size,
                    })

            # 2. Generated subfolder files (scripts, CSVs, images, etc.)
            generated_path = session_path / "generated"
            if generated_path.exists():
                for file_path in generated_path.iterdir():
                    if file_path.is_file():
                        result.append({
                            "filename": file_path.name,
                            "path": f"generated/{file_path.name}",
                            "size_bytes": file_path.stat().st_size,
                        })

            return result

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing session files: {str(e)}")

    @staticmethod
    def update_coordinator_goals(session_id: str, extracted_goals: List[str]) -> Dict[str, Any]:
        """Update DiscoverySession with coordinator-extracted goals."""
        session = get_session()
        try:
            db_session = session.query(DiscoverySession).filter(DiscoverySession.id == session_id).first()
            if not db_session:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

            # Update target_params with extracted goals
            current_params = db_session.target_params or {}
            current_params["coordinator_extracted_goals"] = extracted_goals
            db_session.target_params = current_params

            session.commit()
            session.refresh(db_session)

            return {
                "session_id": session_id,
                "extracted_goals": extracted_goals,
                "updated_at": db_session.updated_at.isoformat() if db_session.updated_at else None,
            }

        except HTTPException:
            raise
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"Error updating session goals: {str(e)}")
        finally:
            session.close()


# ============================================================================
# Shared Session Memory (Multi-Agent Coordination)
# ============================================================================

class CorpusContext(BaseModel):
    """Corpus scan results from coordinator initialization."""
    entities: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    summary: str = ""


class ExperimentalFinding(BaseModel):
    """A single wetlab result recorded back into the session."""
    compound_id: str
    smiles: Optional[str] = None
    assay_type: str                  # e.g. "gag_priming", "cell_viability"
    value: Optional[float] = None    # numeric result
    units: Optional[str] = None
    active: Optional[bool] = None    # True = hit, False = inactive
    notes: Optional[str] = None
    recorded_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SessionMemoryData(BaseModel):
    """Shared memory state accessible to all agents in a discovery session.

    This enables efficient multi-agent coordination by providing a single
    source of truth for session context. All agents (Coordinator, Executor,
    future agents) can read/write to this shared state.
    """
    session_id: str
    initialized_at: str
    domain: Optional[str] = None
    corpus_context: Optional[CorpusContext] = None
    research_goals: List[str] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    agents_completed: List[str] = Field(default_factory=list)
    current_stage: str = "initializing"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    experimental_findings: List[ExperimentalFinding] = Field(default_factory=list)


class SessionMemoryService:
    """Manages shared session memory for multi-agent coordination.

    Session memory is persisted to disk in two formats:
    - session_memory.json: Machine-readable state for agent consumption
    - SESSION_CONTEXT.md: Perpetual living context file read by all agents
    """

    @staticmethod
    def _get_session_path(session_id: str) -> Path:
        """Get base path for session directory."""
        return Path(settings.DATA_DIR) / "discovery" / session_id

    @staticmethod
    def _get_memory_json_path(session_id: str) -> Path:
        """Get path to session_memory.json."""
        return SessionMemoryService._get_session_path(session_id) / "session_memory.json"

    @staticmethod
    def _get_init_md_path(session_id: str) -> Path:
        """Get path to SESSION_CONTEXT.md."""
        return SessionMemoryService._get_session_path(session_id) / "SESSION_CONTEXT.md"

    @staticmethod
    def save_session_memory(session_id: str, memory_data: SessionMemoryData) -> None:
        """Save session memory to both JSON and Markdown formats.

        Args:
            session_id: Discovery session ID
            memory_data: Complete session memory state
        """
        session_path = SessionMemoryService._get_session_path(session_id)
        session_path.mkdir(parents=True, exist_ok=True)

        # Save JSON (machine-readable)
        json_path = SessionMemoryService._get_memory_json_path(session_id)
        json_path.write_text(memory_data.model_dump_json(indent=2), encoding="utf-8")

        # Save Markdown (human-readable)
        md_path = SessionMemoryService._get_init_md_path(session_id)
        md_content = SessionMemoryService._generate_markdown_report(memory_data)
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("Session memory saved for %s", session_id)

    @staticmethod
    def load_session_memory(session_id: str) -> Optional[SessionMemoryData]:
        """Load session memory from disk.

        Args:
            session_id: Discovery session ID

        Returns:
            SessionMemoryData if exists, None otherwise
        """
        json_path = SessionMemoryService._get_memory_json_path(session_id)

        if not json_path.exists():
            return None

        try:
            json_data = json.loads(json_path.read_text(encoding="utf-8"))
            return SessionMemoryData(**json_data)
        except json.JSONDecodeError as e:
            logger.warning(
                "Corrupt session_memory.json for %s: %s", session_id, e
            )
            return None
        except Exception as e:
            logger.warning(
                "Failed to load session memory for %s: %s", session_id, e
            )
            return None

    @staticmethod
    def update_session_memory(
        session_id: str,
        updates: Dict[str, Any]
    ) -> SessionMemoryData:
        """Update session memory with partial changes.

        Args:
            session_id: Discovery session ID
            updates: Dictionary of fields to update

        Returns:
            Updated SessionMemoryData
        """
        # Load existing memory or create new
        memory = SessionMemoryService.load_session_memory(session_id)

        if memory is None:
            # Initialize new memory if doesn't exist
            memory = SessionMemoryData(
                session_id=session_id,
                initialized_at=datetime.utcnow().isoformat()
            )

        # Apply updates
        for key, value in updates.items():
            if hasattr(memory, key):
                setattr(memory, key, value)

        # Save updated memory
        SessionMemoryService.save_session_memory(session_id, memory)

        return memory

    @staticmethod
    def _generate_markdown_report(memory: SessionMemoryData) -> str:
        """Generate SESSION_CONTEXT.md — the living context file shared by all agents.

        This file is the single source of truth for the session. Agents read it at the
        start of every run. It can be updated by agents or the researcher at any time.

        Args:
            memory: Session memory data

        Returns:
            Markdown-formatted context document
        """
        lines = [
            "# Session Context",
            "",
            "> This is a living document. Agents and researchers can update it at any time.",
            "> All agents read this file at the start of every run.",
            "",
            f"**Session ID**: `{memory.session_id}`  ",
            f"**Initialized**: {memory.initialized_at}  ",
        ]

        if memory.domain:
            lines.append(f"**Domain**: {memory.domain}  ")

        lines.append("")

        # Research Goals — most important section, first
        if memory.research_goals:
            lines.extend([
                "## Research Goals",
                "",
                "These goals were extracted during the coordinator initialization conversation.",
                "",
            ])
            for i, goal in enumerate(memory.research_goals, 1):
                lines.append(f"{i}. {goal}")
            lines.append("")

        # Constraints
        if memory.constraints:
            lines.extend([
                "## Constraints",
                "",
            ])
            for key, value in memory.constraints.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        # Corpus Context
        if memory.corpus_context:
            lines.extend([
                "## Corpus Context",
                "",
            ])

            if memory.corpus_context.entities:
                entities_str = ", ".join(memory.corpus_context.entities[:20])
                lines.append(f"**Key Entities**: {entities_str}  ")

            if memory.corpus_context.document_ids:
                lines.append(f"**Documents Indexed**: {len(memory.corpus_context.document_ids)}  ")

            if memory.corpus_context.summary:
                lines.extend([
                    "",
                    "**Corpus Summary**:",
                    "",
                    memory.corpus_context.summary,
                    "",
                ])

        # Session Status
        lines.extend([
            "## Session Status",
            "",
            f"**Current Stage**: {memory.current_stage}  ",
        ])

        if memory.agents_completed:
            completed_str = ", ".join(memory.agents_completed)
            lines.append(f"**Agents Completed**: {completed_str}  ")

        lines.extend([
            "",
            "---",
            "",
            "## Agent Notes",
            "",
            "*(Agents append findings, hypotheses, and observations here across iterations.)*",
        ])

        return "\n".join(lines)
