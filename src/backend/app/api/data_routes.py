"""Slim non-orchestrator API routes for the Atlas Framework workspace UI.

Per-project isolation: each project owns its own folder, SQLite, and Qdrant
storage. The list-of-known-projects lives in the JSON registry at app level.
Endpoints that touch per-project data require a `project_id` so the right
database/qdrant client can be resolved.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.core import project_paths, registry
from app.core.config import settings
from app.core.database import (
    Document,
    close_project_engine,
    get_project_session,
    init_project_db,
)
from app.core.qdrant_store import close_qdrant_client
from app.services.chat import ChatService
from app.services.context_engine import ContextEngineService
from app.services.document import DocumentService
from app.services.graph import GraphService
from app.services.ingest import IngestionService
from app.services.llm import get_llm_service
from app.services.workspace import WorkspaceService
from app.services.workspace_manager import get_workspace_manager

router = APIRouter()
logger = logging.getLogger(__name__)

SUPPORTED_FILE_EXTENSIONS = (".pdf", ".txt", ".docx", ".doc")

_document_service: Optional[DocumentService] = None
_graph_service: Optional[GraphService] = None
_ingestion_service: Optional[IngestionService] = None
_workspace_service: Optional[WorkspaceService] = None
_context_engine_service: Optional[ContextEngineService] = None
_chat_service: Optional[ChatService] = None


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


def get_graph_service() -> GraphService:
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service


def get_workspace_service() -> WorkspaceService:
    global _workspace_service
    if _workspace_service is None:
        _workspace_service = WorkspaceService()
    return _workspace_service


def get_context_engine_service() -> ContextEngineService:
    global _context_engine_service
    if _context_engine_service is None:
        _context_engine_service = ContextEngineService()
    return _context_engine_service


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: str


class ModelLoadRequest(BaseModel):
    model_name: str


class RouteIntentRequest(BaseModel):
    query: str
    project_id: str


class RouteIntentResponse(BaseModel):
    intent: str


class WorkspaceDraftSaveRequest(BaseModel):
    content: Dict[str, Any]


class ChatRequest(BaseModel):
    query: str
    project_id: Optional[str] = None


class ChatCitation(BaseModel):
    source: str
    page: int
    doc_id: Optional[str] = None
    text: Optional[str] = None


class ChatRelationship(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    reasoning: str
    citations: List[ChatCitation] = Field(default_factory=list)
    relationships: List[ChatRelationship] = Field(default_factory=list)
    context_sources: Dict[str, int] = Field(default_factory=dict)


class WorkspaceRegisterRequest(BaseModel):
    path: str


def _list_model_inventory(models_dir: Path) -> Dict[str, Any]:
    if not models_dir.exists():
        return {
            "models_dir": str(models_dir),
            "llm": [],
            "embeddings": [],
            "ner": [],
            "other": [],
            "message": "Models directory not found",
        }

    llm_models = [{"name": path.name, "path": str(path)} for path in sorted(models_dir.glob("*.gguf"))]
    embedding_models = [
        {"name": path.name, "path": str(path)}
        for path in sorted(models_dir.glob("nomic-embed-text*"))
        if path.is_dir()
    ]
    ner_models = [
        {"name": path.name, "path": str(path)}
        for path in sorted(models_dir.glob("gliner*"))
        if path.is_dir()
    ]
    known_paths = {
        *[item["path"] for item in llm_models],
        *[item["path"] for item in embedding_models],
        *[item["path"] for item in ner_models],
    }
    other_items = [
        {"name": path.name, "path": str(path)}
        for path in sorted(models_dir.iterdir())
        if str(path) not in known_paths
    ]

    return {
        "models_dir": str(models_dir),
        "llm": llm_models,
        "embeddings": embedding_models,
        "ner": ner_models,
        "other": other_items,
    }


def _classify_intent(query: str) -> str:
    lowered = query.lower()
    if any(token in lowered for token in ("hypothesis", "synthesize", "compare", "evaluate", "tradeoff")):
        return "BROAD_RESEARCH"
    if any(token in lowered for token in ("graph", "relationship", "pathway", "connection", "knowledge")):
        return "DEEP_DISCOVERY"
    return "SIMPLE"


def _project_response(entry: registry.ProjectEntry) -> ProjectResponse:
    return ProjectResponse(
        id=entry.id,
        name=entry.name,
        description=entry.description,
        created_at=entry.created_at,
    )


def _resolve_project_or_404(project_id: str) -> registry.ProjectEntry:
    entry = registry.get_project(project_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return entry


async def _process_document_background(
    file_path: str,
    filename: str,
    project_id: str,
    doc_id: str,
) -> None:
    ingestion_service = get_ingestion_service()
    graph_service = get_graph_service()
    try:
        await ingestion_service.ingest_document(
            file_path=file_path,
            filename=filename,
            project_id=project_id,
            predefined_doc_id=doc_id,
        )
        graph_service.invalidate_cache()
    except Exception as exc:
        logger.error("Background ingestion failed for %s: %s", filename, exc, exc_info=True)
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            try:
                file_path_obj.unlink()
            except Exception:
                pass


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    return _list_model_inventory(Path(settings.MODELS_DIR))


@router.get("/models/status")
async def get_model_status() -> Dict[str, Any]:
    return get_llm_service().get_status()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Legacy health endpoint kept for compatibility with older UI surfaces."""
    return {
        "status": "healthy",
        "message": "Atlas Framework API is online.",
    }


@router.post("/models/load")
async def load_model(body: ModelLoadRequest) -> Dict[str, Any]:
    llm = get_llm_service()
    try:
        await llm.load_model(body.model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return llm.get_status()


@router.get("/models/registry")
async def get_model_registry() -> Dict[str, Any]:
    llm = get_llm_service()
    return {
        "local": [{"name": name, "source": "local", "provider": "local"} for name in llm.list_available_models()],
        "api": llm.list_available_api_models(),
        "active": llm.get_status(),
    }


@router.post("/chat", response_model=ChatResponse)
@router.post("/api/chat", response_model=ChatResponse)
async def chat_query(body: ChatRequest) -> ChatResponse:
    """Grounded chat endpoint — single orchestrator path."""
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    if not body.project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    _resolve_project_or_404(body.project_id)

    try:
        result = await get_chat_service().chat(
            body.query.strip(),
            project_id=body.project_id,
        )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chat query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat query failed: {exc}") from exc


# ----------------------------------------------------------------------
# Project lifecycle (registry-backed)
# ----------------------------------------------------------------------


@router.post("/projects", response_model=ProjectResponse)
async def create_project(body: ProjectCreate) -> ProjectResponse:
    """Provision a new self-contained project folder and register it."""
    if registry.find_by_name(body.name):
        raise HTTPException(status_code=409, detail=f"Project '{body.name}' already exists")

    project_id = str(uuid.uuid4())
    folder = project_paths.default_project_root(project_id)

    try:
        project_paths.ensure_project_folder(folder)
        # Write the workspace manifest before the registry entry, so partial
        # failures leave a recognizable folder rather than a dangling registry row.
        get_workspace_manager().create_folder(project_id, body.name, body.description)
        entry = registry.add_project(
            project_id=project_id,
            name=body.name,
            description=body.description,
            path=folder,
        )
        init_project_db(project_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Project creation failed: %s", exc)
        # Best-effort cleanup so the next attempt isn't blocked by a stale folder.
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        registry.remove_project(project_id)
        raise HTTPException(status_code=500, detail=f"Error creating project: {exc}") from exc

    return _project_response(entry)


@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects() -> List[ProjectResponse]:
    entries = registry.list_projects()
    entries.sort(key=lambda e: e.created_at, reverse=True)
    return [_project_response(e) for e in entries]


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str) -> ProjectResponse:
    return _project_response(_resolve_project_or_404(project_id))


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str) -> Dict[str, str]:
    """Close all open handles for the project and rm-rf its folder."""
    entry = _resolve_project_or_404(project_id)

    # Release SQLite + Qdrant locks so Windows lets us delete the files.
    close_project_engine(project_id)
    close_qdrant_client(project_id)

    folder = entry.folder()
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)

    registry.remove_project(project_id)
    return {"status": "success", "message": f"Deleted project {project_id}"}


# ----------------------------------------------------------------------
# Ingestion + documents
# ----------------------------------------------------------------------


@router.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: str = Query(..., description="Project to ingest into"),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Strip any path components — only the basename is allowed. Prevents
    # `../../etc/foo` and absolute paths from escaping `upload_dir`.
    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename in {".", ".."} or safe_filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not safe_filename.lower().endswith(SUPPORTED_FILE_EXTENSIONS):
        supported = ", ".join(SUPPORTED_FILE_EXTENSIONS)
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported formats: {supported}")

    _resolve_project_or_404(project_id)

    manager = get_workspace_manager()
    # Self-heal for projects whose `files/` dir was wiped externally.
    upload_dir = manager.files_path(project_id)
    upload_path = (upload_dir / safe_filename).resolve()
    if upload_dir.resolve() not in upload_path.parents:
        raise HTTPException(status_code=400, detail="Resolved upload path escaped project directory")
    ingestion_service = get_ingestion_service()

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_hash = ingestion_service._calculate_hash(str(upload_path))
        session = get_project_session(project_id)
        try:
            existing = (
                session.query(Document)
                .filter(Document.file_hash == file_hash)
                .first()
            )
            if existing:
                if upload_path.exists():
                    upload_path.unlink()
                return {
                    "status": "duplicate",
                    "message": "Document already exists",
                    "filename": safe_filename,
                    "doc_id": str(existing.id),
                }

            file_size = upload_path.stat().st_size
            _, mime_type = ingestion_service._get_file_type_and_mime(safe_filename)
            doc_id = str(uuid.uuid4())
            document = Document(
                id=doc_id,
                filename=safe_filename,
                file_hash=file_hash,
                file_path=str(upload_path),
                file_size=file_size,
                mime_type=mime_type,
                status="processing",
                project_id=project_id,
                uploaded_at=datetime.utcnow(),
            )
            session.add(document)
            session.commit()
        finally:
            session.close()

        background_tasks.add_task(
            _process_document_background,
            str(upload_path),
            safe_filename,
            project_id,
            doc_id,
        )
        return {
            "status": "processing",
            "message": f"Document '{safe_filename}' uploaded and processing started",
            "filename": safe_filename,
            "doc_id": doc_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ingestion error for %s: %s", safe_filename, exc, exc_info=True)
        if upload_path.exists():
            try:
                upload_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc


@router.get("/files", response_model=List[Dict[str, Any]])
async def list_files(
    project_id: str = Query(..., description="Project scope"),
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    _resolve_project_or_404(project_id)
    return get_document_service().list_documents(
        project_id=project_id, status=status, limit=100
    )


@router.get("/files/{doc_id}")
async def get_file(
    doc_id: str,
    project_id: str = Query(..., description="Project that owns this document"),
) -> Any:
    _resolve_project_or_404(project_id)
    file_response = get_document_service().get_document_file(project_id, doc_id)
    if not file_response:
        raise HTTPException(status_code=404, detail="Document not found")
    return file_response


@router.delete("/files/{doc_id}")
async def delete_file(
    doc_id: str,
    project_id: str = Query(..., description="Project that owns this document"),
) -> Dict[str, str]:
    _resolve_project_or_404(project_id)
    success = get_document_service().delete_document(project_id, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    get_graph_service().invalidate_cache()
    return {"status": "success", "message": f"Deleted document {doc_id}"}


# ----------------------------------------------------------------------
# Graph / entities
# ----------------------------------------------------------------------


@router.get("/entities", response_model=List[Dict[str, Any]])
async def list_entities(
    project_id: str = Query(..., description="Project scope"),
    entity_type: Optional[str] = None,
    document_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
) -> List[Dict[str, Any]]:
    _resolve_project_or_404(project_id)
    return get_graph_service().list_nodes(
        project_id=project_id,
        label=entity_type,
        document_id=document_id,
        limit=limit,
    )


@router.get("/entities/{entity_id}/relationships", response_model=List[Dict[str, Any]])
async def get_entity_relationships(
    entity_id: str,
    project_id: str = Query(..., description="Project scope"),
    direction: str = "both",
) -> List[Dict[str, Any]]:
    _resolve_project_or_404(project_id)
    return get_graph_service().get_node_relationships(
        project_id, entity_id, direction=direction
    )


@router.get("/graph/types")
async def get_entity_types(
    project_id: str = Query(..., description="Project scope"),
) -> Dict[str, Any]:
    _resolve_project_or_404(project_id)
    return {"entity_types": get_graph_service().get_node_types(project_id=project_id)}


@router.get("/graph/full")
async def get_full_graph(
    project_id: str = Query(..., description="Project scope"),
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    _resolve_project_or_404(project_id)
    return await get_graph_service().get_full_graph_cached(
        project_id=project_id, document_id=document_id
    )


@router.get("/files/{doc_id}/structure")
async def get_document_structure(
    doc_id: str,
    project_id: str = Query(..., description="Project that owns this document"),
) -> Dict[str, Any]:
    _resolve_project_or_404(project_id)
    result = await get_context_engine_service().get_document_structure(project_id, doc_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@router.get("/files/{doc_id}/related")
async def get_related_passages(
    doc_id: str,
    project_id: str = Query(..., description="Project that owns this document"),
    text: str = Query(..., min_length=3),
    limit: int = Query(8, ge=1, le=20),
) -> List[Dict[str, Any]]:
    _resolve_project_or_404(project_id)
    return await get_context_engine_service().get_related_passages(
        project_id=project_id,
        doc_id=doc_id,
        text=text,
        limit=limit,
    )


@router.get("/files/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    project_id: str = Query(..., description="Project that owns this document"),
    page: Optional[int] = Query(None, ge=1),
    limit: int = Query(50, ge=1, le=200),
) -> List[Dict[str, Any]]:
    _resolve_project_or_404(project_id)
    return await get_context_engine_service().get_document_chunks(
        project_id=project_id,
        doc_id=doc_id,
        page_number=page,
        limit=limit,
    )


@router.post("/api/context")
async def get_context_suggestions(request: Dict[str, Any]) -> Dict[str, Any]:
    project_id = request.get("project_id") or ""
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    _resolve_project_or_404(project_id)
    suggestions = await get_context_engine_service().get_context_suggestions(
        project_id=project_id,
        selected_text=request.get("selected_text"),
        current_doc_id=request.get("current_doc_id"),
        current_page=request.get("current_page"),
    )
    return {"status": "success", "suggestions": suggestions}


# ----------------------------------------------------------------------
# Workspace drafts
# ----------------------------------------------------------------------


@router.get("/api/workspace/{project_id}/drafts")
async def list_workspace_drafts(project_id: str) -> List[Dict[str, Any]]:
    _resolve_project_or_404(project_id)
    return get_workspace_service().list_drafts(project_id)


@router.get("/api/workspace/{project_id}/drafts/{draft_id}")
async def get_workspace_draft(project_id: str, draft_id: str) -> Dict[str, Any]:
    _resolve_project_or_404(project_id)
    draft = get_workspace_service().get_draft(project_id, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft


@router.post("/api/workspace/{project_id}/drafts/{draft_id}")
async def save_workspace_draft(
    project_id: str,
    draft_id: str,
    body: WorkspaceDraftSaveRequest,
) -> Dict[str, Any]:
    _resolve_project_or_404(project_id)
    return get_workspace_service().save_draft(project_id, draft_id, body.content)


@router.delete("/api/workspace/{project_id}/drafts/{draft_id}")
async def delete_workspace_draft(project_id: str, draft_id: str) -> Dict[str, Any]:
    _resolve_project_or_404(project_id)
    deleted = get_workspace_service().delete_draft(project_id, draft_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Draft not found")
    return {"status": "deleted", "id": draft_id}


# ----------------------------------------------------------------------
# Workspace folder export / import / register
# ----------------------------------------------------------------------


def _safe_filename(name: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "-" for c in name.strip()]
    slug = "".join(keep).strip("-") or "workspace"
    return slug[:64]


@router.get("/api/workspaces/{project_id}/folder")
async def get_workspace_folder(project_id: str) -> Dict[str, Any]:
    """Return the on-disk folder + manifest for a project."""
    entry = _resolve_project_or_404(project_id)

    folder = entry.folder()
    if not folder.exists():
        project_paths.ensure_project_folder(folder)
        get_workspace_manager().create_folder(project_id, entry.name, entry.description)

    return {
        "workspace_id": project_id,
        "path": str(folder),
        "files_path": str(project_paths.project_files_path(folder)),
        "drafts_path": str(project_paths.project_drafts_path(folder)),
        "manifest": get_workspace_manager().read_manifest(project_id),
    }


@router.get("/api/workspaces/{project_id}/export")
async def export_workspace(project_id: str) -> Response:
    """Stream a `.atlas` zip of the entire project folder."""
    entry = _resolve_project_or_404(project_id)

    # Flush handles so the SQLite DB inside the folder is consistent on disk.
    close_project_engine(project_id)
    close_qdrant_client(project_id)

    try:
        archive = get_workspace_manager().export_archive(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Workspace export failed for %s: %s", project_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    filename = f"{_safe_filename(entry.name)}.atlas"
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/api/workspaces/import")
async def import_workspace(file: UploadFile = File(...)) -> ProjectResponse:
    """Import a `.atlas` archive: unzip into a fresh project folder + register."""
    try:
        archive_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {exc}") from exc

    if not archive_bytes:
        raise HTTPException(status_code=400, detail="Archive is empty")

    try:
        entry = get_workspace_manager().import_archive(archive_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Workspace import failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import failed: {exc}") from exc

    return _project_response(entry)


@router.post("/api/workspaces/register", response_model=ProjectResponse)
async def register_workspace(body: WorkspaceRegisterRequest) -> ProjectResponse:
    """Register an existing project folder with the app — VSCode 'open folder'.

    The folder must already contain a `workspace.json` manifest.
    """
    try:
        entry = get_workspace_manager().register_existing(Path(body.path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Workspace register failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Register failed: {exc}") from exc

    return _project_response(entry)


@router.post("/api/route", response_model=RouteIntentResponse)
async def route_intent_endpoint(request: RouteIntentRequest) -> RouteIntentResponse:
    return RouteIntentResponse(intent=_classify_intent(request.query))
