"""FastAPI route handlers for Atlas Sidecar API."""
from __future__ import annotations  # Deferred annotation evaluation — type hints are strings, not evaluated at import
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import shutil

# Heavy ML service imports are deferred to ensure_services() to avoid importing
# the entire ML library tree (transformers, sentence_transformers, qdrant, docling, etc.)
# at module load time. This cuts ~90s off startup.
from app.services.llm import get_llm_service
from app.core.config import settings, get_env_path
router = APIRouter()

# Services initialized lazily
chat_service = None
ingestion_service = None
document_service = None
graph_service = None
context_engine_service = None
workspace_service = None

logger = logging.getLogger(__name__)


# ============================================================
# FOLLOW-UP TAXONOMY MODELS & UTILITIES (D4)
# ============================================================

class FollowUpSuggestion(BaseModel):
    label: str
    query: str


class FollowUpSuggestions(BaseModel):
    depth: FollowUpSuggestion
    breadth: FollowUpSuggestion
    opposition: FollowUpSuggestion


FOLLOW_UPS_REGEX = re.compile(r'FOLLOW_UPS:\s*(\{.*?\})\s*$', re.DOTALL)

FOLLOW_UPS_PROMPT_APPENDIX = """

After your response, output a JSON block on a new line:
FOLLOW_UPS: {"depth": {"label": "...", "query": "..."}, "breadth": {"label": "...", "query": "..."}, "opposition": {"label": "...", "query": "..."}}
"""


def parse_follow_ups(raw_text: str) -> tuple[str, Optional[FollowUpSuggestions]]:
    """Parse and strip FOLLOW_UPS block from LLM output.
    
    Returns:
        Tuple of (cleaned_text, follow_ups_object_or_none)
    """
    match = FOLLOW_UPS_REGEX.search(raw_text)
    if not match:
        return raw_text.strip(), None
    
    json_str = match.group(1)
    cleaned_text = raw_text[:match.start()].strip()
    
    try:
        data = json.loads(json_str)
        follow_ups = FollowUpSuggestions(
            depth=FollowUpSuggestion(label=data['depth']['label'], query=data['depth']['query']),
            breadth=FollowUpSuggestion(label=data['breadth']['label'], query=data['breadth']['query']),
            opposition=FollowUpSuggestion(label=data['opposition']['label'], query=data['opposition']['query']),
        )
        return cleaned_text, follow_ups
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse FOLLOW_UPS JSON: {e}")
        return raw_text.strip(), None


async def monitor_disconnect(request: Request, cancel_event: asyncio.Event, poll_interval: float = 0.5):
    """Poll for client disconnect and set the cancel event when detected."""
    while not cancel_event.is_set():
        if await request.is_disconnected():
            cancel_event.set()
            logger.info("Client disconnected - cancellation signal sent")
            return
        await asyncio.sleep(poll_interval)



def ensure_services():
    """Initialize services lazily - imports are deferred to avoid slow startup."""
    global chat_service, ingestion_service, document_service, graph_service, context_engine_service, workspace_service

    if document_service is None:
        from app.services.document import DocumentService
        document_service = DocumentService()
        logger.info("DocumentService initialized")
    if graph_service is None:
        from app.services.graph import GraphService
        graph_service = GraphService()
        logger.info("GraphService initialized")
    if chat_service is None:
        from app.services.chat import ChatService
        chat_service = ChatService()
        logger.info("ChatService initialized")
    if ingestion_service is None:
        from app.services.ingest import IngestionService
        ingestion_service = IngestionService()
        logger.info("IngestionService initialized")
    if context_engine_service is None:
        from app.services.context_engine import ContextEngineService
        context_engine_service = ContextEngineService()
        logger.info("ContextEngineService initialized")
    if workspace_service is None:
        from app.services.workspace import WorkspaceService
        workspace_service = WorkspaceService()
        logger.info("WorkspaceService initialized")


# ============================================================
# PYDANTIC MODELS
# ============================================================

class ChatRequest(BaseModel):
    query: str
    project_id: Optional[str] = None
    stage_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    answer: str
    reasoning: str
    citations: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    context_sources: Dict[str, Any]
    # Follow-up taxonomy (D4)
    follow_ups: Optional[FollowUpSuggestions] = None


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: str


class SwarmRequest(BaseModel):
    project_id: str
    query: str
    

class SwarmStreamRequest(BaseModel):
    project_id: str
    query: str
    session_id: Optional[str] = None
    stage_context: Optional[Dict[str, Any]] = None


class SwarmResponse(BaseModel):
    brain_used: str  # "librarian", "navigator", or "cortex"
    hypothesis: str
    evidence: List[Dict[str, Any]]
    reasoning_trace: List[str]
    status: str
    # Task 0.3: New fields for Navigator 2.0 / Cortex 2.0
    confidence_score: Optional[float] = None
    iterations: Optional[int] = None
    contradictions: List[Dict[str, Any]] = []
    # Follow-up taxonomy (D4)
    follow_ups: Optional[FollowUpSuggestions] = None


class ModelLoadRequest(BaseModel):
    model_name: str


class ModelLoadResponse(BaseModel):
    active_model: Optional[str]
    model_type: str
    device: str
    gpu_layers: int
    fallback: bool
    model_source: str = "local"           # Atlas 3.0: "local" or "api"
    api_models_available: bool = False     # Atlas 3.0: Whether API models are configured


class ContextRequest(BaseModel):
    project_id: str
    selected_text: Optional[str] = None
    current_doc_id: Optional[str] = None
    current_page: Optional[int] = None


class ConfigKeys(BaseModel):
    has_openai: bool
    has_anthropic: bool
    has_deepseek: bool
    has_minimax: bool

class ConfigKeysUpdate(BaseModel):
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    MINIMAX_API_KEY: Optional[str] = None


# Discovery OS — Stage 7 Feedback (D5)
class DiscoveryFeedbackRequest(BaseModel):
    hit_id: str
    epoch_id: str
    result_name: str
    result_value: float
    unit: str
    passed: bool
    notes: str


class DiscoveryFeedbackResponse(BaseModel):
    status: str
    updated_node_ids: List[str]


# Discovery OS — Wetlab Loop
class RecordResultRequest(BaseModel):
    compound_id: str
    smiles: Optional[str] = None
    assay_type: str
    value: Optional[float] = None
    units: Optional[str] = None
    active: Optional[bool] = None
    notes: Optional[str] = None


# Discovery OS — Domain Tools (B3)
class CapabilityGapRequest(BaseModel):
    run_id: str
    stage: int
    required_function: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    standard_reference: Optional[str] = None


class ToolRegisterRequest(BaseModel):
    gap_id: str
    method: str  # 'local_script' | 'api_endpoint' | 'plugin' | 'skip'
    config: Dict[str, Any]

# Discovery OS — Spectroscopy (B6)
class RoutePlanningRequest(BaseModel):
    candidate_id: str
    smiles: str
    epoch_id: str

class ValidateSpectroscopyRequest(BaseModel):
    hit_id: str
    file_content: str
    file_type: str

# Discovery OS — Candidate Generation & Screen (C6)
class GenerateCandidatesRequest(BaseModel):
    session_id: str
    epoch_id: str
    project_id: Optional[str] = None

class ScreenCandidatesRequest(BaseModel):
    session_id: str
    epoch_id: str
    smiles_list: List[str]


# Discovery OS — Editor Structural Completion (D3)
class SuggestStructureRequest(BaseModel):
    text: str
    cursor_position: int
    stage_context: Optional[Dict[str, Any]] = None


class SuggestStructureResponse(BaseModel):
    suggestion: Optional[str] = None


class ParseBrainstormRequest(BaseModel):
    text: str
    domain: str


class ParseBrainstormResponse(BaseModel):
    objective: str
    propertyConstraints: List[Dict[str, Any]]
    domainSpecificConstraints: Dict[str, str]

# ============================================================
# HEALTH / INFO
# ============================================================

@router.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Atlas API - Agentic MoE Knowledge Engine",
        "version": "3.0.0-sidecar",
        "architecture": {
            "database": "SQLite (embedded)",
            "vector_store": "Qdrant (embedded, in-process)",
            "knowledge_graph": "Evidence-Bound GraphRAG",
            "llm": "Hybrid (local GGUF + LiteLLM API)",
            "workspace": "Persistent Markdown/JSON Drafts",
            "swarm": "LangGraph MoE (Supervisor + Expert Agents)",
            "retrieval": "BM25 + Vector + RRF + FlashRank",
        },
    }


@router.get("/health")
async def health_check():
    """Lightweight health check - confirms the server is running.
    
    Does NOT initialize heavy ML services; that happens lazily on first
    real request.  Tauri's start_backend loop polls this endpoint.
    """
    return {
        "status": "healthy",
        "services": {
            "api": "online",
            "database": "sqlite",
            "vector_store": "qdrant_embedded",
            "llm": "bundled",
        },
    }


@router.get("/config/keys", response_model=ConfigKeys)
async def get_config_keys():
    """Return flags indicating which API keys are stored."""
    return {
        "has_openai": bool(settings.OPENAI_API_KEY),
        "has_anthropic": bool(settings.ANTHROPIC_API_KEY),
        "has_deepseek": bool(settings.DEEPSEEK_API_KEY),
        "has_minimax": bool(settings.MINIMAX_API_KEY),
    }


class ConfigKeysVerifyResponse(BaseModel):
    """Result of verifying each provider's API key with a minimal API call."""
    openai: bool
    anthropic: bool
    deepseek: bool
    minimax: bool


@router.post("/config/keys/verify", response_model=ConfigKeysVerifyResponse)
async def verify_config_keys():
    """Verify each configured API key with a minimal API call. Does not persist state."""
    import os
    result = {"openai": False, "anthropic": False, "deepseek": False, "minimax": False}

    _litellm = None
    try:
        import litellm as _litellm
    except ImportError:
        pass  # Fall back to direct clients below

    async def _verify_direct_openai_compat(key: str, base_url: str, model: str) -> bool:
        """Verify key using openai-compatible client (no litellm required)."""
        if not key or not key.strip():
            return False
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=key, base_url=base_url)
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                timeout=20.0,
            )
            return bool(resp and resp.choices and resp.choices[0].message)
        except Exception as e:
            logger.warning("Direct API key verify failed (%s): %s", model, e)
            return False

    async def _verify_litellm(model: str, env_key: str) -> bool:
        key = getattr(settings, env_key, None) or ""
        if not key or not key.strip():
            return False
        os.environ[env_key] = key
        try:
            resp = await _litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                timeout=20.0,
            )
            verified = bool(resp and resp.choices and resp.choices[0].message)
            logger.info(f"API key verify {env_key} ({model}): {'OK' if verified else 'empty response'}")
            return verified
        except Exception as e:
            logger.warning(
                "API key verify %s (%s) failed: %s. Key may still work; verification is best-effort.",
                env_key, model, e,
            )
            return False

    # DeepSeek: use direct openai-compatible client (no litellm required)
    if settings.DEEPSEEK_API_KEY:
        result["deepseek"] = await _verify_direct_openai_compat(
            settings.DEEPSEEK_API_KEY,
            "https://api.deepseek.com",
            "deepseek-chat",
        )

    if _litellm is not None:
        if settings.OPENAI_API_KEY:
            result["openai"] = await _verify_litellm("gpt-4o-mini", "OPENAI_API_KEY")
        if settings.ANTHROPIC_API_KEY:
            result["anthropic"] = await _verify_litellm("claude-3-haiku-20240307", "ANTHROPIC_API_KEY")
        if settings.MINIMAX_API_KEY:
            result["minimax"] = await _verify_litellm("minimax/MiniMax-M2.5", "MINIMAX_API_KEY")

    return ConfigKeysVerifyResponse(**result)


@router.post("/config/keys")
async def update_config_keys(keys: ConfigKeysUpdate):
    """Update API keys in the .env file and application memory."""
    env_path = get_env_path()
    
    # Read existing env
    env_lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            env_lines = f.readlines()
            
    import os as _os
    updates = {}
    if keys.OPENAI_API_KEY is not None:
        updates["OPENAI_API_KEY"] = keys.OPENAI_API_KEY
        settings.OPENAI_API_KEY = keys.OPENAI_API_KEY
    if keys.ANTHROPIC_API_KEY is not None:
        updates["ANTHROPIC_API_KEY"] = keys.ANTHROPIC_API_KEY
        settings.ANTHROPIC_API_KEY = keys.ANTHROPIC_API_KEY
    if keys.DEEPSEEK_API_KEY is not None:
        updates["DEEPSEEK_API_KEY"] = keys.DEEPSEEK_API_KEY
        settings.DEEPSEEK_API_KEY = keys.DEEPSEEK_API_KEY
    if keys.MINIMAX_API_KEY is not None:
        updates["MINIMAX_API_KEY"] = keys.MINIMAX_API_KEY
        settings.MINIMAX_API_KEY = keys.MINIMAX_API_KEY

    # Sync to os.environ so LiteLLM can find the keys at runtime
    # (LiteLLM reads env vars, not the settings object)
    for env_key, env_val in updates.items():
        if env_val:
            _os.environ[env_key] = env_val
        elif env_key in _os.environ:
            del _os.environ[env_key]
        
    if not updates:
        return {"status": "success", "message": "No keys provided to update"}
        
    # Write updated env
    try:
        new_lines = []
        updated_keys = set()
        for line in env_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue
                
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                if key in updates:
                    new_lines.append(f'{key}="{updates[key]}"\n')
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
                
        # Append keys that weren't already in the file
        for key, value in updates.items():
            if key not in updated_keys:
                new_lines.append(f'{key}="{value}"\n')
                
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
            
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to update API keys: {e}")
        raise HTTPException(status_code=500, detail="Failed to save API keys configuration")


@router.get("/models")
async def list_models():
    """List bundled/available models in the models directory."""
    models_dir = Path(settings.MODELS_DIR)
    if not models_dir.exists():
        return {
            "models_dir": str(models_dir),
            "llm": [],
            "embeddings": [],
            "ner": [],
            "other": [],
            "message": "Models directory not found",
        }

    llm_models = [{"name": p.name, "path": str(p)} for p in sorted(models_dir.glob("*.gguf"))]
    embedding_models = [
        {"name": p.name, "path": str(p)}
        for p in sorted(models_dir.glob("nomic-embed-text*"))
        if p.is_dir()
    ]
    ner_models = [
        {"name": p.name, "path": str(p)}
        for p in sorted(models_dir.glob("gliner*"))
        if p.is_dir()
    ]
    known_paths = {
        *[m["path"] for m in llm_models],
        *[m["path"] for m in embedding_models],
        *[m["path"] for m in ner_models],
    }
    other_items = [
        {"name": p.name, "path": str(p)}
        for p in sorted(models_dir.iterdir())
        if str(p) not in known_paths
    ]

    return {
        "models_dir": str(models_dir),
        "llm": llm_models,
        "embeddings": embedding_models,
        "ner": ner_models,
        "other": other_items,
    }


@router.get("/models/status", response_model=ModelLoadResponse)
async def get_model_status():
    """Return current LLM status (active model, device, and fallback state)."""
    llm = get_llm_service()
    return llm.get_status()


@router.post("/models/load", response_model=ModelLoadResponse)
async def load_model(body: ModelLoadRequest):
    """Load a model - either a local GGUF or an API model.

    Atlas 3.0: Accepts both local model filenames (e.g., "Phi-3.5-mini-instruct.Q4_K_M.gguf")
    and API model identifiers (e.g., "deepseek/deepseek-chat").
    Models with "/" are treated as API models and routed through LiteLLM.
    """
    llm = get_llm_service()
    try:
        # Atlas 3.0: Route API models (containing "/") to load_api_model
        if "/" in body.model_name:
            await llm.load_api_model(body.model_name)
        else:
            await llm.load_model(body.model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return llm.get_status()


@router.get("/models/registry")
async def get_model_registry():
    """Atlas 3.0: Get the full model registry (local + cloud API models).

    Returns grouped lists for the frontend model selector:
    - local: GGUF models in the models directory
    - api: Cloud API models available via LiteLLM (with key status)
    - active: Currently loaded model info
    """
    llm = get_llm_service()

    # Local models
    local_models = [
        {"name": name, "source": "local", "provider": "local"}
        for name in llm.list_available_models()
    ]

    # API models
    api_models = llm.list_available_api_models()

    # Current status
    status = llm.get_status()

    return {
        "local": local_models,
        "api": api_models,
        "active": status,
    }


# ============================================================
# PROJECT MANAGEMENT
# ============================================================

@router.post("/projects", response_model=ProjectResponse)
async def create_project(body: ProjectCreate):
    """Create a new research project."""
    from app.core.database import get_session, Project
    import uuid
    from datetime import datetime

    session = get_session()
    try:
        existing = session.query(Project).filter(Project.name == body.name).first()
        if existing:
            raise HTTPException(status_code=409, detail=f"Project '{body.name}' already exists")

        project = Project(
            id=str(uuid.uuid4()),
            name=body.name,
            description=body.description,
            created_at=datetime.utcnow(),
        )
        session.add(project)
        session.commit()
        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            created_at=project.created_at.isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating project: {str(e)}")
    finally:
        session.close()


@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects():
    """List all projects."""
    from app.core.database import get_session, Project

    session = get_session()
    try:
        projects = session.query(Project).order_by(Project.created_at.desc()).all()
        return [
            ProjectResponse(
                id=p.id,
                name=p.name,
                description=p.description,
                created_at=p.created_at.isoformat() if p.created_at else "",
            )
            for p in projects
        ]
    finally:
        session.close()


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and all its data."""
    from app.core.database import get_session, Project

    session = get_session()
    try:
        project = session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        session.delete(project)  # Cascade deletes documents, nodes, edges
        session.commit()
        return {"status": "success", "message": f"Deleted project {project_id}"}
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting project: {str(e)}")
    finally:
        session.close()


# ============================================================
# CHAT (Librarian RAG)
# ============================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Query the knowledge layer using hybrid RAG."""
    from app.services.stage_context import format_stage_context_preamble, set_stage_context_preamble
    ensure_services()
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Chat service unavailable")
    set_stage_context_preamble(format_stage_context_preamble(request.stage_context))
    try:
        result = await chat_service.chat(request.query, project_id=request.project_id)

        # Parse and strip FOLLOW_UPS from answer (D4)
        answer = result.get("answer", "")
        cleaned_answer, follow_ups = parse_follow_ups(answer)
        result["answer"] = cleaned_answer
        result["follow_ups"] = follow_ups

        return ChatResponse(**result)
    except Exception as e:
        logger.exception(f"Query error for request: {request.query[:100]}...")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")
    finally:
        set_stage_context_preamble(None)


@router.post("/api/discovery/parse-brainstorm", response_model=ParseBrainstormResponse)
async def parse_brainstorm(request: ParseBrainstormRequest):
    """Parse unstructured brainstorm text into structured ProjectTargetParams."""
    llm = get_llm_service()
    
    prompt = f"""You are an expert scientific assistant. The researcher has provided a brainstorming text for a new project in the domain '{request.domain}'.
Extract the core objective and constraints into a JSON object matching this schema:
{{
  "objective": "A concise, formal statement of the target objective",
  "propertyConstraints": [
    {{"property": "property_name", "operator": "< | > | <= | >= | between", "value": 123}}
  ],
  "domainSpecificConstraints": {{
    "Forbidden Scaffolds": "...",
    "Allowed Elements": "..."
  }}
}}

Researcher's Brainstorm:
{request.text}

Output ONLY valid JSON.
"""
    try:
        response_text = await llm.generate(
            prompt=prompt,
            system_prompt="You are a JSON-only extraction agent.",
            temperature=0.1
        )
        # Find JSON block
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return ParseBrainstormResponse(
                objective=data.get("objective", "New Objective"),
                propertyConstraints=data.get("propertyConstraints", []),
                domainSpecificConstraints=data.get("domainSpecificConstraints", {})
            )
        else:
            raise ValueError("No JSON found in LLM response")
    except Exception as e:
        logger.error(f"Failed to parse brainstorm: {e}")
        # Return fallback on error
        return ParseBrainstormResponse(
            objective=request.text[:100] + "...",
            propertyConstraints=[],
            domainSpecificConstraints={}
        )


# ============================================================
# DOCUMENT INGESTION
# ============================================================

# Supported file extensions for document ingestion
SUPPORTED_FILE_EXTENSIONS = ('.pdf', '.txt', '.docx', '.doc')

@router.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_id: Optional[str] = Query(None),
):
    """Upload and process a document (PDF, DOCX, DOC, TXT) through the knowledge layer."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Check if file extension is supported
    file_lower = file.filename.lower()
    if not any(file_lower.endswith(ext) for ext in SUPPORTED_FILE_EXTENSIONS):
        supported = ', '.join(SUPPORTED_FILE_EXTENSIONS)
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported formats: {supported}"
        )

    file_path = Path(settings.UPLOAD_DIR) / file.filename

    ensure_services()
    if ingestion_service is None:
        raise HTTPException(status_code=503, detail="Ingestion service unavailable")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Check for duplicate and create DB row synchronously
        from app.core.database import get_session, Document
        import uuid
        from datetime import datetime
        
        file_hash = ingestion_service._calculate_hash(str(file_path))
        session = get_session()
        doc_id = None
        is_duplicate = False
        
        try:
            dup_query = session.query(Document).filter(Document.file_hash == file_hash)
            if project_id:
                dup_query = dup_query.filter(Document.project_id == project_id)
            existing_doc = dup_query.first()
            
            if existing_doc:
                is_duplicate = True
                doc_id = str(existing_doc.id)
            else:
                doc_id = str(uuid.uuid4())
                file_size = file_path.stat().st_size
                _, mime_type = ingestion_service._get_file_type_and_mime(file.filename)
                
                document = Document(
                    id=doc_id,
                    filename=file.filename,
                    file_hash=file_hash,
                    file_path=str(file_path),
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

        if is_duplicate:
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass
            return {
                "status": "duplicate",
                "message": f"Document already exists",
                "filename": file.filename,
                "doc_id": doc_id,
            }

        background_tasks.add_task(
            _process_document_background,
            str(file_path),
            file.filename,
            ingestion_service,
            project_id,
            doc_id,
        )

        return {
            "status": "processing",
            "message": f"Document '{file.filename}' uploaded and processing started",
            "filename": file.filename,
            "doc_id": doc_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion error for {file.filename}: {str(e)}", exc_info=True)
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


async def _process_document_background(
    file_path: str,
    filename: str,
    ingestion_svc: IngestionService,
    project_id: Optional[str] = None,
    doc_id: Optional[str] = None,
):
    """Background task to process document ingestion."""
    try:
        result = await ingestion_svc.ingest_document(file_path, filename, project_id=project_id, predefined_doc_id=doc_id)
        if result.get("status") == "failed":
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                try:
                    file_path_obj.unlink()
                except Exception:
                    pass
        else:
            # Invalidate graph cache so new entities appear immediately
            if graph_service is not None:
                graph_service.invalidate_cache()
                logger.info(f"Graph cache invalidated after ingesting {filename}")
    except Exception as e:
        logger.error(f"Background ingestion failed for {filename}: {str(e)}", exc_info=True)
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            try:
                file_path_obj.unlink()
            except Exception:
                pass


# ============================================================
# FILE MANAGEMENT
# ============================================================

@router.get("/files", response_model=List[Dict[str, Any]])
async def list_files(
    status: Optional[str] = None,
    project_id: Optional[str] = None,
):
    """List all uploaded documents with their status."""
    ensure_services()
    if document_service is None:
        raise HTTPException(status_code=503, detail="Document service unavailable")
    try:
        return document_service.list_documents(status=status, project_id=project_id, limit=100)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@router.get("/files/{doc_id}")
async def get_file(doc_id: str):
    """Stream a PDF file for viewing."""
    ensure_services()
    if document_service is None:
        raise HTTPException(status_code=503, detail="Document service unavailable")
    try:
        file_response = document_service.get_document_file(doc_id)
        if not file_response:
            raise HTTPException(status_code=404, detail="Document not found")
        return file_response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")


@router.delete("/files/{doc_id}")
async def delete_file(doc_id: str):
    """Delete a document from all knowledge layers."""
    ensure_services()
    if document_service is None:
        raise HTTPException(status_code=503, detail="Document service unavailable")
    try:
        success = document_service.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "success", "message": f"Deleted document {doc_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion error: {str(e)}")


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

@router.get("/entities", response_model=List[Dict[str, Any]])
async def list_entities(
    entity_type: Optional[str] = None,
    document_id: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
):
    """List nodes (entities) in the knowledge graph."""
    ensure_services()
    if graph_service is None:
        raise HTTPException(status_code=503, detail="Graph service unavailable")
    try:
        return graph_service.list_nodes(
            label=entity_type, document_id=document_id, project_id=project_id, limit=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing entities: {str(e)}")


@router.get("/entities/{entity_id}/relationships", response_model=List[Dict[str, Any]])
async def get_entity_relationships(entity_id: str, direction: str = "both"):
    """Get all relationships for a node."""
    ensure_services()
    if graph_service is None:
        raise HTTPException(status_code=503, detail="Graph service unavailable")
    try:
        return graph_service.get_node_relationships(entity_id, direction=direction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting relationships: {str(e)}")


@router.get("/graph/types")
async def get_entity_types(project_id: Optional[str] = None):
    """Get all node labels (types) with counts."""
    ensure_services()
    if graph_service is None:
        raise HTTPException(status_code=503, detail="Graph service unavailable")
    try:
        return {"entity_types": graph_service.get_node_types(project_id=project_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting types: {str(e)}")


@router.get("/graph/full")
async def get_full_graph(
    document_id: Optional[str] = None,
    project_id: Optional[str] = None,
):
    """Get the complete knowledge graph (nodes + edges)."""
    ensure_services()
    if graph_service is None:
        raise HTTPException(status_code=503, detail="Graph service unavailable")
    try:
        return await graph_service.get_full_graph_cached(document_id=document_id, project_id=project_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting graph: {str(e)}")


# ============================================================
# SWARM (Two-Brain Agentic RAG)
# ============================================================

@router.post("/api/swarm/stream")
async def stream_swarm(body: SwarmStreamRequest, request: Request):
    """Stream swarm execution via Server-Sent Events.

    Event types:
      - routing:  {"brain": "navigator", "intent": "DEEP_DISCOVERY"}
      - progress: {"node": "planner", "message": "Planning research strategy..."}
      - thinking: {"content": "Step 1: Analyzing the query..."}
      - chunk:    {"content": "partial answer text"}  (token streaming)
      - evidence: {"source": "file.pdf", "page": 5, "excerpt": "..."}
      - grounding: {"claim": "...", "status": "GROUNDED", "confidence": 0.9}
      - complete: {"hypothesis": "full answer", "confidence": 0.85, ...}
      - cancelled: {"message": "Generation stopped by user"}
      - error:    {"message": "..."}
    """
    ensure_services()
    if chat_service is None or graph_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized. Check backend logs.")

    from app.services.stage_context import format_stage_context_preamble, set_stage_context_preamble

    async def event_generator():
        set_stage_context_preamble(format_stage_context_preamble(body.stage_context))
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            from app.services.swarm import run_swarm_query_streaming

            async for event_type, event_data in run_swarm_query_streaming(
                query=body.query,
                project_id=body.project_id,
                session_id=body.session_id,
                graph_service=graph_service,
                llm_service=chat_service.retrieval_service.llm_service,
                qdrant_client=chat_service.retrieval_service.qdrant_client,
                collection_name=chat_service.retrieval_service.collection_name,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    yield f"event: cancelled\ndata: {json.dumps({'message': 'Generation stopped by user'})}\n\n"
                    break
                yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            set_stage_context_preamble(None)
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/api/swarm/run", response_model=SwarmResponse)
async def run_swarm(request: SwarmRequest):
    """Run the Two-Brain Swarm on a query within a project.

    The router classifies the query as DEEP_DISCOVERY (Navigator) or
    BROAD_RESEARCH (Cortex) and dispatches accordingly.
    """
    ensure_services()
    if chat_service is None or graph_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized. Check backend logs.")
    try:
        from app.services.swarm import run_swarm_query

        result = await run_swarm_query(
            query=request.query,
            project_id=request.project_id,
            graph_service=graph_service,
            llm_service=chat_service.retrieval_service.llm_service,
            qdrant_client=chat_service.retrieval_service.qdrant_client,
            collection_name=chat_service.retrieval_service.collection_name,
        )
        return SwarmResponse(**result)
    except Exception as e:
        logger.exception("Swarm execution failed")
        err_msg = f"{type(e).__name__}: {str(e)}" if str(e) else type(e).__name__
        raise HTTPException(status_code=500, detail=f"Swarm error: {err_msg}")


# ============================================================
# ATLAS 3.0: MoE (Mixture of Experts) Agentic RAG
# ============================================================

class MoERequest(BaseModel):
    project_id: str
    query: str
    session_id: Optional[str] = None
    stage_context: Optional[Dict[str, Any]] = None


@router.post("/api/moe/run")
async def run_moe(request: MoERequest):
    """Atlas 3.0: Run the Mixture of Experts pipeline on a query.

    The MoE pipeline uses a Supervisor agent to orchestrate specialized
    Expert agents (Hypothesis, Retrieval, Writer, Critic) for multi-turn
    research with grounding verification.
    """
    ensure_services()
    if chat_service is None or graph_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized. Check backend logs.")
    try:
        from app.services.swarm import run_moe_query

        result = await run_moe_query(
            query=request.query,
            project_id=request.project_id,
            graph_service=graph_service,
            llm_service=chat_service.retrieval_service.llm_service,
            qdrant_client=chat_service.retrieval_service.qdrant_client,
            collection_name=chat_service.retrieval_service.collection_name,
        )
        return result
    except Exception as e:
        logger.exception("MoE execution failed")
        raise HTTPException(status_code=500, detail=f"MoE error: {str(e)}")


@router.post("/api/moe/stream")
async def stream_moe(body: MoERequest, request: Request):
    """Atlas 3.0: Stream MoE execution via Server-Sent Events.

    Event types:
      - routing:    {"brain": "moe_supervisor", "intent": "MoE"}
      - progress:   {"node": "hypothesis", "message": "Generating hypotheses..."}
      - thinking:   {"content": "[Supervisor] Planning..."}
      - hypotheses: {"items": [...], "selected": "..."}
      - evidence:   {"items": [...], "count": N}
      - grounding:  {"results": [...], "verdict": "PASS|REVISE"}
      - complete:   {"hypothesis": "final answer", "confidence": 0.85, ...}
      - cancelled:  {"message": "Generation stopped by user"}
      - error:      {"message": "..."}
    """
    ensure_services()
    if chat_service is None or graph_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized. Check backend logs.")

    from app.services.stage_context import format_stage_context_preamble, set_stage_context_preamble

    async def event_generator():
        set_stage_context_preamble(format_stage_context_preamble(body.stage_context))
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            from app.services.swarm import run_moe_query_streaming

            async for event_type, event_data in run_moe_query_streaming(
                query=body.query,
                project_id=body.project_id,
                session_id=body.session_id,
                graph_service=graph_service,
                llm_service=chat_service.retrieval_service.llm_service,
                qdrant_client=chat_service.retrieval_service.qdrant_client,
                collection_name=chat_service.retrieval_service.collection_name,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    yield f"event: cancelled\ndata: {json.dumps({'message': 'Generation stopped by user'})}\n\n"
                    break
                yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
        except Exception as e:
            logger.error(f"MoE streaming error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            set_stage_context_preamble(None)
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/api/moe/hypotheses")
async def stream_moe_hypotheses(body: MoERequest, request: Request):
    """Atlas 3.0: Stream MoE interactive hypotheses generation via SSE.
    """
    ensure_services()
    if chat_service is None or graph_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized. Check backend logs.")

    from app.services.stage_context import format_stage_context_preamble, set_stage_context_preamble

    async def event_generator():
        set_stage_context_preamble(format_stage_context_preamble(body.stage_context))
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            from app.services.swarm import generate_moe_hypotheses

            async for event_type, event_data in generate_moe_hypotheses(
                query=body.query,
                project_id=body.project_id,
                session_id=body.session_id,
                graph_service=graph_service,
                llm_service=chat_service.retrieval_service.llm_service,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    yield f"event: cancelled\ndata: {json.dumps({'message': 'Generation stopped by user'})}\n\n"
                    break
                yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
        except Exception as e:
            logger.error(f"MoE hypotheses streaming error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            set_stage_context_preamble(None)
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# ============================================================
# INTENT ROUTING (Lightweight, no execution)
# ============================================================

class RouteIntentRequest(BaseModel):
    query: str
    project_id: str


class RouteIntentResponse(BaseModel):
    intent: str


@router.post("/api/route", response_model=RouteIntentResponse)
async def route_intent_endpoint(request: RouteIntentRequest):
    """Classify a query into an intent without starting execution.

    Returns one of: SIMPLE, DEEP_DISCOVERY, BROAD_RESEARCH, MULTI_STEP, DISCOVERY.
    Used by the frontend to preview auto-routing before the user confirms.
    """
    ensure_services()
    from app.services.agents.meta_router import route_intent
    llm_service = chat_service.retrieval_service.llm_service
    intent = await route_intent(request.query, llm_service)
    return RouteIntentResponse(intent=intent)


# ============================================================
# DISCOVERY OS (Phase 1: Deterministic Tool-Calling)
# ============================================================

class DiscoveryRequest(BaseModel):
    project_id: str
    query: str
    session_id: Optional[str] = None
    spectrum_file_path: Optional[str] = None
    stage_context: Optional[Dict[str, Any]] = None


class DiscoveryCandidateModel(BaseModel):
    smiles: str
    properties: Optional[dict] = None
    toxicity: Optional[dict] = None


class DiscoveryResponse(BaseModel):
    brain_used: str = "discovery"
    hypothesis: str
    evidence: List[dict] = []
    reasoning_trace: List[str] = []
    status: str = "completed"
    confidence_score: Optional[float] = None
    candidates: List[DiscoveryCandidateModel] = []
    iterations: int = 0
    session_id: Optional[str] = None


@router.post("/api/discovery/run", response_model=DiscoveryResponse)
async def run_discovery(request: DiscoveryRequest):
    """Discovery OS: Run the ReAct tool-calling loop on a scientific query.

    The Discovery pipeline uses deterministic plugins (RDKit, SMARTS) for
    molecular property prediction and toxicity checking, bridged with the
    existing RAG pipeline for literature search.
    """
    ensure_services()
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized. Check backend logs.")
    try:
        from app.services.agents.discovery_graph import run_discovery_query
        from app.services.discovery_llm import get_discovery_llm_service

        # ISOLATION: Use DiscoveryLLMService (independent from global model selector)
        discovery_llm = get_discovery_llm_service()

        result = await run_discovery_query(
            query=request.query,
            project_id=request.project_id,
            llm_service=discovery_llm,
            retrieval_service=chat_service.retrieval_service,
            spectrum_file_path=request.spectrum_file_path,
        )
        return DiscoveryResponse(**result)
    except Exception as e:
        logger.exception("Discovery execution failed")
        raise HTTPException(status_code=500, detail=f"Discovery error: {str(e)}")


@router.post("/api/discovery/stream")
async def stream_discovery(body: DiscoveryRequest, request: Request):
    """Discovery OS: Stream ReAct execution via Server-Sent Events.

    Event types:
      - routing:     {"brain": "discovery", "intent": "DISCOVERY"}
      - progress:    {"node": "think"|"execute", "message": "..."}
      - thinking:    {"content": "Thought: ..."}
      - tool_call:   {"tool": "predict_properties", "input": {...}}
      - tool_result: {"tool": "predict_properties", "output": {...}}
      - evidence:    {"items": [...], "count": N}
      - complete:    {hypothesis, evidence, candidates, reasoning_trace, ...}
      - cancelled:   {"message": "Generation stopped by user"}
      - error:       {"message": "..."}
    """
    ensure_services()
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized. Check backend logs.")

    from app.services.stage_context import format_stage_context_preamble, set_stage_context_preamble

    async def event_generator():
        set_stage_context_preamble(format_stage_context_preamble(body.stage_context))
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            from app.services.agents.discovery_graph import run_discovery_query_streaming
            from app.services.discovery_llm import get_discovery_llm_service

            # ISOLATION: Use DiscoveryLLMService (independent from global model selector)
            discovery_llm = get_discovery_llm_service()

            async for event_type, event_data in run_discovery_query_streaming(
                query=body.query,
                project_id=body.project_id,
                session_id=body.session_id,
                llm_service=discovery_llm,
                retrieval_service=chat_service.retrieval_service,
                cancel_event=cancel_event,
                spectrum_file_path=body.spectrum_file_path,
            ):
                if cancel_event.is_set():
                    yield f"event: cancelled\ndata: {json.dumps({'message': 'Generation stopped by user'})}\n\n"
                    break
                yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
        except Exception as e:
            logger.error(f"Discovery streaming error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            set_stage_context_preamble(None)
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# DISCOVERY OS: Spectrum Upload (Phase 3)
# ============================================================

@router.post("/api/discovery/upload-spectrum")
async def upload_spectrum(
    file: UploadFile = File(...),
    project_id: str = Query(...),
):
    """Upload a .jdx (JCAMP-DX) NMR spectrum file for verification.

    Unlike /ingest, this endpoint stores the raw .jdx file without text
    extraction, chunking, or embedding. The file is read directly by the
    verify_spectrum plugin.
    """
    import uuid as _uuid

    if not file.filename or not file.filename.lower().endswith(".jdx"):
        raise HTTPException(
            status_code=400,
            detail="Only .jdx (JCAMP-DX) files are accepted.",
        )

    spectrum_dir = Path(settings.UPLOAD_DIR) / "spectrum"
    spectrum_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(_uuid.uuid4())
    safe_name = file.filename.replace(" ", "_")
    dest = spectrum_dir / f"{file_id}_{safe_name}"

    try:
        contents = await file.read()
        dest.write_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save spectrum file: {e}")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_path": str(dest),
    }


# ============================================================
# CONTEXT ENGINE (Phase 4: Smart Reading & Context)
# ============================================================

@router.get("/files/{doc_id}/structure")
async def get_document_structure(doc_id: str):
    """Get extracted paper structure (title, authors, methods, findings)."""
    ensure_services()
    if context_engine_service is None:
        raise HTTPException(status_code=503, detail="Context engine unavailable")
    try:
        result = await context_engine_service.get_document_structure(doc_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document structure: {str(e)}")


@router.get("/files/{doc_id}/related")
async def get_related_passages(
    doc_id: str,
    text: str = Query(..., min_length=3),
    project_id: Optional[str] = Query(None),
    limit: int = Query(8, ge=1, le=20),
):
    """Find passages in other documents related to selected text."""
    ensure_services()
    if context_engine_service is None:
        raise HTTPException(status_code=503, detail="Context engine unavailable")
    try:
        return await context_engine_service.get_related_passages(
            doc_id=doc_id, text=text, project_id=project_id, limit=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding related passages: {str(e)}")


@router.get("/files/{doc_id}/chunks")
async def get_document_chunks(
    doc_id: str,
    page: Optional[int] = Query(None, ge=1),
    limit: int = Query(50, ge=1, le=200),
):
    """Get chunks for a document, optionally filtered by page number."""
    ensure_services()
    if context_engine_service is None:
        raise HTTPException(status_code=503, detail="Context engine unavailable")
    try:
        return await context_engine_service.get_document_chunks(
            doc_id=doc_id, page_number=page, limit=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document chunks: {str(e)}")


@router.post("/api/context")
async def get_context_suggestions(request: ContextRequest):
    """Get context-aware suggestions based on user's current focus.

    Accepts a context snapshot (selected text, current document/page)
    and returns related passages, connected entities, and suggestions.
    """
    ensure_services()
    if context_engine_service is None:
        raise HTTPException(status_code=503, detail="Context engine unavailable")
    try:
        suggestions = await context_engine_service.get_context_suggestions(
            project_id=request.project_id,
            selected_text=request.selected_text,
            current_doc_id=request.current_doc_id,
            current_page=request.current_page,
        )
        return {"status": "success", "suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting context suggestions: {str(e)}")


# ============================================================
# IMPORT / EXPORT (Phase 5)
# ============================================================

class ExportMarkdownRequest(BaseModel):
    content: str
    citations: List[Dict[str, Any]] = []
    project_id: str
    title: str = "Research Synthesis"
    author: str = ""
    style: str = "apa"  # apa, mla, chicago


class ExportChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    project_name: str = "Atlas Research"


class FormatCitationRequest(BaseModel):
    doc_ids: List[str]
    style: str = "apa"


@router.post("/import/bibtex")
async def import_bibtex(
    file: UploadFile = File(...),
    project_id: str = Query(...),
):
    """Import papers from a BibTeX (.bib) or RIS (.ris) file.

    Creates Document records with status='metadata_only' for each entry.
    PDFs can be attached to these records later via the ingest endpoint.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    filename_lower = file.filename.lower()
    content = (await file.read()).decode("utf-8", errors="replace")

    try:
        if filename_lower.endswith(".bib"):
            from app.services.importers.bibtex import BibTeXImporter
            importer = BibTeXImporter()
            result = importer.import_from_string(content, project_id)
        elif filename_lower.endswith(".ris"):
            from app.services.importers.bibtex import RISImporter
            importer = RISImporter()
            result = importer.import_from_string(content, project_id)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported format. Use .bib (BibTeX) or .ris (RIS).",
            )

        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result.get("error", "Import failed"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Import error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Import error: {str(e)}")


@router.get("/export/bibtex/{project_id}")
async def export_bibtex(project_id: str):
    """Export all project citations as a BibTeX (.bib) file."""
    try:
        from app.services.exporters.bibtex import BibTeXExporter
        exporter = BibTeXExporter()
        bib_content = exporter.export_project(project_id)

        if not bib_content:
            return {"status": "empty", "message": "No documents with metadata found in this project"}

        return StreamingResponse(
            iter([bib_content]),
            media_type="application/x-bibtex",
            headers={
                "Content-Disposition": f'attachment; filename="atlas-{project_id[:8]}.bib"',
            },
        )
    except Exception as e:
        logger.error(f"BibTeX export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@router.post("/export/bibtex")
async def export_bibtex_selection(body: FormatCitationRequest):
    """Export specific documents as BibTeX entries."""
    try:
        from app.services.exporters.bibtex import BibTeXExporter
        exporter = BibTeXExporter()
        bib_content = exporter.export_documents(body.doc_ids)

        if not bib_content:
            return {"status": "empty", "message": "No documents found for the given IDs"}

        return StreamingResponse(
            iter([bib_content]),
            media_type="application/x-bibtex",
            headers={
                "Content-Disposition": 'attachment; filename="atlas-selection.bib"',
            },
        )
    except Exception as e:
        logger.error(f"BibTeX export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@router.post("/export/markdown")
async def export_markdown(body: ExportMarkdownRequest):
    """Export synthesis as Pandoc-compatible Markdown.

    Returns a JSON object with:
    - markdown: The full Markdown document with YAML front matter
    - bibtex: Companion .bib file content for Pandoc
    - filename: Suggested filename (without extension)
    """
    try:
        from app.services.exporters.markdown import MarkdownExporter
        exporter = MarkdownExporter()
        result = exporter.export_synthesis(
            content=body.content,
            citations=body.citations,
            project_id=body.project_id,
            title=body.title,
            author=body.author,
            style=body.style,
        )
        return result
    except Exception as e:
        logger.error(f"Markdown export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@router.post("/export/chat")
async def export_chat_history(body: ExportChatRequest):
    """Export a chat conversation as Markdown."""
    try:
        from app.services.exporters.markdown import MarkdownExporter
        exporter = MarkdownExporter()
        md_content = exporter.export_chat_history(
            messages=body.messages,
            project_name=body.project_name,
        )
        return {"markdown": md_content}
    except Exception as e:
        logger.error(f"Chat export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@router.post("/export/citations/format")
async def format_citations(body: FormatCitationRequest):
    """Format citations for specific documents in a given style (APA/MLA/Chicago)."""
    try:
        from app.core.database import get_session, Document as DocModel
        from app.services.exporters.bibtex import BibTeXExporter

        exporter = BibTeXExporter()
        session = get_session()
        try:
            documents = (
                session.query(DocModel)
                .filter(DocModel.id.in_(body.doc_ids))
                .all()
            )
            formatted = []
            for doc in documents:
                meta = doc.doc_metadata or {}
                citation = exporter.format_citation(meta, body.style)
                formatted.append({
                    "doc_id": doc.id,
                    "filename": doc.filename,
                    "citation": citation,
                    "bibtex_key": meta.get("bibtex_key", ""),
                })
            return {"citations": formatted, "style": body.style}
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Citation format error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Format error: {str(e)}")


# ============================================================
# DISCOVERY OS — INITIALIZE & SCHEMA  (B2)
# ============================================================

from app.services.discovery_session import DiscoverySessionService, ProjectTargetParams

@router.get("/api/discovery/schema")
async def get_discovery_schema():
    """Returns the active DomainSchema (e.g., chemistry, materials)."""
    try:
        return DiscoverySessionService.get_schema()
    except Exception as e:
        logger.error(f"Schema error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Schema error: {str(e)}")


@router.post("/api/discovery/initialize")
async def initialize_discovery_session(body: ProjectTargetParams):
    """Accept ProjectTargetParams, store them, return session ID."""
    try:
        return DiscoverySessionService.initialize_session(body)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Initialize session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Initialize session error: {str(e)}")


@router.get("/api/discovery/sessions")
async def list_discovery_sessions(project_id: Optional[str] = Query(None)):
    """List discovery sessions, filtered to the given project when project_id is provided."""
    try:
        return DiscoverySessionService.get_sessions(project_id=project_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List sessions error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"List sessions error: {str(e)}")


@router.get("/api/discovery/{session_id}/files")
async def list_session_files(session_id: str):
    """List all generated files for a specific discovery session."""
    try:
        return DiscoverySessionService.get_session_files(session_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List session files error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"List session files error: {str(e)}")


@router.get("/api/discovery/{session_id}/files/{file_path:path}")
async def read_session_file(session_id: str, file_path: str):
    """Read the contents of a session file (text files only).

    Used by the frontend to display SESSION_INIT.md, logs, CSVs, etc.
    """
    from pathlib import Path as _Path
    session_path = _Path(settings.DATA_DIR) / "discovery" / session_id

    # Resolve and validate path stays within session directory
    resolved = (session_path / file_path).resolve()
    if not str(resolved).startswith(str(session_path.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        return {
            "filename": resolved.name,
            "path": file_path,
            "content": content,
            "size_bytes": resolved.stat().st_size,
        }
    except Exception as e:
        logger.error(f"Read session file error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Read file error: {str(e)}")


@router.get("/api/discovery/{session_id}/memory")
async def get_session_memory(session_id: str):
    """Retrieve shared session memory for multi-agent coordination.

    Returns:
        SessionMemoryData: Shared state including corpus context, research goals,
        constraints, and agent completion status.

    This enables efficient multi-agent coordination by providing a single
    source of truth that all agents (Coordinator, Executor, etc.) can access.
    """
    try:
        from app.services.discovery_session import SessionMemoryService

        memory = SessionMemoryService.load_session_memory(session_id)

        if memory is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session memory not found for {session_id}. Run coordinator first."
            )

        return memory.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session memory error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session memory: {str(e)}")


# ============================================================
# DISCOVERY OS — COORDINATOR AGENT  (Phase 4)
# ============================================================

class CoordinatorChatRequest(BaseModel):
    message: Optional[str] = None
    project_id: Optional[str] = None


@router.post("/api/discovery/{session_id}/coordinator/chat")
async def coordinator_chat(session_id: str, body: CoordinatorChatRequest, request: Request):
    """Coordinator Agent: Interactive session bootstrapping via HITL.

    SSE event types:
      - coordinator_thinking: {"content": "Scanning corpus..."}
      - coordinator_question: {"question": "...", "options": [...], "context": "...", "turn": N}
      - coordinator_complete: {"extracted_goals": [...], "summary": "..."}
      - cancelled: {"message": "..."}
      - error: {"message": "..."}
    """
    ensure_services()
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized.")

    # Resolve project_id: prefer request body, fall back to session target_params
    project_id = body.project_id or ""
    if not project_id:
        db = None
        try:
            from app.core.database import get_session as get_db_session, DiscoverySession
            db = get_db_session()
            ds = db.query(DiscoverySession).filter(DiscoverySession.id == session_id).first()
            if ds and ds.target_params:
                # Try to extract project context from target params
                project_id = ds.target_params.get("project_id", "")
        except Exception as exc:
            logger.warning("Could not resolve project_id for session %s: %s", session_id, exc)
        finally:
            if db is not None:
                db.close()

    async def event_generator():
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            from app.services.agents.coordinator import run_coordinator_streaming
            from app.services.discovery_llm import get_discovery_llm_service

            # ISOLATION: Use DiscoveryLLMService (independent from global model selector)
            discovery_llm = get_discovery_llm_service()

            async for event_type, event_data in run_coordinator_streaming(
                session_id=session_id,
                project_id=project_id,
                user_message=body.message,
                llm_service=discovery_llm,
                retrieval_service=chat_service.retrieval_service,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    yield f"event: cancelled\ndata: {json.dumps({'message': 'Cancelled'})}\n\n"
                    break
                yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
        except Exception as e:
            logger.error(f"Coordinator streaming error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# DISCOVERY OS — UNIFIED CHAT  (Chat-First Overhaul)
# ============================================================

class DiscoveryChatRequest(BaseModel):
    message: Optional[str] = None
    project_id: Optional[str] = None
    action: Optional[str] = None  # "accept_plan", "reject_plan", "generate_plan"
    smiles_list: Optional[List[str]] = None


@router.post("/api/discovery/{session_id}/chat")
async def discovery_chat(session_id: str, body: DiscoveryChatRequest, request: Request):
    """Unified Discovery Chat — single endpoint for the entire session lifecycle.

    Routes internally based on session state (setup, ready, executing, complete).
    Supports free-form Q&A, plan generation, pipeline execution, and AI analysis.

    SSE event types:
      - session_update:    {"stage": "setup"|"ready"|"executing"|"complete"}
      - thinking:          {"content": "..."}
      - message:           {"content": "..."}
      - coordinator_thinking / coordinator_question / coordinator_complete (setup phase)
      - plan_proposed:     {"plan_id", "stages", "summary", "warnings", ...}
      - tool_start:        {"stage_id", "plugin", "description"}
      - tool_complete:     {"stage_id", "plugin", "summary"}
      - pipeline_complete: {"stages_completed", "candidates", ...}
      - analysis:          {"key_findings", "top_candidates", "concerns", "recommendations"}
      - recommendation:    {"recommendations" | "missing_capabilities"}
      - error:             {"message": "..."}
      - cancelled:         {"message": "..."}
    """
    ensure_services()
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized.")

    project_id = body.project_id or ""
    if not project_id:
        db = None
        try:
            from app.core.database import get_session as get_db_session, DiscoverySession
            db = get_db_session()
            ds = db.query(DiscoverySession).filter(DiscoverySession.id == session_id).first()
            if ds and ds.target_params:
                project_id = ds.target_params.get("project_id", "")
        except Exception as exc:
            logger.warning("Could not resolve project_id for session %s: %s", session_id, exc)
        finally:
            if db is not None:
                db.close()

    async def event_generator():
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            from app.services.agents.discovery_chat import run_discovery_chat
            from app.services.discovery_llm import get_discovery_llm_service

            discovery_llm = get_discovery_llm_service()

            async for event_type, event_data in run_discovery_chat(
                session_id=session_id,
                project_id=project_id,
                user_message=body.message,
                llm_service=discovery_llm,
                retrieval_service=chat_service.retrieval_service,
                action=body.action,
                smiles_list=body.smiles_list,
                cancel_event=cancel_event,
            ):
                if cancel_event.is_set():
                    yield f"event: cancelled\ndata: {json.dumps({'message': 'Cancelled'})}\n\n"
                    break
                yield f"event: {event_type}\ndata: {json.dumps(event_data, default=str)}\n\n"
        except Exception as e:
            logger.error(f"Discovery chat error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# DISCOVERY OS — EXECUTOR AGENT  (Phase 5 — legacy)
# ============================================================

class ExecutorStartRequest(BaseModel):
    auto_approve: bool = False
    decision: Optional[str] = None
    edited_code: Optional[str] = None


@router.post("/api/discovery/{session_id}/executor/start")
async def executor_start(
    session_id: str,
    body: ExecutorStartRequest,
    request: Request
):
    """Executor Agent: Script generation sandbox after coordinator completes.

    Bootstrap mode: Loads research goals from session memory (shared memory system).
    The coordinator must run first to initialize session memory.

    SSE event types:
      - executor_thinking: {"content": "Planning next task..."}
      - executor_script_generated: {"filename": "...", "code": "...", "description": "..."}
      - executor_awaiting_approval: {"filename": "...", "preview": "..."}
      - executor_executing: {"filename": "...", "iteration": N}
      - executor_artifact: {"filename": "...", "type": "log|csv|py"}
      - executor_complete: {"artifacts": [...], "summary": "..."}
      - error: {"message": "..."}
    """
    ensure_services()
    if chat_service is None:
        raise HTTPException(status_code=503, detail="Services not initialized.")

    # Executor will bootstrap from session memory (no need to load goals here)
    project_id = body.project_id if hasattr(body, 'project_id') else ""

    async def event_generator():
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            from app.services.agents.executor import run_executor_streaming
            from app.services.discovery_llm import get_discovery_llm_service

            # ISOLATION: Use DiscoveryLLMService (independent from global model selector)
            discovery_llm = get_discovery_llm_service()

            # Determine resume command if provided
            resume_command = None
            if body.decision:
                if body.decision == "edit" and body.edited_code:
                    resume_command = f"edit:{body.edited_code}"
                else:
                    resume_command = body.decision

            async for event_type, event_data in run_executor_streaming(
                session_id=session_id,
                project_id=project_id,
                extracted_goals=None,  # Bootstrap from session memory
                llm_service=discovery_llm,
                auto_approve=body.auto_approve,
                cancel_event=cancel_event,
                resume_command=resume_command,
            ):
                if cancel_event.is_set():
                    yield f"event: cancelled\ndata: {json.dumps({'message': 'Cancelled'})}\n\n"
                    break
                yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"

        except Exception as e:
            logger.error(f"Executor streaming error: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )





# ============================================================
# DISCOVERY OS — PIPELINE EXECUTION  (Chemistry Engine Plan)
# ============================================================

class PipelineRunRequest(BaseModel):
    smiles_list: List[str] = []
    auto_run: bool = True


@router.post("/api/discovery/{session_id}/pipeline/run")
async def run_pipeline(session_id: str, body: PipelineRunRequest, request: Request):
    """Execute the deterministic plugin pipeline for a discovery session.

    This replaces the old LLM-based script sandbox with a hardened, rule-based
    pipeline that selects and executes registered plugins in the correct order.

    SSE event types:
      - pipeline_planning: {"message": "...", "stage_count": N}
      - pipeline_stage_start: {"stage": N, "plugin": "...", "description": "..."}
      - pipeline_stage_complete: {"stage": N, "plugin": "...", "summary": "..."}
      - pipeline_complete: {"stages_completed": N, "final_summary": "...", "results": {...}}
      - error: {"message": "..."}
    """
    ensure_services()

    async def event_generator():
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            from app.services.discovery_session import SessionMemoryService
            from app.services.agents.pipeline_planner import build_pipeline, PipelinePlan
            from app.services.plugins import get_plugin_manager

            memory = SessionMemoryService.load_session_memory(session_id)
            if memory is None:
                yield f"event: error\ndata: {json.dumps({'message': 'Session memory not found. Run coordinator first.'})}\n\n"
                return

            memory_dict = memory.model_dump()
            pm = get_plugin_manager()
            available = pm.get_registered_names()

            plan = build_pipeline(memory_dict, available)

            yield f"event: pipeline_planning\ndata: {json.dumps({'message': plan.reasoning, 'stage_count': len(plan.stages), 'iteration': getattr(plan, 'iteration', 1), 'has_prior_findings': bool(getattr(plan, 'prior_findings_digest', ''))})}\n\n"

            if cancel_event.is_set():
                return

            smiles_list = body.smiles_list
            if not smiles_list:
                smiles_list = _extract_smiles_from_corpus(session_id)
            if not smiles_list:
                # No SMILES in corpus — run explicit LLM candidate proposal stage
                yield f"event: pipeline_stage_start\ndata: {json.dumps({'stage': 0, 'plugin': 'propose_candidates', 'description': 'Proposing candidate molecules from corpus knowledge and research goals...'})}\n\n"
                llm_svc = chat_service.retrieval_service.llm_service if chat_service else None
                if llm_svc is None:
                    yield f"event: error\ndata: {json.dumps({'message': 'No SMILES in corpus and LLM service unavailable. Provide SMILES via smiles_list or upload documents containing SMILES strings.'})}\n\n"
                    return
                smiles_list = await _propose_candidates_from_memory(session_id, llm_svc)
                if not smiles_list:
                    yield f"event: error\ndata: {json.dumps({'message': 'Could not propose candidates: session has no research goals or corpus context. Complete the setup conversation first.'})}\n\n"
                    return
                yield f"event: pipeline_stage_complete\ndata: {json.dumps({'stage': 0, 'plugin': 'propose_candidates', 'summary': f'Proposed {len(smiles_list)} candidate molecules from corpus knowledge.'})}\n\n"

            stage_data: dict = {"smiles_list": smiles_list}
            stage_results = []

            for stage in plan.stages:
                if cancel_event.is_set():
                    break

                yield f"event: pipeline_stage_start\ndata: {json.dumps({'stage': stage.stage_id, 'plugin': stage.plugin, 'description': stage.description})}\n\n"

                try:
                    merged_kwargs = {**stage.config, **stage_data}
                    result = await pm.invoke(stage.plugin, **merged_kwargs)

                    if isinstance(result, dict):
                        if "molecules" in result:
                            stage_data["molecules"] = result["molecules"]
                            stage_data["smiles_list"] = [
                                m.get("smiles", "") for m in result["molecules"] if m.get("valid", True)
                            ]
                        if "predictions" in result:
                            stage_data["predictions"] = result["predictions"]
                        if "scores" in result:
                            stage_data["scores"] = result["scores"]
                        if "properties" in result:
                            stage_data["properties"] = result["properties"]
                        if "toxicity_results" in result:
                            stage_data["toxicity_results"] = result["toxicity_results"]

                    summary = result.get("summary", f"Stage {stage.stage_id} complete") if isinstance(result, dict) else str(result)
                    stage_results.append({"stage": stage.stage_id, "plugin": stage.plugin, "result": result, "summary": summary})

                    yield f"event: pipeline_stage_complete\ndata: {json.dumps({'stage': stage.stage_id, 'plugin': stage.plugin, 'summary': summary})}\n\n"

                    # Push live candidates to the UCSO table after data-producing stages
                    candidates = _build_candidates_for_frontend(stage_data)
                    if candidates:
                        yield f"event: executor_candidates\ndata: {json.dumps({'candidates': candidates})}\n\n"

                except Exception as exc:
                    logger.error("Pipeline stage %s failed: %s", stage.plugin, exc, exc_info=True)
                    stage_results.append({"stage": stage.stage_id, "plugin": stage.plugin, "error": str(exc)})
                    yield f"event: pipeline_stage_complete\ndata: {json.dumps({'stage': stage.stage_id, 'plugin': stage.plugin, 'summary': f'ERROR: {exc}'})}\n\n"

            _save_pipeline_results(session_id, plan, stage_results, stage_data)

            final_candidates = _build_candidates_for_frontend(stage_data)
            final_summary = f"Pipeline completed {len(stage_results)}/{len(plan.stages)} stages. {len(final_candidates)} candidates ranked."
            yield f"event: pipeline_complete\ndata: {json.dumps({'stages_completed': len(stage_results), 'final_summary': final_summary, 'candidates': final_candidates, 'stage_results': [{k: v for k, v in sr.items() if k != 'result'} for sr in stage_results]})}\n\n"

        except Exception as e:
            logger.error("Pipeline execution error: %s", e, exc_info=True)
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _build_candidates_for_frontend(stage_data: dict) -> List[dict]:
    """Build candidate objects for the UCSO table from accumulated pipeline data.

    Merges molecule identity, properties, toxicity, ADMET predictions, and SA scores
    into a single flat candidate list that the DiscoveryWorkbench can render.

    Stage data keys consumed:
      molecules       — from standardize_smiles (list of {smiles, compound_id, valid, inchikey})
      properties      — from predict_properties batch mode (list of {smiles, MolWt, LogP, ...})
      toxicity_results— from check_toxicity batch mode (list of {smiles, clean, alert_count, ...})
      predictions     — from predict_admet (list of {smiles, herg_risk, dili_risk, ...})
      scores          — from score_synthesizability (list of {smiles, sa_score, feasible})
    """
    molecules = stage_data.get("molecules", [])

    # Build lookup dicts keyed by canonical SMILES
    prop_by_smiles = {
        p["smiles"]: p
        for p in stage_data.get("properties", [])
        if isinstance(p, dict) and p.get("smiles")
    }
    tox_by_smiles = {
        t["smiles"]: t
        for t in stage_data.get("toxicity_results", [])
        if isinstance(t, dict) and t.get("smiles")
    }
    pred_by_smiles = {
        p["smiles"]: p
        for p in stage_data.get("predictions", [])
        if isinstance(p, dict) and p.get("smiles")
    }
    score_by_smiles = {
        s["smiles"]: s
        for s in stage_data.get("scores", [])
        if isinstance(s, dict) and s.get("smiles")
    }

    candidates = []
    for mol in molecules:
        if not isinstance(mol, dict):
            continue
        smi = mol.get("smiles", "")
        if not smi:
            continue

        prop = prop_by_smiles.get(smi, {})
        tox = tox_by_smiles.get(smi, {})
        pred = pred_by_smiles.get(smi, {})
        sa = score_by_smiles.get(smi, {})

        # Properties — populated from batch predict_properties
        properties = {
            "MolWt": prop.get("MolWt"),
            "LogP": prop.get("LogP"),
            "TPSA": prop.get("TPSA"),
            "QED": prop.get("QED"),
            "HBD": prop.get("HBD"),
            "HBA": prop.get("HBA"),
            "Lipinski": prop.get("Lipinski"),
        }

        # Toxicity — populated from batch check_toxicity
        if tox:
            toxicity = {
                "clean": tox.get("clean", True),
                "alert_count": tox.get("alert_count", 0),
                "pains_hits": tox.get("pains_hits", 0),
                "alerts": [a.get("name") for a in tox.get("structural_alerts", [])],
            }
        else:
            toxicity = None

        candidates.append({
            "smiles": smi,
            "compound_id": mol.get("compound_id", ""),
            "inchikey": mol.get("inchikey", ""),
            "source": mol.get("source", "unknown"),
            "valid": mol.get("valid", True),
            "properties": properties,
            "toxicity": toxicity,
            "admet": {
                "herg_risk": pred.get("herg_risk", "N/A"),
                "dili_risk": pred.get("dili_risk", "N/A"),
                "caco2": pred.get("caco2_permeability", "N/A"),
                "cyp3a4": pred.get("cyp3a4_inhibition", "N/A"),
                "overall": pred.get("overall_risk", "N/A"),
            } if pred else None,
            "sa_score": sa.get("sa_score"),
            "feasible": sa.get("feasible"),
        })

    return candidates


def _extract_smiles_from_corpus(session_id: str) -> List[str]:
    """Extract SMILES from session corpus document chunks via regex + RDKit validation.

    Queries DocumentChunk.text for the session's corpusDocumentIds, then applies
    a SMILES pattern and validates every hit through RDKit.  Returns [] (not demo
    data) when nothing is found so the caller can fall back to LLM generation.
    """
    import re

    # ------------------------------------------------------------------
    # 1. Load corpus document IDs from the DiscoverySession row
    # ------------------------------------------------------------------
    doc_ids: list = []
    try:
        from app.core.database import get_session as _get_db, DiscoverySession
        db = _get_db()
        try:
            ds = db.query(DiscoverySession).filter(
                DiscoverySession.id == session_id
            ).first()
            if ds and ds.target_params:
                doc_ids = ds.target_params.get("corpusDocumentIds") or []
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Failed to load DiscoverySession for SMILES extraction: %s", exc)

    # Also check session_memory in case doc IDs were stored there
    if not doc_ids:
        try:
            from app.services.discovery_session import SessionMemoryService
            mem = SessionMemoryService.load_session_memory(session_id)
            if mem and mem.corpus_context:
                doc_ids = mem.corpus_context.document_ids or []
        except Exception as exc:
            logger.warning("Failed to load session_memory for doc IDs: %s", exc)

    if not doc_ids:
        logger.info("No corpus doc IDs for session %s — corpus extraction skipped.", session_id)
        return []

    # ------------------------------------------------------------------
    # 2. Fetch all text chunks for those documents
    # ------------------------------------------------------------------
    all_text = ""
    try:
        from app.core.database import get_session as _get_db2, DocumentChunk
        db2 = _get_db2()
        try:
            rows = db2.query(DocumentChunk.text).filter(
                DocumentChunk.document_id.in_(doc_ids)
            ).all()
            all_text = " ".join(r.text for r in rows if r.text)
        finally:
            db2.close()
    except Exception as exc:
        logger.warning("Failed to query DocumentChunks: %s", exc)
        return []

    if not all_text:
        return []

    # ------------------------------------------------------------------
    # 3. Regex: pull SMILES-like tokens (validate with RDKit below)
    # ------------------------------------------------------------------
    # Pattern: starts with a SMILES-valid heavy-atom char, min 6 chars total,
    # composed only of characters legal in SMILES.  False positives are
    # eliminated by RDKit validation in the next step.
    _SMILES_RE = re.compile(
        r"(?<![A-Za-z])"
        r"([BCNOPSFIbcnosp*][A-Za-z0-9@+\-\[\]()=#$\/\\\.%:]{5,})"
        r"(?![A-Za-z])"
    )
    raw_candidates = [m.group(1) for m in _SMILES_RE.finditer(all_text)]

    if not raw_candidates:
        logger.info(
            "No SMILES-like tokens in corpus for session %s (%d chars scanned).",
            session_id, len(all_text),
        )
        return []

    # ------------------------------------------------------------------
    # 4. RDKit validation and deduplication
    # ------------------------------------------------------------------
    valid_smiles: List[str] = []
    try:
        from rdkit import Chem
        seen_keys: set = set()
        for smi in raw_candidates:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                canonical = Chem.MolToSmiles(mol, canonical=True)
                ikey = Chem.MolToInchiKey(mol)
                if ikey and ikey not in seen_keys:
                    seen_keys.add(ikey)
                    valid_smiles.append(canonical)
            except Exception:
                pass
    except ImportError:
        # RDKit unavailable — return the raw regex hits (less reliable)
        seen: set = set()
        for smi in raw_candidates:
            if smi not in seen:
                seen.add(smi)
                valid_smiles.append(smi)

    logger.info(
        "Corpus SMILES extraction for session %s: %d raw → %d valid.",
        session_id, len(raw_candidates), len(valid_smiles),
    )
    return valid_smiles


async def _propose_candidates_from_memory(
    session_id: str,
    llm_service,
) -> List[str]:
    """Propose candidate SMILES via LLM from session research goals + corpus context.

    This is the explicit 'novel discovery' step: the researcher's goals and literature
    knowledge are distilled into concrete candidate molecules for computational screening.
    Called when no SMILES are literally present in the uploaded corpus documents.

    Returns:
        List of RDKit-validated canonical SMILES (empty list if proposal fails).
    """
    from app.services.discovery_session import SessionMemoryService

    memory = SessionMemoryService.load_session_memory(session_id)
    if memory is None:
        logger.warning("_propose_candidates_from_memory: no session memory for %s", session_id)
        return []

    has_goals = bool(memory.research_goals)
    has_corpus = bool(memory.corpus_context and (
        memory.corpus_context.summary or memory.corpus_context.entities
    ))
    if not has_goals and not has_corpus:
        logger.warning(
            "_propose_candidates_from_memory: session %s has no goals or corpus context", session_id
        )
        return []

    goals_text = "\n".join(f"- {g}" for g in memory.research_goals) if memory.research_goals else "Not specified"
    domain = memory.domain or "scientific research"
    corpus_summary = ""
    entities_text = ""
    if memory.corpus_context:
        corpus_summary = memory.corpus_context.summary or ""
        entities = memory.corpus_context.entities or []
        if entities:
            entities_text = ", ".join(str(e) for e in entities[:30])

    # Build constraints text — support both dict and list formats
    if isinstance(memory.constraints, dict):
        constraints_lines = [f"- {k}: {v}" for k, v in memory.constraints.items()]
    elif isinstance(memory.constraints, list):
        constraints_lines = [f"- {c}" for c in memory.constraints]
    else:
        constraints_lines = []
    constraints_text = "\n".join(constraints_lines) if constraints_lines else "None specified"

    # Extract scaffold hint from goals (domain-agnostic — LLM reads the goals and infers)
    scaffold_hint = ""
    if memory.research_goals:
        goal_str = " ".join(memory.research_goals).lower()
        if "scaffold" in goal_str or "core" in goal_str or "motif" in goal_str:
            scaffold_hint = (
                "\n## Scaffold Requirement\n"
                "The research goals reference a specific scaffold or core structure. "
                "Every proposed candidate MUST contain that scaffold. "
                "Verify this before providing your answer."
            )

    # Include any prior wetlab findings to guide the next round
    findings_text = ""
    if memory.experimental_findings:
        lines = []
        for f in memory.experimental_findings:
            status = "ACTIVE" if f.active else ("INACTIVE" if f.active is False else "UNTESTED")
            val = f"{f.value} {f.units or ''}".strip() if f.value is not None else ""
            lines.append(
                f"  - {f.compound_id} ({f.assay_type}): {val} [{status}]"
                + (f" — {f.notes}" if f.notes else "")
            )
        findings_text = (
            "\n## Prior Experimental Results\n"
            "Use these to guide structural refinement — propose analogues of active hits "
            "and avoid structural classes of inactive compounds.\n"
            + "\n".join(lines)
        )

    prompt = f"""You are an expert computational chemist and scientific researcher.

Based on the research context below, propose 5-10 candidate molecules for computational screening.
Propose realistic, novel molecules that directly advance the stated research goals.
Every SMILES string must be chemically valid. Prefer structural diversity within the required scaffold class.

## Research Domain
{domain}

## Research Goals
{goals_text}

## Literature Context
{corpus_summary[:800] if corpus_summary else 'No corpus summary available.'}

## Known Entities from Literature
{entities_text if entities_text else 'None extracted'}

## Property Constraints
{constraints_text}
{scaffold_hint}{findings_text}

For each candidate provide: a valid SMILES string, a short name, and a one-sentence rationale
explaining how it addresses the research goals."""

    schema = {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "smiles": {"type": "string", "description": "Valid SMILES string for the molecule"},
                        "name": {"type": "string", "description": "Common or systematic name"},
                        "rationale": {"type": "string", "description": "One-sentence rationale for proposing this molecule"},
                    },
                    "required": ["smiles", "name", "rationale"],
                },
                "description": "List of 5-10 proposed candidate molecules",
            },
            "strategy": {
                "type": "string",
                "description": "Overall design strategy — scaffold classes chosen and why",
            },
        },
        "required": ["candidates", "strategy"],
    }

    try:
        result = await llm_service.orchestrate_constrained(
            prompt=prompt,
            schema=schema,
            system_prompt=(
                "You are an expert medicinal chemist. Propose novel, realistic drug candidates "
                "as valid SMILES strings. Every SMILES must be chemically valid."
            ),
            temperature=0.6,
            max_tokens=1500,
        )
    except Exception as exc:
        logger.error("LLM candidate proposal failed for session %s: %s", session_id, exc)
        return []

    raw_candidates = result.get("candidates", [])
    if not raw_candidates:
        logger.warning("LLM returned no candidates for session %s", session_id)
        return []

    validated: List[str] = []
    try:
        from rdkit import Chem
        seen_keys: set = set()
        for c in raw_candidates:
            smi = c.get("smiles", "").strip() if isinstance(c, dict) else str(c).strip()
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            try:
                ikey = Chem.MolToInchiKey(mol)
            except Exception:
                ikey = canonical
            if ikey and ikey not in seen_keys:
                seen_keys.add(ikey)
                validated.append(canonical)
    except ImportError:
        validated = [
            c.get("smiles", "").strip()
            for c in raw_candidates
            if isinstance(c, dict) and c.get("smiles", "").strip()
        ]

    logger.info(
        "LLM proposed %d candidates (%d validated) for session %s. Strategy: %s",
        len(raw_candidates), len(validated), session_id,
        result.get("strategy", "")[:120],
    )
    return validated


def _save_pipeline_results(
    session_id: str,
    plan: Any,
    stage_results: List[dict],
    final_data: dict,
):
    """Save pipeline results and update the living .md knowledge substrate.

    This is the core of the compounding loop:
    1. pipeline_results.json — machine-readable snapshot (overwritten per run)
    2. FINDINGS.md — APPENDED, never overwritten. Accumulates across iterations.
    3. session_memory.json — updated with latest stage, iteration count, findings digest
    4. SESSION_CONTEXT.md — regenerated with updated session_memory
    """
    from pathlib import Path
    from datetime import datetime
    from app.core.config import settings

    session_path = Path(settings.DATA_DIR) / "discovery" / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    generated = session_path / "generated"
    generated.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # --- 1. Save machine-readable pipeline_results.json (latest run) ---
    results_path = generated / "pipeline_results.json"
    try:
        serializable = {
            "plan_id": plan.plan_id if hasattr(plan, "plan_id") else "",
            "timestamp": timestamp,
            "stages_completed": len(stage_results),
            "stage_summaries": [
                {k: v for k, v in sr.items() if k != "result"}
                for sr in stage_results
            ],
            "molecules": final_data.get("molecules", [])[:50],
            "predictions": final_data.get("predictions", [])[:50],
            "scores": final_data.get("scores", [])[:50],
        }
        results_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save pipeline results: %s", exc)

    # --- 2. APPEND to FINDINGS.md (actionable research report) ---
    findings_path = session_path / "FINDINGS.md"
    try:
        header_needed = not findings_path.exists()
        existing_text = ""
        if not header_needed:
            existing_text = findings_path.read_text(encoding="utf-8")
        iteration = existing_text.count("## Run ") + 1

        candidates = _build_candidates_for_frontend(final_data)

        with open(findings_path, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("# Discovery Findings\n\n")
                f.write("Each run appends results. Agents read this before every iteration.\n\n---\n\n")

            f.write(f"## Run {iteration} — {timestamp}\n\n")

            # Pipeline summary
            ok_count = sum(1 for sr in stage_results if "error" not in sr)
            f.write(f"**Pipeline**: {ok_count}/{len(stage_results)} stages completed  \n")
            f.write(f"**Input**: {len(final_data.get('smiles_list', []))} SMILES  \n")
            f.write(f"**Output**: {len(candidates)} ranked candidates  \n\n")

            # Actionable hit list
            clean = [c for c in candidates if c.get("valid", True) and (
                (c.get("toxicity") and c["toxicity"].get("clean")) or
                (c.get("admet") and c["admet"].get("overall") == "LOW")
            )]
            flagged = [c for c in candidates if c not in clean]

            if clean:
                f.write(f"### Candidates to advance ({len(clean)})\n\n")
                f.write("These passed safety filters and are ready for synthesis planning or procurement.\n\n")
                f.write("| # | SMILES | MW | LogP | SA | hERG | ADMET | Action |\n")
                f.write("|---|--------|-----|------|-----|------|-------|--------|\n")
                for i, c in enumerate(clean[:20], 1):
                    mw = f"{c['properties'].get('MolWt', '-')}" if c.get("properties") else "-"
                    logp = f"{c['properties'].get('LogP', '-')}" if c.get("properties") else "-"
                    sa = f"{c.get('sa_score', '-')}"
                    herg = c.get("admet", {}).get("herg_risk", "-") if c.get("admet") else "-"
                    admet = c.get("admet", {}).get("overall", "-") if c.get("admet") else "-"
                    feasible = c.get("feasible")
                    action = "Order/synthesize" if feasible else "Check route"
                    f.write(f"| {i} | `{c['smiles'][:40]}` | {mw} | {logp} | {sa} | {herg} | {admet} | {action} |\n")
                f.write("\n")

            if flagged:
                f.write(f"### Flagged candidates ({len(flagged)})\n\n")
                f.write("These failed one or more filters. Review before advancing.\n\n")
                for c in flagged[:10]:
                    reasons = []
                    if c.get("toxicity") and not c["toxicity"].get("clean"):
                        reasons.append(f"{c['toxicity'].get('alert_count', '?')} tox alerts")
                    if c.get("admet") and c["admet"].get("overall") not in ("LOW", "N/A", None):
                        reasons.append(f"ADMET: {c['admet'].get('overall', '?')}")
                    if c.get("sa_score") and float(c.get("sa_score", 0)) > 6:
                        reasons.append(f"SA={c['sa_score']} (hard to synthesize)")
                    f.write(f"- `{c['smiles'][:50]}` — {', '.join(reasons) or 'review needed'}\n")
                f.write("\n")

            # Next steps
            f.write("### Recommended next steps\n\n")
            if clean:
                f.write(f"1. **Procurement**: {len(clean)} clean candidates ready. Check catalog availability.\n")
                f.write(f"2. **Synthesis planning**: Run retrosynthesis on top hits to confirm routes.\n")
            if flagged:
                f.write(f"3. **Iterate**: Edit CONSTRAINTS.md to tighten filters, then re-run pipeline.\n")
            if not clean and not flagged:
                f.write("1. **Expand library**: Upload more SMILES or relax constraints in CONSTRAINTS.md.\n")
            f.write(f"4. **Annotate**: Edit this file or HYPOTHESES.md to refine direction for the next iteration.\n\n")
            f.write("---\n\n")
    except Exception as exc:
        logger.warning("Failed to append to FINDINGS.md: %s", exc)

    # --- 3. Update session_memory.json (compounding state) ---
    try:
        from app.services.discovery_session import SessionMemoryService
        memory = SessionMemoryService.load_session_memory(session_id)
        if memory:
            meta = memory.metadata or {}
            meta["last_pipeline_run"] = timestamp
            meta["pipeline_iterations"] = meta.get("pipeline_iterations", 0) + 1
            meta["last_stages_completed"] = len(stage_results)
            meta["last_molecule_count"] = len(final_data.get("molecules", []))
            memory.metadata = meta

            if "pipeline_executor" not in memory.agents_completed:
                memory.agents_completed.append("pipeline_executor")
            memory.current_stage = "pipeline_complete"

            SessionMemoryService.save_session_memory(session_id, memory)
    except Exception as exc:
        logger.warning("Failed to update session_memory after pipeline: %s", exc)


# ============================================================
# DISCOVERY OS — DOMAIN TOOLS  (B3)
# ============================================================

@router.get("/api/domain/render")
async def domain_render(
    data: str = Query(..., description="Entity data (e.g. SMILES string, alloy name)"),
    type: str = Query(..., description="Render type: molecule_2d | crystal_3d | polymer_chain | data_table"),
):
    """Render an entity to SVG. Uses RDKit for molecule_2d when available;
    returns a labelled placeholder SVG for all other render types."""
    from app.services.domain_tools import render_molecule_2d_svg, render_placeholder_svg
    from fastapi.responses import Response

    valid_types = {"molecule_2d", "crystal_3d", "polymer_chain", "data_table"}
    if type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid render type '{type}'. Must be one of: {', '.join(sorted(valid_types))}")

    if type == "molecule_2d":
        svg = render_molecule_2d_svg(data)
    else:
        svg = render_placeholder_svg(data, type)

    return Response(content=svg, media_type="image/svg+xml")


@router.post("/api/discovery/{session_id}/record_result")
async def record_experimental_result(session_id: str, body: RecordResultRequest):
    """Record a wetlab result back into session memory (closes the design-make-test-learn loop).

    The finding is appended to session_memory.experimental_findings and written to
    FINDINGS.md so the next pipeline iteration sees it as additional context.
    Active hits drive the next round of candidate proposal toward structure-activity
    refinement; inactives are used to tighten exclusion constraints.
    """
    from app.services.discovery_session import SessionMemoryService, ExperimentalFinding
    from pathlib import Path
    from datetime import datetime

    memory = SessionMemoryService.load_session_memory(session_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    finding = ExperimentalFinding(
        compound_id=body.compound_id,
        smiles=body.smiles,
        assay_type=body.assay_type,
        value=body.value,
        units=body.units,
        active=body.active,
        notes=body.notes,
    )
    memory.experimental_findings.append(finding)
    SessionMemoryService.save_session_memory(session_id, memory)

    # Append to FINDINGS.md for agent visibility
    try:
        session_path = Path(settings.DATA_DIR) / "discovery" / session_id
        findings_path = session_path / "FINDINGS.md"
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        status = "ACTIVE HIT" if body.active else ("INACTIVE" if body.active is False else "RECORDED")
        value_str = f"{body.value} {body.units or ''}".strip() if body.value is not None else "—"
        line = (
            f"\n### Wetlab Result — {timestamp}\n"
            f"- **Compound**: `{body.compound_id}`"
            + (f" (`{body.smiles}`)" if body.smiles else "")
            + f"\n- **Assay**: {body.assay_type}  |  **Result**: {value_str}  |  **Status**: {status}\n"
            + (f"- **Notes**: {body.notes}\n" if body.notes else "")
            + "\n---\n"
        )
        with open(findings_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        logger.warning("Failed to append wetlab result to FINDINGS.md: %s", exc)

    return {
        "status": "recorded",
        "compound_id": body.compound_id,
        "finding_index": len(memory.experimental_findings) - 1,
        "message": (
            f"Result recorded. Session now has {len(memory.experimental_findings)} "
            "experimental finding(s). Re-run the pipeline to incorporate this data."
        ),
    }


@router.post("/api/discovery/capability-gap")
async def create_capability_gap(body: CapabilityGapRequest):
    """Record a capability gap — the pipeline detected it needs a tool that
    isn't registered. Returns a gap_id for subsequent resolution."""
    from app.services.domain_tools import create_capability_gap as _create

    try:
        gap_id = _create(
            run_id=body.run_id,
            stage=body.stage,
            required_function=body.required_function,
            input_schema=body.input_schema,
            output_schema=body.output_schema,
            standard_reference=body.standard_reference,
        )
        return {"gap_id": gap_id}
    except Exception as e:
        logger.error(f"Capability gap creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/tools/register")
async def register_tool(body: ToolRegisterRequest):
    """Resolve a capability gap by registering a tool (local script, API endpoint,
    plugin, or skip)."""
    from app.services.domain_tools import resolve_capability_gap

    allowed_methods = {"local_script", "api_endpoint", "plugin", "skip"}
    if body.method not in allowed_methods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method '{body.method}'. Must be one of: {', '.join(sorted(allowed_methods))}",
        )

    try:
        resolve_capability_gap(
            gap_id=body.gap_id,
            method=body.method,
            config=body.config,
        )
        return {"status": "resolved", "gap_id": body.gap_id}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Tool registration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DISCOVERY OS — ROUTE PLANNING & SPECTROSCOPY  (B6)
# ============================================================

@router.post("/api/domain/route-planning")
async def route_planning(body: RoutePlanningRequest):
    """Stream retrosynthesis route planning via SSE (mock for now)."""
    from app.services.spectroscopy import stream_mock_route_planning

    return StreamingResponse(
        stream_mock_route_planning(body.candidate_id, body.smiles, body.epoch_id),
        media_type="text/event-stream",
    )


@router.post("/api/domain/validate-spectroscopy")
async def validate_spectroscopy_endpoint(body: ValidateSpectroscopyRequest):
    """Validate uploaded spectroscopy data (NMR / JCAMP-DX) against a hit."""
    from app.services.spectroscopy import validate_spectroscopy

    try:
        result = validate_spectroscopy(body.hit_id, body.file_content, body.file_type)
        return result
    except Exception as e:
        logger.error(f"Spectroscopy validation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DISCOVERY OS — CANDIDATE GENERATION & SCREEN  (C6)
# ============================================================

@router.post("/api/discovery/generate-candidates")
async def generate_candidates_endpoint(
    body: GenerateCandidatesRequest,
    request: Request,
    mock: bool = Query(False, description="Return hardcoded SMILES without LLM"),
):
    """Stream candidate generation via SSE.

    Events:
      data: {"type":"progress","message":"..."}
      data: {"type":"candidates","smiles":["CCO",...]}
      data: {"type":"complete"}

    Use ?mock=true to skip the LLM and return hardcoded test molecules.
    """
    from app.services.candidate_generation import generate_candidates

    # Gather optional services for the real (non-mock) path
    retrieval_svc = None
    llm_svc = None
    if not mock:
        ensure_services()
        if chat_service is not None:
            from app.services.discovery_llm import get_discovery_llm_service

            retrieval_svc = chat_service.retrieval_service
            # ISOLATION: Use DiscoveryLLMService (independent from global model selector)
            llm_svc = get_discovery_llm_service()

    async def event_generator():
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            async for chunk in generate_candidates(
                session_id=body.session_id,
                epoch_id=body.epoch_id,
                mock=mock,
                retrieval_service=retrieval_svc,
                llm_service=llm_svc,
                project_id=body.project_id,
            ):
                if cancel_event.is_set():
                    yield f"data: {json.dumps({'type': 'cancelled', 'message': 'Generation stopped by user'})}\n\n"
                    break
                yield chunk
        except Exception as e:
            logger.error(f"generate-candidates error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/discovery/screen")
async def screen_candidates_endpoint(
    body: ScreenCandidatesRequest,
    request: Request,
):
    """Stream deterministic RDKit screening via SSE.

    Events:
      data: {"type":"screen_progress","smiles":"...","properties":{...},"passes_constraints":bool}
      data: {"type":"complete","surviving_candidates":[CandidateArtifact,...]}
    """
    from app.services.candidate_generation import screen_candidates

    async def event_generator():
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(monitor_disconnect(request, cancel_event))
        try:
            async for chunk in screen_candidates(
                session_id=body.session_id,
                epoch_id=body.epoch_id,
                smiles_list=body.smiles_list,
            ):
                if cancel_event.is_set():
                    yield f"data: {json.dumps({'type': 'cancelled', 'message': 'Screening stopped by user'})}\n\n"
                    break
                yield chunk
        except Exception as e:
            logger.error(f"screen error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            cancel_event.set()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# EDITOR — STRUCTURAL COMPLETION (D3)
# ============================================================

_STRUCTURE_SYSTEM_PROMPT = """\
You are a structural document scaffold assistant for scientific research notes.

STRICT OUTPUT CONTRACT — VIOLATION TERMINATES YOUR RESPONSE:
You may ONLY output exactly ONE of the following four forms, and nothing else:

  1. Citation block (when a document reference would logically follow):
       [Insert: <filename>, p.<N> — <N>% relevance to <topic>]

  2. Section header (when a new analytical section should begin):
       \\n\\n## <Suggested Section Name>

  3. Data table placeholder (when tabular comparison is the next logical element):
       [Insert: property comparison table for <items>]

  4. The exact token: null
       (output this when none of the above fits — always prefer null over guessing)

FORBIDDEN — output any of these and your response is invalid:
  - Any continuation of a sentence or scientific argument
  - Any word completions, prose phrases, or paragraph text
  - Any interpretation, claim, conclusion, or explanation
  - More than ONE structural element per response
  - Anything not matching one of the four forms above

When in doubt: output null.
"""

_STRUCTURE_USER_TEMPLATE = """\
The researcher has paused at a structural boundary in their notes.
Text before cursor (last 800 chars):
---
{text_excerpt}
---
Cursor position: {cursor_position} / {text_length} chars into the document.
{context_section}
Output the single most appropriate structural element from the allowed forms, or null.
"""


@router.post("/editor/suggest-structure", response_model=SuggestStructureResponse)
async def suggest_structure(body: SuggestStructureRequest):
    """Return a structural completion suggestion for the editor.

    The LLM is strictly constrained to return only one of:
    - A citation block:   [Insert: <filename>, p.<N> — <N>% relevance]
    - A section header:   \\n\\n## <Section Name>
    - A data table placeholder: [Insert: property comparison table for <items>]
    - null (no appropriate structural suggestion)

    Prose continuation, sentence completions, and interpretations are
    explicitly forbidden by the system prompt.
    """
    ensure_services()

    if chat_service is None:
        return SuggestStructureResponse(suggestion=None)

    llm_service = chat_service.retrieval_service.llm_service

    # Build text excerpt: last 800 chars before cursor (safe context window slice)
    cursor = max(0, min(body.cursor_position, len(body.text)))
    text_excerpt = body.text[max(0, cursor - 800):cursor]

    # Optionally surface the active stage for richer citation relevance
    context_section = ""
    if body.stage_context:
        active_stage = body.stage_context.get("activeStage")
        target_params = body.stage_context.get("targetParams") or {}
        objective = target_params.get("objective", "")
        if active_stage or objective:
            parts: List[str] = []
            if active_stage:
                parts.append(f"Active Golden Path stage: {active_stage}")
            if objective:
                parts.append(f"Research objective: {objective}")
            context_section = "Active research context:\n" + "\n".join(parts) + "\n"

    user_message = _STRUCTURE_USER_TEMPLATE.format(
        text_excerpt=text_excerpt or "(empty)",
        cursor_position=cursor,
        text_length=len(body.text),
        context_section=context_section,
    )

    # Combine system + user using Llama 3 chat template
    full_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{_STRUCTURE_SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    try:
        raw = await llm_service.generate(
            prompt=full_prompt,
            temperature=0.05,   # near-deterministic for structural output
            max_tokens=80,      # structural elements are short; cap hallucinatory drift
            stop=["<|eot_id|>", "\n\n\n"],
        )
    except Exception as exc:
        logger.warning(f"suggest-structure LLM call failed: {exc}")
        return SuggestStructureResponse(suggestion=None)

    suggestion = raw.strip()

    # Null signal from the model
    if not suggestion or suggestion.lower() == "null":
        return SuggestStructureResponse(suggestion=None)

    # Post-generation validation: reject anything that looks like prose.
    # Heuristic: more than one '.' indicates multiple sentences → prose.
    dot_count = suggestion.count(".")
    if dot_count > 1:
        logger.info(
            "suggest-structure: rejected prose-like suggestion (dot_count=%d): %.60s",
            dot_count,
            suggestion,
        )
        return SuggestStructureResponse(suggestion=None)

    # Must start with '[Insert:' or a heading marker to be a valid structural element.
    normalised = suggestion.lstrip("\n").lstrip()
    is_citation_or_table = normalised.startswith("[Insert:")
    is_header = normalised.startswith("#")
    if not (is_citation_or_table or is_header):
        logger.info(
            "suggest-structure: rejected non-structural output: %.60s", suggestion
        )
        return SuggestStructureResponse(suggestion=None)

    return SuggestStructureResponse(suggestion=suggestion)


# ============================================================
# DISCOVERY OS — STAGE 7 FEEDBACK (D5)
# ============================================================

@router.post("/api/discovery/feedback", response_model=DiscoveryFeedbackResponse)
async def discovery_feedback(body: DiscoveryFeedbackRequest):
    """Submit experimental bioassay feedback and update the knowledge graph.

    Creates or updates a knowledge graph node with the experimental result,
    enabling the feedback loop to enrich future generation cycles.
    """
    ensure_services()
    if graph_service is None:
        raise HTTPException(status_code=503, detail="Graph service unavailable")

    try:
        # Get the epoch to find the hit details (including SMILES)
        from app.core.database import get_session, Epoch
        session = get_session()
        smiles = None
        try:
            # Note: Epoch is stored in discoveryStore (frontend), not backend DB
            # The hit_id should contain or reference the SMILES if needed
            # For now, we rely on the hit_id being sufficient to identify the node
            pass
        finally:
            session.close()

        # Create or update the feedback node in the knowledge graph
        updated_node_ids = graph_service.create_or_update_feedback_node(
            hit_id=body.hit_id,
            epoch_id=body.epoch_id,
            result_name=body.result_name,
            result_value=body.result_value,
            unit=body.unit,
            passed=body.passed,
            notes=body.notes,
            smiles=smiles,
        )

        # Invalidate cache so the new feedback appears immediately
        graph_service.invalidate_cache()

        return DiscoveryFeedbackResponse(
            status="ok",
            updated_node_ids=updated_node_ids,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Discovery feedback error for hit {body.hit_id}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


# ============================================================
# DISCOVERY OS — PLUGIN VISIBILITY  (Phase 5 - Part 2)
# ============================================================

class PluginInfo(BaseModel):
    name: str
    description: str
    loaded: bool
    available: bool
    unavailable_reason: Optional[str] = None
    type: str  # "deterministic" | "semantic"
    input_schema: dict
    output_schema: dict


class PluginListResponse(BaseModel):
    plugins: List[PluginInfo]
    orchestrator_provider: str
    orchestrator_model: str
    tool_provider: str
    tool_model: str


@router.get("/api/discovery/plugins", response_model=PluginListResponse)
async def get_plugins():
    """List all registered plugins and their load status.

    Shows which deterministic tools (RDKit, spectroscopy, retrosynthesis) are
    registered and whether they're currently loaded in memory. Also displays
    the isolated Discovery LLM configuration (DeepSeek orchestrator + MiniMax tools).
    """
    from app.services.plugins import get_plugin_manager
    from app.services.discovery_llm import get_discovery_llm_service

    pm = get_plugin_manager()
    discovery_llm = get_discovery_llm_service()
    registered = pm.get_all_names()

    plugins_info = []
    for name in registered:
        plugin = pm.get_plugin(name)
        is_loaded = name in pm._loaded
        unavailable_reason = pm.get_unavailable_reason(name)
        available = plugin is not None

        plugins_info.append(PluginInfo(
            name=name,
            description=plugin.description if plugin else "Plugin failed to initialize",
            loaded=is_loaded,
            available=available,
            unavailable_reason=unavailable_reason,
            type="deterministic",  # All current plugins are deterministic
            input_schema=plugin.input_schema() if plugin else {},
            output_schema=plugin.output_schema() if plugin else {},
        ))

    return PluginListResponse(
        plugins=plugins_info,
        orchestrator_provider=discovery_llm._orchestration_provider,
        orchestrator_model=discovery_llm._orchestration_model,
        tool_provider=discovery_llm._tool_provider,
        tool_model=discovery_llm._tool_model,
    )


class UnloadPluginResponse(BaseModel):
    status: str
    plugin: str


@router.post("/api/discovery/plugins/{plugin_name}/unload", response_model=UnloadPluginResponse)
async def unload_plugin(plugin_name: str):
    """Unload a plugin to free memory.

    Useful for reclaiming RAM after heavy tool usage. Plugins will be
    lazy-loaded again on next invocation.
    """
    from app.services.plugins import get_plugin_manager

    pm = get_plugin_manager()
    if plugin_name not in pm._plugins:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_name}' not registered")

    pm.unload(plugin_name)
    return UnloadPluginResponse(status="unloaded", plugin=plugin_name)
