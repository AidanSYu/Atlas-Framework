"""Configuration settings for Atlas application (Embedded Desktop Sidecar)."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _get_backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _get_app_data_root() -> Path:
    """Return the per-user app data root where managed workspaces live.

    Matches the bundled convention established in run_server.py:
      Windows: %LOCALAPPDATA%/Atlas
      macOS:   ~/Library/Application Support/Atlas
      Linux:   ~/.atlas
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Atlas"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Atlas"
    return Path.home() / ".atlas"


def _get_workspaces_dir() -> str:
    return str(_get_app_data_root() / "workspaces")


def get_env_path() -> Path:
    """Path to .env file (same file used for loading and saving API keys)."""
    return _get_backend_dir().parent.parent / "config" / ".env"


def _resolve_config_path(value: str) -> str:
    """Resolve relative config paths against the backend directory."""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((_get_backend_dir() / path).resolve())


def _get_models_dir() -> str:
    return str(_get_backend_dir().parent.parent / "models")

def _get_db_path() -> str:
    return str(_get_backend_dir() / "atlas.db")

def _get_qdrant_path() -> str:
    return str(_get_backend_dir() / "qdrant_storage")

def _get_data_dir() -> str:
    return str(_get_backend_dir() / "data")


def _get_upload_dir() -> str:
    return str(_get_backend_dir() / "data" / "uploads")


def _get_plugins_dir() -> str:
    return str(_get_backend_dir() / "plugins")


def _get_domains_dir() -> Path:
    return _get_backend_dir() / "domains"


def _load_domain_profile(domain_name: str) -> dict:
    """Load base.json + the named domain profile, merging edge_types and entity_labels."""
    domains_dir = _get_domains_dir()
    base_path = domains_dir / "base.json"

    edge_types: list[str] = []
    entity_labels: list[str] = []

    # Always load base
    if base_path.exists():
        with open(base_path, "r") as f:
            base = json.load(f)
        edge_types.extend(base.get("edge_types", []))
        entity_labels.extend(base.get("entity_labels", []))

    # Layer the vertical on top (skip if domain_name is "base" or empty)
    if domain_name and domain_name != "base":
        vertical_path = domains_dir / f"{domain_name}.json"
        if vertical_path.exists():
            with open(vertical_path, "r") as f:
                vertical = json.load(f)
            edge_types.extend(vertical.get("edge_types", []))
            entity_labels.extend(vertical.get("entity_labels", []))
        else:
            logger.warning(f"Domain profile '{domain_name}' not found at {vertical_path}")

    # Deduplicate while preserving order
    seen_e, seen_l = set(), set()
    edge_types = [t for t in edge_types if not (t in seen_e or seen_e.add(t))]
    entity_labels = [l for l in entity_labels if not (l in seen_l or seen_l.add(l))]

    return {
        "edge_types": ",".join(edge_types),
        "entity_labels": ",".join(entity_labels),
    }

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=str(_get_backend_dir().parent.parent / "config" / ".env"),
        case_sensitive=True,
        extra="ignore"
    )

    # SQLite Configuration (embedded - no external server)
    DATABASE_PATH: str = Field(default_factory=_get_db_path)
    
    @property
    def database_url(self) -> str:
        """Construct SQLite connection URL."""
        return f"sqlite:///{self.DATABASE_PATH}"
    
    # Qdrant Configuration (embedded - runs in-process via path mode)
    QDRANT_STORAGE_PATH: str = Field(default_factory=_get_qdrant_path)
    QDRANT_COLLECTION: str = "atlas_documents"
    
    # Model Storage (for bundled LLM and NER models)
    MODELS_DIR: str = Field(default_factory=_get_models_dir)
    
    # File Storage
    DATA_DIR: str = Field(default_factory=_get_data_dir)
    UPLOAD_DIR: str = Field(default_factory=_get_upload_dir)
    
    # API Configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    
    # Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5

    # Prompt Engineering Configuration (Phase A3)
    USE_PROMPT_TEMPLATES: bool = True      # Enable few-shot prompt templates
    ENABLE_OUTPUT_VALIDATION: bool = True  # Enable structured output validation with retries
    MAX_VALIDATION_RETRIES: int = 1        # Max retries for malformed LLM outputs

    # RAG Optimization Configuration (Phase B)
    ENABLE_RERANKING: bool = True          # Enable FlashRank reranking
    RERANK_TOP_N: int = 5                  # Number of chunks to keep after reranking
    GRAPH_CACHE_TTL: int = 300             # Cache duration for graph queries (seconds)

    # Phase B2: Document Parsing
    USE_DOCLING: bool = True               # Enable Docling VLM parsing (fallback: pdfplumber)

    # Phase B3: Semantic Chunking
    USE_SEMANTIC_CHUNKING: bool = True     # Enable semantic chunking (fallback: fixed-size)
    SEMANTIC_CHUNK_TOKENS: int = 512       # Target tokens per semantic chunk

    # Phase B4: RAPTOR Hierarchical Summarization
    USE_RAPTOR: bool = True                # Enable RAPTOR hierarchy on ingestion
    RAPTOR_CLUSTERS: int = 5              # Number of L1 clusters per document

    # Atlas 3.0: LLM Performance Configuration
    LLM_CONTEXT_SIZE: int = 8192           # Context window (tokens). 8192 for 7B, 4096 for 3B if OOM
    LLM_N_BATCH: int = 512                 # Batch size for prompt processing
    LLM_USE_MLOCK: bool = True             # Keep model weights pinned in RAM (prevents swapping)
    LLM_VERBOSE: bool = False              # Disable verbose llama.cpp logging in production

    # Atlas 3.0: Hybrid LLM Layer (Phase 1 - API Model Support)
    # API keys for cloud models (optional - leave empty for local-only mode)
    DEEPSEEK_API_KEY: str = ""             # DeepSeek V3/R1 API key
    MINIMAX_API_KEY: str = ""              # MiniMax 2.5 API key
    OPENAI_API_KEY: str = ""               # OpenAI API key (optional)
    ANTHROPIC_API_KEY: str = ""            # Anthropic API key (optional)

    # Cloud model registry: models available when API keys are configured
    # Format: "provider/model-name" as used by LiteLLM
    CLOUD_MODELS: str = "deepseek/deepseek-chat,deepseek/deepseek-reasoner,minimax/MiniMax-M2.5,openai/gpt-4o,openai/gpt-4o-mini,anthropic/claude-sonnet-4-20250514"

    # Atlas Domain Profile — controls ontology and entity labels per vertical
    # Set to "base" for domain-agnostic, "chemistry" for drug discovery, "manufacturing" for Prometheus, etc.
    ATLAS_DOMAIN: str = "base"

    # Atlas 3.0: GraphRAG Configuration (Phase 2)
    # Populated at startup from domains/base.json + domains/{ATLAS_DOMAIN}.json
    GRAPH_ONTOLOGY_EDGE_TYPES: str = "CAUSES,ENABLES,PART_OF,RELATED_TO,CONTRADICTS,SUPPORTS,MEASURED_BY"
    GRAPH_ENTITY_LABELS: str = "Person,Organization,Location,Concept,Method,Date,Event,Work,Title,Institution"
    ENABLE_EVIDENCE_BOUND_EXTRACTION: bool = True   # Require evidence quotes for edges
    ENABLE_GRAPH_CRITIC: bool = True                 # Validate edges before committing

    # Atlas 3.0: Workspace Configuration (Phase 4)
    DRAFTS_DIR: str = Field(default_factory=lambda: str(Path(__file__).resolve().parent.parent.parent / "data" / "drafts"))

    # Managed Workspace Storage (AppData-rooted, isolated per workspace)
    ATLAS_WORKSPACES_DIR: str = Field(default_factory=_get_workspaces_dir)

    # Atlas Framework Configuration
    ATLAS_PLUGIN_DIR: str = Field(default_factory=_get_plugins_dir)
    ATLAS_ORCHESTRATOR_MODEL: str = "nvidia_Orchestrator-8B-IQ2_M.gguf"
    # Nemotron-8B-IQ2_M weights ≈ 3 GB. KV cache + compute buffer scale with n_ctx.
    # At 32k the total VRAM footprint is ~3.9 GB, which OOMs a 4 GB card (RTX 3050)
    # the moment anything else holds VRAM (embedder, Windows compositor) and llama.cpp
    # silently drops to CPU. 8192 keeps the load ~3.0 GB with flash_attn and leaves
    # margin. Users with bigger GPUs can raise via env (ATLAS_ORCHESTRATOR_CONTEXT_SIZE).
    ATLAS_ORCHESTRATOR_CONTEXT_SIZE: int = 8192
    ATLAS_ORCHESTRATOR_MAX_TOKENS: int = 2048     # Room for <think> reasoning + <tool_call> + text. Don't raise without raising n_ctx — the math breaks past iteration 1 (sys ~2500 + history ~1500 + max_tokens) on 8192 ctx.
    ATLAS_ORCHESTRATOR_MAX_ITERATIONS: int = 12   # Safety bound — model decides when to stop
    ATLAS_ORCHESTRATOR_TEMPERATURE: float = 0.15
    ATLAS_ORCHESTRATOR_RESPONSE_MAX_CHARS: int = 1500
    # Llama-cpp thread count for the orchestrator. 0 = use os.cpu_count() // 2
    # (avoid hyperthreaded virtual cores; they typically hurt throughput).
    ATLAS_ORCHESTRATOR_N_THREADS: int = 0
    # How long to consider a plugin catalog scan fresh before re-checking the
    # plugin directory's mtime. Avoids per-request disk rescans.
    ATLAS_PLUGIN_CATALOG_TTL_SECONDS: float = 5.0

    # Discovery OS LLM Configuration (Phase 5 - Part 1: Isolated from global model selector)
    # These settings are ONLY used by Discovery OS agents (Coordinator, Executor)
    # and do NOT affect chat/retrieval/global model selection
    DISCOVERY_ORCHESTRATION_PROVIDER: str = "deepseek"  # Provider for planning/reasoning
    DISCOVERY_ORCHESTRATION_MODEL: str = "deepseek-chat"  # DeepSeek V3 — supports JSON output, system prompts, temperature
    DISCOVERY_TOOL_PROVIDER: str = "minimax"  # Provider for tool calling/constrained generation
    DISCOVERY_TOOL_MODEL: str = "MiniMax-M2.5"  # Model for structured outputs and tool calls
    DISCOVERY_SCRIPT_TIMEOUT: int = 300  # Seconds before executor script subprocess is killed


settings = Settings()

# Apply domain profile — merges base.json + vertical into ontology settings
_domain = _load_domain_profile(settings.ATLAS_DOMAIN)
settings.GRAPH_ONTOLOGY_EDGE_TYPES = _domain["edge_types"]
settings.GRAPH_ENTITY_LABELS = _domain["entity_labels"]
logger.info(f"Atlas domain: {settings.ATLAS_DOMAIN} — {len(_domain['edge_types'].split(','))} edge types, {len(_domain['entity_labels'].split(','))} entity labels")

settings.DATABASE_PATH = _resolve_config_path(settings.DATABASE_PATH)
settings.QDRANT_STORAGE_PATH = _resolve_config_path(settings.QDRANT_STORAGE_PATH)
settings.MODELS_DIR = _resolve_config_path(settings.MODELS_DIR)
settings.DATA_DIR = _resolve_config_path(settings.DATA_DIR)
settings.UPLOAD_DIR = _resolve_config_path(settings.UPLOAD_DIR)
settings.DRAFTS_DIR = _resolve_config_path(settings.DRAFTS_DIR)
settings.ATLAS_PLUGIN_DIR = _resolve_config_path(settings.ATLAS_PLUGIN_DIR)
# ATLAS_WORKSPACES_DIR is absolute by default (AppData/~/Library); keep user overrides as-is if absolute.
if not Path(settings.ATLAS_WORKSPACES_DIR).is_absolute():
    settings.ATLAS_WORKSPACES_DIR = _resolve_config_path(settings.ATLAS_WORKSPACES_DIR)

# Prefer the repository-root models directory when an older relative env path
# resolves to a non-existent location such as src/models.
_repo_models_dir = (_get_backend_dir().parent.parent / "models").resolve()
if not Path(settings.MODELS_DIR).exists() and _repo_models_dir.exists():
    settings.MODELS_DIR = str(_repo_models_dir)

# Ensure directories exist
Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.QDRANT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(settings.DRAFTS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.ATLAS_PLUGIN_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.ATLAS_WORKSPACES_DIR).mkdir(parents=True, exist_ok=True)
