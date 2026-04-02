"""Configuration settings for Atlas application (Embedded Desktop Sidecar)."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


def _get_backend_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_env_path() -> Path:
    """Path to .env file (same file used for loading and saving API keys)."""
    return _get_backend_dir().parent.parent / "config" / ".env"


def _get_models_dir() -> str:
    return str(_get_backend_dir() / "models")

def _get_db_path() -> str:
    return str(_get_backend_dir() / "atlas.db")

def _get_qdrant_path() -> str:
    return str(_get_backend_dir() / "qdrant_storage")

def _get_data_dir() -> str:
    return str(_get_backend_dir() / "data")


def _get_upload_dir() -> str:
    return str(_get_backend_dir() / "data" / "uploads")

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

    # Agent Reasoning Configuration (Phase A1: Navigator 2.0)
    ENABLE_NAVIGATOR_REFLECTION: bool = True  # Enable multi-turn reflection loops
    MAX_REFLECTION_ITERATIONS: int = 3        # Max reflection cycles (prevents infinite loops)
    NAVIGATOR_CONFIDENCE_THRESHOLD: float = 0.75  # Auto-pass threshold for high confidence

    # Cortex Configuration (Phase A2: Cortex 2.0)
    ENABLE_CORTEX_CROSSCHECK: bool = True  # Enable cross-checking and contradiction detection
    CORTEX_NUM_SUBTASKS: int = 5           # Number of sub-tasks to decompose query into

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

    # Default model source preference: "local" or "api"
    DEFAULT_MODEL_SOURCE: str = "local"

    # Atlas 3.0: GraphRAG Configuration (Phase 2)
    # Strict ontology for evidence-bound extraction
    GRAPH_ONTOLOGY_EDGE_TYPES: str = "CAUSES,INHIBITS,ENABLES,PART_OF,RELATED_TO,CONTRADICTS,SUPPORTS,CLINICAL_TRIAL_FOR,TREATS,DIAGNOSES,MEASURED_BY,AUTHORED_BY,PUBLISHED_IN,FUNDED_BY"
    ENABLE_EVIDENCE_BOUND_EXTRACTION: bool = True   # Require evidence quotes for edges
    ENABLE_GRAPH_CRITIC: bool = True                 # Validate edges before committing

    # Atlas 3.0: MoE Configuration (Phase 3)
    MOE_MAX_EXPERT_ROUNDS: int = 5         # Max rounds of expert delegation
    MOE_HYPOTHESIS_COUNT: int = 3          # Number of hypotheses to generate
    ENABLE_AUTONOMOUS_MODE: bool = False   # Allow agents to pursue hypotheses without user approval

    # Atlas 3.0: Workspace Configuration (Phase 4)
    DRAFTS_DIR: str = Field(default_factory=lambda: str(Path(__file__).resolve().parent.parent.parent / "data" / "drafts"))

    # Discovery OS Configuration (Phase 1)
    MAX_TOOL_ITERATIONS: int = 8           # Max ReAct loop iterations before forced final_answer
    ENABLE_DISCOVERY_MODE: bool = True     # Enable the Discovery OS pipeline
    DISCOVERY_DEFAULT_PHASE: str = "hit_identification"  # Default workflow phase

    # Discovery OS LLM Configuration (Phase 5 - Part 1: Isolated from global model selector)
    # These settings are ONLY used by Discovery OS agents (Coordinator, Executor)
    # and do NOT affect chat/retrieval/global model selection
    DISCOVERY_ORCHESTRATION_PROVIDER: str = "deepseek"  # Provider for planning/reasoning
    DISCOVERY_ORCHESTRATION_MODEL: str = "deepseek-chat"  # DeepSeek V3 — supports JSON output, system prompts, temperature
    DISCOVERY_TOOL_PROVIDER: str = "minimax"  # Provider for tool calling/constrained generation
    DISCOVERY_TOOL_MODEL: str = "MiniMax-M2.5"  # Model for structured outputs and tool calls
    DISCOVERY_SCRIPT_TIMEOUT: int = 300  # Seconds before executor script subprocess is killed


settings = Settings()

# Ensure directories exist
Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.QDRANT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(settings.DRAFTS_DIR).mkdir(parents=True, exist_ok=True)