"""
LLM Service - Bundled LLM and embedding service using llama-cpp-python.

Replaces Ollama for standalone desktop app deployment.
Uses:
- llama-cpp-python for text generation (Llama 3 GGUF)
- sentence-transformers for embeddings (nomic-embed-text-v1.5)
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import os
import sys
import asyncio
import threading
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure CUDA runtime DLLs (installed via pip nvidia-* packages) are findable
# on Windows before llama_cpp tries to load its shared libraries.
# ---------------------------------------------------------------------------
_cuda_dll_dirs_added = False


def _add_cuda_dll_directories():
    """Register NVIDIA pip-package DLL directories with Windows DLL loader.
    Looks in Python site-packages (where pip installs nvidia-* wheels) and
    in the project/backend folder in case libs were extracted there.
    """
    global _cuda_dll_dirs_added
    if _cuda_dll_dirs_added or sys.platform != "win32":
        return
    _cuda_dll_dirs_added = True

    # Pip nvidia-* wheels put DLLs in bin/ (cudart64_12.dll, cublas64_12.dll, etc.)
    nvidia_lib_subdirs = [
        "nvidia/cublas/bin",
        "nvidia/cuda_runtime/bin",
        "nvidia/cublas/lib",
        "nvidia/cuda_runtime/lib",
        "nvidia/cuda_runtime/lib/x64",
        "nvidia/cufft/lib",
        "nvidia/cusparse/lib",
    ]

    dirs_to_check = []

    try:
        import site
        # User site-packages (pip --user installs)
        user_site = site.getusersitepackages()
        if user_site:
            dirs_to_check.append(Path(user_site))
        # System site-packages
        for sp in site.getsitepackages():
            dirs_to_check.append(Path(sp))
    except Exception:
        pass

    # Project folder: backend/app/services/llm.py -> backend = parents[2], project root = parents[3]
    try:
        _backend = Path(__file__).resolve().parents[2]
        _project_root = _backend.parent
        dirs_to_check.extend([_project_root, _backend])
    except Exception:
        pass

    for base in dirs_to_check:
        for subdir in nvidia_lib_subdirs:
            candidate = base / subdir
            if candidate.is_dir():
                try:
                    os.add_dll_directory(str(candidate))
                    logger.info("Added CUDA DLL directory: %s", candidate)
                except Exception as e:
                    logger.debug("Could not add DLL dir %s: %s", candidate, e)

    # CRITICAL: Add llama_cpp/lib directory itself to DLL search path
    try:
        import importlib.util
        spec = importlib.util.find_spec("llama_cpp")
        if spec and spec.origin:
            lc_lib = Path(spec.origin).resolve().parent / "lib"
            if lc_lib.is_dir():
                try:
                    os.add_dll_directory(str(lc_lib))
                    logger.info("Added llama_cpp lib directory: %s", lc_lib)
                except Exception as e:
                    logger.warning("Could not add llama_cpp lib dir: %s", e)
                
                # Also prepend to PATH as a fallback
                existing_path = os.environ.get("PATH", "")
                if str(lc_lib) not in existing_path:
                    os.environ["PATH"] = str(lc_lib) + os.pathsep + existing_path
                    logger.info("Added llama_cpp lib to PATH")
    except Exception as e:
        logger.debug("Could not locate llama_cpp lib directory: %s", e)

    # Prepend nvidia bin dirs to PATH so the loader finds cudart/cublas when loading ggml-cuda.dll
    path_additions = []
    for base in dirs_to_check:
        for subdir in ("nvidia/cuda_runtime/bin", "nvidia/cublas/bin"):
            candidate = base / subdir
            if candidate.is_dir():
                path_additions.append(str(candidate))
    if path_additions:
        existing = os.environ.get("PATH", "")
        new_path = os.pathsep.join(path_additions) + os.pathsep + existing
        os.environ["PATH"] = new_path
        logger.info("Prepended %d CUDA bin dir(s) to PATH", len(path_additions))

    # Copy CUDA runtime DLLs next to llama_cpp lib so ggml-cuda.dll can load them
    try:
        import shutil
        import importlib.util
        spec = importlib.util.find_spec("llama_cpp")
        lc_lib = Path(spec.origin).resolve().parent / "lib" if spec and spec.origin else None
        if lc_lib and (lc_lib / "llama.dll").exists():
            for subdir, dll_name in [
                ("nvidia/cuda_runtime/bin", "cudart64_12.dll"),
                ("nvidia/cublas/bin", "cublas64_12.dll"),
                ("nvidia/cublas/bin", "cublasLt64_12.dll"),
            ]:
                for base in dirs_to_check:
                    src = base / subdir / dll_name
                    if src.is_file():
                        dest = lc_lib / dll_name
                        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
                            try:
                                shutil.copy2(src, dest)
                                logger.info("Copied CUDA DLL to llama_cpp lib: %s", dll_name)
                            except Exception as e:
                                logger.debug("Could not copy %s: %s", src, e)
                        break
    except Exception as e:
        logger.debug("Could not copy CUDA DLLs: %s", e)


# Call DLL setup at module import time to ensure directories are registered
# before any code tries to import llama_cpp
if sys.platform == "win32":
    try:
        _add_cuda_dll_directories()
    except Exception as e:
        logger.debug("Early DLL directory setup failed (non-fatal): %s", e)


# Lazy imports to avoid loading heavy models at import time
_llm_instance = None
_embedder_instance = None

# Default GPU layers for partial offload
# RTX 3050 (4GB VRAM): Use 35 layers for Llama 3 8B-Instruct (Q4_K_M quantization)
# This allows ~3.2GB GPU usage, leaving headroom for inference
DEFAULT_GPU_LAYERS = 35


def _resolve_gpu_layers() -> int:
    """Resolve GPU layer count from env with a safe default."""
    raw = os.environ.get("ATLAS_GPU_LAYERS", "").strip()
    if not raw:
        return DEFAULT_GPU_LAYERS
    if raw.lower() == "auto":
        return -1
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_GPU_LAYERS


class LLMService:
    """Hybrid LLM service supporting both local GGUF models and cloud API models.

    Atlas 3.0: This service provides:
    - Text generation using local Llama/Qwen/Phi GGUF models (llama-cpp-python)
    - Text generation using cloud APIs via LiteLLM (DeepSeek, MiniMax, OpenAI, etc.)
    - Text embeddings using nomic-embed-text-v1.5 (always local)
    - Runtime model switching between local and API models

    Models are loaded from the MODELS_DIR specified in settings.
    API models are routed through LiteLLM when API keys are configured.
    """

    _instance: Optional['LLMService'] = None

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize LLM service.

        Args:
            models_dir: Path to directory containing models.
                       Defaults to settings.MODELS_DIR.
        """
        self.models_dir = Path(models_dir or settings.MODELS_DIR)
        self._llm = None
        self._embedder = None
        self._embedding_dim = 768  # nomic-embed-text dimension
        self._active_model_name: Optional[str] = None
        self._model_type: str = "llama"
        self._gpu_layers: int = _resolve_gpu_layers()
        self._device: str = "unloaded"
        self._llm_last_error: Optional[str] = None
        # Lock to prevent concurrent model loading/switching
        self._model_lock = asyncio.Lock()
        # Lock to prevent concurrent LLM inference (llama_cpp is NOT thread-safe)
        self._llm_lock = threading.Lock()
        # Lock to prevent concurrent embedding generation (Nomic v1.5 race condition fix)
        self._embed_lock = threading.Lock()
        # Lock to prevent concurrent embedding MODEL LOADING (race condition:
        # multiple async requests see _embedder=None and all try to load)
        self._embed_load_lock = threading.Lock()
        self._is_initializing = False

        # Status tracking for background services
        self._is_generating = False

        # Atlas 3.0: Hybrid LLM Layer - API model support
        self._model_source: str = "local"  # "local" or "api"
        self._api_model_name: Optional[str] = None  # e.g., "deepseek/deepseek-chat"
        self._litellm_available = False
        self._setup_litellm()

        logger.info(f"LLMService initializing with models_dir: {self.models_dir}")

    # ----------------------------------------------------------------
    # Atlas 3.0: LiteLLM Setup & API Model Support
    # ----------------------------------------------------------------

    def _setup_litellm(self):
        """Configure LiteLLM with available API keys."""
        try:
            import litellm
            self._litellm_available = True
            litellm.set_verbose = False

            # Set API keys from config
            if settings.DEEPSEEK_API_KEY:
                os.environ["DEEPSEEK_API_KEY"] = settings.DEEPSEEK_API_KEY
            if settings.MINIMAX_API_KEY:
                os.environ["MINIMAX_API_KEY"] = settings.MINIMAX_API_KEY
            if settings.OPENAI_API_KEY:
                os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
            if settings.ANTHROPIC_API_KEY:
                os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY

            logger.info("LiteLLM initialized for API model support")
        except ImportError:
            logger.info("LiteLLM not installed - API model support disabled")
            self._litellm_available = False

    def list_available_api_models(self) -> List[Dict[str, str]]:
        """List cloud API models that have valid API keys configured.

        Returns:
            List of dicts with 'name', 'provider', and 'requires_key' fields.
        """
        models = []
        configured = settings.CLOUD_MODELS.split(",") if settings.CLOUD_MODELS else []

        # Map provider prefixes to their API key settings
        key_map = {
            "deepseek": settings.DEEPSEEK_API_KEY,
            "minimax": settings.MINIMAX_API_KEY,
            "openai": settings.OPENAI_API_KEY,
            "anthropic": settings.ANTHROPIC_API_KEY,
        }

        for model_id in configured:
            model_id = model_id.strip()
            if not model_id:
                continue
            provider = model_id.split("/")[0] if "/" in model_id else model_id
            has_key = bool(key_map.get(provider, ""))
            models.append({
                "name": model_id,
                "provider": provider,
                "has_key": has_key,
                "source": "api",
            })

        return models

    async def load_api_model(self, model_name: str) -> str:
        """Switch to an API model via LiteLLM.

        Args:
            model_name: LiteLLM model identifier (e.g., "deepseek/deepseek-chat")

        Returns:
            The active model name.

        Raises:
            RuntimeError: If LiteLLM is not available or API key is missing.
        """
        if not self._litellm_available:
            raise RuntimeError("LiteLLM not installed. Install with: pip install litellm")

        provider = model_name.split("/")[0] if "/" in model_name else ""
        key_map = {
            "deepseek": settings.DEEPSEEK_API_KEY,
            "minimax": settings.MINIMAX_API_KEY,
            "openai": settings.OPENAI_API_KEY,
            "anthropic": settings.ANTHROPIC_API_KEY,
        }

        if provider in key_map and not key_map[provider]:
            raise RuntimeError(
                f"API key for '{provider}' is not configured. "
                f"Set {provider.upper()}_API_KEY in your .env file."
            )

        # Unload local model to free VRAM if switching from local
        if self._model_source == "local" and self._llm is not None and self._llm != "FALLBACK":
            logger.info(f"Unloading local model to switch to API: {model_name}")
            del self._llm
            self._llm = None
            import gc
            gc.collect()

        self._model_source = "api"
        self._api_model_name = model_name
        self._active_model_name = model_name
        self._device = "cloud"
        self._model_type = "api"

        logger.info(f"Switched to API model: {model_name}")
        return model_name

    async def _generate_via_api(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Generate text using a cloud API model via LiteLLM.

        Args:
            messages: Chat messages in OpenAI format [{role, content}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            Generated text string
        """
        import litellm

        kwargs: Dict[str, Any] = {
            "model": self._api_model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API generation failed ({self._api_model_name}): {e}")
            raise RuntimeError(f"API model generation failed: {e}") from e
    
    # ----------------------------------------------------------------
    # Llama 3 Chat Template
    # ----------------------------------------------------------------

    @staticmethod
    def _format_llama3_prompt(system_msg: str, user_msg: str) -> str:
        """Format a prompt using the Llama 3 Instruct chat template.
        
        Args:
            system_msg: The system instruction (role/persona).
            user_msg: The user's message / query with context.
            
        Returns:
            Formatted prompt string with Llama 3 special tokens.
        """
        return (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    @staticmethod
    def _format_qwen_prompt(system_msg: str, user_msg: str) -> str:
        """Format a prompt using the Qwen ChatML template."""
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _format_phi3_prompt(system_msg: str, user_msg: str) -> str:
        """Format a prompt using the Phi-3 chat template.
        
        Phi-3 template:
        <|system|>
        failed to find system prompt!
        <|end|>
        <|user|>
        {msg}<|end|>
        <|assistant|>
        """
        # Phi-3 typically doesn't use a system prompt in the standard template,
        # but the instruct version supports it via <|system|>.
        return (
            f"<|system|>\n{system_msg}<|end|>\n"
            f"<|user|>\n{user_msg}<|end|>\n"
            "<|assistant|>\n"
        )

    # ----------------------------------------------------------------
    # Model Loading
    # ----------------------------------------------------------------

    def _load_llm(self, model_name: Optional[str] = None):
        """Load the LLM model (lazy loading).
        
        Args:
            model_name: Optional specific GGUF filename to load.
                       If None, auto-detects from models_dir.
        """
        if self._llm is not None and self._llm != "FALLBACK":
            return
        
        # Ensure CUDA DLL directories are registered before importing llama_cpp
        _add_cuda_dll_directories()
        
        try:
            from llama_cpp import Llama
        except ImportError as e:
            error_msg = str(e)
            self._llm_last_error = error_msg
            if "llama.dll" in error_msg or "shared library" in error_msg.lower():
                logger.error("Failed to load llama.dll - missing dependencies: %s", error_msg)
                logger.error("Check that CUDA DLLs are in .venv/Lib/site-packages/llama_cpp/lib/")
            
            import sys
            if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
                logger.warning(
                    "LLM runtime not available in this installation (bundled app). "
                    "Use development mode from source for full LLM support, or reinstall with a build that includes the LLM backend. Error: %s",
                    e,
                )
            else:
                logger.warning(
                    "llama-cpp-python not installed. Text generation will use fallback. "
                    "Install with: pip install llama-cpp-python (in your backend venv or environment)."
                )
            self._llm = "FALLBACK"  # Mark as fallback mode
            self._device = "cpu"
            return
        
        model_path = None
        
        # If a specific model was requested, look for it directly
        if model_name:
            candidate = self.models_dir / model_name
            if candidate.exists():
                model_path = candidate
            else:
                logger.warning(f"Requested model '{model_name}' not found in {self.models_dir}")

        # Auto-detect if no specific model or requested model not found.
        # Ingestion jobs (relationship extraction, RAPTOR summaries, metadata)
        # share this service, so prefer a small model that leaves VRAM for the
        # orchestrator. The orchestrator loads its Nemotron GGUF separately.
        if model_path is None:
            model_patterns = [
                "qwen2.5-1.5b-instruct*.gguf",  # Priority 1: Small, fast, frees VRAM for orchestrator
                "Qwen2.5-3B-Instruct*.gguf",
                "Phi-3.5-mini-instruct*.gguf",
                "llama-3-8b-instruct*.gguf",
                "Meta-Llama-3-8B-Instruct*.gguf",
                "*.gguf"
            ]
            for pattern in model_patterns:
                matches = [
                    p for p in self.models_dir.glob(pattern)
                    if "orchestrator" not in p.name.lower()
                ]
                if matches:
                    model_path = matches[0]
                    break
        
        if not model_path or not model_path.exists():
            logger.warning(
                f"LLM model not found in {self.models_dir}. Using fallback mode. "
                f"Download a .gguf model for full functionality."
            )
            self._llm = "FALLBACK"
            self._active_model_name = None
            self._device = "cpu"
            return
        
        logger.info(f"Loading LLM from {model_path}")
        logger.info(f"GPU layers config: {self._gpu_layers}")
        # Task 0.1: Increase Context Window (Configurable)
        # Task 0.5: Enable Prompt Caching
        n_ctx = int(os.environ.get("ATLAS_N_CTX", "8192"))
        logger.info(f"Llama Context Size: {n_ctx}")

        from app.services import model_slot
        model_slot.acquire("ingest", self.unload)

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=self._gpu_layers,
            n_batch=512,         # Increased batch size for speed
            use_mlock=True,      # Keep model in RAM to prevent swapping
            check_tensors=False, # Skip tensor checks for faster loading
            cache=True,          # Enable KV cache for multi-turn speedup
            flash_attn=True,     # Halves KV cache VRAM — required for 4GB cards
            verbose=settings.LLM_VERBOSE,
            n_threads=4
        )
        self._active_model_name = model_path.name
        
        # Detect model type for chat templates
        name_lower = model_path.name.lower()
        if "phi" in name_lower:
             self._model_type = "phi3"
        elif "qwen" in name_lower:
             self._model_type = "qwen"
        else:
             self._model_type = "llama"
        
        # Prefer llama_cpp's own GPU detection: it reflects whether THIS backend was
        # built with CUDA and can use the GPU. Do NOT use torch.cuda.is_available():
        # requirements.txt installs CPU-only PyTorch by default, so torch would report
        # False even when llama-cpp-python is using the GPU.
        has_cuda = False
        try:
            from llama_cpp.llama_cpp import llama_supports_gpu_offload
            has_cuda = bool(llama_supports_gpu_offload())
        except (ImportError, AttributeError, Exception):
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except ImportError:
                pass

        if self._gpu_layers > 0 and has_cuda:
            self._device = "gpu"
        else:
            self._device = "cpu"
            if self._gpu_layers > 0:
                logger.warning(
                    "GPU layers requested but CUDA not available/detected. "
                    "Install llama-cpp-python with CUDA (e.g. pip install llama-cpp-python "
                    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121). Reporting CPU."
                )

        logger.info(f"LLM loaded successfully: {self._active_model_name}")
        logger.info(f"Model type: {self._model_type}, Device: {self._device}")
        logger.info(f"GPU layers: {self._gpu_layers}, actual device: {self._device}")
    
    def unload(self) -> None:
        """Drop the resident LLM so the GPU slot is free.

        Called by the model slot coordinator when another owner (e.g. the
        Nemotron orchestrator) needs the GPU. Safe to call when no model is
        loaded.
        """
        if self._llm is None or self._llm == "FALLBACK":
            return
        logger.info("Unloading ingestion LLM: %s", self._active_model_name)
        try:
            if hasattr(self._llm, "close"):
                self._llm.close()
            del self._llm
        except Exception as e:
            logger.warning("Error closing ingestion LLM: %s", e)
        self._llm = None
        self._active_model_name = None
        self._device = "unloaded"
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _load_embedder(self):
        """Load the embedding model (lazy loading, thread-safe).

        Uses _embed_load_lock to prevent multiple concurrent requests from
        loading the model simultaneously (which wastes RAM and can crash).
        """
        # Fast path: already loaded
        if self._embedder is not None:
            return

        with self._embed_load_lock:
            # Double-check after acquiring lock (another thread may have loaded it)
            if self._embedder is not None:
                return

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

            # Look for embedding model directory
            embed_patterns = [
                "nomic-embed-text-v1.5",
                "nomic-embed-text*",
            ]

            embed_path = None
            for pattern in embed_patterns:
                matches = list(self.models_dir.glob(pattern))
                if matches:
                    embed_path = matches[0]
                    break

            # nomic-embed uses custom code (nomic-bert-2048) - trust_remote_code required
            # Must pass as direct arg; model_kwargs is not forwarded to AutoConfig.from_pretrained
            # Default to CPU: nomic-embed is 137M params, fast enough on CPU, and on
            # low-VRAM cards (RTX 3050 / 4GB) keeping it off the GPU is required so
            # the Nemotron orchestrator can fit. Override with ATLAS_EMBED_DEVICE=cuda.
            embed_device = os.environ.get("ATLAS_EMBED_DEVICE", "cpu").strip() or "cpu"
            st_kwargs = {"trust_remote_code": True, "device": embed_device}
            if embed_path and embed_path.exists():
                logger.info(f"Loading embedding model from {embed_path} on {embed_device}")
                self._embedder = SentenceTransformer(str(embed_path), **st_kwargs)
            else:
                # Fall back to downloading from HuggingFace (first run)
                logger.warning(
                    f"Embedding model not found in {self.models_dir}. "
                    "Downloading from HuggingFace..."
                )
                self._embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", **st_kwargs)

            logger.info("Embedding model loaded successfully")
    
    @property
    def active_model_name(self) -> Optional[str]:
        """Return the name of the currently loaded model."""
        return self._active_model_name

    @property
    def is_generating(self) -> bool:
        """Return True if the LLM is actively generating text."""
        return self._is_generating

    def list_available_models(self) -> List[str]:
        """List all .gguf models in the models directory."""
        if not self.models_dir.exists():
            return []
        return [f.name for f in self.models_dir.glob("*.gguf")]

    async def load_model(self, model_name: str) -> bool:
        """Swap to a different model.
        
        Args:
            model_name: Name of the .gguf file to load.
            
        Returns:
            True if model was loaded successfully (or already loaded), False otherwise.
        """
        if self._active_model_name == model_name:
            return True
            
        model_path = self.models_dir / model_name
        if not model_path.exists():
            logger.error(f"Cannot allow swap: Model {model_name} not found.")
            return False
            
        logger.info(f"Swapping model from {self._active_model_name} to {model_name}...")
        
        # Unload current model if it exists
        if self._llm and hasattr(self._llm, "__del__"):
             self._llm = None
             import gc
             gc.collect()
             try:
                 import torch
                 if torch.cuda.is_available():
                     torch.cuda.empty_cache()
             except ImportError:
                 pass

        # Load new model
        try:
            self._load_llm(model_name=model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to swap to model {model_name}: {e}")
            self._llm = "FALLBACK"
            return False

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate text completion.

        Atlas 3.0: Routes to either local llama.cpp or cloud API via LiteLLM
        based on the current model source.

        Args:
            prompt: The prompt to complete (should already be formatted
                   with the appropriate chat template for local models).
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
            stop: List of stop sequences. Defaults to Llama 3 stop tokens.

        Returns:
            Generated text string
        """
        # Atlas is offline-first: the cloud LLM stack was removed.
        # Lazy-load the local GGUF model.
        if self._llm is None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_llm)

        # Local model unavailable — surface the real failure instead of returning a
        # canned string that hides the problem from callers (the user's "no silent
        # fallback" doctrine).
        if self._llm == "FALLBACK":
            import sys
            if getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"):
                detail = (
                    "LLM is not available in this installation. Run Atlas from "
                    "source (development mode) for full LLM support."
                )
            else:
                detail = (
                    "LLM model is not available. Install llama-cpp-python "
                    "(pip install llama-cpp-python) and add a GGUF model to the "
                    "models folder."
                )
            raise RuntimeError(f"LLMService.generate: {detail}")

        # Default stop tokens based on loaded model
        if stop is None:
            if self._model_type == "qwen":
                stop = ["<|im_end|>", "<|endoftext|>"]
            elif self._model_type == "phi3":
                stop = ["<|end|>", "<|endoftext|>"]
            else:
                stop = ["<|eot_id|>", "<|end_of_text|>"]

        # Run generation in executor to not block event loop
        loop = asyncio.get_running_loop()

        def _generate():
            self._is_generating = True
            try:
                with self._llm_lock:
                    output = self._llm(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                        echo=False
                    )
                    return output["choices"][0]["text"].strip()
            finally:
                self._is_generating = False

        return await loop.run_in_executor(None, _generate)

    async def generate_constrained(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate JSON output strictly conforming to a JSON schema.

        Atlas 3.0: For local models, uses llama-cpp-python's GBNF grammar.
        For API models, uses response_format=json_object and parses the output.

        Args:
            prompt: The prompt (should request JSON output).
            schema: JSON Schema dict describing the expected output shape.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Parsed dict guaranteed to match the schema structure.
        """
        import json as _json

        # Atlas is offline-first: route only to the local GGUF model.
        # Lazy-load it.
        if self._llm is None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_llm)

        if self._llm == "FALLBACK":
            logger.warning("generate_constrained called in fallback mode")
            return {}

        loop = asyncio.get_running_loop()

        def _generate():
            try:
                from llama_cpp import LlamaGrammar
                grammar = LlamaGrammar.from_json_schema(_json.dumps(schema))
            except (ImportError, Exception) as e:
                logger.warning(f"Grammar creation failed, falling back to unconstrained: {e}")
                grammar = None

            # Determine stop tokens
            if self._model_type == "qwen":
                stop = ["<|im_end|>", "<|endoftext|>"]
            elif self._model_type == "phi3":
                stop = ["<|end|>", "<|endoftext|>"]
            else:
                stop = ["<|eot_id|>", "<|end_of_text|>"]

            with self._llm_lock:
                output = self._llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    grammar=grammar,
                    echo=False,
                )
                return output["choices"][0]["text"].strip()

        raw = await loop.run_in_executor(None, _generate)

        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            # Grammar should prevent this, but handle edge cases
            logger.warning(f"Constrained generation produced invalid JSON: {raw[:200]}")
            # Try extracting JSON substring
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return _json.loads(raw[start:end])
                except _json.JSONDecodeError:
                    pass
            return {}

    async def generate_chat(
        self,
        system_message: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate a response using the active model's chat template.

        Atlas 3.0: For API models, uses LiteLLM's native chat completion API
        with proper system/user message formatting. For local models, formats
        with the appropriate chat template before calling generate().

        Args:
            system_message: System instruction (role, persona, rules).
            user_message: User query / content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop: Optional custom stop sequences.

        Returns:
            Generated text string
        """
        if self._llm is None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_llm)

        if self._model_type == "qwen":
            prompt = self._format_qwen_prompt(system_message, user_message)
        elif self._model_type == "phi3":
            prompt = self._format_phi3_prompt(system_message, user_message)
        else:
            prompt = self._format_llama3_prompt(system_message, user_message)

        return await self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # Lazy load model
        if self._embedder is None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_embedder)
        
        # Run embedding in executor to not block event loop
        loop = asyncio.get_running_loop()
        
        def _embed():
            with self._embed_lock:
                return self._embedder.encode(text).tolist()
        
        return await loop.run_in_executor(None, _embed)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Lazy load model
        if self._embedder is None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_embedder)
        
        # Run batch embedding in executor
        loop = asyncio.get_running_loop()
        
        def _embed_batch():
            with self._embed_lock:
                return self._embedder.encode(texts).tolist()
        
        return await loop.run_in_executor(None, _embed_batch)
    
    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim
    
    @property
    def active_model_name(self) -> Optional[str]:
        """Return the name of the currently loaded GGUF model, or None."""
        return self._active_model_name

    def get_status(self) -> dict:
        """Return LLM status for UI display.

        Atlas 3.0: Now includes model_source and api_models_available fields.
        """
        return {
            "active_model": self._active_model_name,
            "model_type": self._model_type,
            "device": self._device,
            "gpu_layers": self._gpu_layers,
            "fallback": self._llm == "FALLBACK" and self._model_source == "local",
            "model_source": self._model_source,
            "api_models_available": len(self.list_available_api_models()) > 0,
        }

    def list_available_models(self) -> List[str]:
        """List all .gguf files in the models directory."""
        if not self.models_dir.exists():
            return []
        return sorted(p.name for p in self.models_dir.glob("*.gguf"))

    async def load_model(self, model_name: str) -> str:
        """Load a model - either local GGUF or API model.

        Atlas 3.0: Detects API model names (containing '/') and routes to
        load_api_model(). Otherwise loads a local GGUF model.

        Args:
            model_name: Filename of GGUF model OR LiteLLM model ID (e.g., "deepseek/deepseek-chat")

        Returns:
            Name of the newly loaded model.
        """
        # Atlas 3.0: Detect API models by provider/model format
        if "/" in model_name:
            return await self.load_api_model(model_name)

        # Switch back to local mode if currently on API
        if self._model_source == "api":
            self._model_source = "local"
            self._api_model_name = None
            logger.info("Switching from API model back to local model")

        # Acquire lock with timeout to prevent permanent blocking
        try:
            await asyncio.wait_for(self._model_lock.acquire(), timeout=120)
            try:
                logger.info(f"Model switch lock acquired, switching to: {model_name}")
                return await self._do_load_model(model_name)
            finally:
                self._model_lock.release()
        except asyncio.TimeoutError:
            raise RuntimeError(
                "Model switch timed out - a query or operation is holding the model lock. "
                "Please wait for the current operation to complete before switching models."
            )
    
    async def _do_load_model(self, model_name: str) -> str:
        """Internal model loading with lock already held."""
        model_path = self.models_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found in {self.models_dir}")
        
        # If already loaded, skip
        if self._active_model_name == model_name and self._llm is not None and self._llm != "FALLBACK":
            logger.info(f"Model '{model_name}' is already loaded")
            return model_name
        
        # Unload current model to free memory
        if self._llm is not None and self._llm != "FALLBACK":
            logger.info(f"Unloading current model: {self._active_model_name}")
            del self._llm
            self._llm = None
            self._active_model_name = None
            # Encourage garbage collection to free VRAM
            import gc
            gc.collect()
            logger.info("Previous model unloaded and memory freed")
        
        # Load the new model
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_llm, model_name)
        
        if self._llm == "FALLBACK":
            detail = self._llm_last_error or "llama-cpp-python failed to initialize"
            raise RuntimeError(
                "Failed to load model. "
                "Ensure llama-cpp-python is installed and its llama.dll dependencies are available. "
                f"Last error: {detail}"
            )
        
        return self._active_model_name
    
    async def initialize_default_model(self) -> str:
        """Load the default GPU-accelerated model at startup.
        
        Returns:
            Name of the loaded model, or empty string if no model available.
        """
        if self._is_initializing:
            logger.info("Model initialization already in progress")
            return self._active_model_name or ""
        
        self._is_initializing = True
        try:
            available = self.list_available_models()
            if not available:
                logger.warning("No GGUF models found in models directory")
                self._device = "cpu"
                self._active_model_name = None
                return ""

            # Prefer a small ingestion model so VRAM is left for the Nemotron
            # orchestrator. Skip any *Orchestrator*.gguf — that file belongs to
            # AtlasOrchestratorService, not this one.
            priority = [
                "qwen2.5-1.5b-instruct",
                "qwen2.5-3b-instruct",
                "phi-3.5-mini-instruct",
                "llama-3-8b-instruct",
                "meta-llama-3-8b-instruct",
            ]
            lower_to_real = {n.lower(): n for n in available if "orchestrator" not in n.lower()}
            model_to_load: Optional[str] = None
            for pref in priority:
                for lower_name, real_name in lower_to_real.items():
                    if lower_name.startswith(pref):
                        model_to_load = real_name
                        break
                if model_to_load:
                    break
            if model_to_load is None:
                # Fall back to the first non-orchestrator model on disk
                model_to_load = next(iter(lower_to_real.values()), available[0])

            logger.info(f"Loading default model at startup: {model_to_load}")

            async with self._model_lock:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._load_llm, model_to_load)
            
            if self._llm == "FALLBACK":
                logger.error(f"Failed to load default model: {self._llm_last_error}")
                self._device = "cpu"
                return ""
            
            logger.info(f"Default model loaded successfully: {self._active_model_name} on {self._device}")
            return self._active_model_name or ""
        finally:
            self._is_initializing = False
    
    @classmethod
    def get_instance(cls) -> 'LLMService':
        """Get or create singleton instance.
        
        Note: Does not initialize the model. Call initialize_default_model() separately
        during application startup.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Convenience function for getting the service
def get_llm_service() -> LLMService:
    """Get the singleton LLM service instance."""
    return LLMService.get_instance()
