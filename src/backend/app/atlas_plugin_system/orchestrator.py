"""Native Nemotron-Orchestrator tool delegation for the Atlas Framework.

This module uses nvidia_Orchestrator-8B (Nemotron-Orchestrator), an 8B model
that was RL-trained via GRPO specifically for tool orchestration.  Unlike a
prompted ReAct loop, the model natively emits <tool_call> tags and decides
autonomously when to stop calling tools.  We render prompts in the model's
native Qwen3/ChatML format and parse its structured output directly.

Reference: "ToolOrchestra: Elevating Intelligence via Efficient Model and
Tool Orchestration" (arxiv:2511.21689) — NVIDIA / University of Hong Kong.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.atlas_plugin_system.catalog import ToolCatalog, get_tool_catalog
from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_GPU_LAYERS = 35

# ---------------------------------------------------------------------------
# Output parsing — the model emits <think>, <tool_call>, and free text
# ---------------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _resolve_gpu_layers() -> int:
    raw = os.environ.get("ATLAS_ORCHESTRATOR_GPU_LAYERS", "").strip()
    if not raw:
        raw = os.environ.get("ATLAS_GPU_LAYERS", "").strip()
    if not raw:
        return DEFAULT_GPU_LAYERS
    if raw.lower() == "auto":
        return -1
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_GPU_LAYERS


class AtlasOrchestratorService:
    """Tool-delegation orchestrator powered by Nemotron-Orchestrator-8B.

    The model was trained with GRPO to select tools, route arguments, and
    decide when enough information has been gathered — all as a learned
    policy, not prompt engineering.  This class provides:

    * ChatML prompt rendering matching the Qwen3 template the model was
      fine-tuned on.
    * OpenAI-compatible ``<tools>`` schema injection so the model sees
      Atlas core tools and plugins in its trained format.
    * ``<tool_call>`` / ``<tool_response>`` parsing for the multi-turn
      tool loop.
    * A safety-bound iteration limit (the model usually finishes earlier).
    """

    def __init__(self, catalog: Optional[ToolCatalog] = None):
        self.catalog = catalog or get_tool_catalog()
        self._llama: Any = None
        self._model_name: Optional[str] = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = threading.Lock()
        self._gpu_layers = _resolve_gpu_layers()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    async def ensure_model_loaded(self) -> None:
        if self._llama is not None:
            return
        async with self._load_lock:
            if self._llama is not None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._load_model_sync)

    def _resolve_model_path(self) -> Optional[Path]:
        model_path = Path(settings.MODELS_DIR) / settings.ATLAS_ORCHESTRATOR_MODEL
        if model_path.exists():
            return model_path

        matches = sorted(Path(settings.MODELS_DIR).glob("*Orchestrator*.gguf"))
        if matches:
            logger.warning(
                "Configured orchestrator model '%s' not found; using %s",
                settings.ATLAS_ORCHESTRATOR_MODEL,
                matches[0].name,
            )
            return matches[0]

        return None

    def _load_model_sync(self) -> None:
        """Load the local Nemotron GGUF. Raise an actionable error on failure.

        Doctrine: no silent fallback to a different model. Nemotron is trained
        on a specific <tool_call>{...}</tool_call> JSON format; a cloud chat
        model emits a different format and produces infinite supervisor revise
        loops that look like orchestration bugs. Fail loud, point the user at
        the fix.
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. The Atlas orchestrator requires "
                "it to load the Nemotron-Orchestrator GGUF locally. Install with:\n"
                "    pip install llama-cpp-python\n"
                "No API fallback is used — Nemotron's tool-call format is model-specific."
            ) from exc

        model_path = self._resolve_model_path()
        if model_path is None:
            raise FileNotFoundError(
                f"Nemotron GGUF not found. Expected "
                f"'{settings.ATLAS_ORCHESTRATOR_MODEL}' (or any '*Orchestrator*.gguf') "
                f"in MODELS_DIR={settings.MODELS_DIR}. "
                f"Download nvidia_Orchestrator-8B-IQ2_M.gguf from HuggingFace and drop "
                f"it in that directory. No API fallback — Nemotron's tool-call format "
                f"is model-specific and a swap produces broken orchestration."
            )

        try:
            from app.services.llm import _add_cuda_dll_directories
            _add_cuda_dll_directories()

            logger.info("Loading Atlas orchestrator from %s (gpu_layers=%s)", model_path, self._gpu_layers)
            self._llama = Llama(
                model_path=str(model_path),
                n_ctx=settings.ATLAS_ORCHESTRATOR_CONTEXT_SIZE,
                n_gpu_layers=self._gpu_layers,
                n_batch=settings.LLM_N_BATCH,
                use_mlock=settings.LLM_USE_MLOCK,
                check_tensors=False,
                cache=True,
                verbose=settings.LLM_VERBOSE,
                n_threads=4,
            )
            self._model_name = model_path.name
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Nemotron GGUF at {model_path}: {exc}. "
                f"Check that the file is not corrupted and that your GPU has "
                f"enough VRAM (or lower ATLAS_ORCHESTRATOR_GPU_LAYERS). "
                f"No API fallback — see doctrine."
            ) from exc

    # ------------------------------------------------------------------
    # Main orchestration entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        prompt: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
        conversation: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute tool-delegation orchestration.

        The model decides which tools to call and when to stop.  The
        iteration limit is a safety bound, not a prompting mechanism.
        """
        self.catalog.refresh()
        await self.ensure_model_loaded()

        # ----- Build message history in the model's native format -----
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._build_system_message()},
        ]
        for item in conversation or []:
            role = item.get("role", "user")
            content = item.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        trace: List[Dict[str, Any]] = []
        iterations = max_iterations or settings.ATLAS_ORCHESTRATOR_MAX_ITERATIONS

        for iteration in range(1, iterations + 1):
            raw_output = await self._generate(messages)

            thinking = self._extract_thinking(raw_output)
            tool_calls = self._extract_tool_calls(raw_output)
            final_text = self._extract_final_text(raw_output)

            trace_entry: Dict[str, Any] = {
                "iteration": iteration,
                "thinking": thinking,
                "tool_calls": [
                    {"name": name, "arguments": args}
                    for name, args in tool_calls
                ],
            }

            # ---- No tool calls → model decided it has enough info ----
            if not tool_calls:
                answer = final_text or thinking or "No response was generated."
                trace_entry["final_answer"] = answer
                trace.append(trace_entry)
                return self._build_result(answer, iteration, trace)

            # ---- Append assistant turn to history --------------------
            messages.append({"role": "assistant", "content": raw_output})

            # ---- Execute each tool call --------------------------
            tool_results: List[str] = []
            for tool_name, tool_args in tool_calls:
                try:
                    result = await self.catalog.invoke(
                        tool_name,
                        tool_args,
                        context={
                            "project_id": project_id,
                            "session_id": session_id,
                            "user_prompt": prompt,
                            "iteration": iteration,
                        },
                    )
                except Exception as exc:
                    logger.error("Tool '%s' raised: %s", tool_name, exc, exc_info=True)
                    result = {"error": str(exc), "tool": tool_name}

                tool_results.append(self._truncate_payload(result))
                trace_entry.setdefault("tool_results", []).append(result)

            trace.append(trace_entry)

            # ---- Feed results back as <tool_response> under user role
            tool_response_content = "\n".join(
                f"<tool_response>\n{r}\n</tool_response>"
                for r in tool_results
            )
            messages.append({"role": "user", "content": tool_response_content})

        # ---- Safety bound reached — force a synthesis ----------------
        answer = await self._force_final_answer(messages)
        trace.append(
            {
                "iteration": iterations,
                "thinking": "Maximum iteration limit reached.",
                "tool_calls": [],
                "final_answer": answer,
                "forced": True,
            }
        )
        return self._build_result(answer, iterations, trace)

    # ------------------------------------------------------------------
    # System message construction
    # ------------------------------------------------------------------
    def _build_system_message(self) -> str:
        """Build the system prompt with tools in Nemotron's trained format.

        The model was fine-tuned expecting OpenAI-compatible tool schemas
        inside ``<tools></tools>`` XML tags, with instructions to respond
        using ``<tool_call></tool_call>`` tags.
        """
        tools_block = self.catalog.build_openai_tools_block()

        return (
            "You are the Atlas Framework Orchestrator, running locally inside "
            "an offline-first research operating system. You have access to a "
            "knowledge substrate (literature search, vector database, knowledge "
            "graph) and domain-specific research plugins.\n\n"
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_block}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    # ------------------------------------------------------------------
    # Generation — local Nemotron only (no API fallback; see doctrine)
    # ------------------------------------------------------------------
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a completion from the local Nemotron GGUF."""
        prompt_text = self._render_chatml(messages)
        loop = asyncio.get_running_loop()

        def _do_generate() -> str:
            with self._inference_lock:
                response = self._llama(
                    prompt_text,
                    max_tokens=settings.ATLAS_ORCHESTRATOR_MAX_TOKENS,
                    temperature=settings.ATLAS_ORCHESTRATOR_TEMPERATURE,
                    stop=["<|im_end|>", "<|endoftext|>"],
                    echo=False,
                )
                return (response["choices"][0]["text"] or "").strip()

        return await loop.run_in_executor(None, _do_generate)

    async def _force_final_answer(self, messages: List[Dict[str, str]]) -> str:
        """When the safety bound is reached, ask the model to synthesize."""
        extended = messages + [
            {
                "role": "user",
                "content": (
                    "You have reached the tool call limit. Based on all the "
                    "tool results above, provide your final comprehensive "
                    "answer now. Do not call any more tools."
                ),
            }
        ]
        raw = await self._generate(extended)
        text = self._extract_final_text(raw)
        return text or raw or "I reached the tool-call limit before producing a final answer."

    # ------------------------------------------------------------------
    # ChatML rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _render_chatml(messages: List[Dict[str, str]]) -> str:
        """Render a message list into Qwen3/ChatML format.

        Format::

            <|im_start|>system
            {content}<|im_end|>
            <|im_start|>user
            {content}<|im_end|>
            <|im_start|>assistant
            ← model completes from here
        """
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        # Open the assistant turn for the model to complete
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_thinking(text: str) -> str:
        """Extract content from ``<think>`` blocks."""
        matches = _THINK_RE.findall(text)
        return "\n".join(m.strip() for m in matches if m.strip())

    @staticmethod
    def _extract_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract ``(name, arguments)`` pairs from ``<tool_call>`` blocks."""
        calls: List[Tuple[str, Dict[str, Any]]] = []
        for match in _TOOL_CALL_RE.finditer(text):
            try:
                parsed = json.loads(match.group(1))
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                if name:
                    calls.append((name, arguments if isinstance(arguments, dict) else {}))
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse <tool_call> JSON: %s",
                    match.group(1)[:200],
                )
        return calls

    @staticmethod
    def _extract_final_text(text: str) -> str:
        """Return response text after stripping ``<think>`` and ``<tool_call>`` blocks."""
        cleaned = _THINK_RE.sub("", text)
        cleaned = _TOOL_CALL_RE.sub("", cleaned)
        return cleaned.strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_result(
        self,
        answer: str,
        iterations: int,
        trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "iterations": iterations,
            "model": self._model_name,
            "available_tools": self.catalog.tool_names(),
            "trace": trace,
        }

    @staticmethod
    def _truncate_payload(payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=True)
        if len(raw) <= settings.ATLAS_ORCHESTRATOR_RESPONSE_MAX_CHARS:
            return raw
        return raw[: settings.ATLAS_ORCHESTRATOR_RESPONSE_MAX_CHARS] + "...(truncated)"


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_atlas_orchestrator: Optional[AtlasOrchestratorService] = None


def get_atlas_orchestrator() -> AtlasOrchestratorService:
    """Return the Atlas Framework local orchestrator singleton."""
    global _atlas_orchestrator
    if _atlas_orchestrator is None:
        _atlas_orchestrator = AtlasOrchestratorService()
    return _atlas_orchestrator
