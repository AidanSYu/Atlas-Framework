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
import queue as _stdlib_queue
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from app.atlas_plugin_system.catalog import ToolCatalog, get_tool_catalog
from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_GPU_LAYERS = 35

# ---------------------------------------------------------------------------
# Output parsing — the model emits <think>, <tool_call>, and free text
# ---------------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.+?)\s*</tool_call>",
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


def _resolve_n_threads() -> int:
    configured = settings.ATLAS_ORCHESTRATOR_N_THREADS
    if configured and configured > 0:
        return configured
    cores = os.cpu_count() or 4
    # Heuristic: use half the logical CPUs to avoid hyperthread contention.
    return max(2, cores // 2)


def _log_vram_snapshot(stage: str) -> None:
    """Log current VRAM allocation if torch+CUDA are available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        for idx in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(idx)
            used_mb = (total - free) // (1024 * 1024)
            total_mb = total // (1024 * 1024)
            logger.info(
                "VRAM[%s] device %d: %d/%d MiB used (%.1f%%)",
                stage,
                idx,
                used_mb,
                total_mb,
                100.0 * used_mb / max(total_mb, 1),
            )
    except Exception as exc:  # noqa: BLE001 — diagnostics only
        logger.debug("VRAM logging skipped: %s", exc)


def _gpu_vram_used_mb() -> Optional[int]:
    """Return device-0 VRAM used in MiB, or None if CUDA isn't available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(0)
        return int((total - free) // (1024 * 1024))
    except Exception:
        return None


def _warn_if_gpu_unused(requested_layers: int, pre_mb: Optional[int], model_name: str) -> None:
    """Surface the case where n_gpu_layers > 0 but VRAM didn't grow.

    Catches two distinct silent failures:
      1. llama-cpp-python built without CUDA support (wrong wheel installed,
         or running from the wrong Python interpreter — common when the
         project venv isn't activated and the global Python's CPU-only
         wheel takes over).
      2. CUDA build OK but the runtime can't fit the model (OOM) and
         silently runs on CPU instead.
    Doctrine: never silent fallbacks. Log loud, actionable errors.
    """
    if requested_layers == 0:
        return

    # (1) Is the installed llama-cpp build even capable of GPU offload?
    try:
        from llama_cpp.llama_cpp import llama_supports_gpu_offload
        if not llama_supports_gpu_offload():
            import sys as _sys
            logger.error(
                "GPU OFFLOAD UNAVAILABLE: n_gpu_layers=%d was requested but the "
                "installed llama-cpp-python (at %s) was built WITHOUT CUDA "
                "support, so %s is running on CPU. Most likely you're using "
                "the wrong Python interpreter (e.g. global Python instead of "
                "the project venv at .venv\\Scripts\\python.exe). Activate the "
                "venv ('.venv\\Scripts\\activate') and re-run, or install a "
                "CUDA-enabled llama-cpp-python wheel in the active interpreter.",
                requested_layers, _sys.executable, model_name,
            )
            return
    except Exception:
        pass

    # (2) Build supports GPU; check whether VRAM actually grew on load.
    if pre_mb is None:
        return
    post_mb = _gpu_vram_used_mb()
    if post_mb is None:
        return
    delta = post_mb - pre_mb
    if delta < 50:
        logger.error(
            "GPU OFFLOAD FAILED: requested n_gpu_layers=%d for %s but VRAM only "
            "grew %+d MiB (pre=%d, post=%d). The model is running on CPU. "
            "Likely cause: not enough free VRAM for n_ctx=%d. Try a smaller "
            "ATLAS_ORCHESTRATOR_CONTEXT_SIZE, lower ATLAS_ORCHESTRATOR_GPU_LAYERS, "
            "or free VRAM held by other processes (embedder, browser, IDE).",
            requested_layers, model_name, delta, pre_mb, post_mb,
            settings.ATLAS_ORCHESTRATOR_CONTEXT_SIZE,
        )


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
            from app.services import model_slot
            _add_cuda_dll_directories()

            # Claim the single GPU slot — evicts the ingestion LLM if resident.
            model_slot.acquire("orchestrator", self.unload)

            n_threads = _resolve_n_threads()
            _log_vram_snapshot("pre-load")
            logger.info(
                "Loading Atlas orchestrator from %s (gpu_layers=%s, n_threads=%d, n_batch=%d, n_ctx=%d)",
                model_path,
                self._gpu_layers,
                n_threads,
                settings.LLM_N_BATCH,
                settings.ATLAS_ORCHESTRATOR_CONTEXT_SIZE,
            )
            load_started = time.monotonic()
            pre_load_vram_mb = _gpu_vram_used_mb()
            self._llama = Llama(
                model_path=str(model_path),
                n_ctx=settings.ATLAS_ORCHESTRATOR_CONTEXT_SIZE,
                n_gpu_layers=self._gpu_layers,
                n_batch=settings.LLM_N_BATCH,
                use_mlock=settings.LLM_USE_MLOCK,
                check_tensors=False,
                cache=True,
                flash_attn=True,
                verbose=settings.LLM_VERBOSE,
                n_threads=n_threads,
            )
            self._model_name = model_path.name
            logger.info(
                "Orchestrator loaded in %.1fs",
                time.monotonic() - load_started,
            )
            _log_vram_snapshot("post-load")
            _warn_if_gpu_unused(self._gpu_layers, pre_load_vram_mb, model_path.name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Nemotron GGUF at {model_path}: {exc}. "
                f"Check that the file is not corrupted and that your GPU has "
                f"enough VRAM (or lower ATLAS_ORCHESTRATOR_GPU_LAYERS). "
                f"No API fallback — see doctrine."
            ) from exc

    def unload(self) -> None:
        """Drop the Nemotron GGUF so the GPU slot is free.

        Holds the inference lock so the C-level model pointer cannot be freed
        while another thread is mid-generate. Safe to call when no model is
        loaded.
        """
        with self._inference_lock:
            if self._llama is None:
                return
            logger.info("Unloading orchestrator model: %s", self._model_name)
            try:
                if hasattr(self._llama, "close"):
                    self._llama.close()
                del self._llama
            except Exception as e:
                logger.warning("Error closing orchestrator model: %s", e)
            self._llama = None
            self._model_name = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Main orchestration entry point — streaming generator
    # ------------------------------------------------------------------
    async def run_stream(
        self,
        prompt: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
        conversation: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run the orchestration loop, yielding SSE-shaped events live.

        Each yielded value is ``{"event": <name>, "data": {...}}``. Event
        names match what the frontend stream-adapter already understands:
        ``thinking``, ``tool_call``, ``tool_result``, ``chunk``, ``error``,
        ``complete``. We also emit ``run_start`` and ``iteration_start`` so
        the UI can show structural progress before any tokens arrive.

        Persists per-iteration (rendered prompt, raw completion, parsed
        outputs, tool results, latency) to ``data/training_traces/{run_id}.jsonl``
        so the same loop produces the training corpus for a future custom
        orchestrator model.
        """
        run_id = str(uuid.uuid4())
        run_started_at = time.monotonic()
        self.catalog.refresh_if_stale()
        try:
            await self.ensure_model_loaded()
        except Exception as exc:
            yield {
                "event": "error",
                "data": {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "where": "model_load",
                },
            }
            return

        trace_path = self._open_training_trace(run_id, prompt, project_id, session_id)

        yield {
            "event": "run_start",
            "data": {
                "run_id": run_id,
                "model": self._model_name,
                "available_tools": self.catalog.tool_names(),
                "trace_path": str(trace_path) if trace_path else None,
            },
        }

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

        iterations = max_iterations or settings.ATLAS_ORCHESTRATOR_MAX_ITERATIONS
        loop = asyncio.get_running_loop()

        for iteration in range(1, iterations + 1):
            yield {"event": "iteration_start", "data": {"iteration": iteration}}

            chatml_prompt = self._render_chatml(messages)
            full_text = ""
            gen_started_at = time.monotonic()
            try:
                async for delta in self._generate_streaming(chatml_prompt):
                    full_text += delta
                    yield {"event": "chunk", "data": {"content": delta, "iteration": iteration}}
            except Exception as exc:
                yield {
                    "event": "error",
                    "data": {
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                        "where": f"model_generate:iteration_{iteration}",
                        "iteration": iteration,
                    },
                }
                return

            gen_ms = int((time.monotonic() - gen_started_at) * 1000)

            thinking = self._extract_thinking(full_text)
            tool_calls = self._extract_tool_calls(full_text)
            final_text = self._extract_final_text(full_text)

            # If the model was cut off mid-<tool_call>, give it one shot to
            # finish that JSON before we accept "no tool calls" as the verdict.
            # Without this, an 8-tool chain can dead-end at iteration 4 just
            # because max_tokens hit between { and }.
            if not tool_calls and self._was_truncated_mid_tool_call(full_text):
                logger.info(
                    "Iteration %d truncated mid <tool_call>; re-prompting to complete",
                    iteration,
                )
                try:
                    continuation = await self._continue_truncated_response(messages, full_text)
                    full_text = full_text + continuation
                    tool_calls = self._extract_tool_calls(full_text)
                    final_text = self._extract_final_text(full_text)
                except Exception as exc:
                    logger.warning("Truncation continuation failed: %s", exc)

            if thinking:
                yield {"event": "thinking", "data": {"content": thinking, "iteration": iteration}}

            iter_record: Dict[str, Any] = {
                "iteration": iteration,
                "chatml_prompt": chatml_prompt,
                "raw_completion": full_text,
                "parsed": {
                    "thinking": thinking,
                    "tool_calls": [{"name": n, "arguments": a} for n, a in tool_calls],
                    "final_text": final_text,
                },
                "generation_latency_ms": gen_ms,
                "ts": time.time(),
            }

            # ---- No tool calls → model decided it has enough info ----
            if not tool_calls:
                # IQ2 frequently produces only a <think> block with no final
                # answer text. Never leak raw <think> content to the user —
                # force a clean synthesis pass instead.
                if not final_text:
                    logger.info(
                        "Iteration %d emitted only <think> with no answer text; "
                        "forcing a final-answer pass",
                        iteration,
                    )
                    try:
                        answer = await self._force_final_answer(messages)
                    except Exception as exc:
                        logger.warning("force_final_answer failed: %s", exc)
                        answer = "The model produced internal reasoning but no final answer. Try rephrasing more imperatively."
                else:
                    answer = final_text
                iter_record["final_answer"] = answer
                pending_followup = self._extract_pending_followup(answer)
                if not pending_followup:
                    logger.warning(
                        "Iteration %d final answer lacks a 'Next step:' line "
                        "(system-prompt rule violated). Continuation chip will be absent.",
                        iteration,
                    )
                self._append_training_trace(trace_path, iter_record)
                yield {
                    "event": "complete",
                    "data": {
                        "answer": answer,
                        "iterations": iteration,
                        "model": self._model_name,
                        "run_id": run_id,
                        "trace_path": str(trace_path) if trace_path else None,
                        "pending_followup": pending_followup,
                        "duration_ms": int((time.monotonic() - run_started_at) * 1000),
                    },
                }
                return

            # ---- Append assistant turn to history --------------------
            messages.append({"role": "assistant", "content": full_text})

            # ---- Execute each tool call -----------------------------
            executed_results: List[Dict[str, Any]] = []
            tool_results_text: List[str] = []
            for idx, (tool_name, tool_args) in enumerate(tool_calls):
                tool_id = f"{iteration}.{idx}"
                yield {
                    "event": "tool_call",
                    "data": {
                        "tool": tool_name,
                        "input": tool_args,
                        "iteration": iteration,
                        "id": tool_id,
                    },
                }
                t_start = time.monotonic()
                tool_error: Optional[str] = None
                tool_traceback: Optional[str] = None
                if self.catalog.is_exclusive_gpu(tool_name):
                    logger.info(
                        "Tool '%s' requests exclusive GPU — unloading orchestrator",
                        tool_name,
                    )
                    await loop.run_in_executor(None, self.unload)
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
                    tool_error = str(exc)
                    tool_traceback = traceback.format_exc()
                    logger.error("Tool '%s' raised: %s", tool_name, exc, exc_info=True)
                    result = {"error": tool_error, "tool": tool_name}

                duration_ms = int((time.monotonic() - t_start) * 1000)
                executed_results.append(result)
                tool_results_text.append(self._truncate_payload(result))

                if tool_error:
                    yield {
                        "event": "error",
                        "data": {
                            "message": tool_error,
                            "traceback": tool_traceback,
                            "where": f"tool_invocation:{tool_name}",
                            "iteration": iteration,
                            "non_fatal": True,
                        },
                    }
                yield {
                    "event": "tool_result",
                    "data": {
                        "tool": tool_name,
                        "output": result,
                        "iteration": iteration,
                        "id": tool_id,
                        "duration_ms": duration_ms,
                    },
                }

            iter_record["tool_results"] = executed_results
            self._append_training_trace(trace_path, iter_record)

            # ---- Feed results back as <tool_response> under user role
            tool_response_content = "\n".join(
                f"<tool_response>\n{r}\n</tool_response>"
                for r in tool_results_text
            )
            messages.append({"role": "user", "content": tool_response_content})

        # ---- Safety bound reached — force a synthesis ----------------
        try:
            answer = await self._force_final_answer(messages)
        except Exception as exc:
            yield {
                "event": "error",
                "data": {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "where": "force_final_answer",
                },
            }
            return
        self._append_training_trace(
            trace_path,
            {
                "iteration": iterations,
                "forced_final": True,
                "final_answer": answer,
                "ts": time.time(),
            },
        )
        yield {
            "event": "complete",
            "data": {
                "answer": answer,
                "iterations": iterations,
                "model": self._model_name,
                "run_id": run_id,
                "trace_path": str(trace_path) if trace_path else None,
                "pending_followup": self._extract_pending_followup(answer),
                "duration_ms": int((time.monotonic() - run_started_at) * 1000),
                "forced": True,
            },
        }

    async def run(
        self,
        prompt: str,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None,
        max_iterations: Optional[int] = None,
        conversation: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Non-streaming wrapper that consumes ``run_stream`` and accumulates.

        Preserves the original return shape so existing callers (curl
        tests, scripts) keep working while the frontend moves to the
        streaming endpoint.
        """
        trace: List[Dict[str, Any]] = []
        current_iter: Optional[Dict[str, Any]] = None
        answer = ""
        final_iters = 0
        last_error: Optional[Dict[str, Any]] = None

        async for event in self.run_stream(
            prompt=prompt,
            project_id=project_id,
            session_id=session_id,
            max_iterations=max_iterations,
            conversation=conversation,
        ):
            name = event["event"]
            data = event["data"]
            if name == "iteration_start":
                current_iter = {"iteration": data["iteration"], "thinking": "", "tool_calls": [], "tool_results": []}
                trace.append(current_iter)
            elif name == "thinking" and current_iter is not None:
                current_iter["thinking"] = data.get("content", "")
            elif name == "tool_call" and current_iter is not None:
                current_iter["tool_calls"].append({"name": data["tool"], "arguments": data.get("input", {})})
            elif name == "tool_result" and current_iter is not None:
                current_iter["tool_results"].append(data.get("output", {}))
            elif name == "complete":
                answer = data.get("answer", "")
                final_iters = data.get("iterations", 0)
                if current_iter is not None:
                    current_iter["final_answer"] = answer
            elif name == "error" and not data.get("non_fatal"):
                last_error = data

        if last_error and not answer:
            raise RuntimeError(
                f"{last_error.get('where', 'orchestrator')}: {last_error.get('message', 'unknown error')}"
            )
        return self._build_result(answer, final_iters or len(trace), trace)

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
            "an offline-first operating system. You have access to a knowledge "
            "substrate (hybrid retrieval, vector database, knowledge graph) and "
            "whatever tools have been registered with the framework. Atlas is "
            "domain-agnostic — the catalog below tells you what is available; "
            "do not assume any particular field of work.\n\n"
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
            "</tool_call>\n\n"
            "# Style rules\n\n"
            "1. Before EVERY <tool_call>, write ONE short natural sentence saying "
            "what you are about to do. Do NOT skip this sentence.\n"
            "2. EXECUTE — do not deliberate. If the user gives you a concrete "
            "task and the catalog has tools that fit, run the natural chain "
            "implied by the tools without asking permission. If the user is "
            "just chatting or greeting, answer conversationally without calling "
            "any tools.\n"
            "3. Keep <think> blocks under 3 short sentences. Long thinking wastes "
            "tokens you need for actual tool calls.\n"
            "4. NEVER declare the work \"done\" or \"complete.\" Every substantive "
            "answer must end with a `Next step:` line — either a concrete piece "
            "of data the user should bring back, or a tool you propose to call "
            "next when they reply. Skip the Next step line only for short "
            "conversational replies (greetings, clarifications)."
        )

    # ------------------------------------------------------------------
    # Generation — local Nemotron only (no API fallback; see doctrine)
    # ------------------------------------------------------------------
    async def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate a completion from the local Nemotron GGUF.

        Idempotent ``ensure_model_loaded`` call so the model reloads if a
        prior exclusive-GPU tool evicted it.
        """
        await self.ensure_model_loaded()
        prompt_text = self._render_chatml(messages)
        chunks: List[str] = []
        async for delta in self._generate_streaming(prompt_text):
            chunks.append(delta)
        return "".join(chunks).strip()

    async def _generate_streaming(self, prompt_text: str) -> AsyncIterator[str]:
        """Yield text deltas from the local Nemotron GGUF as they are produced.

        Uses llama-cpp-python's native streaming. The generation runs on
        the inference executor (holding ``_inference_lock`` for the full
        generation so the C-level model pointer is safe), and deltas are
        handed off to the async caller through a thread-safe queue.
        """
        await self.ensure_model_loaded()
        loop = asyncio.get_running_loop()
        q: "_stdlib_queue.Queue[Any]" = _stdlib_queue.Queue(maxsize=256)
        SENTINEL: Any = object()

        def _producer() -> None:
            try:
                with self._inference_lock:
                    stream = self._llama(
                        prompt_text,
                        max_tokens=settings.ATLAS_ORCHESTRATOR_MAX_TOKENS,
                        temperature=settings.ATLAS_ORCHESTRATOR_TEMPERATURE,
                        stop=["<|im_end|>", "<|endoftext|>"],
                        echo=False,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk["choices"][0].get("text", "")
                        if delta:
                            q.put(delta)
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(SENTINEL)

        producer_future = loop.run_in_executor(None, _producer)
        try:
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            try:
                await producer_future
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Training trace persistence
    # ------------------------------------------------------------------
    def _open_training_trace(
        self,
        run_id: str,
        prompt: str,
        project_id: Optional[str],
        session_id: Optional[str],
    ) -> Optional[Path]:
        """Open a per-run JSONL training-trace file and write the header line."""
        try:
            traces_dir = Path(settings.DATA_DIR) / "training_traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
            path = traces_dir / f"{run_id}.jsonl"
            header = {
                "kind": "run_header",
                "run_id": run_id,
                "ts": time.time(),
                "model": self._model_name,
                "available_tools": self.catalog.tool_names(),
                "project_id": project_id,
                "session_id": session_id,
                "prompt": prompt,
            }
            with path.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(header, ensure_ascii=False) + "\n")
            return path
        except Exception as exc:
            logger.warning("Could not open training trace for %s: %s", run_id, exc)
            return None

    @staticmethod
    def _append_training_trace(path: Optional[Path], record: Dict[str, Any]) -> None:
        if path is None:
            return
        try:
            record_with_kind = {"kind": "iteration", **record}
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record_with_kind, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Could not append training trace at %s: %s", path, exc)

    async def _force_final_answer(self, messages: List[Dict[str, str]]) -> str:
        """Synthesize a final answer when normal iteration didn't produce one.

        On long chains the full message history pushes against n_ctx, so the
        model runs out of room mid-summary. We condense: keep the system
        prompt, the original user prompt, and a compact bullet list of every
        tool call + result, then ask for a fresh summary. This frees enough
        budget for the model to actually write a clean answer.
        """
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        user_prompt = next((m for m in messages if m.get("role") == "user"), None)

        # Walk the message history and pull tool calls + responses into a brief
        # bullet list (avoids dumping full JSON back into the model's view).
        bullets: List[str] = []
        for m in messages:
            content = m.get("content", "") or ""
            for match in _TOOL_CALL_RE.finditer(content):
                try:
                    parsed = json.loads(match.group(1))
                    bullets.append(f"- called {parsed.get('name','?')}({json.dumps(parsed.get('arguments',{}))[:120]})")
                except Exception:
                    pass
            for resp_match in re.finditer(r"<tool_response>\s*(.+?)\s*</tool_response>", content, re.DOTALL):
                snippet = resp_match.group(1).strip()[:200]
                bullets.append(f"  → {snippet}")

        compact_history = "\n".join(bullets) if bullets else "(no tool calls were recorded)"
        condensed: List[Dict[str, str]] = []
        if system_msg:
            condensed.append(system_msg)
        if user_prompt:
            condensed.append(user_prompt)
        condensed.append({
            "role": "user",
            "content": (
                "TOOL TRACE FROM THIS SESSION:\n"
                f"{compact_history}\n\n"
                "Write your final integrated answer NOW based on the trace above. "
                "Do NOT call any more tools. Do NOT emit <think> — go straight to "
                "the answer. Remember to end with a `Next step:` line."
            ),
        })
        raw = await self._generate(condensed)
        text = self._extract_final_text(raw)
        # Never leak raw <think> content — if the strip produced nothing,
        # return the friendly fallback rather than the model's internal scratch.
        if not text:
            return (
                "Reached the context budget before the model could synthesize a "
                "final answer. The tool chain completed — see the trace for "
                "results. Next step: try a shorter prompt or raise n_ctx if "
                "VRAM allows."
            )
        return text

    async def _continue_truncated_response(
        self,
        messages: List[Dict[str, str]],
        partial: str,
    ) -> str:
        """Resume generation when the previous response was cut off mid-emit.

        We append the partial assistant turn so far and prompt the model to
        finish, then return only the continuation (so the caller can concat).
        """
        resumed = messages + [
            {"role": "assistant", "content": partial},
            {
                "role": "user",
                "content": (
                    "Your previous turn was cut off mid-<tool_call>. Resume "
                    "from exactly where you stopped — emit the rest of the "
                    "JSON and close the </tool_call> tag, then continue."
                ),
            },
        ]
        return await self._generate(resumed)

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
    def _extract_pending_followup(answer: str) -> Optional[str]:
        """Pull the ``Next step:`` prose from the final answer.

        Returns the continuation hook the frontend can render as an actionable
        chip. None when the model failed to include one (which the system
        prompt forbids — log a warning so we can monitor compliance).
        """
        if not answer:
            return None
        # Find "Next step:" line, case-insensitive, take everything after it
        # up to the next double newline or end of answer
        match = re.search(
            r"(?im)^\s*next\s*step\s*:\s*(.+?)(?:\n\s*\n|\Z)",
            answer,
            re.DOTALL,
        )
        if match:
            followup = match.group(1).strip()
            return followup or None
        return None

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
        """Return response text after stripping ``<think>`` and ``<tool_call>`` blocks.

        Also drops orphan ``<tool_call>`` openings without a closing tag —
        these appear when the model hits max_tokens mid-emit and would
        otherwise leak raw JSON into the user-visible answer.
        """
        cleaned = _THINK_RE.sub("", text)
        cleaned = _TOOL_CALL_RE.sub("", cleaned)
        # Drop any orphan <tool_call> opening (truncation defense)
        if "<tool_call>" in cleaned:
            cleaned = cleaned.split("<tool_call>", 1)[0]
        # Drop any orphan <think> opening as well
        if "<think>" in cleaned:
            cleaned = cleaned.split("<think>", 1)[0]
        return cleaned.strip()

    @staticmethod
    def _was_truncated_mid_tool_call(text: str) -> bool:
        """True if the model emitted an opening <tool_call> but no closing tag.

        Indicates max_tokens cut off the response mid-JSON. We need to
        re-prompt the model to finish, not silently drop the intended call.
        """
        open_count = text.count("<tool_call>")
        close_count = text.count("</tool_call>")
        return open_count > close_count

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
            "pending_followup": self._extract_pending_followup(answer),
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
