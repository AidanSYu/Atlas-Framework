"""Nemotron-backed tool execution loop with event emission + circuit breakers.

This wraps the existing ``AtlasOrchestratorService`` (which owns the model
loading + ChatML rendering + tool-call parsing) and runs its own loop so
we can emit per-turn events into the task log and enforce circuit breakers
(loop limit, consecutive-same-error threshold, fatal crashes).

Two synthetic tools are injected alongside the scoped manifest so Nemotron
can cleanly escape the loop on its own:

- ``yield_to_supervisor(reason, suggested_options?)`` — the explicit escape
  hatch. Emits TOOL_YIELD, transitions to REVIEWING.
- ``submit_final_answer(answer)`` — Nemotron signals it's done. Emits
  FINAL_ANSWER candidate, transitions to REVIEWING (supervisor reviews).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from app.atlas_plugin_system import get_atlas_orchestrator, get_tool_catalog
from app.atlas_plugin_system.orchestrator import AtlasOrchestratorService
from app.core.config import settings
from app.core.task_events import (
    Actor,
    EventType,
    ToolStatus,
)
from app.core.task_log import TaskLog, get_task_log

logger = logging.getLogger(__name__)


YIELD_TOOL_NAME = "yield_to_supervisor"
FINAL_ANSWER_TOOL_NAME = "submit_final_answer"

_DEFAULT_LOOP_LIMIT = 12  # matches settings.ATLAS_ORCHESTRATOR_MAX_ITERATIONS
_SAME_ERROR_THRESHOLD = 3
_MAX_TOOL_OUTPUT_CHARS = 4000  # before truncation in the log


class ExecutorExit(str):
    """Reasons the executor returns control to the task service."""
    FINAL_ANSWER = "final_answer"
    YIELD = "yield"
    CIRCUIT_BREAKER_LOOP = "circuit_breaker_loop"
    CIRCUIT_BREAKER_ERRORS = "circuit_breaker_errors"
    FATAL = "fatal"
    REQUIRES_HUMAN = "requires_human"


@dataclass
class ExecutorResult:
    exit_reason: str
    final_answer: Optional[str] = None
    yield_reason: Optional[str] = None
    yield_suggestions: Optional[List[str]] = None
    requires_human_question: Optional[str] = None
    fatal_error: Optional[str] = None
    turns: int = 0


@dataclass
class ExecutorBrief:
    """Input to the executor — a distilled view of the goal brief."""
    task_id: str
    goal_statement: str
    definition_of_done: str
    active_manifest: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)  # absolute file paths


class TaskExecutor:
    """Runs Nemotron's tool-calling loop for a single brief."""

    def __init__(
        self,
        orchestrator: Optional[AtlasOrchestratorService] = None,
        log: Optional[TaskLog] = None,
    ):
        self._orchestrator = orchestrator or get_atlas_orchestrator()
        self._catalog = get_tool_catalog()
        self._log = log or get_task_log()

    async def run(self, brief: ExecutorBrief, cancel_event: Optional[asyncio.Event] = None) -> ExecutorResult:
        """Execute Nemotron's tool loop under the given brief until it exits.

        The FSM transitions (EXECUTING → REVIEWING, etc.) are the task
        service's job. This method just runs the loop and reports the exit.
        """
        self._catalog.refresh()
        try:
            await self._orchestrator.ensure_model_loaded()
        except Exception as exc:
            self._emit_circuit_breaker(brief.task_id, "fatal_wrapper_crash", {"error": str(exc)})
            return ExecutorResult(exit_reason=ExecutorExit.FATAL, fatal_error=str(exc))

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._build_system_message(brief)},
            {"role": "user", "content": self._build_user_message(brief)},
        ]

        loop_limit = _DEFAULT_LOOP_LIMIT
        error_tracker: Dict[str, int] = {}  # (tool|args_hash) → consecutive error count

        for turn in range(1, loop_limit + 1):
            if cancel_event is not None and cancel_event.is_set():
                return ExecutorResult(exit_reason=ExecutorExit.FATAL, fatal_error="cancelled", turns=turn - 1)

            try:
                raw = await self._orchestrator._generate(messages)  # uses loaded model
            except Exception as exc:
                self._emit_circuit_breaker(brief.task_id, "fatal_wrapper_crash", {"error": str(exc), "phase": "generate"})
                return ExecutorResult(exit_reason=ExecutorExit.FATAL, fatal_error=str(exc), turns=turn - 1)

            tool_calls = AtlasOrchestratorService._extract_tool_calls(raw)
            # If no tool calls, treat the entire raw text as final answer.
            if not tool_calls:
                answer = AtlasOrchestratorService._extract_final_text(raw) or raw
                return ExecutorResult(exit_reason=ExecutorExit.FINAL_ANSWER, final_answer=answer, turns=turn)

            messages.append({"role": "assistant", "content": raw})

            tool_response_parts: List[str] = []

            for tool_name, tool_args in tool_calls:
                # ---- Synthetic: submit_final_answer ----
                if tool_name == FINAL_ANSWER_TOOL_NAME:
                    answer = str(tool_args.get("answer") or "").strip()
                    if not answer:
                        answer = "(executor submitted final answer with no body)"
                    return ExecutorResult(exit_reason=ExecutorExit.FINAL_ANSWER, final_answer=answer, turns=turn)

                # ---- Synthetic: yield_to_supervisor ----
                if tool_name == YIELD_TOOL_NAME:
                    reason = str(tool_args.get("reason") or "no reason given")
                    suggestions = tool_args.get("suggested_options") or []
                    if not isinstance(suggestions, list):
                        suggestions = [str(suggestions)]
                    self._log.append(
                        brief.task_id,
                        Actor.NEMOTRON,
                        EventType.TOOL_YIELD,
                        {"reason": reason, "suggested_options": [str(s) for s in suggestions]},
                    )
                    return ExecutorResult(
                        exit_reason=ExecutorExit.YIELD,
                        yield_reason=reason,
                        yield_suggestions=[str(s) for s in suggestions],
                        turns=turn,
                    )

                # ---- Real plugin / core tool ----
                call_id = str(uuid.uuid4())
                self._log.append(
                    brief.task_id,
                    Actor.NEMOTRON,
                    EventType.TOOL_CALL_INTENT,
                    {
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "arguments": tool_args if isinstance(tool_args, dict) else {"raw": tool_args},
                    },
                )

                # Scope check — refuse tools not in the active manifest.
                if tool_name not in brief.active_manifest:
                    error_msg = (
                        f"Tool '{tool_name}' is not in the active manifest for this task. "
                        f"Available tools: {', '.join(brief.active_manifest)}. "
                        f"If you need a different toolkit, call {YIELD_TOOL_NAME} "
                        f"with reason='toolkit_insufficient'."
                    )
                    self._log.append(
                        brief.task_id,
                        Actor.TOOL_WRAPPER,
                        EventType.TOOL_EXECUTION_RESULT,
                        {
                            "call_id": call_id,
                            "status": ToolStatus.ERROR_PERMANENT.value,
                            "output": {"summary": error_msg, "truncated": False},
                            "execution_time_ms": 0,
                            "error_detail": "tool_not_in_active_manifest",
                        },
                    )
                    tool_response_parts.append(f"error: {error_msg}")
                    continue

                t_start = time.perf_counter()
                try:
                    result = await self._catalog.invoke(
                        tool_name,
                        tool_args if isinstance(tool_args, dict) else {},
                        context={"task_id": brief.task_id},
                    )
                    status, summary, error_detail = _classify_tool_result(result)
                except Exception as exc:
                    status = ToolStatus.ERROR_PERMANENT
                    summary = f"Tool '{tool_name}' raised: {exc}"
                    error_detail = type(exc).__name__
                    result = {"error": str(exc)}
                elapsed_ms = int((time.perf_counter() - t_start) * 1000)

                summary_text, truncated = _truncate(summary, _MAX_TOOL_OUTPUT_CHARS)
                self._log.append(
                    brief.task_id,
                    Actor.TOOL_WRAPPER,
                    EventType.TOOL_EXECUTION_RESULT,
                    {
                        "call_id": call_id,
                        "status": status.value,
                        "output": {"summary": summary_text, "truncated": truncated},
                        "execution_time_ms": elapsed_ms,
                        "error_detail": error_detail,
                    },
                )

                # ---- requires_human short-circuits immediately ----
                if status == ToolStatus.REQUIRES_HUMAN:
                    return ExecutorResult(
                        exit_reason=ExecutorExit.REQUIRES_HUMAN,
                        requires_human_question=summary_text,
                        turns=turn,
                    )

                # ---- circuit breaker: 3 identical permanent errors ----
                key = _error_key(tool_name, tool_args)
                if status == ToolStatus.ERROR_PERMANENT:
                    error_tracker[key] = error_tracker.get(key, 0) + 1
                    if error_tracker[key] >= _SAME_ERROR_THRESHOLD:
                        self._emit_circuit_breaker(
                            brief.task_id,
                            "error_threshold_exceeded",
                            {"tool_name": tool_name, "consecutive_errors": error_tracker[key]},
                        )
                        return ExecutorResult(exit_reason=ExecutorExit.CIRCUIT_BREAKER_ERRORS, turns=turn)
                else:
                    # Success or transient resets the counter for this (tool, args).
                    error_tracker.pop(key, None)

                tool_response_parts.append(json.dumps(result, ensure_ascii=True)[:_MAX_TOOL_OUTPUT_CHARS])

            # Feed results back in Nemotron's trained format.
            messages.append(
                {
                    "role": "user",
                    "content": "\n".join(f"<tool_response>\n{r}\n</tool_response>" for r in tool_response_parts),
                }
            )

        # Loop limit reached
        self._emit_circuit_breaker(brief.task_id, "loop_limit_exceeded", {"turns": loop_limit})
        return ExecutorResult(exit_reason=ExecutorExit.CIRCUIT_BREAKER_LOOP, turns=loop_limit)

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _build_system_message(self, brief: ExecutorBrief) -> str:
        tools_block = self._build_scoped_tools_block(brief.active_manifest)
        constraints_text = _render_constraints(brief.constraints) if brief.constraints else ""
        return (
            "You are the Atlas Framework tool-orchestration executor. You run inside an "
            "offline-first research operating system. A supervisor has given you a GOAL BRIEF "
            "and a scoped tool manifest. Achieve the goal by calling tools dynamically, "
            "observing results, and deciding the next action based on what you learn. You are "
            "NOT given a pre-baked plan — the order of tool calls is yours to discover.\n\n"
            f"# Goal\n{brief.goal_statement}\n\n"
            f"# Definition of Done\n{brief.definition_of_done}\n"
            f"{constraints_text}\n\n"
            "# How to finish\n"
            f"- When you have a final answer that meets the definition of done, call "
            f"`{FINAL_ANSWER_TOOL_NAME}({{\"answer\": \"...\"}})`.\n"
            f"- If you are stuck, ambiguous, or need a different toolkit, call "
            f"`{YIELD_TOOL_NAME}({{\"reason\": \"...\", \"suggested_options\": [...]}})`.\n"
            "- If a tool returns an error, READ the error message — it usually tells you "
            "what to do next (e.g. 'call tool X first'). Adjust and continue.\n\n"
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

    def _build_user_message(self, brief: ExecutorBrief) -> str:
        base = (
            "Begin executing the goal. Call tools as needed. When done, call "
            f"`{FINAL_ANSWER_TOOL_NAME}`. If you get blocked, call `{YIELD_TOOL_NAME}`."
        )
        if not brief.attachments:
            return base
        # Surface user-uploaded file paths so tools that take file_path / image_path
        # arguments (verify_spectrum, vision_inspector, etc.) can reference them.
        attachments_block = "\n".join(f"- {p}" for p in brief.attachments)
        return (
            f"{base}\n\n"
            "# Attachments\n"
            "The user uploaded these files. Their absolute paths are:\n"
            f"{attachments_block}\n"
            "Pass these paths verbatim as the `file_path` / `image_path` / `reference_dir` "
            "arguments to tools that accept them. Do NOT paraphrase the path."
        )

    def _build_scoped_tools_block(self, manifest: List[str]) -> str:
        full_tools = self._catalog.list_tools()
        selected = [t for t in full_tools if t["name"] in manifest]
        lines: List[str] = []
        for tool in selected:
            entry = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            lines.append(json.dumps(entry, ensure_ascii=True))
        # Synthetic tools.
        lines.append(json.dumps({
            "type": "function",
            "function": {
                "name": FINAL_ANSWER_TOOL_NAME,
                "description": "Submit your final answer to the supervisor for review.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }, ensure_ascii=True))
        lines.append(json.dumps({
            "type": "function",
            "function": {
                "name": YIELD_TOOL_NAME,
                "description": (
                    "Yield control to the supervisor when you are stuck, the available tools "
                    "are insufficient, or the task is ambiguous. Use reason='toolkit_insufficient' "
                    "if you need different tools."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "suggested_options": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["reason"],
                },
            },
        }, ensure_ascii=True))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_circuit_breaker(self, task_id: str, reason: str, context: Dict[str, Any]) -> None:
        self._log.append(
            task_id,
            Actor.SYSTEM_CIRCUIT_BREAKER,
            EventType.SYSTEM_CIRCUIT_BREAKER,
            {"reason": reason, "context": context},
        )


# ----------------------------------------------------------------------
# Tool result classification
# ----------------------------------------------------------------------


def _classify_tool_result(result: Any) -> Tuple[ToolStatus, str, Optional[str]]:
    """Map a tool result dict to a (status, summary, error_detail) triple.

    Convention:
      - explicit status field: "success" | "error_transient" | "error_permanent" | "requires_human"
      - legacy: { error: "..." } → error_permanent
      - anything else → success
    """
    if not isinstance(result, dict):
        return ToolStatus.SUCCESS, _to_str(result), None

    explicit = str(result.get("status") or "").lower().strip()
    if explicit in {s.value for s in ToolStatus}:
        status = ToolStatus(explicit)
        summary = _to_str(result.get("message") or result.get("data") or result)
        error_detail = result.get("error_detail")
        return status, summary, str(error_detail) if error_detail else None

    if "error" in result:
        return ToolStatus.ERROR_PERMANENT, _to_str(result.get("error")), None

    return ToolStatus.SUCCESS, _to_str(result), None


def _to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def _truncate(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "...(truncated)", True


def _error_key(tool_name: str, args: Any) -> str:
    args_repr = json.dumps(args, sort_keys=True, ensure_ascii=True) if isinstance(args, dict) else str(args)
    return f"{tool_name}|{args_repr}"


def _render_constraints(constraints: Dict[str, Any]) -> str:
    lines = ["", "# Constraints"]
    for k, v in constraints.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


# Singleton
_task_executor: Optional[TaskExecutor] = None


def get_task_executor() -> TaskExecutor:
    global _task_executor
    if _task_executor is None:
        _task_executor = TaskExecutor()
    return _task_executor
