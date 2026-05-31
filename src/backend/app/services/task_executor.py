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
import re
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
_MAX_TOOL_OUTPUT_CHARS = 1100  # default cap on the MODEL-facing view of a tool result; structural projection (not a blind byte-cut) keeps it under this. Frontend always gets the full payload via the event log.
_RECITATION_EVERY = 3  # re-state the campaign goal at the END of context every N turns (lost-in-the-middle / context-rot mitigation)
_MODEL_LOAD_TIMEOUT_SECONDS = 180
_GENERATION_TIMEOUT_SECONDS = 180
_TOOL_TIMEOUT_SECONDS = 300

# Rough chars-per-token. Real tokenizer would be precise but slow; this
# heuristic is good enough for budget gating (always errs on the safe side).
_CHARS_PER_TOKEN = 3.5
# Compaction trigger: when estimated context exceeds this fraction of n_ctx,
# evict the oldest tool exchange (anchors stay). 75% leaves headroom for the
# generation itself (max_tokens) plus the next tool call.
_CONTEXT_BUDGET_FRACTION = 0.75


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _detect_stream_phase(text_so_far: str) -> str:
    """Classify what the model is currently emitting based on open/close tags.

    - "thinking": inside an open <think>...</think> block
    - "tool_call": inside an open <tool_call>...</tool_call> block
    - "response": free text between blocks (the user-facing prose)
    """
    if text_so_far.count("<think>") > text_so_far.count("</think>"):
        return "thinking"
    if text_so_far.count("<tool_call>") > text_so_far.count("</tool_call>"):
        return "tool_call"
    return "response"


def _extract_response_prose(raw: str) -> str:
    """Pull the user-facing prose between </think> and the first <tool_call>.

    This is the "let me do X then Y" plan the model writes — the natural
    response that should render as a prominent block above the tool cards,
    matching the Claude-style think → respond → act loop.
    """
    # Strip all <think> blocks first (their content isn't response prose)
    no_think = _THINK_BLOCK_RE.sub("", raw)
    # Take everything before the first <tool_call> if there is one
    if "<tool_call>" in no_think:
        prose = no_think.split("<tool_call>", 1)[0]
    else:
        prose = no_think
    # Drop dangling open tags (truncation defense)
    if "<think>" in prose:
        prose = prose.split("<think>", 1)[0]
    return prose.strip()


def _extract_intent_lines_for_calls(raw: str, tool_names: List[str]) -> List[str]:
    """Pull the one-sentence natural-language intent that precedes each <tool_call>.

    The system prompt tells the model to write ONE short sentence before every
    <tool_call> describing what it's about to do. We split the raw response on
    <tool_call> boundaries and take the last non-empty line of the prose chunk
    immediately before each call. Falls back to "" when none was emitted (the
    UI degrades gracefully — just no subtitle on that tool card).
    """
    if not tool_names:
        return []
    # Strip <think> blocks first — intent lives in the free-text between them
    # and the <tool_call> tags, not inside <think>.
    stripped = _THINK_BLOCK_RE.sub("", raw)
    parts = _TOOL_CALL_BLOCK_RE.split(stripped)
    # parts[i] is the prose preceding tool_call i (parts has len = n+1 generally).
    intents: List[str] = []
    for i in range(len(tool_names)):
        chunk = parts[i] if i < len(parts) else ""
        # Take the last non-empty stripped line of the chunk — that's typically
        # the intent sentence. Skip lines that are obviously not intent
        # (json fragments, tags, etc.).
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        intent = ""
        for ln in reversed(lines):
            if ln.startswith("<") or ln.startswith("{") or ln.startswith("}"):
                continue
            intent = ln
            break
        intents.append(intent)
    return intents


def _synthesize_narration(
    tool_calls: List[Tuple[str, Any]],
    intent_lines: List[str],
) -> str:
    """Guarantee a user-facing 'what I'm about to do' sentence every acting turn.

    The IQ2 orchestrator frequently dumps its whole plan inside <think> and emits
    no prose before the <tool_call>, so _extract_response_prose returns "" and the
    turn renders as REASONING-only — the "it never responds" failure mode. When
    that happens we synthesize a concise narration from the per-tool intent lines,
    falling back to the tool names. Deterministic, no extra generation (keeps the
    laptop loop fast) — and the RESPONSE event is flagged synthesized=True so the
    UI/trace can tell model-authored prose from our fallback.
    """
    real = [tc[0] for tc in tool_calls if tc[0] not in (YIELD_TOOL_NAME, FINAL_ANSWER_TOOL_NAME)]
    if not real:
        return ""
    intents = [ln for ln in intent_lines if ln]
    if intents:
        # The model wrote per-tool intent prose even though the leading block was
        # empty — stitch the first one or two into a sentence.
        return " ".join(intents[:2])
    if len(real) == 1:
        return f"Running {real[0]}."
    return f"Running {', '.join(real[:-1])}, then {real[-1]}."


def _estimate_tokens(messages: List[Dict[str, str]]) -> int:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return int(total_chars / _CHARS_PER_TOKEN)


def _compact_if_over_budget(messages: List[Dict[str, str]], n_ctx: int) -> List[Dict[str, str]]:
    """Sliding-window compaction. Keep system + first user prompt as anchors;
    when total context exceeds the budget, drop the oldest non-anchor message
    pair (assistant turn + corresponding tool_response user turn) and prepend
    a single-line summary so the model retains continuity.

    Returns either the input list unchanged or a new compacted list.
    """
    budget_tokens = int(n_ctx * _CONTEXT_BUDGET_FRACTION)
    if _estimate_tokens(messages) <= budget_tokens or len(messages) < 5:
        return messages

    # Anchors: messages[0] = system, messages[1] = original user prompt.
    # Middle: messages[2:-2] — evict the oldest of these. Always keep the
    # last 4 messages (recent assistant+result+assistant+result) for fidelity.
    if len(messages) <= 6:
        return messages

    anchors = messages[:2]
    recent = messages[-4:]
    middle = messages[2:-4]

    if not middle:
        return messages

    # Drop one pair from the front of middle (typically: 1 assistant turn +
    # 1 user tool_response).
    dropped_count = min(2, len(middle))
    kept_middle = middle[dropped_count:]

    # Find a previous compaction summary we may have prepended earlier so we
    # don't keep accumulating duplicate summaries.
    existing_summary_idx = None
    for i, m in enumerate(kept_middle):
        if isinstance(m.get("content"), str) and m["content"].startswith("[compacted earlier tool calls:"):
            existing_summary_idx = i
            break

    summary_line = f"[compacted earlier tool calls: {dropped_count} message(s) summarized; full trace in event log]"
    summary_msg = {"role": "user", "content": summary_line}

    if existing_summary_idx is None:
        new_messages = anchors + [summary_msg] + kept_middle + recent
    else:
        # Replace the prior summary with an updated one
        kept_middle[existing_summary_idx] = summary_msg
        new_messages = anchors + kept_middle + recent

    logger.info(
        "Context compaction: dropped %d message(s), %d → %d est tokens",
        dropped_count,
        _estimate_tokens(messages),
        _estimate_tokens(new_messages),
    )
    return new_messages


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
    project_id: str
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
            await asyncio.wait_for(
                self._orchestrator.ensure_model_loaded(),
                timeout=_MODEL_LOAD_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            self._emit_circuit_breaker(brief.project_id, brief.task_id, "fatal_wrapper_crash", {"error": str(exc)})
            return ExecutorResult(exit_reason=ExecutorExit.FATAL, fatal_error=str(exc))

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._build_system_message(brief)},
            {"role": "user", "content": self._build_user_message(brief)},
        ]

        # Per-tool model-facing projection specs (salient fields / max_chars),
        # read once from the merged catalog so the loop can shrink each tool
        # result to what the model needs — instead of a blind byte-cut.
        projection_specs: Dict[str, Any] = {
            t["name"]: t.get("to_model_projection")
            for t in self._catalog.list_tools()
        }

        loop_limit = _DEFAULT_LOOP_LIMIT
        error_tracker: Dict[str, int] = {}  # (tool|args_hash) → consecutive error count

        for turn in range(1, loop_limit + 1):
            if cancel_event is not None and cancel_event.is_set():
                return ExecutorResult(exit_reason=ExecutorExit.FATAL, fatal_error="cancelled", turns=turn - 1)

            # Sliding-window compaction so long tool chains can't blow n_ctx.
            messages = _compact_if_over_budget(messages, settings.ATLAS_ORCHESTRATOR_CONTEXT_SIZE)

            # Stream generation token-by-token so the UI can render thinking
            # and response prose live as it appears (Claude-style typing feel).
            # We accumulate chunks for the final raw text while emitting
            # ORCHESTRATOR_STREAM_DELTA events the frontend coalesces in real
            # time. After the iteration completes, canonical THINKING +
            # RESPONSE events get logged as the source-of-truth for replay.
            async def _stream_and_emit() -> str:
                accumulated: List[str] = []
                await self._orchestrator.ensure_model_loaded()
                prompt_text = AtlasOrchestratorService._render_chatml(messages)
                async for delta in self._orchestrator._generate_streaming(prompt_text):
                    accumulated.append(delta)
                    if not delta:
                        continue
                    # Skip structural-tag deltas. When </think> or </tool_call>
                    # arrives as its own token, the phase detector sees the
                    # open/close counts equalize and classifies it as the NEXT
                    # phase — so a closing tag flashes in the response stream
                    # as orphan text. Tags are noise to the live UI; canonical
                    # THINKING/RESPONSE events emitted post-iteration carry the
                    # real content.
                    if any(tag in delta for tag in ("<think>", "</think>", "<tool_call>", "</tool_call>")):
                        continue
                    phase = _detect_stream_phase("".join(accumulated))
                    if phase in ("thinking", "response"):
                        self._log.append(
                            brief.project_id,
                            brief.task_id,
                            Actor.ORCHESTRATOR,
                            EventType.ORCHESTRATOR_STREAM_DELTA,
                            {"iteration": turn, "phase": phase, "delta": delta},
                        )
                return "".join(accumulated).strip()

            try:
                raw = await asyncio.wait_for(_stream_and_emit(), timeout=_GENERATION_TIMEOUT_SECONDS)
            except Exception as exc:
                self._emit_circuit_breaker(brief.project_id, brief.task_id, "fatal_wrapper_crash", {"error": str(exc), "phase": "generate"})
                return ExecutorResult(exit_reason=ExecutorExit.FATAL, fatal_error=str(exc), turns=turn - 1)

            tool_calls = AtlasOrchestratorService._extract_tool_calls(raw)

            # Surface the orchestrator's <think> content as a canonical event
            # (the deltas above were ephemeral; this is the replay-truth).
            thinking_text = AtlasOrchestratorService._extract_thinking(raw)
            if thinking_text:
                self._log.append(
                    brief.project_id,
                    brief.task_id,
                    Actor.ORCHESTRATOR,
                    EventType.ORCHESTRATOR_THINKING,
                    {"content": thinking_text, "iteration": turn},
                )

            # Per-tool intent prose (one sentence the model is asked to write
            # before each <tool_call>). Extracted up-front so it can both
            # subtitle each tool card AND seed a synthesized narration when the
            # model skipped the leading prose.
            intent_lines = _extract_intent_lines_for_calls(raw, [tc[0] for tc in tool_calls]) if tool_calls else []

            # Surface the response prose (between </think> and first <tool_call>)
            # as the user-facing "I'll do X then Y" block. GUARANTEE it: when the
            # model dumps everything into <think> and writes no prose (the IQ2
            # "never responds" failure mode), synthesize a concise narration from
            # the tool intents / names so every acting turn still speaks. When
            # there are no tool calls, the prose becomes the final answer via the
            # no-tool-calls branch below, so we don't emit RESPONSE here.
            response_text = _extract_response_prose(raw)
            if tool_calls:
                narration = response_text or _synthesize_narration(tool_calls, intent_lines)
                if narration:
                    self._log.append(
                        brief.project_id,
                        brief.task_id,
                        Actor.ORCHESTRATOR,
                        EventType.ORCHESTRATOR_RESPONSE,
                        {
                            "content": narration,
                            "iteration": turn,
                            "synthesized": not bool(response_text),
                        },
                    )

            # If no tool calls, treat the cleaned text as final answer. Never
            # fall back to raw — that leaks <think> content when the model
            # produced only internal reasoning with a truncated/orphan tag.
            if not tool_calls:
                answer = AtlasOrchestratorService._extract_final_text(raw)
                if not answer:
                    answer = (
                        "The model produced internal reasoning but no final "
                        "answer text — likely ran out of token budget. Try "
                        "rephrasing more imperatively, or shorten the question."
                    )
                return ExecutorResult(exit_reason=ExecutorExit.FINAL_ANSWER, final_answer=answer, turns=turn)

            messages.append({"role": "assistant", "content": raw})

            tool_response_parts: List[str] = []

            for call_idx, (tool_name, tool_args) in enumerate(tool_calls):
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
                        brief.project_id,
                        brief.task_id,
                        Actor.ORCHESTRATOR,
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
                intent_for_call = intent_lines[call_idx] if call_idx < len(intent_lines) else ""
                self._log.append(
                    brief.project_id,
                    brief.task_id,
                    Actor.ORCHESTRATOR,
                    EventType.TOOL_CALL_INTENT,
                    {
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "intent": intent_for_call,
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
                        brief.project_id,
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
                    result = await asyncio.wait_for(
                        self._catalog.invoke(
                            tool_name,
                            tool_args if isinstance(tool_args, dict) else {},
                            context={"task_id": brief.task_id, "project_id": brief.project_id},
                        ),
                        timeout=_TOOL_TIMEOUT_SECONDS,
                    )
                    status, summary, error_detail = _classify_tool_result(result)
                except Exception as exc:
                    status = ToolStatus.ERROR_PERMANENT
                    summary = f"Tool '{tool_name}' raised: {exc}"
                    error_detail = type(exc).__name__
                    result = {"error": str(exc)}
                elapsed_ms = int((time.perf_counter() - t_start) * 1000)

                summary_text, truncated = _truncate(summary, _MAX_TOOL_OUTPUT_CHARS)
                # Embed the raw structured result alongside the summary so the
                # frontend can extract specific fields (purity_percent, ic50_nm,
                # batch_id, etc.) for inline tool-card chips. Without this only
                # the stringified summary survives and chips can't read numbers.
                result_payload: Dict[str, Any] = {"summary": summary_text, "truncated": truncated}
                if isinstance(result, dict):
                    for k, v in result.items():
                        if k == "summary":
                            continue
                        result_payload[k] = v
                self._log.append(
                    brief.project_id,
                    brief.task_id,
                    Actor.TOOL_WRAPPER,
                    EventType.TOOL_EXECUTION_RESULT,
                    {
                        "call_id": call_id,
                        "status": status.value,
                        "output": result_payload,
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
                            brief.project_id,
                            brief.task_id,
                            "error_threshold_exceeded",
                            {"tool_name": tool_name, "consecutive_errors": error_tracker[key]},
                        )
                        return ExecutorResult(exit_reason=ExecutorExit.CIRCUIT_BREAKER_ERRORS, turns=turn)
                else:
                    # Success or transient resets the counter for this (tool, args).
                    error_tracker.pop(key, None)

                # Project the tool result to what the MODEL needs to see, using
                # the plugin's declared `to_model_projection` (salient fields)
                # when present, else a conservative structural default that keeps
                # scalars/short fields and compresses large lists/dicts to
                # placeholders. Replaces the old blind byte-cut that sliced JSON
                # mid-token and silently dropped fields the next tool needed. The
                # FULL payload still reaches the UI via TOOL_EXECUTION_RESULT.
                model_view, was_compressed = _project_for_model(
                    result,
                    projection_specs.get(tool_name),
                    _MAX_TOOL_OUTPUT_CHARS,
                )
                if was_compressed:
                    # Restorable offload: tell the model how to recover the full
                    # result it can't see inline — the complete payload is in the
                    # event log, fetchable via read_artifact by this call_id.
                    model_view += (
                        f'\n[result summarized — call read_artifact(artifact_id="{call_id}") '
                        f"for all fields]"
                    )
                    if not (projection_specs.get(tool_name) or {}).get("salient_fields"):
                        logger.info(
                            "Tool '%s' result compressed for model context — add "
                            "'to_model_projection' to its manifest for a sharper view.",
                            tool_name,
                        )
                tool_response_parts.append(model_view)

            # Feed results back in Nemotron's trained format. Every Nth turn we
            # also recite the campaign goal at the END of the message (the
            # recency anchor) so the small model doesn't lose the thread on long
            # chains — kept in the same user turn to preserve role alternation.
            tool_response_block = "\n".join(
                f"<tool_response>\n{r}\n</tool_response>" for r in tool_response_parts
            )
            if turn % _RECITATION_EVERY == 0:
                tool_response_block += "\n\n" + _build_recitation(brief, turn)
            messages.append({"role": "user", "content": tool_response_block})

        # Loop limit reached
        self._emit_circuit_breaker(brief.project_id, brief.task_id, "loop_limit_exceeded", {"turns": loop_limit})
        return ExecutorResult(exit_reason=ExecutorExit.CIRCUIT_BREAKER_LOOP, turns=loop_limit)

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _build_system_message(self, brief: ExecutorBrief) -> str:
        tools_block = self._build_scoped_tools_block(brief.active_manifest)
        constraints_text = _render_constraints(brief.constraints) if brief.constraints else ""
        if brief.active_manifest:
            tool_policy = (
                "# Tools\n\n"
                "You may call functions when they would help. Function signatures "
                "are listed in <tools></tools> XML tags:\n"
                "<tools>\n"
                f"{tools_block}\n"
                "</tools>\n\n"
                "To call a function, emit a JSON object with name + arguments inside "
                "<tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call>\n\n"
                "When you have nothing more to do, just respond in plain text — that "
                "ends the turn. No special closing tool required."
            )
        else:
            tool_policy = (
                "# Tools\n"
                "No external tools are available for this turn. Answer the user directly."
            )
        return (
            "You are Atlas, a local research operating system running offline on the "
            "user's machine. Respond to the user naturally. Use tools when they "
            "genuinely help; otherwise just answer in plain text."
            f"{constraints_text}\n\n"
            f"{tool_policy}"
        )

    def _build_user_message(self, brief: ExecutorBrief) -> str:
        base = (brief.goal_statement or "").strip() or "Hello"
        if not brief.attachments:
            return base
        # Surface user-uploaded file paths so tools that take file_path / image_path
        # arguments (verify_spectrum, vision_inspector, etc.) can reference them.
        attachments_block = "\n".join(f"- {p}" for p in brief.attachments)
        return (
            f"{base}\n\n"
            "Attached files (pass these paths verbatim to tools that take "
            "`file_path` / `image_path` / `reference_dir`):\n"
            f"{attachments_block}"
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
        # Synthetic escape hatch. submit_final_answer is intentionally NOT
        # advertised — we treat "no tool_calls in the response" as the final
        # answer, which is the model's natural stopping condition. The
        # synthetic handler in run() still catches it if the model invents
        # the call from training, but listing it pushes the model to
        # over-formalize even short replies.
        lines.append(json.dumps({
            "type": "function",
            "function": {
                "name": YIELD_TOOL_NAME,
                "description": (
                    "Use only when the available tools genuinely cannot make progress "
                    "and you need a different toolkit. Do NOT use for normal uncertainty — "
                    "if you're unsure what the user wants, just ask them in plain text."
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

    def _emit_circuit_breaker(self, project_id: str, task_id: str, reason: str, context: Dict[str, Any]) -> None:
        self._log.append(
            project_id,
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


def _project_for_model(
    result: Any,
    spec: Optional[Dict[str, Any]],
    default_cap: int,
) -> Tuple[str, bool]:
    """Shrink a tool result to the MODEL-facing view (full payload still goes to
    the UI event log). Uses the plugin-declared projection when present::

        {"salient_fields": ["canonical_smiles", "inchi_key"], "max_chars": 200}

    Otherwise a conservative structural default: keep the full payload if it
    already fits, else compress the largest fields (lists/dicts/long strings) to
    placeholders while keeping scalars and list-heads — never a blind mid-JSON
    byte-cut (the old failure that silently dropped fields the next tool needed).

    Returns (model_view_string, was_compressed).
    """
    spec = spec or {}
    cap = default_cap
    spec_max = spec.get("max_chars")
    if isinstance(spec_max, int) and spec_max > 0:
        cap = spec_max

    def _dump(obj: Any) -> str:
        return obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=True, default=str)

    # 1) Declared salient-field projection wins.
    salient = spec.get("salient_fields")
    if salient and isinstance(result, dict):
        view = {k: result[k] for k in salient if k in result}
        if "summary" in result and "summary" not in view:
            view["summary"] = result["summary"]
        rendered = _dump(view)
        if len(rendered) > cap:
            return rendered[:cap] + "...(truncated)", True
        return rendered, True

    # 2) No declared projection: keep the full payload if it already fits.
    full = _dump(result)
    if len(full) <= cap:
        return full, False
    if not isinstance(result, dict):
        return full[:cap] + "...(truncated)", True

    # 3) Compress the largest fields until under cap, keeping list-heads so short
    #    result-chains (e.g. 1-3 standardized molecules) survive intact.
    view: Dict[str, Any] = dict(result)
    for key, value in sorted(view.items(), key=lambda kv: len(_dump(kv[1])), reverse=True):
        if len(_dump(view)) <= cap:
            break
        if isinstance(value, list):
            if len(value) > 3:
                view[key] = value[:3] + [f"...(+{len(value) - 3} more — full payload in event log)"]
            if len(_dump(view)) > cap:
                view[key] = f"<{len(value)} items — full payload in event log>"
        elif isinstance(value, dict):
            view[key] = f"<{len(value)} fields — full payload in event log>"
        elif isinstance(value, str) and len(value) > 200:
            view[key] = value[:200] + "…"
    rendered = _dump(view)
    if len(rendered) > cap:
        rendered = rendered[:cap] + "...(truncated)"
    return rendered, True


def _error_key(tool_name: str, args: Any) -> str:
    args_repr = json.dumps(args, sort_keys=True, ensure_ascii=True) if isinstance(args, dict) else str(args)
    return f"{tool_name}|{args_repr}"


def _render_constraints(constraints: Dict[str, Any]) -> str:
    lines = ["", "# Constraints"]
    for k, v in constraints.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _build_recitation(brief: ExecutorBrief, turn: int) -> str:
    """A compact goal re-statement appended at the END of context periodically.

    Small models lose the thread on long tool chains (lost-in-the-middle /
    context rot — they attend to the start and end of context, not the middle).
    Re-citing the goal at the recency anchor keeps the campaign on-target without
    mutating the pinned system+goal prefix (which would invalidate the KV-cache).
    """
    goal = (brief.goal_statement or "").strip()
    dod = (brief.definition_of_done or "").strip()
    line = f"[Checkpoint · iteration {turn}. Goal: {goal[:200]}"
    if dod:
        line += f" — done when: {dod[:160]}"
    line += ". Keep going; when finished, reply in plain text to end the turn.]"
    return line


# Singleton
_task_executor: Optional[TaskExecutor] = None


def get_task_executor() -> TaskExecutor:
    global _task_executor
    if _task_executor is None:
        _task_executor = TaskExecutor()
    return _task_executor
