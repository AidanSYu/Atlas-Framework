"""DeepSeek supervisor for the two-tier task orchestration.

Responsibilities:
- Librarian: scope the master tool manifest down to 8–12 relevant tools.
- Planner: turn the user prompt into a goal brief (statement, DoD, constraints).
- Reviewer: judge Nemotron's final answer — approve / revise / rescope / ask user.
- Amender: when Nemotron yields or circuit breakers trip, produce a revised brief.
- Classifier: when the user responds from SUSPENDED, pick resume vs replan.

Never produces an ordered step list. The brief contains goals + scope, not a plan.
Nemotron's dynamic loop discovers the order by observing tool results at runtime.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.atlas_plugin_system import get_tool_catalog
from app.core.config import settings
from app.services.discovery_llm import DiscoveryLLMService

logger = logging.getLogger(__name__)

_SCOPE_TARGET_MIN = 4
_SCOPE_TARGET_MAX = 12


@dataclass
class ToolDescriptor:
    name: str
    description: str
    source: str  # "core" | "plugin"


@dataclass
class ScopedManifest:
    candidates: List[str]
    selected: List[str]
    reasoning: str


@dataclass
class GoalBrief:
    brief_id: str
    goal_statement: str
    definition_of_done: str
    active_manifest: List[str]
    constraints: Dict[str, Any]


@dataclass
class ReviewVerdict:
    verdict: str  # "approve" | "revise" | "rescope" | "ask_user"
    reasoning: str
    amendment: Optional[str] = None  # for "revise"
    user_question: Optional[str] = None  # for "ask_user"


@dataclass
class ResumeClassification:
    classification: str  # "resume" | "replan"
    synthetic_tool_result: Optional[str] = None  # for resume: inject as tool result


class TaskSupervisor:
    """DeepSeek-backed supervisor. One instance per process; lazily loaded."""

    def __init__(self, llm: Optional[DiscoveryLLMService] = None):
        self._llm = llm or DiscoveryLLMService(settings)

    # ------------------------------------------------------------------
    # Librarian: scope the manifest
    # ------------------------------------------------------------------

    async def scope_manifest(self, user_prompt: str) -> ScopedManifest:
        """Pick the relevant subset of tools for this task.

        v1: present all tool descriptions to DeepSeek and ask it to select
        up to 12 most relevant names + a one-paragraph reasoning. A future
        upgrade puts an embedding pre-filter in front of this.
        """
        catalog = get_tool_catalog()
        tools = _collect_tool_descriptors(catalog)
        if not tools:
            return ScopedManifest(candidates=[], selected=[], reasoning="no tools available")

        all_names = [t.name for t in tools]
        if len(all_names) <= _SCOPE_TARGET_MAX:
            # Small catalog — just pass everything through.
            return ScopedManifest(
                candidates=all_names,
                selected=all_names,
                reasoning=f"catalog has only {len(all_names)} tools; all included",
            )

        descriptions = "\n".join(
            f"- {t.name} [{t.source}]: {t.description[:160]}" for t in tools
        )
        system = (
            "You are the Tool Librarian for the Atlas Framework, a domain-agnostic "
            "research orchestration system. Given a user's task, select the minimum "
            f"useful subset of tools ({_SCOPE_TARGET_MIN}-{_SCOPE_TARGET_MAX}) from the "
            "manifest that a tool-calling model will need to accomplish the task.\n\n"
            "Rules:\n"
            "- Prefer fewer, more relevant tools over comprehensive coverage.\n"
            "- If two tools share a capability, pick ONE (the more general one).\n"
            "- Include general-purpose tools (retrieval, KG search) when the task is open-ended.\n"
            "- You are NOT writing an execution plan. You are ONLY selecting which tools "
            "should be available. The tool-calling model decides the order at runtime.\n\n"
            "Respond with ONLY a JSON object: "
            '{"selected": ["tool_a", "tool_b"], "reasoning": "one-paragraph why"}'
        )
        prompt = (
            f"User task:\n{user_prompt}\n\n"
            f"Available tools:\n{descriptions}\n\n"
            f"Select {_SCOPE_TARGET_MIN}-{_SCOPE_TARGET_MAX} tools."
        )

        try:
            raw = await self._llm.orchestrate(
                prompt=prompt, system_prompt=system, temperature=0.2, max_tokens=1024
            )
        except Exception as exc:
            logger.warning("Librarian call failed; falling back to all tools: %s", exc)
            return ScopedManifest(
                candidates=all_names,
                selected=all_names[:_SCOPE_TARGET_MAX],
                reasoning=f"librarian unavailable ({exc}); used first {_SCOPE_TARGET_MAX} tools",
            )

        data = _parse_json_object(raw)
        selected_raw = data.get("selected", []) if isinstance(data, dict) else []
        selected = [name for name in selected_raw if isinstance(name, str) and name in all_names]
        if not selected:
            selected = all_names[:_SCOPE_TARGET_MAX]
            reasoning = "librarian returned no valid tools; default fallback"
        else:
            reasoning = str(data.get("reasoning", "")) if isinstance(data, dict) else ""
        return ScopedManifest(
            candidates=all_names, selected=selected[:_SCOPE_TARGET_MAX], reasoning=reasoning
        )

    # ------------------------------------------------------------------
    # Planner: goal brief
    # ------------------------------------------------------------------

    async def build_brief(
        self, user_prompt: str, scoped: ScopedManifest, context_md: Optional[str] = None
    ) -> GoalBrief:
        """Turn the user prompt + scoped tools into a goal brief."""
        system = (
            "You are the planning supervisor for Atlas, a two-tier research orchestration "
            "system. A tool-calling model will execute this task using the tools you scope. "
            "Your job is to produce a GOAL BRIEF, not a step-by-step plan.\n\n"
            "Rules:\n"
            "- goal_statement: 1-3 sentences. What does the user want? Keep scope tight.\n"
            "- definition_of_done: concrete observable criteria. How will we know the task "
            "is finished? Focus on outcomes, not tool calls.\n"
            "- constraints: any hard rules (forbid X, prefer Y, stay under N items).\n"
            "- DO NOT list steps or the order of tool calls. The tool model picks the order "
            "at runtime based on live observations.\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"goal_statement": "...", "definition_of_done": "...", "constraints": {...}}'
        )
        ctx_section = f"\n\nWorkspace context:\n{context_md}" if context_md else ""
        prompt = (
            f"User prompt:\n{user_prompt}{ctx_section}\n\n"
            f"Tools the executor has access to: {', '.join(scoped.selected)}\n\n"
            "Produce the JSON brief."
        )
        try:
            raw = await self._llm.orchestrate(
                prompt=prompt, system_prompt=system, temperature=0.2, max_tokens=1024
            )
            data = _parse_json_object(raw) or {}
        except Exception as exc:
            logger.warning("Planner call failed; using fallback brief: %s", exc)
            data = {}

        goal_statement = str(data.get("goal_statement") or user_prompt.strip())
        definition_of_done = str(
            data.get("definition_of_done")
            or "A substantive answer is produced that directly addresses the user's request."
        )
        constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
        import uuid
        return GoalBrief(
            brief_id=str(uuid.uuid4()),
            goal_statement=goal_statement,
            definition_of_done=definition_of_done,
            active_manifest=list(scoped.selected),
            constraints=constraints or {},
        )

    # ------------------------------------------------------------------
    # Reviewer: judge final answer or yielded trace
    # ------------------------------------------------------------------

    async def review(
        self,
        brief: GoalBrief,
        trace_summary: str,
        candidate_answer: Optional[str],
        yield_reason: Optional[str] = None,
        circuit_breaker_reason: Optional[str] = None,
    ) -> ReviewVerdict:
        """Review Nemotron's output and decide next FSM move.

        Returns one of:
          - approve: answer is good → COMPLETED
          - revise: amend the brief and continue EXECUTING
          - rescope: toolkit is wrong → back to PLANNING
          - ask_user: we need user input → SUSPENDED
        """
        situation = []
        if candidate_answer:
            situation.append(f"Nemotron proposed final answer:\n{candidate_answer}")
        if yield_reason:
            situation.append(f"Nemotron yielded to supervisor with reason: {yield_reason}")
        if circuit_breaker_reason:
            situation.append(f"Circuit breaker tripped: {circuit_breaker_reason}")

        system = (
            "You are the supervisor reviewing a task execution. Decide what happens next.\n\n"
            "Options:\n"
            "- approve: the candidate answer meets the definition_of_done. Task succeeds.\n"
            "- revise: the trajectory is on-track but needs a nudge. Provide an amendment "
            "  string that clarifies or narrows. Executor will keep running with the amended brief.\n"
            "- rescope: the executor is fundamentally missing a tool or approach. Go back to "
            "  planning and re-scope the toolkit.\n"
            "- ask_user: ambiguity requires a human decision. Provide a crisp user_question.\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"verdict": "approve|revise|rescope|ask_user", '
            '"reasoning": "1-3 sentences", '
            '"amendment": "string if verdict=revise", '
            '"user_question": "string if verdict=ask_user"}'
        )
        prompt = (
            f"Goal: {brief.goal_statement}\n"
            f"Definition of done: {brief.definition_of_done}\n"
            f"Available tools: {', '.join(brief.active_manifest)}\n\n"
            f"Execution trace summary:\n{trace_summary}\n\n"
            + "\n\n".join(situation)
        )
        try:
            raw = await self._llm.orchestrate(
                prompt=prompt, system_prompt=system, temperature=0.15, max_tokens=1024
            )
            data = _parse_json_object(raw) or {}
        except Exception as exc:
            logger.warning("Reviewer call failed; defaulting to approve: %s", exc)
            data = {"verdict": "approve", "reasoning": f"reviewer unavailable ({exc})"}

        verdict = str(data.get("verdict", "approve")).strip().lower()
        if verdict not in {"approve", "revise", "rescope", "ask_user"}:
            verdict = "approve"
        return ReviewVerdict(
            verdict=verdict,
            reasoning=str(data.get("reasoning", "")),
            amendment=str(data.get("amendment")) if data.get("amendment") else None,
            user_question=str(data.get("user_question")) if data.get("user_question") else None,
        )

    # ------------------------------------------------------------------
    # Classifier: SUSPENDED user response → resume vs replan
    # ------------------------------------------------------------------

    async def classify_user_response(
        self, question: str, response: str, brief: GoalBrief
    ) -> ResumeClassification:
        system = (
            "A task is SUSPENDED awaiting a user reply to a specific question. Decide whether "
            "the user's response is a targeted answer (injectable as a tool result — resume) "
            "or a scope change / new directive (re-plan needed).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"classification": "resume|replan", '
            '"synthetic_tool_result": "string if resume — concise restatement of the answer"}'
        )
        prompt = (
            f"Goal: {brief.goal_statement}\n"
            f"Question posed:\n{question}\n\n"
            f"User response:\n{response}"
        )
        try:
            raw = await self._llm.orchestrate(
                prompt=prompt, system_prompt=system, temperature=0.1, max_tokens=512
            )
            data = _parse_json_object(raw) or {}
        except Exception:
            data = {}
        cls = str(data.get("classification", "replan")).strip().lower()
        if cls not in {"resume", "replan"}:
            cls = "replan"
        return ResumeClassification(
            classification=cls,
            synthetic_tool_result=str(data.get("synthetic_tool_result")) if data.get("synthetic_tool_result") else None,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _collect_tool_descriptors(catalog) -> List[ToolDescriptor]:
    descriptors: List[ToolDescriptor] = []
    for tool in catalog.list_core_tools():
        descriptors.append(
            ToolDescriptor(
                name=tool.get("name", ""),
                description=tool.get("description", ""),
                source="core",
            )
        )
    for plugin in catalog.list_plugins():
        descriptors.append(
            ToolDescriptor(
                name=plugin.get("name", ""),
                description=plugin.get("description", ""),
                source="plugin",
            )
        )
    return [d for d in descriptors if d.name]


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # Strip ```json fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# Singleton
_supervisor: Optional[TaskSupervisor] = None


def get_task_supervisor() -> TaskSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = TaskSupervisor()
    return _supervisor
