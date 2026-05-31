"""Deterministic task supervisor (post-cloud-LLM rip-out).

The previous implementation called a cloud LLM (DeepSeek / MiniMax) to
"scope" a tool manifest, build a goal brief, review the orchestrator's
final answer, and classify user responses from SUSPENDED.

The current architecture is single-orchestrator (Nemotron-8B running
locally via llama-cpp-python) and offline-first. The orchestrator was
RL-trained to select tools and decide when to stop — there is nothing for
a cloud "supervisor" to do, and CLAUDE.md says no LangGraph / no agents /
no two-tier stacks.

This module preserves the old PUBLIC INTERFACE so callers (task_service,
task_executor) keep working, but every decision is now deterministic:

* ``scope_manifest`` — passes every tool through. No keyword tables, no
  domain-specific fast paths (the user's "domain-agnostic" rule).
* ``build_brief`` — uses the user prompt verbatim as the goal statement;
  the orchestrator's RL-trained loop owns the rest.
* ``review`` — always approves. The orchestrator's FINAL_ANSWER is the
  answer; no second-guessing pass.
* ``classify_user_response`` — always replans. A SUSPENDED → reply is
  treated as a new prompt, fed back to the orchestrator from scratch.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.atlas_plugin_system import get_tool_catalog

logger = logging.getLogger(__name__)


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
    amendment: Optional[str] = None
    user_question: Optional[str] = None


@dataclass
class ResumeClassification:
    classification: str  # "resume" | "replan"
    synthetic_tool_result: Optional[str] = None


class TaskSupervisor:
    """Deterministic supervisor. No cloud calls, no domain heuristics."""

    async def scope_manifest(self, user_prompt: str) -> ScopedManifest:
        catalog = get_tool_catalog()
        names: List[str] = []
        for tool in catalog.list_core_tools():
            name = tool.get("name", "")
            if name:
                names.append(name)
        for plugin in catalog.list_plugins():
            name = plugin.get("name", "")
            if name:
                names.append(name)
        return ScopedManifest(
            candidates=names,
            selected=list(names),
            reasoning="single-orchestrator: full catalog passed through; orchestrator selects at runtime",
        )

    async def build_brief(
        self,
        user_prompt: str,
        scoped: ScopedManifest,
        context_md: Optional[str] = None,
    ) -> GoalBrief:
        goal = (user_prompt or "").strip() or "Answer the user's request."
        return GoalBrief(
            brief_id=str(uuid.uuid4()),
            goal_statement=goal,
            definition_of_done=(
                "A substantive answer is produced that directly addresses the user's request."
            ),
            active_manifest=list(scoped.selected),
            constraints={},
        )

    async def review(
        self,
        brief: GoalBrief,
        trace_summary: str,
        candidate_answer: Optional[str],
        yield_reason: Optional[str] = None,
        circuit_breaker_reason: Optional[str] = None,
    ) -> ReviewVerdict:
        # The orchestrator emits FINAL_ANSWER when it's done. Always accept.
        return ReviewVerdict(
            verdict="approve",
            reasoning="single-orchestrator: FINAL_ANSWER from the orchestrator is taken as final",
        )

    async def classify_user_response(
        self, question: str, response: str, brief: GoalBrief
    ) -> ResumeClassification:
        # A user reply to a SUSPENDED task is treated as a new prompt.
        return ResumeClassification(classification="replan", synthetic_tool_result=None)


_supervisor_singleton: Optional[TaskSupervisor] = None


def get_task_supervisor() -> TaskSupervisor:
    global _supervisor_singleton
    if _supervisor_singleton is None:
        _supervisor_singleton = TaskSupervisor()
    return _supervisor_singleton


def is_conversational_turn(prompt: str) -> bool:
    """Retained for backward compat with task_service. Returns False — the
    orchestrator handles short prompts uniformly without a fast path
    (per the user's "no prompt-type special cases" rule)."""
    return False
