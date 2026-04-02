"""Coordinator Agent — Interactive session bootstrapping via LangGraph HITL.

Implements the Phase 4 Coordinator Initialization Flow:
  1. Scan existing corpus via RetrievalService
  2. LLM identifies missing constraints/goals
  3. interrupt() pauses graph, surfaces question to user
  4. Resume with user answer, loop until goals are complete

Uses MemorySaver checkpointing with thread_id = f"coordinator-{session_id}".
"""
import asyncio
import logging
import re
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

from app.services.discovery_llm import DiscoveryLLMService
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# Living .md Knowledge Helper
# ============================================================

_MD_PRIORITY = [
    "CONSTRAINTS.md",
    "HYPOTHESES.md",
    "RESEARCH_NOTES.md",
    "SESSION_CONTEXT.md",
    "FINDINGS.md",
]


def _read_session_notes(session_id: str, max_chars: int = 2000) -> str:
    """Read user-written .md files from the session root.

    Called during scan_corpus so the coordinator is aware of any constraints,
    hypotheses, or background notes the researcher dropped in the session folder
    before or between runs — without needing to upload them to the vector DB.
    """
    from pathlib import Path
    session_path = Path(settings.DATA_DIR) / "discovery" / session_id
    if not session_path.exists():
        return ""

    parts: list[str] = []
    seen: set[str] = set()

    for fname in _MD_PRIORITY:
        fpath = session_path / fname
        if fpath.exists():
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")[:1000]
                parts.append(f"=== {fname} ===\n{text}")
                seen.add(fname)
            except OSError:
                pass

    for fpath in sorted(session_path.glob("*.md")):
        if fpath.name not in seen:
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")[:400]
                parts.append(f"=== {fpath.name} ===\n{text}")
            except OSError:
                pass

    combined = "\n\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars].rsplit("\n", 1)[0]
    return combined


# ============================================================
# State
# ============================================================

class CoordinatorState(TypedDict, total=False):
    # Conversation history
    messages: List[Dict[str, Any]]

    # Extracted research parameters
    extracted_goals: List[str]

    # Questions still unanswered
    missing_context: List[str]

    # Corpus intelligence
    corpus_summary: str
    corpus_entities: List[str]

    # Control flow
    status: str  # "scanning" | "questioning" | "complete"
    turn_count: int
    max_turns: int

    # Session identifiers
    project_id: str
    session_id: str


# ============================================================
# LLM Constrained Output Schema
# ============================================================

COORDINATOR_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {
            "type": "string",
            "description": "Brief assessment of current research context"
        },
        "new_goals_extracted": {
            "type": "array",
            "items": {"type": "string"},
            "description": "New goals/constraints extracted from the latest user response"
        },
        "still_missing": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Information still needed to define a complete research session"
        },
        "ready_to_proceed": {
            "type": "boolean",
            "description": "True if enough information has been gathered"
        },
        "question": {
            "type": "string",
            "description": "Next question to ask the researcher"
        },
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 multiple-choice options for the question"
        },
    },
    "required": [
        "assessment", "new_goals_extracted", "still_missing",
        "ready_to_proceed", "question", "options"
    ],
}


# ============================================================
# Helper Functions
# ============================================================

def _infer_domain_from_goals(goals: List[str]) -> str:
    """Infer research domain from extracted goals using keyword matching.

    Args:
        goals: List of research goals extracted by coordinator

    Returns:
        Inferred domain (e.g., "organic_chemistry", "materials_science", "general")
    """
    goals_text = " ".join(goals).lower()

    # Domain keyword mapping
    if any(kw in goals_text for kw in ["molecule", "compound", "synthesis", "chemical", "drug", "inhibitor", "kinase"]):
        return "organic_chemistry"
    elif any(kw in goals_text for kw in ["material", "polymer", "crystal", "conductivity", "mechanical"]):
        return "materials_science"
    elif any(kw in goals_text for kw in ["protein", "enzyme", "gene", "biological", "cellular"]):
        return "biochemistry"
    else:
        return "general"


def _build_fallback_analysis(
    messages: List[Dict[str, Any]],
    goals: List[str],
    turn: int,
    max_turns: int,
) -> Dict[str, Any]:
    """Build a deterministic fallback analysis when LLM output fails.

    This prevents coordinator hard-failures from transient provider issues
    (empty body, invalid JSON, network jitter). The fallback keeps the HITL
    flow moving with conservative defaults and focused follow-up questions.
    """
    latest_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user = str(m.get("content", ""))
            break

    merged_goals = list(goals)
    if latest_user and latest_user not in merged_goals:
        merged_goals.append(latest_user)

    goals_text = " ".join(merged_goals).lower()
    has_domain = bool(
        re.search(r"\b(chemistry|molecule|drug|material|protein|biochem|biology|catalyst)\b", goals_text)
    )
    has_target = bool(
        re.search(r"\b(target|objective|against|inhibitor|bind|protein|enzyme|receptor)\b", goals_text)
    )
    has_constraints = bool(
        re.search(r"\b(mw|logp|tpsa|constraint|range|<=|>=|<|>|limit)\b", goals_text)
    )
    has_forbidden = bool(
        re.search(r"\b(forbidden|exclude|avoid|ban|not allow|without|none)\b", goals_text)
    )
    has_success = bool(
        re.search(r"\b(success|metric|criteria|ic50|ec50|kd|affinity|yield|accuracy|threshold)\b", goals_text)
    )

    missing: List[str] = []
    if not has_domain:
        missing.append("research domain")
    if not has_target:
        missing.append("primary objective or target")
    if not has_constraints:
        missing.append("property constraints")
    if not has_forbidden:
        missing.append("forbidden substructures or exclusions")
    if not has_success:
        missing.append("success metrics")

    ready = (len(missing) == 0) or (turn >= max_turns - 1)

    if not missing:
        question = "I have enough to proceed. Do you want to refine any constraints before execution?"
        options = ["Proceed", "Refine constraints", "Add forbidden motifs", "Other..."]
    elif missing[0] == "research domain":
        question = "Which research domain should this discovery session target?"
        options = ["Small-molecule chemistry", "Biochemistry/protein", "Materials science", "Other..."]
    elif missing[0] == "primary objective or target":
        question = "What is the primary target or objective for this run?"
        options = ["Specific biological target", "Property optimization only", "Exploratory hypothesis generation", "Other..."]
    elif missing[0] == "property constraints":
        question = "Which constraints should we enforce (for example MW, LogP, toxicity, stability)?"
        options = ["Use standard drug-like defaults", "I will provide numeric limits", "No hard constraints", "Other..."]
    elif missing[0] == "forbidden substructures or exclusions":
        question = "What should be excluded from candidates (substructures, liabilities, off-target risks)?"
        options = ["No exclusions", "Exclude known toxicophores", "Exclude previous failed scaffolds", "Other..."]
    else:
        question = "How should we define success for this session?"
        options = ["Potency threshold (e.g., IC50/Kd)", "Multi-objective score", "Top-N ranking only", "Other..."]

    return {
        "assessment": (
            "Coordinator fallback mode: used deterministic analysis because the "
            "orchestration model returned an invalid response."
        ),
        "new_goals_extracted": [latest_user] if latest_user else [],
        "still_missing": missing,
        "ready_to_proceed": ready,
        "question": question,
        "options": options,
    }


def _seed_living_md_files(
    session_id: str,
    goals: List[str],
    corpus_summary: str,
) -> None:
    """Create initial CONSTRAINTS.md, HYPOTHESES.md, RESEARCH_NOTES.md from goals.

    These are the living .md files that every agent reads before each run.
    The coordinator seeds them from the HITL conversation. The researcher
    can edit them between runs. Pipeline runs append FINDINGS.md.

    Only writes files that don't already exist (never overwrites user edits).
    """
    from pathlib import Path
    from app.core.config import settings

    session_path = Path(settings.DATA_DIR) / "discovery" / session_id
    session_path.mkdir(parents=True, exist_ok=True)

    goals_lower = [g.lower() for g in goals]
    goals_text = " ".join(goals_lower)

    # --- CONSTRAINTS.md ---
    constraints_path = session_path / "CONSTRAINTS.md"
    if not constraints_path.exists():
        constraint_goals = [
            g for g in goals
            if any(kw in g.lower() for kw in [
                "mw", "logp", "tpsa", "constraint", "<=", ">=", "<", ">",
                "limit", "range", "herg", "dili", "cyp", "solubility",
                "weight", "rotatable", "donor", "acceptor", "lipinski",
                "forbidden", "exclude", "avoid", "ban", "without",
            ])
        ]
        other_goals = [g for g in goals if g not in constraint_goals]
        lines = [
            "# Constraints",
            "",
            "> Auto-generated by the Coordinator from your session setup conversation.",
            "> Edit this file between pipeline runs to add, remove, or refine constraints.",
            "> All agents read this file before every execution.",
            "",
        ]
        if constraint_goals:
            lines.append("## Extracted Constraints\n")
            for g in constraint_goals:
                lines.append(f"- {g}")
            lines.append("")
        if any(kw in goals_text for kw in ["forbidden", "exclude", "avoid", "ban", "without", "not "]):
            lines.append("## Exclusions\n")
            exclusion_goals = [
                g for g in goals
                if any(kw in g.lower() for kw in ["forbidden", "exclude", "avoid", "ban", "without"])
            ]
            for g in exclusion_goals:
                lines.append(f"- {g}")
            lines.append("")
        if not constraint_goals:
            lines.append("*(No specific constraints extracted yet. Add them here.)*\n")
        try:
            constraints_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Seeded CONSTRAINTS.md for session %s", session_id)
        except OSError as exc:
            logger.warning("Failed to seed CONSTRAINTS.md: %s", exc)

    # --- HYPOTHESES.md ---
    hypotheses_path = session_path / "HYPOTHESES.md"
    if not hypotheses_path.exists():
        hypothesis_goals = [
            g for g in goals
            if any(kw in g.lower() for kw in [
                "hypothes", "predict", "expect", "should", "might",
                "improve", "optimize", "better", "reduce", "increase",
            ])
        ]
        lines = [
            "# Hypotheses",
            "",
            "> Working hypotheses for this research session.",
            "> Edit between pipeline runs to refine direction.",
            "> Agents read this to guide iteration strategy.",
            "",
        ]
        if hypothesis_goals:
            for i, g in enumerate(hypothesis_goals, 1):
                lines.append(f"**H{i}**: {g}\n")
        else:
            objective_goals = [g for g in goals if any(kw in g.lower() for kw in ["find", "identify", "discover", "target", "inhibitor", "candidate"])]
            if objective_goals:
                lines.append(f"**H1**: {objective_goals[0]}\n")
            else:
                lines.append("*(Add hypotheses here to guide the discovery pipeline.)*\n")
        try:
            hypotheses_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Seeded HYPOTHESES.md for session %s", session_id)
        except OSError as exc:
            logger.warning("Failed to seed HYPOTHESES.md: %s", exc)

    # --- RESEARCH_NOTES.md ---
    notes_path = session_path / "RESEARCH_NOTES.md"
    if not notes_path.exists():
        lines = [
            "# Research Notes",
            "",
            "> Background knowledge for this session.",
            "> Add literature references, domain context, or prior results here.",
            "> All agents read this file before every execution.",
            "",
        ]
        if corpus_summary and corpus_summary.strip():
            lines.append("## Corpus Summary (auto-extracted)\n")
            lines.append(corpus_summary[:1000])
            lines.append("")
        else:
            lines.append("*(Upload documents and they will be summarized here automatically.)*\n")
        try:
            notes_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Seeded RESEARCH_NOTES.md for session %s", session_id)
        except OSError as exc:
            logger.warning("Failed to seed RESEARCH_NOTES.md: %s", exc)

    # --- FINDINGS.md (empty header only) ---
    findings_path = session_path / "FINDINGS.md"
    if not findings_path.exists():
        try:
            findings_path.write_text(
                "# Research Findings\n\n"
                "Auto-updated log of all pipeline executions. "
                "Agents read this file before planning each next iteration.\n"
                "Researchers can annotate between runs.\n\n---\n\n",
                encoding="utf-8",
            )
            logger.info("Seeded FINDINGS.md for session %s", session_id)
        except OSError as exc:
            logger.warning("Failed to seed FINDINGS.md: %s", exc)


# ============================================================
# Graph Builder
# ============================================================

def _build_coordinator_graph(
    llm_service: DiscoveryLLMService,
    retrieval_service: Any,
) -> StateGraph:
    """Build the Coordinator StateGraph with HITL interrupt."""

    async def scan_corpus(state: CoordinatorState) -> dict:
        """Node 1: Semantic search over project corpus + read session .md notes."""
        project_id = state.get("project_id", "")
        session_id = state.get("session_id", "")

        # Broad query to understand what's in the corpus
        query = (
            "Summarize the key topics, molecules, biological targets, "
            "methodologies, and findings in this research corpus."
        )

        corpus_summary = "No documents found in corpus."
        entity_names: List[str] = []

        try:
            result = await retrieval_service.query_atlas(
                user_question=query,
                project_id=project_id if project_id else None,
            )

            chunks = result.get("context", {}).get("vector_chunks", [])
            entities_raw = result.get("context", {}).get("graph_nodes", [])

            if chunks:
                summary_parts = []
                for c in chunks[:8]:
                    text = c.get("text", "") if isinstance(c, dict) else str(c)
                    summary_parts.append(text[:300])
                corpus_summary = "\n---\n".join(summary_parts)

            if entities_raw:
                for e in entities_raw[:20]:
                    label = e.get("label", "") if isinstance(e, dict) else str(e)
                    if label:
                        entity_names.append(label)

        except Exception as exc:
            logger.warning("Coordinator corpus scan failed: %s", exc)
            corpus_summary = "Corpus scan encountered an error. Proceeding without corpus context."

        # Read user-written .md notes from the session folder.
        # These are appended to the corpus summary so the coordinator is aware of
        # any constraints, hypotheses, or background knowledge the researcher
        # dropped in the session folder (CONSTRAINTS.md, HYPOTHESES.md, etc.)
        # without needing to upload them to the vector DB.
        if session_id:
            session_notes = _read_session_notes(session_id)
            if session_notes:
                corpus_summary = (
                    corpus_summary
                    + "\n\n=== RESEARCHER NOTES (from session .md files) ===\n"
                    + session_notes
                )

        return {
            "corpus_summary": corpus_summary,
            "corpus_entities": entity_names,
            "status": "questioning",
            "turn_count": 0,
        }

    async def analyze_and_ask(state: CoordinatorState) -> dict:
        """Node 2: LLM analyses state, generates question or completes.

        Calls interrupt() to pause execution and surface a question to the user.
        When the graph resumes via Command(resume=answer), interrupt() returns
        the user's answer and execution continues from that exact point.
        """
        messages = state.get("messages", [])
        goals = list(state.get("extracted_goals", []))
        corpus = state.get("corpus_summary", "")
        entities = state.get("corpus_entities", [])
        turn = state.get("turn_count", 0)
        max_turns = state.get("max_turns", 5)

        # Format conversation history
        history_lines = []
        for m in messages[-10:]:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_str = "\n".join(history_lines) if history_lines else "(no conversation yet)"

        # Format entities
        entity_str = ", ".join(entities[:15]) if entities else "none detected"

        system_prompt = """You are a rigorous research coordinator for a scientific discovery project.
Your goal is to extract EXACTLY 5 key pieces of information before allowing the session to proceed:
1. Research domain (e.g., organic chemistry, materials science)
2. Primary objective / biological target
3. Property constraints (MW, LogP, TPSA ranges, etc.)
4. Forbidden substructures or exclusion criteria
5. Success criteria and evaluation metrics

CRITICAL INSTRUCTIONS:
- You must ask deep, probing questions to extract missing information.
- NEVER set 'ready_to_proceed': true until you have reasonably satisfied ALL 5 criteria.
- Ask ONE focused question at a time.
- Generate 2-4 appropriate multiple-choice options for the question (always include "Other..." or "None").

Your output must be valid JSON matching the schema. Do not include any markdown, explanations, or text outside the JSON object."""

        prompt = f"""=== CORPUS CONTEXT ===
Key entities found: {entity_str}

Corpus excerpts:
{corpus[:1500]}

=== GOALS EXTRACTED SO FAR ===
{chr(10).join(f"- {g}" for g in goals) if goals else "None yet."}

=== CONVERSATION HISTORY ===
{history_str}

=== TASK ===
Turn {turn + 1} of {max_turns}.

Analyze the history and goals. Extract any new constraints from the user's last message.
Compare the known goals against the 5 key criteria. What is still missing?
If anything is missing, generate a highly specific question to gather the next piece of missing info.
Only set ready_to_proceed to true if all 5 criteria are substantially answered or the maximum turns are reached.

Output JSON:
{{
  "assessment": "Brief assessment of current research context",
  "new_goals_extracted": ["goal1", "goal2", ...],
  "still_missing": ["missing1", "missing2", ...],
  "ready_to_proceed": true/false,
  "question": "Next question to ask",
  "options": ["option1", "option2", "option3", "Other..."]
}}"""

        try:
            # Use orchestrate_constrained (DeepSeek) — the reasoning model is far
            # more reliable for structured analysis than MiniMax (tool-calling model).
            analysis = await llm_service.orchestrate_constrained(
                prompt=prompt,
                schema=COORDINATOR_ANALYSIS_SCHEMA,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning(
                "Coordinator DeepSeek analysis failed on turn %d; "
                "continuing with fallback analysis: %s",
                turn,
                exc,
            )
            analysis = _build_fallback_analysis(
                messages=messages,
                goals=goals,
                turn=turn,
                max_turns=max_turns,
            )

        # Merge new goals
        new_goals = analysis.get("new_goals_extracted", [])
        for g in new_goals:
            if g and g not in goals:
                goals.append(g)

        # Check if ready or max turns reached
        if analysis.get("ready_to_proceed") or turn >= max_turns - 1:
            return {
                "extracted_goals": goals,
                "missing_context": [],
                "status": "complete",
                "turn_count": turn + 1,
            }

        # Build question payload and pause via interrupt()
        question_payload = {
            "question": analysis.get("question", "Could you provide more details about your research goals?"),
            "options": analysis.get("options", ["Yes", "No", "Other..."]),
            "context": analysis.get("assessment", ""),
            "turn": turn + 1,
            "goals_so_far": goals,
        }

        # interrupt() pauses the graph here and surfaces question_payload.
        # When the graph is resumed via Command(resume=answer), interrupt()
        # returns the user's answer string.
        user_response = interrupt(question_payload)

        # --- Execution resumes here after Command(resume=...) ---

        updated_messages = list(messages) + [
            {"role": "assistant", "content": analysis.get("question", "")},
            {"role": "user", "content": str(user_response)},
        ]

        return {
            "messages": updated_messages,
            "extracted_goals": goals,
            "missing_context": analysis.get("still_missing", []),
            "status": "questioning",
            "turn_count": turn + 1,
        }

    def should_continue(state: CoordinatorState) -> str:
        """Route: loop back to analyze_and_ask or end."""
        if state.get("status") == "complete":
            return "end"
        return "ask"

    # Assemble graph
    sg = StateGraph(CoordinatorState)
    sg.add_node("scan_corpus", scan_corpus)
    sg.add_node("analyze_and_ask", analyze_and_ask)

    sg.set_entry_point("scan_corpus")
    sg.add_edge("scan_corpus", "analyze_and_ask")
    sg.add_conditional_edges("analyze_and_ask", should_continue, {
        "ask": "analyze_and_ask",
        "end": END,
    })

    return sg


# ============================================================
# Finalization Helper
# ============================================================

def _finalize_coordinator(
    session_id: str,
    project_id: str,
    goals: List[str],
    corpus_summary: str,
    corpus_entities: List[str],
    turn_count: int,
):
    """Persist session memory, write SESSION_CONTEXT.md, yield coordinator_complete.

    This is a regular generator (not async) so callers use ``yield from``.
    """
    from app.services.discovery_session import (
        DiscoverySessionService,
        SessionMemoryService,
        SessionMemoryData,
        CorpusContext,
    )

    # Persist goals to database
    try:
        DiscoverySessionService.update_coordinator_goals(session_id, goals)
    except Exception as exc:
        logger.warning("Failed to persist coordinator goals to DB: %s", exc)

    # Save session memory + SESSION_CONTEXT.md
    try:
        memory_data = SessionMemoryData(
            session_id=session_id,
            initialized_at=datetime.utcnow().isoformat(),
            domain=_infer_domain_from_goals(goals),
            corpus_context=CorpusContext(
                entities=corpus_entities,
                document_ids=[],
                summary=corpus_summary[:500],
            ),
            research_goals=goals,
            constraints={},
            agents_completed=["coordinator"],
            current_stage="coordinator_complete",
            metadata={
                "project_id": project_id,
                "coordinator_turns": turn_count,
            },
        )

        SessionMemoryService.save_session_memory(session_id, memory_data)
        logger.info("Session memory saved for %s", session_id)
    except Exception as exc:
        logger.warning("Failed to save session memory: %s", exc)

    # Seed the living .md knowledge substrate from extracted goals.
    # These files are read by ALL agents at the start of every run.
    # The researcher can edit them between runs to steer the session.
    _seed_living_md_files(session_id, goals, corpus_summary)

    yield ("coordinator_complete", {
        "extracted_goals": goals,
        "summary": f"Research session configured with {len(goals)} goals.",
        "corpus_entities": corpus_entities,
        "corpus_summary": corpus_summary[:2000] if corpus_summary else "",
    })


# ============================================================
# Streaming Execution
# ============================================================

async def run_coordinator_streaming(
    session_id: str,
    project_id: str,
    user_message: Optional[str],
    llm_service: DiscoveryLLMService,
    retrieval_service: Any,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncGenerator[Tuple[str, dict], None]:
    """Stream coordinator events. Yields (event_type, event_data) tuples.

    Event types:
        coordinator_thinking: {"content": "Scanning corpus..."}
        coordinator_question: {"question": ..., "options": [...], ...}
        coordinator_complete: {"extracted_goals": [...], "summary": ...}
        error:                {"message": "..."}
    """
    from app.core.memory import get_memory_saver

    memory = await get_memory_saver()
    sg = _build_coordinator_graph(llm_service, retrieval_service)
    compiled = sg.compile(checkpointer=memory)

    thread_id = f"coordinator-{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # Track corpus data across nodes — stream_mode="updates" only yields
    # per-node diffs, so scan_corpus data is lost when analyze_and_ask completes.
    tracked_corpus_summary = ""
    tracked_corpus_entities: List[str] = []

    # Check if this is a resume (user answered a question) or initial trigger
    snapshot = await compiled.aget_state(config)
    is_resume = bool(user_message and snapshot.next)

    if is_resume:
        yield ("coordinator_thinking", {"content": "Processing your answer..."})
        input_value = Command(resume=user_message)
        # Pre-load corpus data from existing state for resumed sessions
        existing_state = snapshot.values or {}
        tracked_corpus_summary = existing_state.get("corpus_summary", "")
        tracked_corpus_entities = existing_state.get("corpus_entities", [])
    else:
        # BUGFIX: Emit a routing event so frontend runManager transitions past 'routing' state
        yield ("routing", {"brain": "discovery", "intent": "DISCOVERY"})
        yield ("coordinator_thinking", {"content": "Initializing Discovery Coordinator — scanning your research corpus for entities, targets, and prior findings..."})
        input_value = {
            "messages": [],
            "extracted_goals": [],
            "missing_context": [],
            "corpus_summary": "",
            "corpus_entities": [],
            "status": "scanning",
            "turn_count": 0,
            "max_turns": 5,
            "project_id": project_id,
            "session_id": session_id,
        }

    try:
        async for event in compiled.astream(input_value, config=config, stream_mode="updates"):
            if cancel_event and cancel_event.is_set():
                return

            # astream with stream_mode="updates" yields {node_name: state_update}
            for node_name, update in event.items():
                if not isinstance(update, dict):
                    continue

                if node_name == "scan_corpus":
                    # Capture corpus data for later use
                    tracked_corpus_summary = update.get("corpus_summary", "")
                    tracked_corpus_entities = update.get("corpus_entities", [])
                    entity_count = len(tracked_corpus_entities)
                    # Emit rich thinking with what was actually found
                    parts = [f"Corpus scan complete — found **{entity_count}** key entities."]
                    if tracked_corpus_entities:
                        parts.append(f"Top entities: {', '.join(tracked_corpus_entities[:6])}")
                    # Check if session .md notes were found
                    if "RESEARCHER NOTES" in tracked_corpus_summary:
                        parts.append("Read existing session notes (CONSTRAINTS.md, HYPOTHESES.md, etc.)")
                    parts.append("Analyzing gaps — preparing first question...")
                    yield ("coordinator_thinking", {
                        "content": "\n".join(parts)
                    })

                elif node_name == "analyze_and_ask":
                    # BUGFIX: Removed duplicate _finalize_coordinator yield here.
                    # It will be handled exclusively in the final_snapshot block below
                    # when the graph execution fully completes.
                    pass

    except Exception as exc:
        logger.exception("Coordinator streaming failed")
        yield ("error", {"message": str(exc)})
        return

    # After stream completes, check if graph is paused at an interrupt.
    # Use the FULL accumulated state snapshot (not per-node updates).
    try:
        final_snapshot = await compiled.aget_state(config)

        # If the graph completed (no next nodes), check for completion
        if not final_snapshot.next:
            final_state = final_snapshot.values or {}
            if final_state.get("status") == "complete":
                goals = final_state.get("extracted_goals", [])
                corpus_summary = final_state.get("corpus_summary", "") or tracked_corpus_summary
                corpus_entities = final_state.get("corpus_entities", []) or tracked_corpus_entities

                yield ("coordinator_thinking", {
                    "content": f"Session complete — extracted {len(goals)} research goals. Writing session files..."
                })

                for evt in _finalize_coordinator(
                    session_id, project_id, goals,
                    corpus_summary, corpus_entities,
                    final_state.get("turn_count", 0),
                ):
                    yield evt
            return

        # Graph is paused — extract interrupt payload from tasks
        for task in (final_snapshot.tasks or []):
            if hasattr(task, "interrupts") and task.interrupts:
                question_data = task.interrupts[0].value
                if isinstance(question_data, dict):
                    yield ("coordinator_question", question_data)
                return

    except Exception as exc:
        logger.warning("Failed to read coordinator snapshot: %s", exc)
        yield ("error", {"message": f"State read error: {exc}"})
