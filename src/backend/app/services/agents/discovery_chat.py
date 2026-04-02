"""Unified Discovery Chat — single conversational endpoint for the entire session lifecycle.

Routes internally based on session state:
  1. Setup phase   → Coordinator LangGraph (interrupt/resume HITL)
  2. Ready phase   → Free-form Q&A (DeepSeek + full session context)
  3. Plan phase    → Generates structured execution plan from build_pipeline()
  4. Execute phase → Runs deterministic plugin pipeline with per-stage SSE
  5. Analyze phase → Post-pipeline AI analysis + recommendations

All events flow through a single SSE stream so the frontend renders everything
inline in one chat conversation.
"""
import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.discovery_llm import DiscoveryLLMService
from app.services.discovery_session import (
    SessionMemoryService,
    SessionMemoryData,
)
from app.services.agents.coordinator import (
    run_coordinator_streaming,
    _read_session_notes,
)

logger = logging.getLogger(__name__)


# ============================================================
# Stage Thinking Text — shown inline as each tool executes
# ============================================================

STAGE_THINKING: Dict[str, str] = {
    "standardize_smiles": (
        "Canonicalizing SMILES via RDKit, computing InChIKeys, deduplicating by structural identity..."
    ),
    "predict_properties": (
        "Computing MW, LogP, TPSA, HBD/HBA, QED drug-likeness, Lipinski Ro5 via RDKit descriptors..."
    ),
    "check_toxicity": (
        "Screening PAINS patterns (Pan-Assay Interference Compounds), Brenk structural alerts, "
        "reactive groups: Michael acceptors, epoxides, acyl halides, nitroaromatics, polyhalogenated carbons..."
    ),
    "score_synthesizability": (
        "Computing SA scores (1 = trivial single-step, 10 = complex multi-step synthesis). "
        "Using RDKit SA_Score fragment contributions where available, else SMILES-length heuristic..."
    ),
    "predict_admet": (
        "Running ADMET screening: hERG channel liability (cardiotoxicity), drug-induced liver injury (DILI), "
        "Caco-2 passive membrane permeability, CYP3A4 metabolic inhibition..."
    ),
    "plan_synthesis": (
        "Running retrosynthetic disconnection against USPTO reaction templates. "
        "Searching for commercial building blocks and known reaction precedents..."
    ),
    "evaluate_strategy": (
        "Ranking retrosynthetic routes by step count, yield confidence, and commercial availability of intermediates..."
    ),
    "propose_candidates": (
        "Querying corpus context and research goals to propose de novo candidate molecules..."
    ),
    "enumerate_fragments": (
        "Generating combinatorial library by assembling core scaffold with curated fragment substituents..."
    ),
}


def _compute_stage_stats(plugin: str, result: dict) -> dict:
    """Extract human-readable per-stage statistics from a plugin result dict.

    These stats are included in tool_complete events so the frontend can render
    a compact summary of what each tool found — like Claude Code's tool output blocks.
    """
    stats: dict = {}
    try:
        if plugin == "standardize_smiles":
            mols = result.get("molecules", [])
            valid = sum(1 for m in mols if m.get("valid"))
            stats = {"valid": valid, "dropped": len(mols) - valid, "total": len(mols)}

        elif plugin == "predict_properties":
            props = [p for p in result.get("properties", []) if p.get("valid")]
            mws = [p["MolWt"] for p in props if p.get("MolWt") is not None]
            logps = [p["LogP"] for p in props if p.get("LogP") is not None]
            lipo = sum(1 for p in props if p.get("Lipinski"))
            stats = {
                "total": len(props),
                "mw_range": f"{min(mws):.0f}–{max(mws):.0f} Da" if mws else "N/A",
                "logp_range": f"{min(logps):.2f}–{max(logps):.2f}" if logps else "N/A",
                "lipinski_pass": lipo,
            }

        elif plugin == "check_toxicity":
            tox = result.get("toxicity_results", [])
            clean = sum(1 for t in tox if t.get("clean"))
            alerts: Dict[str, int] = {}
            for t in tox:
                for a in t.get("structural_alerts", []):
                    name = a.get("name", "unknown")
                    alerts[name] = alerts.get(name, 0) + 1
            stats = {
                "clean": clean,
                "flagged": len(tox) - clean,
                "total": len(tox),
                "top_alerts": sorted(alerts.items(), key=lambda x: -x[1])[:4],
            }

        elif plugin == "score_synthesizability":
            scores = result.get("scores", [])
            feasible = sum(1 for s in scores if s.get("feasible"))
            sa_vals = [s["sa_score"] for s in scores if s.get("sa_score") is not None]
            stats = {
                "feasible": feasible,
                "infeasible": len(scores) - feasible,
                "best_sa": round(min(sa_vals), 2) if sa_vals else None,
                "worst_sa": round(max(sa_vals), 2) if sa_vals else None,
                "total": len(scores),
            }

        elif plugin == "predict_admet":
            preds = result.get("predictions", [])
            risk_counts: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
            for p in preds:
                r = (p.get("overall_risk") or "MEDIUM").upper()
                risk_counts[r] = risk_counts.get(r, 0) + 1
            stats = {
                "total": len(preds),
                "low_risk": risk_counts.get("LOW", 0),
                "medium_risk": risk_counts.get("MEDIUM", 0),
                "high_risk": risk_counts.get("HIGH", 0),
            }

        elif plugin == "enumerate_fragments":
            mols = result.get("smiles_list", result.get("molecules", []))
            stats = {"generated": len(mols)}

    except Exception as exc:
        logger.debug("Stage stats computation failed for %s: %s", plugin, exc)
    return stats


def _rank_candidates(candidates: List[dict]) -> List[dict]:
    """Score and rank candidates by composite score: QED × tox_safety × SA_feasibility × ADMET.

    Returns candidates sorted best-first with _composite_score attached.
    Lower score = filtered out. Used to select top-N for display, bypassing
    LLM hallucination of SMILES strings.
    """
    def _score(c: dict) -> float:
        props = c.get("properties", {}) or {}
        tox = c.get("toxicity") or {}
        sa = float(c.get("sa_score") or 5.0)
        admet = c.get("admet") or {}

        qed = float(props.get("QED") or 0.5)
        tox_ok = 1.0 if tox.get("clean", True) else 0.5
        # SA: 1 is easiest, 10 is hardest. Map to 0–1.
        sa_norm = max(0.05, 1.0 - (sa - 1.0) / 9.0)
        overall = (admet.get("overall") or "MEDIUM").upper()
        admet_mult = 1.0 if overall == "LOW" else (0.8 if overall == "MEDIUM" else 0.4)
        return qed * tox_ok * sa_norm * admet_mult

    ranked = sorted(candidates, key=_score, reverse=True)
    for c in ranked:
        c["_composite_score"] = round(_score(c), 4)
    return ranked


# ============================================================
# Session Stage Resolution
# ============================================================

def _resolve_session_stage(session_id: str) -> str:
    """Determine current session stage from session_memory.json.

    Returns one of: "setup", "ready", "executing", "complete"
    """
    memory = SessionMemoryService.load_session_memory(session_id)
    if memory is None:
        return "setup"

    stage = memory.current_stage or ""

    if stage in ("coordinator_complete", "ready", "plan_accepted"):
        return "ready"
    if stage.startswith("pipeline_"):
        return "executing"
    if stage == "analysis_complete":
        return "complete"

    if "coordinator" in memory.agents_completed:
        return "ready"

    return "setup"


def _detect_execution_intent(message: str) -> bool:
    """Check if the user message signals they want to run the pipeline."""
    if not message:
        return False
    msg = message.lower().strip()
    triggers = [
        "run", "execute", "start", "go", "analyze", "begin",
        "do it", "let's go", "proceed", "launch", "run pipeline",
        "run analysis", "start pipeline", "run the pipeline",
        "accept", "approve", "looks good", "let's run",
    ]
    return any(t in msg for t in triggers)


# ============================================================
# Free-form Q&A (DeepSeek with session context)
# ============================================================

FREEFORM_SYSTEM_PROMPT = """You are a research assistant embedded in a drug discovery / scientific research platform called Atlas.
You have access to the researcher's session context, goals, constraints, and prior findings.
Answer questions thoughtfully, referencing the session data when relevant.
Be concise but thorough. If the researcher asks about running analysis or executing tools,
tell them you can generate an execution plan — ask them to confirm when ready.
When you identify that the user wants to execute/run something, respond with the key phrase
"I can generate an execution plan" so the system knows to offer the plan option."""


async def _run_freeform_qa(
    session_id: str,
    message: str,
    llm_service: DiscoveryLLMService,
) -> AsyncGenerator[Tuple[str, dict], None]:
    """Handle free-form Q&A after coordinator setup is complete."""
    memory = SessionMemoryService.load_session_memory(session_id)
    session_notes = _read_session_notes(session_id, max_chars=3000)

    context_parts = []
    if memory:
        if memory.research_goals:
            context_parts.append("## Research Goals\n" + "\n".join(f"- {g}" for g in memory.research_goals))
        if memory.domain:
            context_parts.append(f"## Domain\n{memory.domain}")
        if memory.corpus_context and memory.corpus_context.summary:
            context_parts.append(f"## Corpus Summary\n{memory.corpus_context.summary[:500]}")
    if session_notes:
        context_parts.append(f"## Session Notes\n{session_notes}")

    context_block = "\n\n".join(context_parts) if context_parts else "(No session context available)"

    prompt = f"""=== SESSION CONTEXT ===
{context_block}

=== USER MESSAGE ===
{message}

Respond helpfully. If the user wants to run tools or analyze compounds, suggest generating an execution plan."""

    yield ("thinking", {"content": "Thinking..."})

    try:
        response = await llm_service.orchestrate(
            prompt=prompt,
            system_prompt=FREEFORM_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=2048,
        )
        yield ("message", {"content": response or "I couldn't generate a response. Please try again."})
    except Exception as exc:
        logger.error("Free-form Q&A failed: %s", exc)
        yield ("error", {"message": f"Failed to generate response: {exc}"})


# ============================================================
# Plan Generation
# ============================================================

PLAN_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "One-sentence summary of what this plan will do"},
        "reasoning": {"type": "string", "description": "Why these stages were selected"},
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Potential issues or limitations"
        },
        "molecule_notes": {"type": "string", "description": "Notes about the input molecules"},
    },
    "required": ["summary", "reasoning", "warnings", "molecule_notes"],
}


async def _generate_plan(
    session_id: str,
    llm_service: DiscoveryLLMService,
    smiles_list: Optional[List[str]] = None,
) -> AsyncGenerator[Tuple[str, dict], None]:
    """Generate a structured execution plan and yield it as a plan_proposed event."""
    from app.services.agents.pipeline_planner import build_pipeline, PipelinePlan
    from app.services.plugins import get_plugin_manager

    yield ("thinking", {"content": "Building execution plan..."})

    memory = SessionMemoryService.load_session_memory(session_id)
    if memory is None:
        yield ("error", {"message": "Session memory not found. Complete the setup conversation first."})
        return

    memory_dict = memory.model_dump()
    pm = get_plugin_manager()
    available = pm.get_registered_names()

    plan = build_pipeline(memory_dict, available)

    if not plan.stages:
        yield ("error", {"message": "No pipeline stages could be selected. Check that plugins are available."})
        return

    if not smiles_list:
        from app.api.routes import _extract_smiles_from_corpus
        smiles_list = _extract_smiles_from_corpus(session_id)

    unavailable_plugins = []
    for name in available:
        info = pm.get_plugin_info(name) if hasattr(pm, 'get_plugin_info') else None
        if info and not info.get('available', True):
            unavailable_plugins.append(name)

    try:
        goals_text = "\n".join(f"- {g}" for g in memory.research_goals) if memory.research_goals else "None"
        ai_analysis = await llm_service.orchestrate_constrained(
            prompt=f"""Analyze this pipeline plan for a scientific discovery session.

Research Goals:
{goals_text}

Pipeline Stages:
{chr(10).join(f"{s.stage_id}. {s.plugin}: {s.description}" for s in plan.stages)}

Input Molecules: {len(smiles_list)} compounds
Iteration: {plan.iteration}
Prior Findings: {plan.prior_findings_digest[:300] if plan.prior_findings_digest else 'None (first run)'}
Unavailable Plugins: {', '.join(unavailable_plugins) if unavailable_plugins else 'None'}

Provide a brief summary, reasoning, any warnings, and notes about molecules.""",
            schema=PLAN_ANALYSIS_SCHEMA,
            system_prompt="You are a scientific pipeline analyst. Be concise and specific.",
            temperature=0.3,
            max_tokens=512,
        )
    except Exception as exc:
        logger.warning("Plan AI analysis failed, using defaults: %s", exc)
        ai_analysis = {
            "summary": f"Run {len(plan.stages)} analysis stages on {len(smiles_list)} molecules",
            "reasoning": plan.reasoning,
            "warnings": [],
            "molecule_notes": f"{len(smiles_list)} molecules queued for analysis",
        }

    raw_warn = ai_analysis.get("warnings", [])
    if isinstance(raw_warn, str):
        warnings = [raw_warn] if raw_warn.strip() else []
    elif isinstance(raw_warn, list):
        warnings = list(raw_warn)
    else:
        warnings = []

    if unavailable_plugins:
        warnings.append(f"Unavailable plugins (will use heuristics): {', '.join(unavailable_plugins)}")
    candidates_source = "provided"
    if not smiles_list:
        candidates_source = "llm_proposed"
        warnings.append(
            "No SMILES found in corpus — AI will propose candidate molecules from research goals at execution time."
        )

    plan_payload = {
        "plan_id": plan.plan_id,
        "summary": ai_analysis.get("summary", ""),
        "reasoning": ai_analysis.get("reasoning", plan.reasoning),
        "molecule_notes": ai_analysis.get("molecule_notes", ""),
        "molecule_count": len(smiles_list) if smiles_list else 0,
        "candidates_source": candidates_source,
        "iteration": plan.iteration,
        "estimated_total_seconds": sum(s.estimated_seconds for s in plan.stages),
        "warnings": warnings,
        "stages": [
            {
                "stage_id": s.stage_id,
                "plugin": s.plugin,
                "description": s.description,
                "estimated_seconds": s.estimated_seconds,
            }
            for s in plan.stages
        ],
        "is_demo_data": False,
    }

    yield ("plan_proposed", plan_payload)


# ============================================================
# Pipeline Execution (inline tool cards)
# ============================================================

async def _execute_pipeline(
    session_id: str,
    smiles_list: Optional[List[str]] = None,
    llm_service: Any = None,
) -> AsyncGenerator[Tuple[str, dict], None]:
    """Execute the deterministic plugin pipeline, yielding tool_start/tool_complete events."""
    from app.services.agents.pipeline_planner import build_pipeline
    from app.services.plugins import get_plugin_manager
    from app.api.routes import (
        _extract_smiles_from_corpus,
        _propose_candidates_from_memory,
        _build_candidates_for_frontend,
        _save_pipeline_results,
    )

    memory = SessionMemoryService.load_session_memory(session_id)
    if memory is None:
        yield ("error", {"message": "Session memory not found."})
        return

    SessionMemoryService.update_session_memory(session_id, {"current_stage": "pipeline_running"})

    memory_dict = memory.model_dump()
    pm = get_plugin_manager()
    available = pm.get_registered_names()
    plan = build_pipeline(memory_dict, available)

    if not smiles_list:
        smiles_list = _extract_smiles_from_corpus(session_id)
        source_label = "corpus_extracted"
    else:
        source_label = "provided"

    if not smiles_list:
        # No SMILES in corpus — run explicit LLM candidate proposal as first pipeline stage
        if llm_service is None:
            yield ("error", {
                "message": (
                    "No SMILES in corpus and no LLM service available. "
                    "Provide SMILES via smiles_list or upload documents containing SMILES strings."
                )
            })
            return
        yield ("tool_start", {
            "stage_id": 0,
            "plugin": "propose_candidates",
            "description": "Proposing candidate molecules from corpus knowledge and research goals...",
            "total_stages": len(plan.stages) + 1,
        })
        smiles_list = await _propose_candidates_from_memory(session_id, llm_service)
        source_label = "llm_proposed"
        if not smiles_list:
            yield ("error", {
                "message": (
                    "Could not propose candidates: session has no research goals or corpus context. "
                    "Complete the setup conversation first."
                )
            })
            return
        yield ("tool_complete", {
            "stage_id": 0,
            "plugin": "propose_candidates",
            "summary": f"Proposed {len(smiles_list)} candidate molecules from corpus knowledge.",
            "total_stages": len(plan.stages) + 1,
            "candidates_so_far": len(smiles_list),
        })

    initial_molecules = [{"smiles": s, "source": source_label, "valid": True} for s in smiles_list]
    stage_data: dict = {
        "smiles_list": smiles_list,
        "molecules": initial_molecules
    }
    stage_results = []
    total_stages = len(plan.stages)

    for stage in plan.stages:
        thinking_text = STAGE_THINKING.get(stage.plugin, f"Running {stage.plugin}...")
        yield ("tool_start", {
            "stage_id": stage.stage_id,
            "plugin": stage.plugin,
            "description": stage.description,
            "thinking": thinking_text,
            "total_stages": total_stages,
        })

        try:
            merged_kwargs = {**stage.config, **stage_data}
            result = await pm.invoke(stage.plugin, **merged_kwargs)

            if isinstance(result, dict):
                if "molecules" in result:
                    stage_data["molecules"] = result["molecules"]
                    stage_data["smiles_list"] = [
                        m.get("smiles", "") for m in result["molecules"]
                        if isinstance(m, dict) and m.get("valid") is not False and m.get("smiles")
                    ]
                if "predictions" in result:
                    stage_data["predictions"] = result["predictions"]
                if "scores" in result:
                    stage_data["scores"] = result["scores"]
                if "properties" in result:
                    stage_data["properties"] = result["properties"]
                if "toxicity_results" in result:
                    stage_data["toxicity_results"] = result["toxicity_results"]

            summary = result.get("summary", f"Stage {stage.stage_id} complete") if isinstance(result, dict) else str(result)
            stats = _compute_stage_stats(stage.plugin, result if isinstance(result, dict) else {})
            stage_results.append({
                "stage": stage.stage_id,
                "plugin": stage.plugin,
                "result": result,
                "summary": summary,
                "stats": stats,
            })

            candidates = _build_candidates_for_frontend(stage_data)

            yield ("tool_complete", {
                "stage_id": stage.stage_id,
                "plugin": stage.plugin,
                "summary": summary,
                "stats": stats,
                "total_stages": total_stages,
                "candidates_so_far": len(candidates),
            })

        except Exception as exc:
            logger.error("Pipeline stage %s failed: %s", stage.plugin, exc, exc_info=True)
            stage_results.append({
                "stage": stage.stage_id,
                "plugin": stage.plugin,
                "error": str(exc),
            })
            yield ("tool_complete", {
                "stage_id": stage.stage_id,
                "plugin": stage.plugin,
                "summary": f"ERROR: {exc}",
                "stats": {},
                "total_stages": total_stages,
                "error": True,
            })

    _save_pipeline_results(session_id, plan, stage_results, stage_data)

    final_candidates = _build_candidates_for_frontend(stage_data)
    SessionMemoryService.update_session_memory(session_id, {"current_stage": "pipeline_complete"})

    yield ("pipeline_complete", {
        "stages_completed": len(stage_results),
        "total_stages": total_stages,
        "candidates": final_candidates,
        "stage_results": [
            {k: v for k, v in sr.items() if k != "result"}
            for sr in stage_results
        ],
    })


# ============================================================
# Post-Pipeline Analysis
# ============================================================

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "3-6 specific findings. Each MUST cite the tool by name "
                "(e.g., 'check_toxicity flagged 3/20 with PAINS', 'score_synthesizability: "
                "best SA=2.1 for ...', 'predict_admet: 15 low-risk, 5 medium-risk'). "
                "Give concrete numbers. Do NOT use vague language like 'the analysis shows'."
            ),
        },
        "concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific problems: liability patterns, property cliffs, coverage gaps. "
                "Reference which tool surfaced each concern."
            ),
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["wetlab_validation", "additional_modeling", "iterate_constraints", "manual_review"],
                    },
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
            "description": "Concrete next steps tied to findings above",
        },
        "missing_capabilities": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific tools not available that would change conclusions: "
                "docking, MD, QSAR model, target-specific assay, etc."
            ),
        },
    },
    "required": ["key_findings", "concerns", "recommendations", "missing_capabilities"],
}


async def _analyze_results(
    session_id: str,
    pipeline_results: dict,
    llm_service: DiscoveryLLMService,
) -> AsyncGenerator[Tuple[str, dict], None]:
    """Post-pipeline AI analysis — tool-attributed findings, recommendations, ranked candidates."""
    yield ("thinking", {"content": "Ranking candidates and building analysis..."})

    memory = SessionMemoryService.load_session_memory(session_id)
    goals_text = ""
    if memory and memory.research_goals:
        goals_text = "\n".join(f"- {g}" for g in memory.research_goals)

    candidates = pipeline_results.get("candidates", [])
    stage_results = pipeline_results.get("stage_results", [])

    # Deterministic ranking — avoids LLM hallucinating empty SMILES
    ranked = _rank_candidates(list(candidates))
    top_n = min(max(3, len(ranked) // 4), 8)  # dynamic: 3–8 depending on set size
    top_candidates = [
        {
            "smiles": c.get("smiles", ""),
            "reasoning": _describe_candidate(c),
            "composite_score": c.get("_composite_score", 0.0),
        }
        for c in ranked[:top_n]
        if c.get("smiles")
    ]

    # Build stages text with stats for context
    stages_text_parts = []
    for sr in stage_results:
        plugin = sr.get("plugin", "?")
        summary = sr.get("summary", "done")
        stats = sr.get("stats", {})
        line = f"- {plugin}: {summary}"
        if stats:
            stat_strs = []
            if "valid" in stats:
                stat_strs.append(f"{stats['valid']} valid / {stats.get('total', '?')} total")
            if "clean" in stats:
                stat_strs.append(f"{stats['clean']} clean / {stats.get('total', '?')} screened")
            if "top_alerts" in stats and stats["top_alerts"]:
                alert_str = ", ".join(f"{n}({c})" for n, c in stats["top_alerts"])
                stat_strs.append(f"alerts: {alert_str}")
            if "feasible" in stats:
                stat_strs.append(f"{stats['feasible']}/{stats.get('total', '?')} SA-feasible")
            if "low_risk" in stats:
                stat_strs.append(
                    f"ADMET: {stats['low_risk']} low / {stats['medium_risk']} med / {stats['high_risk']} high risk"
                )
            if stat_strs:
                line += f" [{'; '.join(stat_strs)}]"
        stages_text_parts.append(line)
    stages_text = "\n".join(stages_text_parts)

    # Build candidate summary with all properties — use None-safe formatting
    def _fmt(val, unit="", decimals=1) -> str:
        if val is None:
            return "N/A"
        try:
            return f"{float(val):.{decimals}f}{unit}"
        except Exception:
            return str(val)

    candidates_summary = []
    for c in ranked[:12]:
        props = c.get("properties", {}) or {}
        tox = c.get("toxicity") or {}
        admet = c.get("admet") or {}
        sa = c.get("sa_score")
        alert_count = tox.get("alert_count", 0)
        safety_str = "clean" if tox.get("clean", True) else f"flagged({alert_count} alerts)"
        parts = [
            f"SMILES: {c.get('smiles', '?')}",
            f"MW={_fmt(props.get('MolWt'), ' Da', 0)}",
            f"LogP={_fmt(props.get('LogP'), decimals=2)}",
            f"TPSA={_fmt(props.get('TPSA'), decimals=0)}",
            f"QED={_fmt(props.get('QED'), decimals=3)}",
            f"SA={_fmt(sa, decimals=2)}",
            f"Safety={safety_str}",
            f"ADMET={admet.get('overall', 'N/A')}",
            f"score={_fmt(c.get('_composite_score'), decimals=3)}",
        ]
        candidates_summary.append(" | ".join(parts))

    prompt = f"""You are reviewing computational chemistry pipeline results for a scientific discovery session.
Your job is to write specific, scientist-level findings — not generic advice.

## Research Goals
{goals_text or 'Not specified'}

## Tools That Ran (with statistics)
{stages_text or 'No stages ran'}

## Candidates Ranked by Composite Score (QED × safety × SA feasibility × ADMET)
{chr(10).join(candidates_summary) if candidates_summary else 'No candidates generated'}
Total: {len(candidates)} candidates

## Instructions
- Each key_finding MUST name the specific tool that produced the data
- Cite actual numbers from the statistics above
- Connect findings to the research goals where possible
- For concerns: be specific about which molecules or patterns are problematic
- For recommendations: tie each action to a specific finding
- Do NOT repeat generic drug discovery boilerplate"""

    try:
        analysis = await llm_service.orchestrate_constrained(
            prompt=prompt,
            schema=ANALYSIS_SCHEMA,
            system_prompt=(
                "You are a senior medicinal chemist and cheminformatics scientist. "
                "Write terse, data-grounded observations. Name tools. Cite numbers. "
                "Never write 'the analysis shows' or 'it is important to note'."
            ),
            temperature=0.3,
            max_tokens=1800,
        )

        # Inject deterministically-ranked top candidates (LLM doesn't pick SMILES)
        analysis["top_candidates"] = top_candidates
        analysis.setdefault("key_findings", [])
        analysis.setdefault("concerns", [])
        analysis.setdefault("recommendations", [])
        analysis.setdefault("missing_capabilities", [])

        yield ("analysis", analysis)

        recs = analysis.get("recommendations", [])
        if recs:
            yield ("recommendation", {"recommendations": recs})

        missing = analysis.get("missing_capabilities", [])
        if missing:
            yield ("recommendation", {"missing_capabilities": missing})

        SessionMemoryService.update_session_memory(session_id, {"current_stage": "analysis_complete"})

    except Exception as exc:
        logger.error("Post-pipeline analysis failed: %s", exc)
        # Fallback: still rank deterministically
        yield ("analysis", {
            "key_findings": [
                f"Pipeline ran {len(stage_results)} stage(s) on {len(candidates)} candidates.",
                "AI analysis failed — review the candidates table manually.",
            ],
            "top_candidates": top_candidates,
            "concerns": [f"Analysis error: {exc}"],
            "recommendations": [
                {"action": "manual_review", "description": "Review the ranked candidates table", "priority": "high"}
            ],
            "missing_capabilities": [],
        })


def _describe_candidate(c: dict) -> str:
    """Build a one-line description of a candidate from its pipeline data."""
    props = c.get("properties", {}) or {}
    tox = c.get("toxicity") or {}
    admet = c.get("admet") or {}
    sa = c.get("sa_score")
    parts = []

    mw = props.get("MolWt")
    if mw is not None:
        parts.append(f"MW {mw:.0f} Da")

    logp = props.get("LogP")
    if logp is not None:
        parts.append(f"LogP {logp:.2f}")

    if sa is not None:
        parts.append(f"SA {sa:.1f}")

    if tox:
        parts.append("clean" if tox.get("clean", True) else f"{tox.get('alert_count', 0)} alerts")

    overall = (admet.get("overall") or "").upper()
    if overall:
        parts.append(f"ADMET {overall}")

    score = c.get("_composite_score")
    if score is not None:
        parts.append(f"score {score:.3f}")

    return " | ".join(parts) if parts else "No property data available"


# ============================================================
# Unified Chat Orchestrator
# ============================================================

async def run_discovery_chat(
    session_id: str,
    project_id: str,
    user_message: Optional[str],
    llm_service: DiscoveryLLMService,
    retrieval_service: Any,
    action: Optional[str] = None,
    smiles_list: Optional[List[str]] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> AsyncGenerator[Tuple[str, dict], None]:
    """Unified discovery chat — single entry point for all session interactions.

    Args:
        session_id: Discovery session ID
        project_id: Project ID for corpus queries
        user_message: User's chat message (None for bootstrap/action-only)
        llm_service: Isolated Discovery LLM service
        retrieval_service: RAG retrieval service for corpus queries
        action: Optional action override: "accept_plan", "reject_plan", "generate_plan"
        smiles_list: Optional explicit SMILES for pipeline
        cancel_event: Cancellation signal

    Yields:
        (event_type, event_data) tuples for SSE streaming
    """
    stage = _resolve_session_stage(session_id)

    yield ("session_update", {"stage": stage, "session_id": session_id})

    # --- Action overrides (plan accept/reject) ---
    if action == "accept_plan":
        yield ("thinking", {"content": "Plan accepted — starting pipeline execution..."})
        pipeline_output: dict = {}
        async for evt in _execute_pipeline(session_id, smiles_list, llm_service):
            if cancel_event and cancel_event.is_set():
                return
            yield evt
            if evt[0] == "pipeline_complete":
                pipeline_output = evt[1]
        async for evt in _analyze_results(session_id, pipeline_output, llm_service):
            if cancel_event and cancel_event.is_set():
                return
            yield evt
        return

    if action == "reject_plan":
        yield ("message", {"content": "Plan rejected. Tell me what you'd like to change, or ask me anything about the session."})
        return

    if action == "generate_plan":
        async for evt in _generate_plan(session_id, llm_service, smiles_list):
            if cancel_event and cancel_event.is_set():
                return
            yield evt
        return

    # --- Stage-based routing ---
    if stage == "setup":
        async for evt in run_coordinator_streaming(
            session_id=session_id,
            project_id=project_id,
            user_message=user_message,
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            cancel_event=cancel_event,
        ):
            if cancel_event and cancel_event.is_set():
                return
            yield evt

            if evt[0] == "coordinator_complete":
                yield ("session_update", {"stage": "ready", "session_id": session_id})
                yield ("message", {
                    "content": (
                        "Setup complete. I now have a clear picture of your research goals. "
                        "You can:\n"
                        "- **Ask questions** about the session, corpus, or methodology\n"
                        "- **Add context** — tell me anything I missed\n"
                        "- **Run the pipeline** — say \"run\" or \"let's go\" when you're ready\n\n"
                        "What would you like to do?"
                    )
                })
        return

    # Ready stage: free-form or plan generation
    if stage == "ready":
        if user_message and _detect_execution_intent(user_message):
            async for evt in _generate_plan(session_id, llm_service, smiles_list):
                if cancel_event and cancel_event.is_set():
                    return
                yield evt
            return

        if user_message:
            async for evt in _run_freeform_qa(session_id, user_message, llm_service):
                if cancel_event and cancel_event.is_set():
                    return
                yield evt
            return

        yield ("message", {
            "content": (
                "Session is ready. Ask me anything about your research, "
                "or say **\"run\"** to generate an execution plan."
            )
        })
        return

    # Completed / post-analysis: allow continued conversation
    if stage in ("complete", "executing"):
        if user_message and _detect_execution_intent(user_message):
            async for evt in _generate_plan(session_id, llm_service, smiles_list):
                if cancel_event and cancel_event.is_set():
                    return
                yield evt
            return

        if user_message:
            async for evt in _run_freeform_qa(session_id, user_message, llm_service):
                if cancel_event and cancel_event.is_set():
                    return
                yield evt
            return

        yield ("message", {"content": "What would you like to do next? You can ask questions or run another iteration."})
