"""Pipeline Planner — deterministic plugin selection and ordering.

Replaces the old Executor's LLM-based script generation with a rule-based
pipeline that selects registered plugins based on session_memory.

The planner reads the living .md knowledge substrate (FINDINGS.md, session_memory)
before each run so that iterations compound — each run is informed by all prior runs.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

from app.services.plugins import get_plugin_manager

logger = logging.getLogger(__name__)


# Canonical pipeline order: generate (optional) → standardize → filter → score → predict → rank
# Plugins not in available_plugins are skipped automatically.
_CANONICAL_ORDER: List[str] = [
    "enumerate_fragments",    # Stage 0: combinatorial generation (only when scaffold goals detected)
    "standardize_smiles",     # Canonicalize + deduplicate
    "predict_properties",     # MW, LogP, TPSA, etc.
    "check_toxicity",         # PAINS/Brenk alerts
    "score_synthesizability", # SA score
    "predict_admet",          # ADMET prediction
    "plan_synthesis",         # Retrosynthesis
    "evaluate_strategy",      # Final ranking
    "verify_spectrum",        # NMR — only when goals mention spectrum/NMR
]

# Maps goal keywords to SCAFFOLD_PRESETS keys in fragment_enumeration.py
_SCAFFOLD_KEYWORD_MAP: Dict[str, str] = {
    "xyloside": "xyloside",
    "xylopyranoside": "xyloside",
    "glycoside": "xyloside",
    "gag priming": "xyloside",
    "glycosaminoglycan": "xyloside",
}


@dataclass
class PipelineStage:
    stage_id: int
    name: str
    plugin: str  # must match a registered plugin name
    config: dict  # passed to plugin.execute()
    depends_on: list  # stage_ids that must complete first
    estimated_seconds: int
    description: str


@dataclass
class PipelinePlan:
    plan_id: str
    session_id: str
    stages: List[PipelineStage]
    created_at: str
    reasoning: str  # explanation of why these stages were selected
    iteration: int = 1  # which run this is (1-indexed, read from session_memory)
    prior_findings_digest: str = ""  # summary of what prior runs discovered


def _extract_session_memory_fields(session_memory: Dict[str, Any]) -> tuple:
    """Extract key fields from session_memory, handling missing/empty data."""
    if not session_memory or not isinstance(session_memory, dict):
        return "", [], {}, None

    session_id = session_memory.get("session_id") or ""
    research_goals = session_memory.get("research_goals") or []
    if not isinstance(research_goals, list):
        research_goals = []

    constraints = session_memory.get("constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}

    domain = session_memory.get("domain")

    return session_id, research_goals, constraints, domain


def _read_prior_findings(session_id: str) -> str:
    """Read the tail of FINDINGS.md to understand what prior runs discovered.

    This is injected into the plan reasoning so each iteration is aware of
    previous results — the compounding loop.
    """
    if not session_id:
        return ""
    try:
        from app.core.config import settings
        findings_path = Path(settings.DATA_DIR) / "discovery" / session_id / "FINDINGS.md"
        if not findings_path.exists():
            return ""
        text = findings_path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 1500:
            text = text[-1500:]
        return text.strip()
    except Exception:
        return ""


def _get_iteration_count(session_memory: Dict[str, Any]) -> int:
    """Get the current iteration number from session_memory metadata."""
    meta = session_memory.get("metadata", {})
    if isinstance(meta, dict):
        return meta.get("pipeline_iterations", 0) + 1
    return 1


def _detect_scaffold_preset(goals: List[str]) -> Optional[str]:
    """Return a named scaffold preset if research goals indicate fragment enumeration.

    Returns the preset key (e.g. 'xyloside') or None if no scaffold match found.
    Extend _SCAFFOLD_KEYWORD_MAP to support new domains without touching this logic.
    """
    goals_text = " ".join(str(g).lower() for g in goals)
    for keyword, preset_name in _SCAFFOLD_KEYWORD_MAP.items():
        if keyword in goals_text:
            return preset_name
    return None


def _goals_mention_spectrum(goals: List[str]) -> bool:
    """Check if research goals mention NMR or spectrum data."""
    keywords = ("nmr", "spectrum", "spectra", ".jdx", "jcamp", "1h-nmr", "13c-nmr")
    goals_text = " ".join(str(g).lower() for g in goals)
    return any(kw in goals_text for kw in keywords)


def _goals_mention_synthesis(goals: List[str]) -> bool:
    """Check if research goals mention synthesis or retrosynthesis."""
    keywords = ("synthesis", "synthesize", "retrosynthesis", "route", "synthetic")
    goals_text = " ".join(str(g).lower() for g in goals)
    return any(kw in goals_text for kw in keywords)


def _build_stage_config(plugin_name: str, constraints: Dict[str, Any]) -> dict:
    """Build config for a stage from session_memory constraints."""
    config: Dict[str, Any] = {}

    # Property constraints (MW, LogP, TPSA) for predict_properties / filtering
    prop_constraints = constraints.get("propertyConstraints") or []
    if isinstance(prop_constraints, list):
        for pc in prop_constraints:
            if isinstance(pc, dict):
                config[f"constraint_{pc.get('property', '')}"] = pc
            elif hasattr(pc, "model_dump"):
                config["property_constraints"] = prop_constraints
                break

    # Forbidden substructures for toxicity
    forbidden = constraints.get("forbiddenSubstructures") or []
    if forbidden:
        config["forbidden_substructures"] = forbidden

    # Domain-specific passthrough
    domain_specific = constraints.get("domainSpecificConstraints") or {}
    if domain_specific:
        config["domain_constraints"] = domain_specific

    return config


def _estimate_seconds(plugin_name: str) -> int:
    """Estimate execution time in seconds per stage."""
    estimates: Dict[str, int] = {
        "enumerate_fragments": 3,
        "standardize_smiles": 5,
        "predict_properties": 30,
        "check_toxicity": 15,
        "score_synthesizability": 20,
        "predict_admet": 60,
        "plan_synthesis": 120,
        "evaluate_strategy": 5,
        "verify_spectrum": 10,
    }
    return estimates.get(plugin_name, 30)


def _stage_description(plugin_name: str, config: Optional[dict] = None) -> str:
    """Human-readable description for a stage."""
    if plugin_name == "enumerate_fragments":
        preset = (config or {}).get("scaffold_name", "custom")
        return f"Generate candidate library by fragment enumeration (preset: {preset})"
    descriptions: Dict[str, str] = {
        "standardize_smiles": "Canonicalize SMILES, deduplicate by InChIKey",
        "predict_properties": "Compute MW, LogP, TPSA, Lipinski",
        "check_toxicity": "PAINS and structural alert screening",
        "score_synthesizability": "Synthetic accessibility score",
        "predict_admet": "ADMET property prediction",
        "plan_synthesis": "Retrosynthesis route planning",
        "evaluate_strategy": "Rank synthesis routes by feasibility",
        "verify_spectrum": "Verify NMR spectrum against structure",
    }
    return descriptions.get(plugin_name, f"Run {plugin_name}")


def build_pipeline(
    session_memory: Dict[str, Any],
    available_plugins: List[str],
) -> PipelinePlan:
    """Build a deterministic pipeline plan from session memory.

    Args:
        session_memory: Loaded session_memory.json contents
        available_plugins: List of registered plugin names from PluginManager

    Returns:
        PipelinePlan with ordered stages
    """
    session_id, research_goals, constraints, domain = _extract_session_memory_fields(
        session_memory
    )

    available_set = set(available_plugins or [])
    include_spectrum = _goals_mention_spectrum(research_goals)
    include_synthesis = _goals_mention_synthesis(research_goals)
    scaffold_preset = _detect_scaffold_preset(research_goals)

    stages: List[PipelineStage] = []
    reasoning_parts: List[str] = []

    # Build stages in canonical order
    prev_ids: List[int] = []
    stage_id = 0

    for plugin_name in _CANONICAL_ORDER:
        if plugin_name not in available_set:
            continue

        # enumerate_fragments: only include when a scaffold preset is detected from goals
        if plugin_name == "enumerate_fragments":
            if not scaffold_preset:
                reasoning_parts.append(
                    "Skipped enumerate_fragments (no scaffold keyword detected in goals)"
                )
                continue
            reasoning_parts.append(
                f"Including enumerate_fragments with preset '{scaffold_preset}' "
                "(scaffold keyword detected in goals)"
            )

        # Skip verify_spectrum if no NMR/spectrum mentioned in goals
        elif plugin_name == "verify_spectrum" and not include_spectrum:
            reasoning_parts.append("Skipped verify_spectrum (no NMR/spectrum in goals)")
            continue

        # Skip plan_synthesis and evaluate_strategy if no synthesis goals
        elif plugin_name in ("plan_synthesis", "evaluate_strategy") and not include_synthesis:
            reasoning_parts.append(
                f"Skipped {plugin_name} (no synthesis/retrosynthesis in goals)"
            )
            continue

        stage_id += 1
        config = _build_stage_config(plugin_name, constraints)

        # Inject scaffold_name for fragment enumeration stage
        if plugin_name == "enumerate_fragments" and scaffold_preset:
            config["scaffold_name"] = scaffold_preset

        stages.append(
            PipelineStage(
                stage_id=stage_id,
                name=plugin_name,
                plugin=plugin_name,
                config=config,
                depends_on=list(prev_ids),
                estimated_seconds=_estimate_seconds(plugin_name),
                description=_stage_description(plugin_name, config),
            )
        )
        prev_ids.append(stage_id)

    # Build reasoning
    if not stages:
        reasoning = (
            "No stages selected. Either no plugins are available or session_memory "
            "is empty. Available plugins: " + ", ".join(sorted(available_set))
        )
    else:
        stage_names = [s.plugin for s in stages]
        reasoning = (
            f"Selected {len(stages)} stages in canonical order: "
            f"{', '.join(stage_names)}. "
        )
        if constraints:
            reasoning += "Config built from session constraints. "
        if not include_spectrum and "verify_spectrum" in available_set:
            reasoning += "Skipped verify_spectrum (goals do not mention NMR/spectrum). "
        if not include_synthesis:
            reasoning += "Skipped synthesis stages (goals do not mention synthesis). "
        reasoning += " ".join(reasoning_parts)

    iteration = _get_iteration_count(session_memory)
    prior_findings = _read_prior_findings(session_id)

    if prior_findings:
        reasoning += f" This is iteration {iteration}. Prior findings available ({len(prior_findings)} chars)."
    else:
        reasoning += " This is the first run (no prior findings)."

    plan_id = str(uuid.uuid4())[:8]
    return PipelinePlan(
        plan_id=plan_id,
        session_id=session_id,
        stages=stages,
        created_at=datetime.utcnow().isoformat() + "Z",
        reasoning=reasoning.strip(),
        iteration=iteration,
        prior_findings_digest=prior_findings[-500:] if prior_findings else "",
    )


def _plugin_expects_single_smiles(plugin_name: str) -> bool:
    """Check if plugin expects a single smiles string (vs batch)."""
    single_smiles_plugins = {
        "predict_properties",
        "check_toxicity",
        "plan_synthesis",
    }
    return plugin_name in single_smiles_plugins


async def _run_stage_with_batching(
    pm: Any,
    stage: PipelineStage,
    current_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a stage, batching over molecules if plugin expects single smiles."""
    # evaluate_strategy expects aggregated routes from plan_synthesis
    if stage.plugin == "evaluate_strategy":
        routes = []
        for r in current_data.get("batch_results", []):
            res = r.get("result", {}) if isinstance(r, dict) else {}
            routes.extend(res.get("routes", []) or [])
        if not routes:
            return {
                "valid": False,
                "error": "No routes to evaluate from plan_synthesis.",
                "routes_evaluated": 0,
            }
        kwargs = {"routes": routes, **stage.config}
        return await pm.invoke(stage.plugin, **kwargs)

    molecules = current_data.get("molecules") or []
    # Fallback: if no molecules but smiles_list present, treat as raw molecules for batching
    if not molecules and current_data.get("smiles_list"):
        smiles_list = current_data["smiles_list"]
        if isinstance(smiles_list, list):
            molecules = [
                {"smiles": str(s).strip(), "valid": True}
                for s in smiles_list
                if s and str(s).strip()
            ]
    valid_molecules = [m for m in molecules if isinstance(m, dict) and m.get("valid")]

    if not valid_molecules or not _plugin_expects_single_smiles(stage.plugin):
        # No batching: pass data as-is
        kwargs = {**current_data, **stage.config}
        return await pm.invoke(stage.plugin, **kwargs)

    # Batch over valid molecules
    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for m in valid_molecules:
        smiles = m.get("smiles") or m.get("smiles_list", [""])[0]
        if not smiles:
            continue
        try:
            out = await pm.invoke(stage.plugin, smiles=smiles, **stage.config)
            results.append({"smiles": smiles, "result": out})
        except Exception as e:
            errors.append(f"{smiles}: {e}")
            logger.warning("Stage %s failed for %s: %s", stage.plugin, smiles, e)

    # Merge back into molecules for downstream stages
    result_by_smiles: Dict[str, Any] = {r["smiles"]: r["result"] for r in results}
    augmented_molecules = []
    for m in molecules:
        smi = m.get("smiles") or ""
        aug = dict(m)
        if smi in result_by_smiles:
            aug[f"{stage.plugin}_result"] = result_by_smiles[smi]
        augmented_molecules.append(aug)

    return {
        "molecules": augmented_molecules,
        "batch_results": results,
        "batch_errors": errors,
        "summary": f"Processed {len(results)} molecules, {len(errors)} errors",
    }


async def execute_pipeline(
    plan: PipelinePlan,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a pipeline plan stage by stage.

    Args:
        plan: The PipelinePlan to execute
        input_data: Initial data (e.g. {"smiles_list": [...]})

    Returns:
        Combined results from all stages
    """
    pm = get_plugin_manager()
    stage_results: List[Dict[str, Any]] = []
    stages_completed: List[str] = []
    all_errors: List[str] = []
    current_data: Dict[str, Any] = dict(input_data)

    for stage in plan.stages:
        try:
            result = await _run_stage_with_batching(pm, stage, current_data)
            stage_results.append({
                "stage_id": stage.stage_id,
                "plugin": stage.plugin,
                "success": True,
                "result": result,
            })
            stages_completed.append(stage.plugin)
            current_data = result
        except Exception as e:
            err_msg = f"{stage.plugin}: {e}"
            all_errors.append(err_msg)
            logger.exception("Pipeline stage %s failed: %s", stage.plugin, e)
            stage_results.append({
                "stage_id": stage.stage_id,
                "plugin": stage.plugin,
                "success": False,
                "error": str(e),
            })
            # Stop on first failure to avoid cascading bad data
            break

    # Build final summary
    summaries = []
    for sr in stage_results:
        if sr.get("success") and isinstance(sr.get("result"), dict):
            s = sr["result"].get("summary")
            if s:
                summaries.append(f"{sr['plugin']}: {s}")

    final_summary = "; ".join(summaries) if summaries else "Pipeline completed."

    return {
        "stages_completed": stages_completed,
        "stage_results": stage_results,
        "final_summary": final_summary,
        "errors": all_errors,
        "final_data": current_data,
    }
