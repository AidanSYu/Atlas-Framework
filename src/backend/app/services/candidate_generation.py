"""
Candidate Generation & Screening Service (Agent C6).

Provides two async generators that yield SSE-formatted events:
  1. generate_candidates() — uses LLM + corpus RAG to propose SMILES strings
  2. screen_candidates()  — deterministic RDKit property calculations + constraint filtering

Both support a `mock=True` flag for frontend testing without LLM tokens.
"""
import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from app.core.database import get_project_session, DiscoverySession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock SMILES for ?mock=true  (common small molecules)
# ---------------------------------------------------------------------------
MOCK_SMILES: List[Dict[str, str]] = [
    {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "Aspirin"},
    {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "name": "Ibuprofen"},
    {"smiles": "CCO", "name": "Ethanol"},
    {"smiles": "CC(=O)Nc1ccc(O)cc1", "name": "Acetaminophen"},
    {"smiles": "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34", "name": "Pyrene"},
    {"smiles": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "name": "Testosterone"},
    {"smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "name": "Glucose"},
]


def _load_session_params(project_id: str, session_id: str) -> Dict[str, Any]:
    """Load ProjectTargetParams from the per-project DiscoverySession row."""
    db = get_project_session(project_id)
    try:
        row = db.query(DiscoverySession).filter(DiscoverySession.id == session_id).first()
        if not row:
            raise ValueError(f"DiscoverySession not found: {session_id}")
        return row.target_params
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 1. Generate Candidates
# ---------------------------------------------------------------------------

async def generate_candidates(
    session_id: str,
    epoch_id: str,
    mock: bool = False,
    retrieval_service=None,
    llm_service=None,
    project_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE-formatted strings for candidate generation.

    Events emitted:
      data: {"type":"progress","message":"..."}
      data: {"type":"candidates","smiles":[...]}
      data: {"type":"complete"}
    """

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    yield _sse({"type": "progress", "message": "Loading session parameters..."})
    await asyncio.sleep(0)  # yield control

    try:
        if not project_id:
            yield _sse({"type": "error", "message": "project_id is required"})
            return
        params = _load_session_params(project_id, session_id)
    except ValueError as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    objective = params.get("objective", "")
    yield _sse({"type": "progress", "message": f"Objective: {objective}"})

    # ------------------------------------------------------------------
    # MOCK PATH — return hardcoded SMILES without hitting the LLM
    # ------------------------------------------------------------------
    if mock:
        yield _sse({"type": "progress", "message": "Mock mode: returning hardcoded candidates..."})
        await asyncio.sleep(0.2)
        smiles_list = [m["smiles"] for m in MOCK_SMILES]
        yield _sse({"type": "candidates", "smiles": smiles_list})
        yield _sse({"type": "complete"})
        return

    # ------------------------------------------------------------------
    # REAL PATH — RAG retrieval + LLM candidate generation
    # ------------------------------------------------------------------
    yield _sse({"type": "progress", "message": "Retrieving corpus context..."})

    corpus_context = ""
    if retrieval_service is not None:
        try:
            rag_result = await retrieval_service.query_atlas(
                user_question=objective,
                project_id=project_id,
            )
            chunks = rag_result.get("context", {}).get("vector_chunks", [])
            corpus_context = "\n---\n".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in chunks[:5]
            )
        except Exception as e:
            logger.warning(f"RAG retrieval failed, continuing without context: {e}")

    yield _sse({"type": "progress", "message": "Generating candidate structures via LLM..."})

    # Build constraints description
    constraints_desc = ""
    for pc in params.get("propertyConstraints", []):
        prop = pc.get("property", "")
        op = pc.get("operator", "")
        val = pc.get("value", "")
        constraints_desc += f"  - {prop} {op} {val}\n"

    prompt = f"""You are a medicinal chemistry expert. Based on the research objective and corpus context below, propose 5-7 candidate molecular structures as valid SMILES strings.

Research Objective: {objective}

Property Constraints:
{constraints_desc if constraints_desc else "  (none specified)"}

Corpus Context:
{corpus_context if corpus_context else "(no corpus context available)"}

Return ONLY a JSON array of SMILES strings, e.g.:
["CCO", "c1ccccc1", "CC(=O)O"]

Do NOT include explanations — only the JSON array."""

    smiles_list: List[str] = []

    if llm_service is not None:
        try:
            raw = await llm_service.generate(prompt=prompt, temperature=0.7, max_tokens=1024)
            # Parse the JSON array from LLM output
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                smiles_list = json.loads(raw[start:end])
                # Filter to strings only
                smiles_list = [s for s in smiles_list if isinstance(s, str) and len(s) > 0]
        except Exception as e:
            logger.error(f"LLM candidate generation failed: {e}")
            yield _sse({"type": "error", "message": f"LLM generation failed: {e}"})
            return
    else:
        logger.warning("No LLM service available — falling back to mock candidates")
        smiles_list = [m["smiles"] for m in MOCK_SMILES]

    if not smiles_list:
        yield _sse({"type": "error", "message": "LLM returned no valid SMILES candidates."})
        return

    yield _sse({"type": "candidates", "smiles": smiles_list})
    yield _sse({"type": "complete"})


# ---------------------------------------------------------------------------
# 2. Screen Candidates  (deterministic RDKit)
# ---------------------------------------------------------------------------

def _compute_rdkit_properties(smiles: str) -> Optional[Dict[str, Any]]:
    """Compute deterministic molecular properties using RDKit.

    Returns None if the SMILES is invalid / RDKit not available.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        logger.error("RDKit is not installed — cannot compute properties")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def _check_constraint(value: float, constraint: Dict[str, Any]) -> bool:
    """Evaluate a single PropertyConstraint against a computed value."""
    op = constraint.get("operator", "")
    target = constraint.get("value")

    if target is None:
        return True

    if op == "<":
        return value < float(target)
    elif op == ">":
        return value > float(target)
    elif op == "<=":
        return value <= float(target)
    elif op == ">=":
        return value >= float(target)
    elif op == "between":
        if isinstance(target, (list, tuple)) and len(target) == 2:
            return float(target[0]) <= value <= float(target[1])
        return True
    return True


def _passes_all_constraints(
    props: Dict[str, Any], constraints: List[Dict[str, Any]]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Check all constraints against computed properties.

    Returns (passes_all, predicted_properties_list) where each entry
    matches the PredictedProperty TypeScript interface.
    """
    predicted: List[Dict[str, Any]] = []
    all_pass = True

    # Build a map of property name -> constraint
    constraint_map: Dict[str, Dict[str, Any]] = {}
    for c in constraints:
        constraint_map[c.get("property", "")] = c

    # Standard property metadata
    units = {"MW": "Da", "LogP": None, "TPSA": "A^2", "HBD": None, "HBA": None, "RotBonds": None}

    for prop_name, prop_value in props.items():
        constraint = constraint_map.get(prop_name)
        passes: Optional[bool] = None
        if constraint:
            passes = _check_constraint(prop_value, constraint)
            if not passes:
                all_pass = False

        predicted.append({
            "name": prop_name,
            "value": prop_value,
            "unit": units.get(prop_name),
            "passesConstraint": passes,
            "model": "RDKit (deterministic)",
        })

    return all_pass, predicted


async def screen_candidates(
    project_id: str,
    session_id: str,
    epoch_id: str,
    smiles_list: List[str],
) -> AsyncGenerator[str, None]:
    """Async generator yielding SSE-formatted strings for candidate screening.

    Events emitted:
      data: {"type":"progress","message":"..."}
      data: {"type":"screen_progress","smiles":"...","properties":{...},"passes_constraints":bool}
      data: {"type":"complete","surviving_candidates":[CandidateArtifact,...]}
    """

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    yield _sse({"type": "progress", "message": f"Screening {len(smiles_list)} candidates with RDKit..."})
    await asyncio.sleep(0)

    # Load constraints from session
    try:
        if not project_id:
            yield _sse({"type": "error", "message": "project_id is required"})
            return
        params = _load_session_params(project_id, session_id)
    except ValueError as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    constraints = params.get("propertyConstraints", [])

    surviving: List[Dict[str, Any]] = []
    rank = 0

    for idx, smiles in enumerate(smiles_list):
        # Yield control to avoid blocking the event loop
        await asyncio.sleep(0)

        props = _compute_rdkit_properties(smiles)
        if props is None:
            yield _sse({
                "type": "screen_progress",
                "smiles": smiles,
                "properties": None,
                "passes_constraints": False,
                "note": "Invalid SMILES or RDKit unavailable",
            })
            continue

        passes_all, predicted_properties = _passes_all_constraints(props, constraints)

        yield _sse({
            "type": "screen_progress",
            "smiles": smiles,
            "properties": props,
            "passes_constraints": passes_all,
        })

        if passes_all:
            rank += 1
            # Compute a mock score based on how many properties pass
            pass_count = sum(1 for p in predicted_properties if p["passesConstraint"] is True)
            total_constrained = sum(1 for p in predicted_properties if p["passesConstraint"] is not None)
            score = round(pass_count / max(total_constrained, 1), 2)
            # Clamp to [0.5, 1.0] range to make surviving candidates look reasonable
            score = round(0.5 + score * 0.5, 2)

            candidate_artifact = {
                "id": str(uuid.uuid4()),
                "rank": rank,
                "score": score,
                "renderType": "molecule_2d",
                "renderData": smiles,
                "properties": predicted_properties,
                "sourceReasoning": f"Candidate {smiles} passed all {total_constrained} property constraints during deterministic RDKit screening.",
                "sourceDocumentIds": [],
                "status": "pending",
            }
            surviving.append(candidate_artifact)

    yield _sse({
        "type": "progress",
        "message": f"Screening complete: {len(surviving)}/{len(smiles_list)} candidates survived.",
    })
    yield _sse({"type": "complete", "surviving_candidates": surviving})
