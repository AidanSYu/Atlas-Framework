"""Synthetic accessibility scoring plugin — fully self-contained, no app imports.

Computes SA Score (1-10, higher = harder) using RDKit SA_Score or heuristic fallback.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _heuristic_sa_score(smiles: str) -> float:
    length = len(smiles or "")
    if length <= 10:
        return 1.5
    if length <= 25:
        return 2.0 + (length - 10) / 30.0
    if length <= 50:
        return 3.5 + (length - 25) / 25.0
    if length <= 80:
        return 5.5 + (length - 50) / 30.0
    return min(10.0, 6.5 + (length - 80) / 20.0)


class ScoreSynthesizabilityWrapper:

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        smiles_list = args.get("smiles_list", [])
        if not isinstance(smiles_list, list):
            smiles_list = [smiles_list] if smiles_list else []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._score, smiles_list)

    @staticmethod
    def _score(smiles_list: List[str]) -> dict:
        scores: List[dict] = []
        use_sascorer = False
        try:
            from rdkit import Chem
            from rdkit.Contrib.SA_Score import sascorer
            use_sascorer = True
        except ImportError:
            logger.info("RDKit SA_Score not available; using heuristic.")

        for smi in smiles_list:
            smi = (smi or "").strip()
            if not smi:
                scores.append({
                    "smiles": "", "sa_score": None, "feasible": False,
                    "engine_used": "none", "valid": False,
                    "error": "empty SMILES",
                })
                continue
            try:
                if use_sascorer:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        sa = round(sascorer.calculateScore(mol), 2)
                        engine = "rdkit_sascorer"
                        valid = True
                    else:
                        sa = round(_heuristic_sa_score(smi), 2)
                        engine = "heuristic_smiles_length"
                        valid = False
                else:
                    sa = round(_heuristic_sa_score(smi), 2)
                    engine = "heuristic_smiles_length"
                    valid = False
                scores.append({
                    "smiles": smi, "sa_score": sa, "feasible": sa <= 6.0,
                    "engine_used": engine, "valid": valid,
                })
            except Exception as e:
                logger.exception("Error scoring SMILES %r: %s", smi, e)
                sa = round(_heuristic_sa_score(smi), 2)
                scores.append({
                    "smiles": smi, "sa_score": sa, "feasible": sa <= 6.0,
                    "engine_used": "heuristic_smiles_length", "valid": False,
                    "error": str(e),
                })

        feasible_count = sum(1 for s in scores if s.get("feasible"))
        heuristic_count = sum(1 for s in scores if s.get("engine_used", "").startswith("heuristic"))
        engine_note = (
            " WARNING: SA scores are SMILES-length heuristic, not from rdkit.Contrib.SA_Score."
            if heuristic_count == len(scores) and scores
            else (f" Note: {heuristic_count}/{len(scores)} scores are heuristic." if heuristic_count else "")
        )
        return {
            "scores": scores,
            "engine_used": "rdkit_sascorer" if heuristic_count == 0 else ("mixed" if heuristic_count < len(scores) else "heuristic"),
            "valid": heuristic_count == 0,
            "summary": (
                f"SA scores for {len(scores)} molecules: {feasible_count} feasible (SA <= 6), "
                f"{len(scores) - feasible_count} infeasible." + engine_note
            ),
        }


PLUGIN = ScoreSynthesizabilityWrapper()
