"""Synthetic Accessibility Score Plugin.

Computes SA Score (1-10, higher = harder to synthesize) using RDKit
Contrib.SA_Score when available, or a simple heuristic fallback.
"""
import asyncio
import logging
from typing import Any, List

from app.services.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


def _heuristic_sa_score(smiles: str) -> float:
    """Heuristic: longer SMILES = harder to synthesize. Capped 1-10."""
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


class SAScorePlugin(BasePlugin):
    """Score synthetic accessibility for SMILES using SA Score or heuristic."""

    @property
    def name(self) -> str:
        return "score_synthesizability"

    @property
    def description(self) -> str:
        return (
            "Score synthetic accessibility (SA Score 1-10) for a list of SMILES. "
            "Uses RDKit SA_Score when available, else a length-based heuristic."
        )

    async def load(self) -> Any:
        """No model to load; SA_Score is used at execute time."""
        return None

    async def execute(self, model: Any, **kwargs) -> dict:
        """Score synthetic accessibility for a list of SMILES.

        Args (in kwargs):
            smiles_list: list[str] -- SMILES strings to score.

        Returns:
            Dict with scores, summary.
        """
        smiles_list = kwargs.get("smiles_list", [])
        if not isinstance(smiles_list, list):
            smiles_list = [smiles_list] if smiles_list else []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._score, smiles_list)

    def _score(self, smiles_list: List[str]) -> dict:
        """Synchronous scoring logic."""
        scores: List[dict] = []
        use_sascorer = False

        try:
            from rdkit import Chem
            from rdkit.Contrib.SA_Score import sascorer
            use_sascorer = True
        except ImportError:
            logger.info(
                "RDKit SA_Score not available; using heuristic (SMILES length)."
            )

        for smi in smiles_list:
            smi = (smi or "").strip()
            if not smi:
                scores.append({
                    "smiles": "",
                    "sa_score": 5.0,
                    "feasible": False,
                })
                continue

            try:
                if use_sascorer:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        sa = _heuristic_sa_score(smi)
                    else:
                        sa = round(sascorer.calculateScore(mol), 2)
                else:
                    sa = round(_heuristic_sa_score(smi), 2)

                feasible = sa <= 6.0
                scores.append({
                    "smiles": smi,
                    "sa_score": sa,
                    "feasible": feasible,
                })
            except Exception as e:
                logger.exception("Error scoring SMILES %r: %s", smi, e)
                sa = round(_heuristic_sa_score(smi), 2)
                scores.append({
                    "smiles": smi,
                    "sa_score": sa,
                    "feasible": sa <= 6.0,
                })

        feasible_count = sum(1 for s in scores if s.get("feasible"))
        return {
            "scores": scores,
            "summary": (
                f"SA scores for {len(scores)} molecules: "
                f"{feasible_count} feasible (SA <= 6), "
                f"{len(scores) - feasible_count} infeasible."
            ),
        }

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SMILES strings to score for synthesizability",
                },
            },
            "required": ["smiles_list"],
        }

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "smiles": {"type": "string"},
                            "sa_score": {"type": "number"},
                            "feasible": {"type": "boolean"},
                        },
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["scores", "summary"],
        }
