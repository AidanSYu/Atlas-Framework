"""RDKit SMARTS Toxicity Checker Plugin.

Pure algorithmic plugin -- uses SMARTS pattern matching for structural alerts
and RDKit's built-in PAINS FilterCatalog. Zero model loading.
All computation runs on CPU via run_in_executor().
"""
import asyncio
from typing import Any, Dict, List

from app.services.plugins.base import BasePlugin

# Curated SMARTS patterns for structural alerts (subset of Brenk + common reactive groups)
STRUCTURAL_ALERTS: Dict[str, str] = {
    "michael_acceptor": "[CH2]=[CH][C,S,N]=O",
    "epoxide": "C1OC1",
    "aldehyde": "[CH1](=O)",
    "azide": "[N-]=[N+]=[N-]",
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]",
    "nitro_aromatic": "c[N+](=O)[O-]",
    "aniline": "c1ccccc1[NH2]",
    "polyhalogenated": "[#6]([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I]",
}


class ToxicityCheckerPlugin(BasePlugin):

    @property
    def name(self) -> str:
        return "check_toxicity"

    @property
    def description(self) -> str:
        return (
            "Check a molecule for structural toxicity alerts and PAINS patterns "
            "using SMARTS matching. Returns alert list and clean/dirty verdict."
        )

    async def load(self) -> Any:
        """Pre-compile SMARTS patterns (fast, pure code)."""
        from rdkit import Chem

        compiled = {}
        for alert_name, smarts in STRUCTURAL_ALERTS.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is not None:
                compiled[alert_name] = pat
        return compiled

    async def execute(self, model: Any, **kwargs) -> dict:
        """Check one or many SMILES for structural alerts and PAINS.

        Args (in kwargs):
            smiles_list: list[str] -- Batch mode (preferred in pipeline).
            smiles: str            -- Single-molecule mode (legacy).

        Returns:
            Batch mode:  {"toxicity_results": [...], "summary": "..."}
            Single mode: {"smiles": ..., "clean": ..., "alert_count": ..., ...}
        """
        smiles_list = kwargs.get("smiles_list")
        if smiles_list and isinstance(smiles_list, list):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._check_batch, model, smiles_list)
        smiles = kwargs.get("smiles", "")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._check, model, smiles)

    def _check_batch(self, compiled_patterns: dict, smiles_list: list) -> dict:
        results = [self._check(compiled_patterns, smi) for smi in smiles_list if (smi or "").strip()]
        clean_count = sum(1 for r in results if r.get("clean"))
        flagged_count = len(results) - clean_count
        return {
            "toxicity_results": results,
            "summary": f"Toxicity check: {clean_count} clean, {flagged_count} flagged.",
        }

    def _check(self, compiled_patterns: dict, smiles: str) -> dict:
        from rdkit import Chem
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"smiles": smiles, "valid": False, "error": "Invalid SMILES string"}

        # Check structural alerts
        alerts: List[Dict[str, str]] = []
        for alert_name, pattern in compiled_patterns.items():
            if mol.HasSubstructMatch(pattern):
                alerts.append({
                    "name": alert_name,
                    "smarts": STRUCTURAL_ALERTS[alert_name],
                })

        # Check PAINS (RDKit built-in FilterCatalog)
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        pains_entry = catalog.GetFirstMatch(mol)
        pains_hits = 1 if pains_entry else 0
        pains_description = pains_entry.GetDescription() if pains_entry else None

        return {
            "smiles": smiles,
            "valid": True,
            "structural_alerts": alerts,
            "alert_count": len(alerts),
            "pains_hits": pains_hits,
            "pains_description": pains_description,
            "clean": len(alerts) == 0 and pains_hits == 0,
        }

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES representation of the molecule",
                }
            },
            "required": ["smiles"],
        }

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "valid": {"type": "boolean"},
                "structural_alerts": {"type": "array"},
                "alert_count": {"type": "integer"},
                "pains_hits": {"type": "integer"},
                "pains_description": {"type": "string"},
                "clean": {"type": "boolean"},
            },
        }
