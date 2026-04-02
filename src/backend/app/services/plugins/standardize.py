"""SMILES Standardization Plugin.

Standardizes SMILES strings using RDKit: canonical SMILES, InChIKey,
deduplication. Falls back to passthrough if RDKit is not installed.
"""
import asyncio
import logging
from typing import Any, List
from uuid import uuid4

from app.services.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


class StandardizePlugin(BasePlugin):
    """Standardize SMILES strings to canonical form with InChIKey and deduplication."""

    @property
    def name(self) -> str:
        return "standardize_smiles"

    @property
    def description(self) -> str:
        return (
            "Standardize a list of SMILES strings: canonical SMILES, InChIKey, "
            "compound IDs, and deduplication by InChIKey."
        )

    async def load(self) -> Any:
        """No model to load; RDKit is used at execute time."""
        return None

    async def execute(self, model: Any, **kwargs) -> dict:
        """Standardize a list of SMILES strings.

        Args (in kwargs):
            smiles_list: list[str] -- SMILES strings to standardize.
            existing_inchikeys: set[str] or list[str] (optional) -- InChIKeys to
                treat as duplicates; molecules matching these are skipped.

        Returns:
            Dict with molecules, summary, failed_count.
        """
        smiles_list = kwargs.get("smiles_list", [])
        if not isinstance(smiles_list, list):
            smiles_list = [smiles_list] if smiles_list else []

        existing = kwargs.get("existing_inchikeys") or []
        existing_set = set(existing) if isinstance(existing, (list, set)) else set()

        input_molecules = kwargs.get("molecules", [])
        metadata = {}
        for m in input_molecules:
            if isinstance(m, dict) and m.get("smiles"):
                metadata[m["smiles"]] = {
                    "source": m.get("source", "unknown"),
                    "name": m.get("name", "")
                }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._standardize, smiles_list, existing_set, metadata
        )

    def _standardize(
        self, smiles_list: List[str], existing_inchikeys: set, metadata: dict
    ) -> dict:
        """Synchronous standardization logic."""
        molecules: List[dict] = []
        failed_count = 0
        seen_inchikeys: set = set()

        try:
            from rdkit import Chem
        except ImportError:
            logger.warning("RDKit not installed; using passthrough mode.")
            for i, smi in enumerate(smiles_list):
                smi = (smi or "").strip()
                if not smi:
                    failed_count += 1
                    continue
                meta = metadata.get(smi, {})
                compound_id = f"ATLAS-{uuid4().hex[:8].upper()}"
                molecules.append({
                    "smiles": smi,
                    "inchikey": "",
                    "compound_id": compound_id,
                    "valid": True,
                    "source": meta.get("source", "unknown"),
                    "name": meta.get("name", ""),
                })
            return {
                "molecules": molecules,
                "summary": (
                    f"Passthrough mode (RDKit unavailable): {len(molecules)} processed, "
                    f"{failed_count} failed."
                ),
                "failed_count": failed_count,
            }

        for smi in smiles_list:
            smi = (smi or "").strip()
            if not smi:
                failed_count += 1
                continue

            try:
                mol = Chem.MolFromSmiles(smi)
                meta = metadata.get(smi, {})
                if mol is None:
                    molecules.append({
                        "smiles": smi,
                        "inchikey": "",
                        "compound_id": "",
                        "valid": False,
                        "source": meta.get("source", "unknown"),
                        "name": meta.get("name", ""),
                    })
                    failed_count += 1
                    continue

                canonical = Chem.MolToSmiles(mol, canonical=True)
                inchikey = Chem.MolToInchiKey(mol)

                if inchikey in existing_inchikeys:
                    molecules.append({
                        "smiles": canonical,
                        "inchikey": inchikey,
                        "compound_id": "",
                        "valid": False,
                        "source": meta.get("source", "unknown"),
                        "name": meta.get("name", ""),
                    })
                    failed_count += 1
                    continue

                if inchikey in seen_inchikeys:
                    molecules.append({
                        "smiles": canonical,
                        "inchikey": inchikey,
                        "compound_id": "",
                        "valid": False,
                        "source": meta.get("source", "unknown"),
                        "name": meta.get("name", ""),
                    })
                    failed_count += 1
                    continue

                seen_inchikeys.add(inchikey)
                compound_id = f"ATLAS-{uuid4().hex[:8].upper()}"

                molecules.append({
                    "smiles": canonical,
                    "inchikey": inchikey,
                    "compound_id": compound_id,
                    "valid": True,
                    "source": meta.get("source", "unknown"),
                    "name": meta.get("name", ""),
                })
            except Exception as e:
                logger.exception("Error standardizing SMILES %r: %s", smi, e)
                meta = metadata.get(smi, {})
                molecules.append({
                    "smiles": smi,
                    "inchikey": "",
                    "compound_id": "",
                    "valid": False,
                    "source": meta.get("source", "unknown"),
                    "name": meta.get("name", ""),
                })
                failed_count += 1

        valid_count = sum(1 for m in molecules if m.get("valid"))
        return {
            "molecules": molecules,
            "summary": (
                f"Standardized {len(molecules)} molecules: {valid_count} valid, "
                f"{failed_count} failed or duplicate."
            ),
            "failed_count": failed_count,
        }

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SMILES strings to standardize",
                },
                "existing_inchikeys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional InChIKeys to treat as duplicates",
                },
            },
            "required": ["smiles_list"],
        }

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "molecules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "smiles": {"type": "string"},
                            "inchikey": {"type": "string"},
                            "compound_id": {"type": "string"},
                            "valid": {"type": "boolean"},
                        },
                    },
                },
                "summary": {"type": "string"},
                "failed_count": {"type": "integer"},
            },
            "required": ["molecules", "summary", "failed_count"],
        }
