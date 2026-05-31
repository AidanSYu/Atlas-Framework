"""Standardize SMILES plugin — fully self-contained, no app imports.

Canonicalizes SMILES strings using RDKit: canonical SMILES, InChIKey,
deduplication. Falls back to passthrough if RDKit is not installed.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class StandardizeSmilesWrapper:
    """New-style Atlas plugin wrapper. All logic is self-contained."""

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        smiles_list = args.get("smiles_list", [])
        if not isinstance(smiles_list, list):
            smiles_list = [smiles_list] if smiles_list else []

        existing = args.get("existing_inchikeys") or []
        existing_set = set(existing) if isinstance(existing, (list, set)) else set()

        input_molecules = args.get("molecules", [])
        metadata: Dict[str, dict] = {}
        for m in input_molecules:
            if isinstance(m, dict) and m.get("smiles"):
                metadata[m["smiles"]] = {
                    "source": m.get("source", "unknown"),
                    "name": m.get("name", ""),
                }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._standardize, smiles_list, existing_set, metadata
        )

    @staticmethod
    def _standardize(
        smiles_list: List[str], existing_inchikeys: set, metadata: dict
    ) -> dict:
        molecules: List[dict] = []
        failed_count = 0
        seen_inchikeys: set = set()

        try:
            from rdkit import Chem
        except ImportError:
            logger.warning("RDKit not installed; using passthrough mode.")
            for smi in smiles_list:
                smi = (smi or "").strip()
                if not smi:
                    failed_count += 1
                    continue
                meta = metadata.get(smi, {})
                compound_id = f"ATLAS-{uuid4().hex[:8].upper()}"
                molecules.append({
                    "smiles": smi, "inchikey": "", "compound_id": compound_id,
                    # passthrough: SMILES are NOT canonicalized and there's no
                    # InChIKey, so downstream dedup/lookup will be wrong. Mark
                    # valid=False so consumers know to reject these.
                    "valid": False,
                    "passthrough": True,
                    "engine_used": "passthrough_no_rdkit",
                    "source": meta.get("source", "unknown"),
                    "name": meta.get("name", ""),
                })
            return {
                "molecules": molecules,
                "engine_used": "passthrough_no_rdkit",
                "valid": False,
                "summary": (
                    f"Passthrough mode (RDKit unavailable): {len(molecules)} processed, "
                    f"{failed_count} failed. WARNING: SMILES were NOT canonicalized and have no "
                    "InChIKey; downstream dedup/lookup will be unreliable. Install RDKit."
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
                        "smiles": smi, "inchikey": "", "compound_id": "",
                        "valid": False, "source": meta.get("source", "unknown"),
                        "name": meta.get("name", ""),
                    })
                    failed_count += 1
                    continue
                canonical = Chem.MolToSmiles(mol, canonical=True)
                inchikey = Chem.MolToInchiKey(mol)
                if inchikey in existing_inchikeys or inchikey in seen_inchikeys:
                    molecules.append({
                        "smiles": canonical, "inchikey": inchikey, "compound_id": "",
                        "valid": False, "source": meta.get("source", "unknown"),
                        "name": meta.get("name", ""),
                    })
                    failed_count += 1
                    continue
                seen_inchikeys.add(inchikey)
                compound_id = f"ATLAS-{uuid4().hex[:8].upper()}"
                molecules.append({
                    "smiles": canonical, "inchikey": inchikey,
                    "compound_id": compound_id, "valid": True,
                    "source": meta.get("source", "unknown"),
                    "name": meta.get("name", ""),
                })
            except Exception as e:
                logger.exception("Error standardizing SMILES %r: %s", smi, e)
                meta = metadata.get(smi, {})
                molecules.append({
                    "smiles": smi, "inchikey": "", "compound_id": "",
                    "valid": False, "source": meta.get("source", "unknown"),
                    "name": meta.get("name", ""),
                })
                failed_count += 1

        for m in molecules:
            m.setdefault("engine_used", "rdkit")
        valid_count = sum(1 for m in molecules if m.get("valid"))
        return {
            "molecules": molecules,
            "engine_used": "rdkit",
            "valid": True,
            "summary": f"Standardized {len(molecules)} molecules: {valid_count} valid, {failed_count} failed or duplicate.",
            "failed_count": failed_count,
        }


PLUGIN = StandardizeSmilesWrapper()
