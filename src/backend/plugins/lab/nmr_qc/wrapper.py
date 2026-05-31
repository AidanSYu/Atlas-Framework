"""1H NMR QC stub — pretends to acquire a spectrum on a Bruker 400 MHz.

Deterministic on batch_id. Generates a plausible chemical shift list with
peaks in the aromatic (6.5–8.0 ppm) and aliphatic (1.0–4.5 ppm) regions
proportional to the SMILES character classes — enough to look real in the
demo without invoking RDKit.
"""
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _seed(batch_id: str) -> int:
    return int(hashlib.sha1(batch_id.encode("utf-8")).hexdigest()[:12], 16)


def _shifts(expected_smiles: str, seed: int) -> List[float]:
    # Aromatic protons: count lowercase aromatic atoms minus N/O which don't carry H
    aromatics = sum(1 for c in expected_smiles if c in "cn")
    # Aliphatic carbons: count uppercase C not followed by lowercase
    aliphatics = len(re.findall(r"C(?![a-z])", expected_smiles))
    shifts: List[float] = []
    s = seed
    for i in range(min(aromatics, 6)):
        shifts.append(round(6.5 + ((s >> (i * 3)) & 0xFF) / 256.0 * 1.5, 2))
    for i in range(min(aliphatics, 8)):
        shifts.append(round(1.0 + ((s >> (i * 5 + 11)) & 0xFF) / 256.0 * 3.5, 2))
    return sorted(shifts, reverse=True)


class NMRQCWrapper:

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        batch_id = (args.get("batch_id") or "").strip()
        expected = (args.get("expected_smiles") or "").strip()
        if not batch_id or not expected:
            return {"error": "batch_id and expected_smiles are both required"}

        s = _seed(batch_id)
        shifts = _shifts(expected, s)
        # 88% of stubs confirm the structure — the rest flag a mismatch
        confirmed = (s % 100) < 88
        notes = (
            "Spectrum consistent with proposed structure."
            if confirmed
            else "Aromatic region shows an extra peak — possible regioisomer; recommend recrystallization or repurification."
        )
        logger.info("nmr_qc: %s confirmed=%s shifts=%s", batch_id, confirmed, shifts)
        return {
            "batch_id": batch_id,
            "chemical_shifts_ppm": shifts,
            "proton_count": len(shifts),
            "structure_confirmed": confirmed,
            "notes": notes,
            "instrument": "Bruker AVANCE NEO 400 MHz",
        }


PLUGIN = NMRQCWrapper()
