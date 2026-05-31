"""Plate reader stub — pretends to run a binding assay on a BMG CLARIOstar.

Deterministic on (batch_id, target). IC50 distribution skews to "interesting
but not magical" range (30 nM – 5 uM) so the orchestrator's downstream
recommendation has something to chew on.
"""
import hashlib
import logging
import math
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _seed(batch_id: str, target: str) -> int:
    h = hashlib.sha1(f"{batch_id}|{target}".encode("utf-8")).hexdigest()[:12]
    return int(h, 16)


def _classify(ic50_nm: float) -> str:
    if ic50_nm < 100:
        return "strong_hit"
    if ic50_nm < 1000:
        return "moderate_hit"
    if ic50_nm < 10000:
        return "weak_hit"
    return "inactive"


class PlateReaderWrapper:

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        batch_id = (args.get("batch_id") or "").strip()
        target = (args.get("target") or "").strip()
        if not batch_id or not target:
            return {"error": "batch_id and target are both required"}

        s = _seed(batch_id, target)
        # Log-uniform IC50 over [30 nM, 5000 nM]
        frac = (s % 10_000) / 10_000.0
        ic50 = round(30.0 * math.exp(frac * math.log(5000.0 / 30.0)), 1)
        # % inhibition at 10 uM — derived from IC50 with a sigmoid (Hill n=1)
        inhibition = round(100.0 / (1.0 + ic50 / 10_000.0), 1)
        classification = _classify(ic50)
        logger.info("plate_reader: %s vs %s IC50=%.1f nM inhib=%.1f%% class=%s",
                    batch_id, target, ic50, inhibition, classification)
        return {
            "batch_id": batch_id,
            "target": target,
            "ic50_nm": ic50,
            "percent_inhibition_at_10uM": inhibition,
            "hit_classification": classification,
            "instrument": "BMG CLARIOstar",
        }


PLUGIN = PlateReaderWrapper()
