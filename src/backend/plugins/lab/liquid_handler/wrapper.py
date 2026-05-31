"""Liquid handler stub — pretends to queue a synthesis on a Hamilton STARlet.

Deterministic on (target_smiles): same input always returns same batch_id so
downstream QC tools can be called repeatedly with consistent identity.
"""
import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _batch_id_for(smiles: str) -> str:
    h = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:8].upper()
    return f"LH-{h}"


def _eta_minutes(smiles: str) -> int:
    # 30–180 min based on SMILES length (proxy for complexity)
    n = max(1, len(smiles))
    return 30 + (n % 150)


class LiquidHandlerWrapper:

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        smiles = (args.get("target_smiles") or "").strip()
        if not smiles:
            return {"error": "target_smiles is required"}

        batch_id = _batch_id_for(smiles)
        eta = _eta_minutes(smiles)
        logger.info("liquid_handler: queued %s -> %s (eta %d min)", smiles, batch_id, eta)
        return {
            "batch_id": batch_id,
            "status": "queued",
            "estimated_completion_min": eta,
            "instrument": "Hamilton STARlet (bench 2)",
        }


PLUGIN = LiquidHandlerWrapper()
