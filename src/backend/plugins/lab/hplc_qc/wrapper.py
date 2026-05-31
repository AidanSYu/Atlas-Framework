"""HPLC purity QC stub — pretends to run an Agilent 1290 on a batch.

Deterministic on batch_id so repeated calls return the same readout.
Purity is biased to look "mostly good" (90–99 %) so the demo flows past QC
to the assay step most of the time.
"""
import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _seed(batch_id: str) -> int:
    return int(hashlib.sha1(batch_id.encode("utf-8")).hexdigest()[:8], 16)


class HPLCQCWrapper:

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        batch_id = (args.get("batch_id") or "").strip()
        if not batch_id:
            return {"error": "batch_id is required (call liquid_handler first)"}

        s = _seed(batch_id)
        # Purity in 90.0–99.5 %, two decimals
        purity = round(90.0 + (s % 950) / 100.0, 2)
        # Retention time 2.0–8.0 min
        retention = round(2.0 + ((s >> 4) % 600) / 100.0, 2)
        # 1–3 peaks (impurity proxy)
        peaks = 1 + ((s >> 8) % 3)
        passed = purity >= 95.0 and peaks == 1
        logger.info("hplc_qc: %s purity=%.2f%% rt=%.2f peaks=%d pass=%s",
                    batch_id, purity, retention, peaks, passed)
        return {
            "batch_id": batch_id,
            "purity_percent": purity,
            "retention_time_min": retention,
            "peak_count": peaks,
            "passed_qc": passed,
            "instrument": "Agilent 1290 Infinity II",
        }


PLUGIN = HPLCQCWrapper()
