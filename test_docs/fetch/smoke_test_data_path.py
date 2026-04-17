"""Smoke test for the new 'data_path' parameter on two Prometheus plugins.

Invokes each plugin with data_path pointing at the curated SECOM fixtures and
asserts the wrapper returns valid: true (or an explicit, non-crashing dep-gap
error that mentions the missing package). Run once to confirm the file-loader
branch works end-to-end.

Usage:
    python test_docs/fetch/smoke_test_data_path.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "src" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

FIX_DIR = REPO_ROOT / "test_docs" / "manufacturing"
UNIVARIATE = FIX_DIR / "reflow_sensor_series.csv"
MULTIVARIATE = FIX_DIR / "sensor_multivariate.csv"


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


async def _test_mwm() -> dict:
    from plugins.prometheus.manufacturing_world_model.wrapper import PLUGIN
    # anomaly mode doesn't require a ML backend — tests the CSV loader path
    # without needing chronos/timesfm/etc to be installed.
    result = await PLUGIN.invoke(
        {
            "mode": "anomaly",
            "data_path": str(UNIVARIATE),
            "value_column": "value",
            "timestamp_column": "timestamp",
        }
    )
    print("[MWM] keys:", sorted(result.keys()))
    print("[MWM] summary:", result.get("summary"))
    _assert(result.get("valid") is True, f"MWM returned valid=False: {result}")
    _assert("anomaly_scores" in result, "MWM response missing anomaly_scores")
    _assert(
        len(result["anomaly_scores"]) == 500,
        f"expected 500 anomaly scores, got {len(result['anomaly_scores'])}",
    )
    return result


async def _test_causal() -> dict:
    from plugins.prometheus.causal_discovery.wrapper import (
        PLUGIN,
        _load_table_from_csv,
    )
    # The wrapper short-circuits on a missing tigramite/pysr install before
    # ever reaching the CSV loader. To prove the loader works, invoke it
    # directly first.
    data, variable_names = _load_table_from_csv(str(MULTIVARIATE))
    _assert(len(data) == 500, f"expected 500 rows, got {len(data)}")
    _assert(len(variable_names) == 10, f"expected 10 columns, got {len(variable_names)}")
    _assert("defect_rate" in variable_names, f"defect_rate missing: {variable_names}")
    _assert(all(isinstance(row, list) and len(row) == 10 for row in data),
            "rows are not uniform 10-wide lists")
    print(f"[CD-loader] loaded {len(data)}x{len(variable_names)} matrix; cols={variable_names}")

    # Now also exercise the invoke() path: it may fall through on the missing
    # tigramite/pysr stack, which is fine — but we want confidence the wrapper
    # doesn't crash on the new arguments.
    result = await PLUGIN.invoke(
        {
            "mode": "discover",
            "data_path": str(MULTIVARIATE),
            "target_column": "defect_rate",
            "max_lag": 2,
        }
    )
    print("[CD] keys:", sorted(result.keys()))
    print("[CD] summary:", result.get("summary"))
    if result.get("valid") is False:
        err = result.get("error") or ""
        if "tigramite" in err or "pysr" in err:
            print("[CD] tigramite/pysr stack not installed — loader already "
                  "validated directly above, invoke path returns an explicit "
                  "dep-gap error without crashing.")
            return result
        raise AssertionError(f"CD returned valid=False unexpectedly: {result}")
    _assert("variable_names" in result.get("causal_graph", {}), "missing variable_names")
    names = result["causal_graph"]["variable_names"]
    _assert("defect_rate" in names, f"defect_rate not in variable_names: {names}")
    return result


async def _test_bad_path() -> None:
    """data_path pointing at a non-existent file must raise loudly."""
    from plugins.prometheus.manufacturing_world_model.wrapper import PLUGIN
    try:
        await PLUGIN.invoke({"mode": "anomaly", "data_path": "/nope/does_not_exist.csv"})
    except RuntimeError as exc:
        msg = str(exc)
        _assert("does not point to an existing file" in msg, f"wrong error message: {msg!r}")
        print("[loud-fail] MWM bad path raised as expected:", msg[:80])
        return
    raise AssertionError("MWM did not raise on missing data_path")


async def main() -> int:
    _assert(UNIVARIATE.exists(), f"fixture missing: {UNIVARIATE}")
    _assert(MULTIVARIATE.exists(), f"fixture missing: {MULTIVARIATE}")
    await _test_mwm()
    await _test_causal()
    await _test_bad_path()
    print("\nSMOKE TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
