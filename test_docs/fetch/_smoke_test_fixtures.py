"""Smoke tests for the new data_path code paths.

Not a pytest module — just an end-to-end script:

    python test_docs/fetch/_smoke_test_fixtures.py

Exits non-zero on any failure so CI can pick it up.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "backend"))

DARCY = ROOT / "test_docs" / "physics" / "darcy_small.npz"
OBS = ROOT / "test_docs" / "manufacturing" / "sandbox_observations.csv"


def test_physics_simulator_training_data_path() -> None:
    from plugins.prometheus.physics_simulator.wrapper import PLUGIN  # type: ignore
    from plugins.prometheus.physics_simulator.wrapper import PhysicsSimulatorWrapper  # type: ignore

    # --- 1. Direct loader contract ---
    loaded = PhysicsSimulatorWrapper._load_training_data_from_path(str(DARCY))
    assert "inputs" in loaded and "outputs" in loaded
    assert loaded["inputs"].shape == (100, 64, 64), loaded["inputs"].shape
    assert loaded["outputs"].shape == (100, 64, 64), loaded["outputs"].shape
    print(f"[physics] loader returns inputs={loaded['inputs'].shape}, outputs={loaded['outputs'].shape}")

    # --- 2. End-to-end FNO train via training_data_path ---
    #     Darcy is (N, H, W) image data -> only the FNO code path supports it.
    #     Skip if neuralop isn't installed (auto would fall through to PINN
    #     which can't handle image data — that's a pre-existing limitation of
    #     PINN, not a regression in our new loader).
    try:
        import neuralop  # type: ignore  # noqa: F401
        has_neuralop = True
    except ImportError:
        has_neuralop = False

    if has_neuralop:
        result_fno = asyncio.run(PLUGIN.invoke({
            "mode": "train",
            "model_type": "fno",
            "training_data_path": str(DARCY),
            "parameters": {"epochs": 2, "lr": 1e-3},
        }))
        assert result_fno.get("valid") is True, result_fno
        print(f"[physics] FNO trained from training_data_path OK. summary: {result_fno.get('summary')}")
    else:
        print("[physics] neuralop not installed; skipping end-to-end FNO train. "
              "Direct loader check in step 1 already validated the path wiring.")

    # --- 3. Negative: missing file should fail loudly ---
    bad = asyncio.run(PLUGIN.invoke({
        "mode": "train",
        "model_type": "auto",
        "training_data_path": "c:/this/path/does/not/exist.npz",
    }))
    assert "error" in bad and "does not exist" in bad["error"], bad
    print(f"[physics] missing-file negative test OK: {bad['error'][:80]}")

    # --- 4. Negative: both in-memory + path -> refuse ---
    dual = asyncio.run(PLUGIN.invoke({
        "mode": "train",
        "model_type": "auto",
        "training_data_path": str(DARCY),
        "training_data": {"inputs": [[1.0, 2.0]], "outputs": [3.0]},
    }))
    assert "error" in dual and "either" in dual["error"].lower(), dual
    print(f"[physics] dual-source refusal OK: {dual['error'][:80]}")


def test_sandbox_lab_observations_path() -> None:
    from plugins.prometheus.sandbox_lab.wrapper import PLUGIN  # type: ignore

    payload = {
        "mode": "pareto",  # pareto doesn't need botorch stack to be useful
        "parameters": [
            {"name": "peak_c", "bounds": [220.0, 260.0], "type": "continuous"},
            {"name": "tal_s", "bounds": [20.0, 120.0], "type": "continuous"},
            {"name": "ramp_c_per_s", "bounds": [0.5, 4.0], "type": "continuous"},
        ],
        "objectives": [
            {"name": "defect_rate", "minimize": True},
            {"name": "throughput", "minimize": False},
        ],
        "observations_path": str(OBS),
    }
    result = asyncio.run(PLUGIN.invoke(payload))
    if not result.get("valid"):
        # If botorch stack is missing, invoke short-circuits BEFORE the loader
        # runs. Accept that outcome but still flag it.
        err = result.get("error") or result.get("summary", "")
        if "botorch" in err.lower() or "gpytorch" in err.lower():
            print(f"[sandbox] botorch stack unavailable — loader not exercised via "
                  f"wrapper. Calling loader directly instead. ({err[:80]})")
            from plugins.prometheus.sandbox_lab.wrapper import _load_observations_from_path  # type: ignore
            loaded = _load_observations_from_path(
                str(OBS),
                parameter_columns=None,
                objective_columns=None,
                parameters=payload["parameters"],
                objectives=payload["objectives"],
            )
            assert "X" in loaded and "Y" in loaded, loaded
            assert len(loaded["X"]) >= 30 and len(loaded["Y"]) >= 30, (
                f"expected >=30 rows, got {len(loaded['X'])}"
            )
            assert len(loaded["X"][0]) == 3, loaded["X"][0]
            assert len(loaded["Y"][0]) == 2, loaded["Y"][0]
            print(f"[sandbox] direct loader test OK: X shape ({len(loaded['X'])}, 3), "
                  f"Y shape ({len(loaded['Y'])}, 2)")
            return
        raise AssertionError(f"[sandbox] pareto invocation failed: {result}")

    assert "pareto_front" in result, result
    assert "pareto_X" in result, result
    pareto_front = result["pareto_front"]
    assert len(pareto_front) >= 1, pareto_front
    print(f"[sandbox] pareto run from observations_path OK. "
          f"Pareto size={len(pareto_front)}. summary: {result.get('summary')}")

    # Negative: missing file.
    bad_payload = {**payload, "observations_path": "c:/nope.csv"}
    bad = asyncio.run(PLUGIN.invoke(bad_payload))
    # Pareto may propagate the raise as an error field OR as an exception
    # caught by the wrapper and returned as {valid:False, error:...}.
    err = bad.get("error") or bad.get("summary", "")
    assert "does not exist" in err or "not a file" in err, bad
    print(f"[sandbox] missing-file negative test OK: {err[:80]}")


def main() -> None:
    print(f"Darcy fixture: {DARCY}  exists={DARCY.exists()}  bytes={DARCY.stat().st_size if DARCY.exists() else 'N/A'}")
    print(f"Obs fixture: {OBS}  exists={OBS.exists()}  bytes={OBS.stat().st_size if OBS.exists() else 'N/A'}")
    print()
    print("=== physics_simulator ===")
    test_physics_simulator_training_data_path()
    print()
    print("=== sandbox_lab ===")
    test_sandbox_lab_observations_path()
    print()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
