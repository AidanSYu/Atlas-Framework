"""Sandbox Lab - multi-objective Bayesian optimization with BoTorch."""

from __future__ import annotations

import ast
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _load_observations_from_path(
    path: str,
    parameter_columns: Optional[List[str]],
    objective_columns: Optional[List[str]],
    parameters: Optional[List[dict]],
    objectives: Optional[List[dict]],
) -> Dict[str, Any]:
    """Load an observations CSV into the {X, Y} format the BO loop expects.

    Fails LOUDLY on missing file, missing columns, or ambiguous column
    assignment. Additive — callers that pass observations in-memory bypass
    this entirely.

    Column selection precedence:
      1. Explicit `parameter_columns` + `objective_columns` arguments.
      2. Inferred from `parameters[*].name` + `objectives[*].name`.
      3. Otherwise: raise — we refuse to guess.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"observations_path does not exist or is not a file: {path}"
        )
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "observations_path requires pandas. Install with: pip install pandas"
        ) from exc
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read observations_path '{path}' as CSV: {exc}"
        ) from exc
    if df.empty:
        raise ValueError(f"observations_path '{path}' is empty.")

    # Resolve column names.
    if parameter_columns is None and parameters:
        parameter_columns = [p["name"] for p in parameters]
    if objective_columns is None and objectives:
        objective_columns = [o["name"] for o in objectives]
    if not parameter_columns or not objective_columns:
        raise ValueError(
            "observations_path requires parameter_columns + objective_columns "
            "(or the `parameters` + `objectives` blocks) to know which CSV "
            "columns are inputs (X) and which are outputs (Y). "
            f"Got parameter_columns={parameter_columns}, "
            f"objective_columns={objective_columns}."
        )

    missing_X = [c for c in parameter_columns if c not in df.columns]
    missing_Y = [c for c in objective_columns if c not in df.columns]
    if missing_X or missing_Y:
        raise KeyError(
            f"observations_path '{path}' is missing required column(s). "
            f"Missing parameter columns: {missing_X}. "
            f"Missing objective columns: {missing_Y}. "
            f"CSV columns: {list(df.columns)}."
        )

    X = df[list(parameter_columns)].to_numpy(dtype=float)
    Y = df[list(objective_columns)].to_numpy(dtype=float)
    if np.isnan(X).any() or np.isnan(Y).any():
        raise ValueError(
            f"observations_path '{path}' contains NaN values in the selected "
            "parameter/objective columns. Clean the CSV and retry."
        )
    return {"X": X.tolist(), "Y": Y.tolist()}


def _maybe_hydrate_observations_from_path(payload: dict) -> None:
    """If the payload carries an observations_path, populate observations in-place.

    Mutates `payload` so the downstream handlers (_run_suggest, _run_pareto)
    don't need to know about the path. Raises on any load failure — no silent
    fallback.
    """
    path = payload.get("observations_path")
    if not path:
        return
    if payload.get("observations"):
        raise ValueError(
            "Pass either observations OR observations_path, not both. "
            "Refusing to silently pick one."
        )
    loaded = _load_observations_from_path(
        path,
        parameter_columns=payload.get("parameter_columns"),
        objective_columns=payload.get("objective_columns"),
        parameters=payload.get("parameters"),
        objectives=payload.get("objectives"),
    )
    payload["observations"] = loaded


def _has_botorch_stack() -> bool:
    for pkg in ("torch", "gpytorch", "botorch"):
        try:
            __import__(pkg)
        except ImportError:
            return False
    return True


def _missing_botorch_stack() -> List[str]:
    missing: List[str] = []
    for pkg in ("torch", "gpytorch", "botorch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def _safe_eval_expr(expr: str, namespace: Dict[str, Any]) -> Any:
    """Evaluate a basic arithmetic expression without using eval()."""

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError(f"Unsupported operator in constraint: {ast.dump(node.op)}")
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
        if isinstance(node, ast.Name):
            if node.id not in namespace:
                raise ValueError(f"Unknown variable '{node.id}' in constraint expression")
            return namespace[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def _compile_constraints(
    constraints: List[dict],
    param_names: List[str],
) -> List[Tuple[Callable, bool]]:
    """Compile constraints into BoTorch nonlinear callables."""
    import torch

    compiled: List[Tuple[Callable, bool]] = []
    for constraint in constraints:
        expr = constraint["expression"]
        ctype = constraint["type"]
        rhs = float(constraint["value"])
        idx_map = {name: i for i, name in enumerate(param_names)}

        def _make_fn(expression: str, operator: str, value: float, mapping: Dict[str, int]):
            def fn(X: "torch.Tensor") -> "torch.Tensor":
                namespace = {name: X[..., idx] for name, idx in mapping.items()}
                lhs = _safe_eval_expr(expression, namespace)
                if operator == "<=":
                    return value - lhs
                if operator == ">=":
                    return lhs - value
                return 1e-3 - (lhs - value).abs()

            return fn

        compiled.append((_make_fn(expr, ctype, rhs, idx_map), True))
    return compiled


def _constraints_mask(
    X: np.ndarray,
    constraints: List[dict],
    param_names: List[str],
) -> np.ndarray:
    if not constraints:
        return np.ones(len(X), dtype=bool)
    mask = np.ones(len(X), dtype=bool)
    for idx, row in enumerate(X):
        namespace = {name: row[i] for i, name in enumerate(param_names)}
        for constraint in constraints:
            lhs = float(_safe_eval_expr(constraint["expression"], namespace))
            rhs = float(constraint["value"])
            operator = constraint["type"]
            ok = (
                lhs <= rhs + 1e-8 if operator == "<="
                else lhs >= rhs - 1e-8 if operator == ">="
                else abs(lhs - rhs) <= 1e-3
            )
            if not ok:
                mask[idx] = False
                break
    return mask


def _build_bounds_array(parameters: List[dict]) -> np.ndarray:
    return np.array([p["bounds"] for p in parameters], dtype=float)


def _build_bounds_tensor(parameters: List[dict]) -> "Any":
    import torch

    bounds = np.array([p["bounds"] for p in parameters], dtype=float).T
    return torch.tensor(bounds, dtype=torch.double)


def _postprocess_candidates(X: np.ndarray, parameters: List[dict]) -> np.ndarray:
    processed = X.copy()
    for i, parameter in enumerate(parameters):
        low, high = parameter["bounds"]
        processed[:, i] = np.clip(processed[:, i], low, high)
        if parameter.get("type") == "integer":
            processed[:, i] = np.round(processed[:, i])
    return processed


def _candidates_to_dicts(candidates: np.ndarray, parameters: List[dict]) -> List[dict]:
    rows = _postprocess_candidates(np.asarray(candidates, dtype=float), parameters)
    results: List[dict] = []
    for row in rows:
        entry: Dict[str, Any] = {}
        for i, parameter in enumerate(parameters):
            value = float(row[i])
            if parameter.get("type") == "integer":
                entry[parameter["name"]] = int(round(value))
            else:
                entry[parameter["name"]] = round(value, 6)
        results.append(entry)
    return results


def _transform_Y(raw_Y: np.ndarray, objectives: List[dict]) -> np.ndarray:
    signs = np.array([1.0 if obj["minimize"] else -1.0 for obj in objectives], dtype=float)
    return raw_Y * signs.reshape(1, -1)


def _extract_pareto(train_Y: np.ndarray, objectives: List[dict]) -> np.ndarray:
    Y = _transform_Y(train_Y, objectives)
    n = len(Y)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominates = np.all(Y <= Y[i], axis=1) & np.any(Y < Y[i], axis=1)
        if np.any(dominates):
            mask[i] = False
    return mask


def _synthetic_reflow(X: np.ndarray) -> np.ndarray:
    """Synthetic two-objective reflow surface used for demo mode."""
    temp = X[:, 0]
    speed = X[:, 1]
    paste = X[:, 2]
    yield_pct = (
        95.0
        - 0.015 * (temp - 245.0) ** 2
        - 14.0 * (speed - 1.0) ** 2
        - 600.0 * (paste - 0.14) ** 2
        + np.random.normal(0.0, 0.5, size=len(X))
    )
    energy = 0.1 * temp / np.maximum(speed, 0.1) + np.random.normal(0.0, 0.2, size=len(X))
    return np.column_stack([yield_pct, energy])


def _best_observed(
    train_X: np.ndarray,
    train_Y: np.ndarray,
    objectives: List[dict],
    parameters: List[dict],
) -> dict:
    best: Dict[str, Any] = {}
    for i, obj in enumerate(objectives):
        col = train_Y[:, i]
        idx = int(np.argmin(col) if obj["minimize"] else np.argmax(col))
        point = {
            parameters[j]["name"]: (
                int(round(train_X[idx, j])) if parameters[j].get("type") == "integer"
                else round(float(train_X[idx, j]), 6)
            )
            for j in range(len(parameters))
        }
        best[obj["name"]] = {"value": round(float(train_Y[idx, i]), 4), "parameters": point}
    return best


def _sample_candidate_pool(bounds: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    low = bounds[:, 0]
    high = bounds[:, 1]
    return rng.uniform(low=low, high=high, size=(n_samples, bounds.shape[0]))


def _suggest_botorch_batch(
    train_X_np: np.ndarray,
    train_Y_np: np.ndarray,
    objectives: List[dict],
    parameters: List[dict],
    batch_size: int,
    acquisition: str,
    constraints: List[dict],
) -> np.ndarray:
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.utils.sampling import sample_simplex
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    train_X = torch.tensor(train_X_np, dtype=torch.double)
    train_Y = torch.tensor(_transform_Y(train_Y_np, objectives), dtype=torch.double)
    bounds = _build_bounds_tensor(parameters)
    models = []
    for idx in range(train_Y.shape[1]):
        gp = SingleTaskGP(train_X, train_Y[:, idx : idx + 1], outcome_transform=Standardize(m=1))
        models.append(gp)
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))

    constraint_callables = _compile_constraints(constraints, [p["name"] for p in parameters]) if constraints else None
    nonlinear_constraints = [fn for fn, _ in constraint_callables] if constraint_callables else None

    use_nehvi = acquisition in ("auto", "qnehvi") and len(objectives) <= 3
    if use_nehvi:
        ref_point = []
        for idx, obj in enumerate(objectives):
            col = train_Y_np[:, idx]
            if obj["minimize"]:
                ref_point.append(float(np.max(col) + 0.1 * (np.max(col) - np.min(col) + 1e-6)))
            else:
                ref_point.append(float(-(np.min(col) - 0.1 * (np.max(col) - np.min(col) + 1e-6))))
        partitioning = FastNondominatedPartitioning(ref_point=torch.tensor(ref_point, dtype=torch.double), Y=train_Y)
        _ = partitioning
        acqf = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_X,
            prune_baseline=True,
        )
    else:
        weights = sample_simplex(len(objectives), dtype=torch.double).squeeze()
        scalarization = get_chebyshev_scalarization(weights=weights, Y=train_Y)
        acqf = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=train_X,
            objective=GenericMCObjective(scalarization),
            prune_baseline=True,
        )

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=batch_size,
        num_restarts=20,
        raw_samples=256,
        nonlinear_inequality_constraints=nonlinear_constraints,
    )
    return candidates.detach().cpu().numpy()


def _run_suggest(payload: dict) -> dict:
    _maybe_hydrate_observations_from_path(payload)
    parameters = payload["parameters"]
    objectives = payload["objectives"]
    constraints = payload.get("constraints", [])
    observations = payload["observations"]
    batch_size = int(payload.get("batch_size", 4))
    acquisition = payload.get("acquisition", "auto")

    train_X = np.asarray(observations["X"], dtype=float)
    train_Y = np.asarray(observations["Y"], dtype=float)
    candidate_matrix = _suggest_botorch_batch(
        train_X,
        train_Y,
        objectives,
        parameters,
        batch_size,
        acquisition,
        constraints,
    )
    engine_used = "botorch"

    pareto_mask = _extract_pareto(train_Y, objectives)
    return {
        "valid": True,
        "engine_used": engine_used,
        "suggestions": _candidates_to_dicts(candidate_matrix, parameters),
        "pareto_front": train_Y[pareto_mask].tolist(),
        "pareto_X": train_X[pareto_mask].tolist(),
        "best_observed": _best_observed(train_X, train_Y, objectives, parameters),
        "summary": (
            f"Suggested {len(candidate_matrix)} experiment(s) using {engine_used}. "
            f"Current Pareto front has {int(pareto_mask.sum())} point(s)."
        ),
    }


def _run_optimize(payload: dict) -> dict:
    parameters = payload["parameters"]
    objectives = payload["objectives"]
    constraints = payload.get("constraints", [])
    batch_size = int(payload.get("batch_size", 4))
    n_iterations = int(payload.get("n_iterations", 20))
    acquisition = payload.get("acquisition", "auto")

    bounds = _build_bounds_array(parameters)
    rng = np.random.default_rng(123)
    n_initial = max(2 * len(parameters), 6)
    train_X = _postprocess_candidates(_sample_candidate_pool(bounds, n_initial, rng), parameters)
    mask = _constraints_mask(train_X, constraints, [p["name"] for p in parameters])
    train_X = train_X[mask] if np.any(mask) else train_X
    train_Y = _synthetic_reflow(train_X)

    trace: List[dict] = []
    engine_used = "botorch"
    for iteration in range(n_iterations):
        candidates = _suggest_botorch_batch(
            train_X,
            train_Y,
            objectives,
            parameters,
            batch_size,
            acquisition,
            constraints,
        )
        candidates = _postprocess_candidates(candidates, parameters)
        new_Y = _synthetic_reflow(candidates)
        train_X = np.vstack([train_X, candidates])
        train_Y = np.vstack([train_Y, new_Y])

        pareto_mask = _extract_pareto(train_Y, objectives)
        trace.append(
            {
                "iteration": iteration + 1,
                "n_pareto": int(pareto_mask.sum()),
                "best_yield": round(float(np.max(train_Y[:, 0])), 3),
                "best_energy": round(float(np.min(train_Y[:, 1])), 3),
            }
        )

    pareto_mask = _extract_pareto(train_Y, objectives)
    return {
        "valid": True,
        "engine_used": engine_used,
        "suggestions": [],
        "pareto_front": train_Y[pareto_mask].tolist(),
        "pareto_X": train_X[pareto_mask].tolist(),
        "optimization_trace": trace,
        "best_observed": _best_observed(train_X, train_Y, objectives, parameters),
        "summary": (
            f"Completed {n_iterations} optimization iteration(s) using {engine_used}. "
            f"Pareto front has {int(pareto_mask.sum())} non-dominated solution(s)."
        ),
    }


def _run_pareto(payload: dict) -> dict:
    _maybe_hydrate_observations_from_path(payload)
    objectives = payload["objectives"]
    observations = payload["observations"]
    parameters = payload.get("parameters", [])

    train_X = np.asarray(observations["X"], dtype=float)
    train_Y = np.asarray(observations["Y"], dtype=float)
    pareto_mask = _extract_pareto(train_Y, objectives)
    return {
        "valid": True,
        "engine_used": "pareto_only",
        "pareto_front": train_Y[pareto_mask].tolist(),
        "pareto_X": train_X[pareto_mask].tolist(),
        "best_observed": _best_observed(train_X, train_Y, objectives, parameters),
        "summary": f"Extracted {int(pareto_mask.sum())} Pareto-optimal points from {len(train_Y)} observation(s).",
    }


def _run_self_test(_payload: dict) -> dict:
    demo_payload = {
        "parameters": [
            {"name": "temperature", "bounds": [200.0, 280.0], "type": "continuous"},
            {"name": "belt_speed", "bounds": [0.5, 2.0], "type": "continuous"},
            {"name": "paste_thickness", "bounds": [0.08, 0.2], "type": "continuous"},
        ],
        "objectives": [
            {"name": "yield_pct", "minimize": False},
            {"name": "energy_kWh", "minimize": True},
        ],
        "constraints": [
            {"expression": "temperature", "type": "<=", "value": 260.0},
        ],
        "batch_size": 4,
        "n_iterations": 12,
        "acquisition": "auto",
    }
    result = _run_optimize(demo_payload)
    result["self_test"] = True
    result["summary"] = (
        "SELF-TEST (Solder Reflow Optimization): "
        + result["summary"]
        + " Constraint enforced: temperature <= 260 C."
    )
    return result


class SandboxLabWrapper:
    """Atlas-compatible wrapper for offline experiment planning."""

    name = "sandbox_lab"

    async def invoke(self, payload: Optional[dict] = None, _context: Optional[dict] = None) -> dict:
        args = payload or {}
        mode = args.get("mode", "suggest")
        missing = _missing_botorch_stack()
        if missing:
            return {
                "valid": False,
                "error": (
                    "sandbox_lab is unavailable because the required optimization stack is missing: "
                    + ", ".join(missing)
                    + ". Install them with: pip install "
                    + " ".join(missing)
                ),
                "summary": (
                    "Sandbox Lab is not ready on this machine. "
                    f"Missing dependencies: {', '.join(missing)}."
                ),
            }
        dispatch = {
            "suggest": _run_suggest,
            "optimize": _run_optimize,
            "pareto": _run_pareto,
            "self_test": _run_self_test,
        }
        handler = dispatch.get(mode)
        if handler is None:
            return {
                "valid": False,
                "summary": f"Unknown mode '{mode}'. Choose from: {', '.join(dispatch)}.",
            }

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(_EXECUTOR, handler, args)
        except Exception as exc:
            logger.exception("sandbox_lab failed in mode '%s'", mode)
            return {"valid": False, "summary": f"Error in {mode}: {exc}", "error": str(exc)}


PLUGIN = SandboxLabWrapper()
