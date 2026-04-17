"""Causal discovery with Tigramite PCMCI+ and PySR."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy

logger = logging.getLogger(__name__)


def _has_tigramite() -> bool:
    try:
        import tigramite  # noqa: F401
        return True
    except ImportError:
        return False


def _has_pysr() -> bool:
    try:
        import pysr  # noqa: F401
        return True
    except ImportError:
        return False


def _missing_stack() -> List[str]:
    missing: List[str] = []
    if not _has_tigramite():
        missing.append("tigramite")
    if not _has_pysr():
        missing.append("pysr")
    return missing


def _run_pcmci(data_array, variable_names: List[str], max_lag: int) -> dict:
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    data_np = np.asarray(data_array, dtype=np.float64)
    dataframe = pp.DataFrame(data_np, var_names=variable_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmciplus(tau_min=1, tau_max=max_lag, pc_alpha=0.05)

    graph = results["graph"]
    val_matrix = results["val_matrix"]
    p_matrix = results["p_matrix"]
    n_vars = len(variable_names)
    edges = []
    parent_sets: Dict[str, List[Dict[str, Any]]] = {name: [] for name in variable_names}
    for tau in range(graph.shape[2]):
        for source_idx in range(n_vars):
            for target_idx in range(n_vars):
                edge_type = graph[source_idx, target_idx, tau]
                if edge_type not in ("-->", "<--", "o-o"):
                    continue
                edge = {
                    "source": variable_names[source_idx],
                    "target": variable_names[target_idx],
                    "lag": int(tau),
                    "type": str(edge_type),
                    "strength": float(val_matrix[source_idx, target_idx, tau]),
                    "p_value": float(p_matrix[source_idx, target_idx, tau]),
                }
                edges.append(edge)
                if edge_type == "-->":
                    parent_sets[variable_names[target_idx]].append(
                        {"variable": variable_names[source_idx], "lag": int(tau), "strength": edge["strength"]}
                    )
    return {
        "engine_used": "pcmci+pysr",
        "causal_graph": {"edges": edges, "variable_names": variable_names},
        "parent_sets": parent_sets,
    }


def _get_parents_for_target(parent_sets: dict, target: str) -> List[Tuple[str, int]]:
    raw = parent_sets.get(target, [])
    return [(entry["variable"], int(entry["lag"])) for entry in raw]


def _build_lagged_features(
    data_array,
    variable_names: List[str],
    parents: List[Tuple[str, int]],
    target: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data_np = np.asarray(data_array, dtype=np.float64)
    var_idx = {name: i for i, name in enumerate(variable_names)}
    if not parents:
        raise ValueError(f"No causal parents discovered for target '{target}'.")

    max_parent_lag = max(max(lag for _, lag in parents), 1)
    start = max_parent_lag
    feature_names: List[str] = []
    feature_columns = []
    for parent_name, lag in parents:
        idx = var_idx[parent_name]
        feature_names.append(f"{parent_name}_lag{lag}")
        feature_columns.append(data_np[start - lag : data_np.shape[0] - lag, idx])
    X = np.column_stack(feature_columns)
    y = data_np[start:, var_idx[target]]
    return X, y, feature_names


def _run_pysr(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> dict:
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["exp", "log", "sqrt"],
        maxsize=20,
        verbosity=0,
        temp_equation_file=True,
    )
    model.fit(X, y, variable_names=feature_names)
    equations = []
    if getattr(model, "equations_", None) is not None:
        for _, row in model.equations_.iterrows():
            equations.append(
                {
                    "complexity": int(row["complexity"]),
                    "loss": float(row["loss"]),
                    "equation": str(row["equation"]),
                }
            )
    predictions = np.asarray(model.predict(X), dtype=float)
    r2 = 1.0 - float(np.var(y - predictions) / max(np.var(y), 1e-9))
    return {
        "engine_used": "pysr",
        "best_equation": str(model.sympy()),
        "pareto_front": equations,
        "feature_names": feature_names,
        "r2": round(r2, 4),
    }


def _rank_interventions(equation_str: str, feature_names: List[str], X: np.ndarray) -> List[dict]:
    symbols = {name: sympy.Symbol(name) for name in feature_names}
    try:
        expr = sympy.sympify(equation_str, locals=symbols)
    except (sympy.SympifyError, TypeError):
        return [{"variable": name, "sensitivity": None} for name in feature_names]

    means = {name: float(np.mean(X[:, idx])) for idx, name in enumerate(feature_names)}
    ranking = []
    for name in feature_names:
        derivative = sympy.diff(expr, symbols[name])
        try:
            value = float(derivative.evalf(subs={symbols[key]: means[key] for key in feature_names}))
            sensitivity = abs(value)
        except Exception:
            sensitivity = None
        ranking.append(
            {
                "variable": name,
                "partial_derivative": str(derivative),
                "sensitivity": sensitivity,
            }
        )
    ranking.sort(key=lambda item: item["sensitivity"] if item["sensitivity"] is not None else -1, reverse=True)
    return ranking


def _load_table_from_csv(data_path: str) -> Tuple[List[List[float]], List[str]]:
    """Load a tabular CSV into (2D float list, variable_names). Fails loud.

    The CSV must have a header row and contain only numeric columns; NaNs are
    rejected because PCMCI+ cannot consume them.
    """
    if not isinstance(data_path, str) or not data_path.strip():
        raise RuntimeError("data_path must be a non-empty string.")
    if not os.path.isfile(data_path):
        raise RuntimeError(
            f"data_path does not point to an existing file: {data_path!r}."
        )
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Loading 'data_path' requires pandas (pip install pandas)."
        ) from exc
    try:
        df = pd.read_csv(data_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse CSV at {data_path!r}: {exc}"
        ) from exc
    if df.shape[1] < 2:
        raise RuntimeError(
            f"CSV at {data_path!r} needs >= 2 columns for causal discovery, "
            f"got {df.shape[1]}."
        )
    non_numeric = [c for c in df.columns if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric:
        raise RuntimeError(
            f"Non-numeric columns in {data_path!r}: {non_numeric}. "
            "Drop or numerically encode them before passing to causal_discovery."
        )
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise RuntimeError(
            f"CSV at {data_path!r} contains NaN values in columns: {bad}. "
            "Clean the data before passing it to causal_discovery."
        )
    variable_names = [str(c) for c in df.columns]
    data = df.astype(float).values.tolist()
    return data, variable_names


def _generate_demo_data(T: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    names = ["oven_temperature", "belt_speed", "paste_thickness", "defect_rate"]
    data = np.zeros((T, 4))
    data[0, 0] = 200.0
    data[0, 1] = 50.0
    for t in range(1, T):
        data[t, 0] = data[t - 1, 0] + rng.normal(0, 1.0)
        data[t, 1] = data[t - 1, 1] + rng.normal(0, 0.5)
        data[t, 2] = 0.7 * data[t - 1, 0] + rng.normal(0, 2.0)
    for t in range(2, T):
        data[t, 3] = 0.0005 * data[t - 1, 2] ** 2 + 0.5 * data[t - 2, 1] + rng.normal(0, 1.5)
    return data.tolist(), names


class CausalDiscoveryWrapper:
    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        _context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        mode = args.get("mode", "full")
        missing = _missing_stack()
        if missing:
            return {
                "valid": False,
                "error": (
                    "causal_discovery is unavailable because the required stack is missing: "
                    + ", ".join(missing)
                    + ". Install them with: pip install "
                    + " ".join(missing)
                ),
                "summary": (
                    "Causal Discovery is not ready on this machine. "
                    f"Missing dependencies: {', '.join(missing)}."
                ),
            }
        if mode == "self_test":
            return await self._run_self_test()

        data = args.get("data")
        variable_names = args.get("variable_names")
        target_variable = args.get("target_variable")
        target_column = args.get("target_column")
        data_path = args.get("data_path")
        max_lag = int(args.get("max_lag", 5))

        if data is None and data_path:
            # File-loading path: loud failure on missing file / bad columns.
            data, variable_names = _load_table_from_csv(data_path)

        if data is None or variable_names is None:
            return {
                "valid": False,
                "error": (
                    "Provide either ('data' + 'variable_names') or 'data_path' "
                    "pointing at a header-rowed numeric CSV."
                ),
            }

        if target_variable is None and target_column is not None:
            target_variable = target_column

        if target_variable is None:
            target_variable = variable_names[-1]
        if isinstance(target_variable, int):
            target_variable = variable_names[target_variable]
        if target_variable not in variable_names:
            return {"valid": False, "error": f"Target variable '{target_variable}' not found."}

        loop = asyncio.get_running_loop()

        if mode == "discover":
            discovery = await loop.run_in_executor(None, _run_pcmci, data, variable_names, max_lag)
            discovery["valid"] = True
            discovery["summary"] = self._summarize_discovery(discovery, target_variable)
            return discovery

        if mode == "equations":
            parent_sets_raw = args.get("parent_sets")
            if parent_sets_raw is None:
                return {"valid": False, "error": "Mode 'equations' requires 'parent_sets'."}
            parents = _get_parents_for_target(parent_sets_raw, target_variable)
            if not parents:
                return {"valid": False, "error": f"No parents found for target '{target_variable}'."}
            return await self._run_equations(data, variable_names, target_variable, parents)

        return await self._run_full(data, variable_names, target_variable, max_lag)

    async def _run_equations(
        self,
        data,
        variable_names,
        target_variable: str,
        parents: List[Tuple[str, int]],
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        X, y, feature_names = await loop.run_in_executor(None, _build_lagged_features, data, variable_names, parents, target_variable)
        equation_result = await loop.run_in_executor(None, _run_pysr, X, y, feature_names)
        ranking = await loop.run_in_executor(None, _rank_interventions, equation_result["best_equation"], feature_names, X)
        return {
            "valid": True,
            "engine_used": equation_result.get("engine_used"),
            "equations": equation_result,
            "intervention_ranking": ranking,
            "summary": f"Built equation for {target_variable} using {equation_result.get('engine_used')}.",
        }

    async def _run_full(self, data, variable_names, target_variable: str, max_lag: int) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        discovery = await loop.run_in_executor(None, _run_pcmci, data, variable_names, max_lag)
        parents = _get_parents_for_target(discovery["parent_sets"], target_variable)
        if not parents:
            return {
                "valid": True,
                **discovery,
                "equations": None,
                "intervention_ranking": [],
                "summary": f"No causal parents discovered for '{target_variable}'.",
            }

        equation_bundle = await self._run_equations(data, variable_names, target_variable, parents)
        ranking = equation_bundle["intervention_ranking"]
        top_variable = ranking[0]["variable"] if ranking else "n/a"
        return {
            "valid": True,
            **discovery,
            "engine_used": discovery.get("engine_used"),
            "equations": equation_bundle["equations"],
            "intervention_ranking": ranking,
            "summary": (
                f"Recovered {len(parents)} causal parent(s) for '{target_variable}' using {discovery.get('engine_used')}. "
                f"Top intervention lever: {top_variable}."
            ),
        }

    async def _run_self_test(self) -> Dict[str, Any]:
        data, names = _generate_demo_data()
        result = await self._run_full(data, names, "defect_rate", max_lag=5)
        ground_truth = {("paste_thickness", 1), ("belt_speed", 2)}
        discovered = set(_get_parents_for_target(result.get("parent_sets", {}), "defect_rate"))
        hits = len(ground_truth & discovered)
        parent_recall = hits / len(ground_truth)
        result["self_test"] = True
        result["parent_recall"] = round(parent_recall, 4)
        result["valid"] = bool(result.get("equations")) and parent_recall >= 0.5
        result["ground_truth"] = sorted(
            [{"variable": variable, "lag": lag} for variable, lag in ground_truth],
            key=lambda item: (item["variable"], item["lag"]),
        )
        status = "PASS" if result["valid"] else "FAIL"
        result["summary"] += f" Synthetic benchmark {status}; parent recall: {parent_recall:.2%}."
        return result

    @staticmethod
    def _summarize_discovery(discovery: dict, target: str) -> str:
        parents = discovery["parent_sets"].get(target, [])
        if not parents:
            return f"No causal parents discovered for '{target}'."
        return (
            f"Discovered {len(parents)} parent(s) for '{target}' using {discovery.get('engine_used')}: "
            + ", ".join(f"{item['variable']} (lag {item['lag']})" for item in parents)
            + "."
        )


PLUGIN = CausalDiscoveryWrapper()
