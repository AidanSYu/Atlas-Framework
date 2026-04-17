"""Manufacturing World Model — time-series forecasting + anomaly detection plugin.

Fully self-contained: imports NOTHING from app.*.
Supports multiple backends: chronos2, timesfm25, ttm, moment, stats.
Includes built-in demo with synthetic reflow oven temperature data.
"""

import asyncio
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)

# Backend priority order for auto-selection.
# TimesFM 2.5 is the flagship backend — strongest zero-shot accuracy and a 16K
# context window suit slow-drift manufacturing signatures. Others are fallbacks.
_BACKEND_ORDER = ["timesfm25", "chronos2", "ttm", "moment", "stats"]


# ---------------------------------------------------------------------------
# Backend availability probes
# ---------------------------------------------------------------------------

def _probe_backend(name: str) -> bool:
    """Check if a backend's dependencies are importable."""
    try:
        if name == "chronos2":
            import chronos  # noqa: F401
            import torch  # noqa: F401
            return True
        elif name == "timesfm25":
            import timesfm  # noqa: F401
            return True
        elif name == "ttm":
            import tsfm_public  # noqa: F401
            return True
        elif name == "moment":
            import momentfm  # noqa: F401
            return True
        elif name == "stats":
            import pandas  # noqa: F401
            import statsforecast  # noqa: F401
            return True
    except ImportError:
        pass
    return False


def _has_ruptures() -> bool:
    try:
        import ruptures  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_auto_backend() -> str:
    """Return the first importable backend, or raise."""
    for name in _BACKEND_ORDER:
        if _probe_backend(name):
            logger.info("Auto-selected backend: %s", name)
            return name
    raise RuntimeError("No forecasting backend is available.")


# ---------------------------------------------------------------------------
# Backend loaders — each returns an opaque model object
# ---------------------------------------------------------------------------

def _load_chronos2() -> Any:
    from chronos import ChronosPipeline
    import torch
    return ChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )


def _load_timesfm25(horizon: int, adapter_path: Optional[str] = None) -> Any:
    import timesfm
    import torch
    from huggingface_hub import hf_hub_download

    # NOTE: timesfm 2.5's `from_pretrained` leaks HuggingFace hub kwargs
    # (`proxies`, `resume_download`, ...) into the class __init__, which
    # rejects them. Workaround: download weights explicitly, then init the
    # class directly and call its load_checkpoint path.
    cls = timesfm.TimesFM_2p5_200M_torch
    weights_path = hf_hub_download(
        repo_id="google/timesfm-2.5-200m-pytorch",
        filename=cls.WEIGHTS_FILENAME,
    )
    use_compile = torch.cuda.is_available()
    model = cls(torch_compile=use_compile)
    model.model.load_checkpoint(weights_path, torch_compile=use_compile)

    if adapter_path:
        # Optional: load a LoRA adapter produced by finetune.py on top of the
        # frozen base weights. Absent adapter → zero-shot inference.
        try:
            from peft import PeftModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "adapter_path supplied but 'peft' is not installed (pip install peft)."
            ) from exc
        inner = getattr(model, "model", model)
        wrapped = PeftModel.from_pretrained(inner, adapter_path)
        if hasattr(model, "model"):
            model.model = wrapped
        else:
            model = wrapped
        logger.info("Loaded LoRA adapter from %s", adapter_path)
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=max(256, horizon),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def _load_ttm() -> Any:
    from tsfm_public import TinyTimeMixerForPrediction
    return TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2"
    )


def _load_moment(task: str = "forecasting") -> Any:
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", model_task=task
    )
    model.init()
    return model


def _load_stats() -> Any:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS

    return StatsForecast(models=[AutoETS(), AutoARIMA()], freq=1)


# ---------------------------------------------------------------------------
# Backend predict functions — all return (median, lower, upper) np arrays
# ---------------------------------------------------------------------------

def _predict_chronos2(
    model: Any, values: np.ndarray, horizon: int, confidence: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    context = torch.tensor(values, dtype=torch.float32)
    forecast = model.predict(
        context=context, prediction_length=horizon, num_samples=20
    )
    samples = forecast.numpy()
    alpha = (1 - confidence) / 2
    low = np.quantile(samples, alpha, axis=1).flatten()[:horizon]
    median = np.quantile(samples, 0.5, axis=1).flatten()[:horizon]
    high = np.quantile(samples, 1 - alpha, axis=1).flatten()[:horizon]
    return median, low, high


def _predict_timesfm25(
    model: Any, values: np.ndarray, horizon: int, confidence: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    point, quantile = model.forecast(horizon=horizon, inputs=[values])
    # point: (batch=1, horizon); quantile: (batch=1, horizon, 10) — [mean, q10..q90]
    median = np.asarray(point)[0, :horizon]
    q = np.asarray(quantile)[0]  # (horizon, 10)
    # Map confidence to the nearest decile available in the quantile head.
    # Indices: 0=mean, 1=q10, 2=q20, ..., 9=q90.
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(1, min(9, int(round(alpha * 10))))
    hi_idx = max(1, min(9, int(round((1.0 - alpha) * 10))))
    low = q[:horizon, lo_idx]
    high = q[:horizon, hi_idx]
    return median, low, high


def _z_value(confidence: float) -> float:
    return float(NormalDist().inv_cdf(1 - (1 - confidence) / 2))


def _predict_ttm(
    model: Any, values: np.ndarray, horizon: int, confidence: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    context_length = 512
    arr = values[-context_length:] if len(values) > context_length else values
    padded = np.zeros(context_length)
    padded[-len(arr):] = arr
    # Shape: (batch=1, context_length, channels=1)
    input_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        output = model(input_tensor)
    preds = output.prediction_outputs.squeeze().cpu().numpy()
    median = preds[:horizon] if len(preds) >= horizon else np.pad(preds, (0, horizon - len(preds)), mode="edge")
    std_est = np.std(values[-100:]) if len(values) >= 100 else np.std(values)
    z = _z_value(confidence)
    low = median - z * std_est
    high = median + z * std_est
    return median, low, high


def _predict_moment(
    model: Any, values: np.ndarray, horizon: int, confidence: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    input_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        output = model(input_tensor)
    preds = output.forecast.squeeze().cpu().numpy()
    median = preds[:horizon] if len(preds) >= horizon else np.pad(preds, (0, horizon - len(preds)), mode="edge")
    std_est = np.std(values[-100:]) if len(values) >= 100 else np.std(values)
    z = _z_value(confidence)
    low = median - z * std_est
    high = median + z * std_est
    return median, low, high


def _predict_stats(
    model: Any, values: np.ndarray, horizon: int, confidence: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.DataFrame(
        {
            "unique_id": ["series_0"] * len(values),
            "ds": list(range(len(values))),
            "y": values,
        }
    )
    fcst = model.forecast(df=df, h=horizon, level=[int(confidence * 100)])
    forecast_columns = [
        column
        for column in fcst.columns
        if column not in ("unique_id", "ds") and "-lo-" not in column and "-hi-" not in column
    ]
    if not forecast_columns:
        raise RuntimeError("statsforecast did not return a point forecast column")
    col_med = "AutoETS" if "AutoETS" in forecast_columns else forecast_columns[0]
    median = fcst[col_med].values[:horizon]
    level_pct = int(confidence * 100)
    lo_col = f"{col_med}-lo-{level_pct}"
    hi_col = f"{col_med}-hi-{level_pct}"
    if lo_col in fcst.columns:
        low = fcst[lo_col].values[:horizon]
        high = fcst[hi_col].values[:horizon]
    else:
        std_est = np.std(values[-100:]) if len(values) >= 100 else np.std(values)
        z = _z_value(confidence)
        low = median - z * std_est
        high = median + z * std_est
    return median, low, high


# ---------------------------------------------------------------------------
# Anomaly scoring (z-score on forecast residuals)
# ---------------------------------------------------------------------------

def _compute_anomaly_scores(
    values: np.ndarray, forecast_median: np.ndarray, window: int = 100
) -> Tuple[List[float], List[int]]:
    """Compute rolling z-score anomaly scores on residuals.

    Returns (scores, flagged_indices) where flagged means |z| > 3.
    """
    n = min(len(values), len(forecast_median))
    residuals = values[:n] - forecast_median[:n]
    scores = np.zeros(n)

    for i in range(n):
        start = max(0, i - window)
        chunk = residuals[start:i + 1]
        mu = np.mean(chunk)
        sigma = np.std(chunk)
        if sigma < 1e-9:
            scores[i] = 0.0
        else:
            scores[i] = (residuals[i] - mu) / sigma

    flagged = [int(i) for i in range(n) if abs(scores[i]) > 3.0]
    return scores.tolist(), flagged


# ---------------------------------------------------------------------------
# Change-point detection via ruptures PELT
# ---------------------------------------------------------------------------

def _detect_changepoints(values: np.ndarray, pen: float = 10.0) -> List[int]:
    """Detect change-points using the PELT algorithm from the ruptures library."""
    try:
        import ruptures as rpt
    except ImportError as exc:
        raise RuntimeError(
            "Changepoint detection requires the 'ruptures' package. Install it with: pip install ruptures"
        ) from exc
    algo = rpt.Pelt(model="rbf").fit(np.array(values))
    changepoints = algo.predict(pen=pen)
    # ruptures returns the last index as len(values), drop it
    return [int(cp) for cp in changepoints if cp < len(values)]


def _available_backends() -> List[str]:
    return [name for name in _BACKEND_ORDER if _probe_backend(name)]


def _self_test_readiness() -> List[str]:
    missing: List[str] = []
    if not _available_backends():
        missing.append(
            "No supported forecasting backend is installed. Install one of: "
            "chronos-forecasting, timesfm, tsfm_public, momentfm, or statsforecast."
        )
    if not _has_ruptures():
        missing.append("Changepoint detection requires 'ruptures'. Install it with: pip install ruptures")
    return missing


# ---------------------------------------------------------------------------
# Conformal prediction intervals
# ---------------------------------------------------------------------------

def _conformal_intervals(
    values: np.ndarray,
    predict_fn,
    horizon: int,
    confidence: float,
) -> Dict[str, Any]:
    """Calibrate prediction intervals via split-conformal method.

    Hold out the last 20% as calibration set, fit on the first 80%.
    """
    n = len(values)
    cal_size = max(int(n * 0.2), horizon + 1)
    train = values[:-cal_size]
    cal_actual = values[-cal_size:]

    if len(train) < 10:
        return {"calibrated": False, "reason": "insufficient training data for conformal calibration"}

    # Produce rolling one-step forecasts on cal set
    errors = []
    for i in range(len(cal_actual)):
        context = np.concatenate([train, cal_actual[:i]]) if i > 0 else train
        med, _, _ = predict_fn(context, 1)
        errors.append(abs(cal_actual[i] - med[0]))

    errors = np.array(errors)
    # Quantile of absolute errors at the requested confidence level
    q = np.quantile(errors, confidence)

    # Now produce the actual forecast
    med, _, _ = predict_fn(values, horizon)
    return {
        "calibrated": True,
        "median": med.tolist(),
        "lower": (med - q).tolist(),
        "upper": (med + q).tolist(),
        "conformity_score": float(q),
    }


# ---------------------------------------------------------------------------
# Synthetic reflow oven demo data
# ---------------------------------------------------------------------------

def _load_series_from_csv(
    data_path: str,
    value_column: str,
    timestamp_column: Optional[str],
) -> Tuple[List[float], Optional[List[str]]]:
    """Load a univariate time-series from a CSV. Fails loud on any data issue.

    Returns (values, timestamps_or_None).
    """
    if not isinstance(data_path, str) or not data_path.strip():
        raise RuntimeError("data_path must be a non-empty string.")
    if not os.path.isfile(data_path):
        raise RuntimeError(
            f"data_path does not point to an existing file: {data_path!r}."
        )
    try:
        import pandas as pd  # local import — pandas is already required for 'stats' backend
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

    if value_column not in df.columns:
        raise RuntimeError(
            f"value_column {value_column!r} not found in {data_path!r}. "
            f"Available columns: {list(df.columns)}"
        )
    col = df[value_column]
    try:
        values = col.astype(float).tolist()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"value_column {value_column!r} is not numeric (dtype={col.dtype})."
        ) from exc
    if any(v != v for v in values):  # NaN check
        raise RuntimeError(
            f"value_column {value_column!r} contains NaN values in {data_path!r}. "
            "Clean the CSV or forward-fill before handing it to the plugin."
        )

    timestamps: Optional[List[str]] = None
    if timestamp_column and timestamp_column in df.columns:
        timestamps = [str(v) for v in df[timestamp_column].tolist()]
    return values, timestamps


def _generate_reflow_demo() -> Tuple[np.ndarray, str]:
    """Generate 500-point synthetic reflow oven temperature profile.

    - Normal operating range ~240C with sinusoidal cycle
    - Drift starting at t=350: gradual +5C over 50 points
    - Sudden spike at t=420
    """
    rng = np.random.default_rng(42)
    n = 500
    t = np.arange(n, dtype=float)

    # Base: sinusoidal around 240C
    base = 240.0 + 3.0 * np.sin(2.0 * np.pi * t / 60.0)

    # Noise
    noise = rng.normal(0, 0.5, size=n)

    # Drift: gradual +5C from t=350 to t=400
    drift = np.zeros(n)
    for i in range(350, 400):
        drift[i] = 5.0 * (i - 350) / 50.0
    drift[400:] = 5.0

    # Spike at t=420
    spike = np.zeros(n)
    spike[420] = 25.0
    spike[421] = 12.0

    values = base + noise + drift + spike

    description = (
        "Synthetic reflow oven profile: 500 points at ~240C with sinusoidal cycle. "
        "Injected gradual drift (+5C) starting at t=350 and sudden spike at t=420."
    )
    return values, description


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class ManufacturingWorldModelWrapper:
    """Time-series forecasting and anomaly detection for manufacturing data."""

    def __init__(self) -> None:
        self._backends: Dict[str, Any] = {}

    def _get_or_load_backend(
        self,
        name: str,
        horizon: int = 64,
        task: str = "forecasting",
        adapter_path: Optional[str] = None,
    ) -> Any:
        """Load and cache a backend model. Adapter is part of the cache key."""
        cache_key = f"{name}_{task}_{adapter_path or ''}"
        if cache_key in self._backends:
            return self._backends[cache_key]

        if not _probe_backend(name):
            raise RuntimeError(
                f"Backend '{name}' is not installed. "
                f"Required package: {_backend_package(name)}. "
                f"Install it and retry."
            )

        logger.info("Loading backend: %s (task=%s, adapter=%s)", name, task, adapter_path or "none")
        if name == "chronos2":
            model = _load_chronos2()
        elif name == "timesfm25":
            model = _load_timesfm25(horizon, adapter_path=adapter_path)
        elif name == "ttm":
            model = _load_ttm()
        elif name == "moment":
            model = _load_moment(task)
        elif name == "stats":
            model = _load_stats()
        else:
            raise ValueError(f"Unknown backend: {name}")

        self._backends[cache_key] = model
        return model

    def _predict_with_backend(
        self, name: str, values: np.ndarray, horizon: int, confidence: float,
        task: str = "forecasting", adapter_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run prediction through the named backend."""
        model = self._get_or_load_backend(name, horizon, task, adapter_path=adapter_path)

        dispatch = {
            "chronos2": _predict_chronos2,
            "timesfm25": _predict_timesfm25,
            "ttm": _predict_ttm,
            "moment": _predict_moment,
            "stats": _predict_stats,
        }
        fn = dispatch[name]
        return fn(model, values, horizon, confidence)

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        mode = args.get("mode", "forecast")

        if mode == "self_test":
            return await self._run_self_test()

        values_raw = args.get("values")
        timestamps_raw = args.get("timestamps")
        data_path = args.get("data_path")

        if (values_raw is None or len(values_raw) == 0) and data_path:
            # File-loading path: surface errors loudly — no silent fallback to
            # self_test or synthetic data.
            value_column = args.get("value_column") or "value"
            timestamp_column = args.get("timestamp_column") or "timestamp"
            values_raw, loaded_timestamps = _load_series_from_csv(
                data_path, value_column, timestamp_column
            )
            if timestamps_raw is None:
                timestamps_raw = loaded_timestamps

        if values_raw is None or len(values_raw) == 0:
            return {
                "valid": False,
                "error": (
                    "No 'values' provided. Supply an array of numeric time-series "
                    "values OR a 'data_path' pointing at a CSV with a 'value' column."
                ),
            }

        values = np.array(values_raw, dtype=float)
        horizon = int(args.get("horizon", 64))
        backend_name = args.get("backend", "auto")
        confidence = float(args.get("confidence_level", 0.9))
        adapter_path = args.get("adapter_path") or None

        if mode in ("changepoint", "full") and not _has_ruptures():
            return {
                "valid": False,
                "error": "Changepoint detection requires the 'ruptures' package. Install it with: pip install ruptures",
                "summary": "Manufacturing World Model is not ready for changepoint analysis on this machine.",
            }

        # Resolve backend
        if backend_name == "auto":
            try:
                backend_name = _resolve_auto_backend()
            except RuntimeError as exc:
                return {"valid": False, "error": str(exc)}
        elif not _probe_backend(backend_name):
            return {
                "valid": False,
                "error": (
                    f"Backend '{backend_name}' is not installed. "
                    f"Required package: {_backend_package(backend_name)}. "
                    f"Install it or use backend='auto'."
                ),
            }

        loop = asyncio.get_running_loop()

        result: Dict[str, Any] = {"valid": True, "backend_used": backend_name}
        summaries: List[str] = []

        # --- Forecast ---
        if mode in ("forecast", "full"):
            try:
                median, low, high = await loop.run_in_executor(
                    _executor,
                    lambda: self._predict_with_backend(
                        backend_name, values, horizon, confidence, adapter_path=adapter_path,
                    ),
                )
                result["forecast"] = {
                    "median": median.tolist(),
                    "lower": low.tolist(),
                    "upper": high.tolist(),
                    "horizon": horizon,
                }

                # Conformal calibration
                def _conformal_predict_fn(v: np.ndarray, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    return self._predict_with_backend(
                        backend_name, v, h, confidence, adapter_path=adapter_path,
                    )

                try:
                    conf_result = await loop.run_in_executor(
                        _executor,
                        lambda: _conformal_intervals(values, _conformal_predict_fn, horizon, confidence),
                    )
                    result["prediction_intervals"] = conf_result
                except Exception as exc:
                    logger.warning("Conformal calibration failed: %s", exc)
                    result["prediction_intervals"] = {
                        "calibrated": False,
                        "reason": str(exc),
                    }

                summaries.append(f"Forecast {horizon} steps via {backend_name}")
            except Exception as exc:
                result["forecast"] = {"error": str(exc)}
                summaries.append(f"Forecast failed: {exc}")

        # --- Anomaly scoring ---
        if mode in ("anomaly", "full"):
            try:
                # Use one-step-ahead residual scoring
                # Produce in-sample "forecast" by sliding window
                window_size = min(100, len(values) // 2)
                if window_size < 5:
                    result["anomaly_scores"] = []
                    summaries.append("Anomaly: insufficient data")
                else:
                    in_sample_median = np.full(len(values), np.nan)
                    # Simple rolling mean as fast in-sample proxy
                    for i in range(window_size, len(values)):
                        in_sample_median[i] = np.mean(values[i - window_size:i])
                    # Fill initial points with first valid
                    first_valid = np.nanmean(values[:window_size])
                    in_sample_median[:window_size] = first_valid

                    scores, flagged = _compute_anomaly_scores(values, in_sample_median)
                    result["anomaly_scores"] = scores
                    result["anomaly_flagged_indices"] = flagged
                    summaries.append(f"Anomaly: {len(flagged)} points flagged (|z|>3)")
            except Exception as exc:
                result["anomaly_scores"] = []
                summaries.append(f"Anomaly failed: {exc}")

        # --- Changepoint detection ---
        if mode in ("changepoint", "full"):
            try:
                cps = await loop.run_in_executor(
                    _executor,
                    lambda: _detect_changepoints(values),
                )
                result["changepoints"] = cps
                summaries.append(f"Changepoints: {len(cps)} detected at {cps}")
            except RuntimeError as exc:
                result["changepoints"] = []
                result["changepoint_error"] = str(exc)
                summaries.append(f"Changepoint: {exc}")

        result["summary"] = "; ".join(summaries) + f" [confidence={confidence}]"
        return result

    async def _run_self_test(self) -> Dict[str, Any]:
        """Run built-in demo with synthetic reflow oven temperature data."""
        missing = _self_test_readiness()
        if missing:
            return {
                "valid": False,
                "self_test": True,
                "error": " ".join(missing),
                "missing_requirements": missing,
                "summary": "Manufacturing World Model self-test is blocked until its required runtime stack is installed.",
            }

        values, description = _generate_reflow_demo()
        result: Dict[str, Any] = {
            "valid": True,
            "self_test": True,
            "demo_description": description,
            "data_points": len(values),
        }

        summaries: List[str] = [description]

        # Anomaly scoring (no ML backend needed)
        window_size = 100
        in_sample_median = np.full(len(values), np.nan)
        for i in range(window_size, len(values)):
            in_sample_median[i] = np.mean(values[i - window_size:i])
        in_sample_median[:window_size] = np.mean(values[:window_size])

        scores, flagged = _compute_anomaly_scores(values, in_sample_median)
        result["anomaly_scores"] = scores
        result["anomaly_flagged_indices"] = flagged
        summaries.append(f"Anomaly detection: {len(flagged)} points flagged")

        # Check that spike at 420-421 is detected
        spike_detected = any(i in flagged for i in [420, 421])
        result["spike_detected"] = spike_detected

        # Changepoint detection
        cps = _detect_changepoints(values)
        result["changepoints"] = cps
        drift_detected = any(300 <= cp <= 400 for cp in cps)
        result["drift_detected"] = drift_detected
        summaries.append(f"Changepoints: {cps}")

        backend = _resolve_auto_backend()
        loop = asyncio.get_running_loop()
        train = values[:-50]
        actual = values[-50:]
        median, low, high = await loop.run_in_executor(
            _executor,
            lambda: self._predict_with_backend(backend, train, 50, 0.9),
        )
        forecast_mae = float(np.mean(np.abs(actual - median)))
        result["forecast"] = {
            "median": median.tolist(),
            "lower": low.tolist(),
            "upper": high.tolist(),
            "horizon": 50,
        }
        result["backend_used"] = backend
        result["forecast_mae"] = round(forecast_mae, 4)
        summaries.append(f"Forecast: 50 holdout steps via {backend} (MAE={forecast_mae:.3f})")

        result["summary"] = "; ".join(summaries)
        return result


def _backend_package(name: str) -> str:
    """Map backend name to its pip-installable package."""
    return {
        "chronos2": "chronos-forecasting",
        "timesfm25": "timesfm",
        "ttm": "tsfm_public",
        "moment": "momentfm",
        "stats": "statsforecast pandas",
    }.get(name, name)


PLUGIN = ManufacturingWorldModelWrapper()
