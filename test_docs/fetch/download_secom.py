"""Download the UCI SECOM dataset and derive curated CSV fixtures.

Running this script (idempotent) will:
  1. Download raw SECOM files into ./cache/ (gitignored).
  2. Select one high-quality sensor column for a univariate series.
  3. Select a set of 9 multivariate sensor columns + pass/fail label.
  4. Write two derived CSVs under ../manufacturing/:
       - reflow_sensor_series.csv  (timestamp,value)
       - sensor_multivariate.csv   (9 sensors + defect_rate)

Sources (public domain):
  https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data
  https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data

No third-party deps beyond pandas/numpy. If the raw data already exists in cache,
no network calls are made.
"""

from __future__ import annotations

import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


SECOM_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
SECOM_LABELS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

THIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = THIS_DIR / "cache"
OUT_DIR = THIS_DIR.parent / "manufacturing"

N_ROWS = 500                # rows to retain in each fixture
N_MULTIVARIATE_COLS = 9     # sensor columns to keep in multivariate fixture
MAX_NAN_RATE = 0.01         # reject columns with > 1% NaNs for shortlist
START_ISO = "2026-01-01T00:00:00+00:00"


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[cache] {dest.name} already present ({dest.stat().st_size} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())


def _load_secom() -> Tuple[pd.DataFrame, pd.Series]:
    data_path = CACHE_DIR / "secom.data"
    labels_path = CACHE_DIR / "secom_labels.data"
    _download(SECOM_DATA_URL, data_path)
    _download(SECOM_LABELS_URL, labels_path)

    # secom.data: whitespace-separated floats, NaN represented as the literal 'NaN'
    df = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN"], engine="python")
    df.columns = [f"sensor_{i:03d}" for i in range(df.shape[1])]

    # secom_labels.data: "<label> <timestamp>"; label is -1 (pass) or 1 (fail)
    labels_raw = pd.read_csv(labels_path, sep=r"\s+", header=None, engine="python")
    labels = labels_raw.iloc[:, 0].astype(int)
    # convert -1 -> 0 (pass), 1 -> 1 (fail)  so we have a clean defect_rate field
    defect_rate = (labels > 0).astype(float)
    defect_rate.name = "defect_rate"
    return df, defect_rate


def _rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Score columns by signal quality; return sorted DataFrame of metadata.

    All statistics are computed on the first N_ROWS slice we'll actually
    export, so a sensor that happens to be flat during our chosen window is
    disqualified.

      - NaN rate < 1%
      - no sparse-zero bursts (> 2% zeros disqualifies)
      - coefficient-of-variation in a realistic sensor band (1e-3 .. 0.5)
      - > 50 unique values in the window (drops categorical / quantized sensors)
      - dynamic range (p95 - p5) > small fraction of mean
    """
    head = df.iloc[:N_ROWS]
    nan_rate = head.isna().mean()
    uniques = head.nunique()
    mean = head.mean(numeric_only=True)
    std = head.std(numeric_only=True)
    variance = head.var(numeric_only=True)
    zero_rate = (head == 0).mean()
    p5 = head.quantile(0.05)
    p95 = head.quantile(0.95)
    dyn_range = (p95 - p5).abs()

    meta = pd.DataFrame(
        {
            "nan_rate": nan_rate,
            "variance": variance,
            "uniques": uniques,
            "mean": mean,
            "std": std,
            "zero_rate": zero_rate,
            "dyn_range": dyn_range,
        }
    )
    meta = meta[meta["nan_rate"] <= MAX_NAN_RATE]
    meta = meta[meta["uniques"] > 50]
    meta = meta[meta["zero_rate"] <= 0.02]
    meta = meta[meta["variance"] > 1e-6]
    meta = meta[meta["mean"].abs() > 1e-6]
    meta["cv"] = (meta["std"] / meta["mean"].abs()).astype(float)
    meta = meta[(meta["cv"] >= 1e-3) & (meta["cv"] <= 0.5)]
    meta = meta[meta["dyn_range"] > (meta["mean"].abs() * 1e-3).clip(lower=1e-6)]
    return meta


def _select_univariate(_df: pd.DataFrame, meta: pd.DataFrame) -> str:
    """Pick a sensor whose signature looks like a reflow-oven thermocouple.

    Target: CV near 2% (drift-dominated process signal) AND mean in the
    100..1000 range (plausible temperature magnitude).
    """
    temp_like = meta[(meta["mean"] > 100.0) & (meta["mean"] < 1000.0)].copy()
    temp_like["score"] = (temp_like["cv"] - 0.02).abs()
    temp_like = temp_like.sort_values("score")
    if temp_like.empty:
        # fallback: closest to CV=0.02 regardless of magnitude
        meta = meta.copy()
        meta["score"] = (meta["cv"] - 0.02).abs()
        return meta.sort_values("score").index[0]
    return temp_like.index[0]


def _select_multivariate(_df: pd.DataFrame, meta: pd.DataFrame, skip: str) -> List[str]:
    """Pick 9 well-behaved sensors that vary across different magnitudes.

    We bucket candidates by mean order-of-magnitude and sample evenly so the
    resulting frame contains a spread of scales (useful for causal discovery).
    """
    candidates = meta.drop(index=[skip], errors="ignore").copy()
    candidates["log_mean"] = np.log10(candidates["mean"].abs().clip(lower=1e-6))
    candidates = candidates.sort_values("cv")  # stable, low-noise first
    picks: List[str] = []
    # Two passes: first prefer columns spread across log-magnitude buckets
    buckets = pd.cut(candidates["log_mean"], bins=6)
    for _, group in candidates.groupby(buckets, observed=True):
        if group.empty:
            continue
        picks.append(group.index[0])
        if len(picks) >= N_MULTIVARIATE_COLS:
            break
    for col in candidates.index:
        if len(picks) >= N_MULTIVARIATE_COLS:
            break
        if col not in picks:
            picks.append(col)
    return picks[:N_MULTIVARIATE_COLS]


def _build_timestamps(n: int, start_iso: str = START_ISO, step_s: int = 1) -> List[str]:
    start = datetime.fromisoformat(start_iso)
    return [(start + timedelta(seconds=i * step_s)).isoformat() for i in range(n)]


def _write_univariate(df: pd.DataFrame, col: str) -> Path:
    series = df[col].ffill().bfill().iloc[:N_ROWS].astype(float).reset_index(drop=True)
    if series.isna().any():
        raise RuntimeError(f"column '{col}' still has NaNs after ffill/bfill")
    out = pd.DataFrame(
        {
            "timestamp": _build_timestamps(len(series)),
            "value": series.values,
        }
    )
    path = OUT_DIR / "reflow_sensor_series.csv"
    out.to_csv(path, index=False)
    print(f"[write] {path}  ({len(out)} rows, column '{col}')")
    return path


def _write_multivariate(df: pd.DataFrame, cols: List[str], defect_rate: pd.Series) -> Path:
    sub = df[cols].copy().iloc[:N_ROWS].ffill().bfill().astype(float).reset_index(drop=True)
    if sub.isna().any().any():
        raise RuntimeError("multivariate slice still has NaNs after ffill/bfill")
    sub["defect_rate"] = defect_rate.iloc[:N_ROWS].reset_index(drop=True).astype(float)
    path = OUT_DIR / "sensor_multivariate.csv"
    sub.to_csv(path, index=False)
    print(f"[write] {path}  ({len(sub)} rows, {len(sub.columns)} columns)")
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df, defect_rate = _load_secom()
    print(f"[info] raw SECOM shape: {df.shape}, defect rate: {defect_rate.mean():.3%}")

    meta = _rank_columns(df)
    print(f"[info] {len(meta)} columns pass NaN/variance gate")

    uni_col = _select_univariate(df, meta)
    multi_cols = _select_multivariate(df, meta, skip=uni_col)
    print(f"[pick] univariate: {uni_col}")
    print(f"[pick] multivariate: {multi_cols}")

    uni_path = _write_univariate(df, uni_col)
    multi_path = _write_multivariate(df, multi_cols, defect_rate)

    print("[done]")
    print(f"  {uni_path}")
    print(f"  {multi_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
