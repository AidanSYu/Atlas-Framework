# Manufacturing fixtures (derived from UCI SECOM)

These two CSV fixtures are curated slices of the **UCI SECOM** semiconductor
manufacturing dataset (public domain, [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/machine-learning-databases/secom/)).
They exist so the Atlas Framework can exercise the file-loading path
(`data_path` parameter) on the `manufacturing_world_model` and
`causal_discovery` plugins without relying on the plugin's built-in synthetic
data.

To regenerate from raw SECOM, run:

```bash
python test_docs/fetch/download_secom.py
```

The raw download is cached in `test_docs/fetch/cache/` (gitignored). Only the
derived CSVs below are tracked.

---

## `reflow_sensor_series.csv` — univariate time-series (500 rows)

| column      | type   | description                                 |
|-------------|--------|---------------------------------------------|
| `timestamp` | string | synthetic ISO-8601 UTC, 1 second spacing    |
| `value`     | float  | SECOM column `sensor_046` (forward-filled)  |

**Why `sensor_046`?** Selection criteria (computed over the first 500 rows we
export, not the whole 1567-row table):

- 0 % NaNs in the export window
- 0 % zero-valued samples (rules out bursty / on-off sensors)
- Coefficient of variation (std / |mean|) ≈ **1.9 %** — realistic for a reflow
  oven thermocouple in steady-state, where we expect small drift around a
  setpoint rather than wild excursions.
- Mean ≈ **732**, std ≈ **14**, range ≈ **700..790** — plausible temperature
  magnitudes after choosing the candidate closest to a CV of 2 % among columns
  with mean in the 100..1000 band.
- `>` 50 unique values (no categorical / quantized masquerading as numeric).

Consumer contract: point this at `manufacturing_world_model` with
`data_path` pointing at the CSV; the wrapper reads `timestamp` + `value`.

## `sensor_multivariate.csv` — tabular, causal-ready (500 rows × 10 columns)

| column         | type   | description                                  |
|----------------|--------|----------------------------------------------|
| `sensor_354`   | float  | SECOM sensor, mean ≈ 0.04, std ≈ 0.01        |
| `sensor_582`   | float  | SECOM sensor, mean ≈ 0.50, std ≈ 0.004       |
| `sensor_057`   | float  | SECOM sensor, mean ≈ 0.95, std ≈ 0.004       |
| `sensor_037`   | float  | SECOM sensor, mean ≈ 66.2, std ≈ 0.36        |
| `sensor_038`   | float  | SECOM sensor, mean ≈ 87.0, std ≈ 0.60        |
| `sensor_133`   | float  | SECOM sensor, mean ≈ 1005, std ≈ 5.2         |
| `sensor_056`   | float  | SECOM sensor, mean ≈ 0.93, std ≈ 0.005       |
| `sensor_119`   | float  | SECOM sensor, mean ≈ 0.97, std ≈ 0.006       |
| `sensor_053`   | float  | SECOM sensor, mean ≈ 4.60, std ≈ 0.04        |
| `defect_rate`  | float  | 0.0 = pass, 1.0 = fail (SECOM label, −1→0/+1→1) |

**Why these 9?** After filtering the 590 SECOM sensors by the same gate above
(NaN rate ≤ 1 %, zero-rate ≤ 2 %, CV ∈ [0.001, 0.5], > 50 unique values,
genuine dynamic range), we bucket the survivors by log-mean and sample one
low-CV column from each bucket. That produces a spread across scales
(0.04 ↔ 1 000), which is what `causal_discovery` (PCMCI+) benefits from —
homogeneous columns make the independence tests degenerate.

`defect_rate` is the SECOM pass/fail label (−1 → 0.0, +1 → 1.0). Mean ≈
0.12 in the first 500 rows. This is the natural target column for the
causal-discovery plugin.

Consumer contract: point `causal_discovery` at this CSV with
`data_path=<this file>` and (optionally) `target_column="defect_rate"`. The
wrapper will set `variable_names` from the headers and pass the 2-D float
matrix to PCMCI+.

---

## Provenance

- Source: UCI SECOM ([`secom.data`](https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data),
  [`secom_labels.data`](https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data)).
- License: UCI ML Repository datasets are distributed without restrictions
  beyond citing the repository; SECOM has no additional license file.
- Slice: rows 0 .. 499 (chronological order as shipped by UCI).
- Forward-fill applied only to the univariate file where a trailing NaN
  survived the initial filter; multivariate file has no NaNs in the chosen
  columns.
- Timestamps in `reflow_sensor_series.csv` are **synthetic** (1 Hz starting
  `2026-01-01T00:00:00+00:00`) because SECOM's native timestamps are in a
  separate file and irregularly spaced. The regular cadence makes the series
  suitable for the forecasting backends (TimesFM-2.5 etc) that assume uniform
  sampling.
