# Atlas Manufacturing End-to-End Demo

**One prompt. One task session. Every manufacturing plugin gets called — each one on real fixtures, not synthetic self_test hacks.**

This demo exercises the orchestration doctrine: Nemotron picks order at
runtime, threads output from one plugin into the next where the schemas
compose, and loads real data from disk where the plugin expects a file
path.

## Plugins called and how they connect

| # | Plugin | Mode | Source of input |
|---|---|---|---|
| 1 | `walk_knowledge_graph` / `search_literature` | — | Seeded KG (`reflow-soldering-sac305.txt` ingested via Files panel) |
| 2 | `physics_simulator` | `reflow_defect_physics` | Internal parametric thermal model (generates 20 labeled profiles) |
| 3 | `manufacturing_world_model` | `full` | **Real SECOM data**: `data_path` → `reflow_sensor_series.csv` |
| 4 | `causal_discovery` | `full` | **Real SECOM data**: `data_path` → `sensor_multivariate.csv`, `target_column: defect_rate` |
| 5 | `sandbox_lab` | `suggest` | **Real SAC305 history**: `observations_path` → `sandbox_observations.csv` + causal findings framed as context |
| 6 | `vision_inspector` | `train_reference` → `inspect` | **Real VisA PCB images**: `reference_dir` + `image_path` on disk |
| 7 | `traceability_compliance` | `report` | **Real SMT lineage**: `graph_data` loaded from `smt_lineage_graph.json`, root `board-SN48291` |

---

## Setup

1. **Load Nemotron.** `llama-cpp-python` installed, `nvidia_Orchestrator-8B-IQ2_M.gguf` in `MODELS_DIR`. Missing → task fails at `ensure_model_loaded` with an install hint. No API fallback. Doctrine.

2. **Install plugin deps.** Any plugin missing its optional deps fails loud with a `pip install` hint. For this demo you need:
   - `pandas`, `numpy` (universal)
   - `tigramite`, `pysr` (causal_discovery)
   - `torch`, `botorch`, `gpytorch` (sandbox_lab)
   - `neuraloperator` (physics_simulator; PINN fallback works without)
   - `anomalib`, `transformers`, `bitsandbytes`, `qwen-vl-utils` (vision_inspector)
   - `ruptures` (MWM changepoint detection)
   - `timesfm` or `chronos-forecasting` (MWM forecasting backend)

3. **Seed the KG.** Upload `test_docs/kg-seed/reflow-soldering-sac305.txt` via the **Files panel** (NOT the paperclip). Wait for "completed".

4. **One-time vision reference training.** Before running the demo, execute `train_reference` once to populate `vision_inspector`'s PatchCore checkpoint. You can do this via its own short chat prompt:
   ```
   Train the vision_inspector reference model on
   C:/Code/ContAInuumAtlas/test_docs/pcb/reference — call mode="train_reference"
   with reference_dir set to that path.
   ```
   Wait for it to finish (PatchCore saves to `.vision_model/`). This step is separate so that later the demo's inspect call is fast.

---

## The demo prompt

Copy verbatim into the chat input:

```
I'm investigating a SAC305 tombstoning outbreak on our SMT line and I
want a full closed-loop workup in a single session — diagnose, recommend,
verify, audit. Use every manufacturing plugin available. Absolute file
paths are given below; pass them verbatim to the tools that accept them.
Do not fabricate or synthesize data when a path is provided.

1. Pull from the knowledge graph: what is SAC305 tombstoning, what
   causes it, and which sensor signatures precede it?

2. Call physics_simulator in reflow_defect_physics mode (count=20,
   seed=42). Report how many of each defect class appeared in the
   20 generated profiles.

3. Call manufacturing_world_model in 'full' mode on the real univariate
   reflow sensor series at
   C:/Code/ContAInuumAtlas/test_docs/manufacturing/reflow_sensor_series.csv
   (pass as data_path; value_column="value", timestamp_column="timestamp").
   Give me the forecast, anomaly z-scores, and any detected changepoints.

4. Call causal_discovery in 'full' mode on the real multivariate sensor
   data at
   C:/Code/ContAInuumAtlas/test_docs/manufacturing/sensor_multivariate.csv
   with target_column="defect_rate", max_lag=3. Report the causal parents
   of defect_rate and, if PySR returned an equation, the symbolic form.

5. Call sandbox_lab in 'suggest' mode. Parameters:
   - peak_c: [215, 255] continuous
   - tal_s: [30, 90] continuous
   - ramp_c_per_s: [0.5, 3.5] continuous
   Objectives:
   - defect_rate (minimize=true)
   - throughput (minimize=false)
   Warm-start with observations_path=
   C:/Code/ContAInuumAtlas/test_docs/manufacturing/sandbox_observations.csv,
   parameter_columns=["peak_c","tal_s","ramp_c_per_s"],
   objective_columns=["defect_rate","throughput"], batch_size=4.
   Propose 4 next-batch recipes. Briefly note whether the suggested
   parameter ranges align with what causal_discovery flagged in step 4.

6. Call vision_inspector in 'inspect' mode on a real defective PCB:
   image_path=C:/Code/ContAInuumAtlas/test_docs/pcb/defects/defect_0.jpg
   reference_dir=C:/Code/ContAInuumAtlas/test_docs/pcb/reference
   Report verdict (PASS/FAIL/UNCERTAIN), anomaly_score, stage_reached,
   and the VLM explanation if the cascade fired. If VLM OOMs on a 4GB
   GPU, retry with skip_vlm=true and report PatchCore-only.

7. Call traceability_compliance in 'report' mode to bundle the audit:
   - graph_data: load from
     C:/Code/ContAInuumAtlas/test_docs/manufacturing/smt_lineage_graph.json
     (strip the top-level "description" field — plugin only consumes
     "nodes" and "edges")
   - root_node_id: "board-SN48291"
   - domain_profile: "manufacturing"
   Give me bundle_id, content_hash, and the narrative_report. Do NOT
   fabricate a graph from prior-step summaries; this fixture is the
   canonical audit trail.

You decide the tool-call order and argument shapes where not specified.
At each step tell me which plugin you called, which mode, and which
path(s) or prior outputs you passed.
```

---

## What success looks like

- ~7–9 tool calls in the event stream.
- Every `OBSERVATION` carries real structured data from a named fixture or a parametric generator — no self_test smoke-test paths.
- Tool arguments visible in the log:
  - MWM's `data_path` = `reflow_sensor_series.csv`
  - causal_discovery's `data_path` = `sensor_multivariate.csv`, `target_column: "defect_rate"`
  - sandbox_lab's `observations_path` = `sandbox_observations.csv`
  - vision_inspector's `image_path` + `reference_dir` under `test_docs/pcb/`
  - traceability_compliance's `graph_data.nodes[*].id` from `smt_lineage_graph.json`, `root_node_id: "board-SN48291"`
- Final assistant message synthesizes: diagnosis of the tombstoning cause (from KG + causal), MWM's take on sensor anomalies, 4 recommended next-batch recipes (from sandbox), vision verdict on a real defective PCB, and a PROV-DM audit bundle.

## Fail-loud behavior (not a bug)

- Any `data_path` / `observations_path` / `image_path` that doesn't resolve → `RuntimeError` with the missing path.
- `causal_discovery` without `tigramite`/`pysr` → `ERROR_PERMANENT` with a `pip install` hint.
- `vision_inspector` without `anomalib`/`transformers` → explicit failure, no VLM-less substitution.
- Nemotron GGUF missing → task fails at load, no deepseek-chat fallback.

Install what's flagged and re-run.

## Hardware notes (RTX 3050 4GB laptop)

- Nemotron IQ2_M: partial GPU offload, `n_gpu_layers=15` gives ~3–5 tok/sec. Full demo ~5–15 min wall time.
- Vision_inspector + Qwen2-VL can OOM on 4GB. If it does, the prompt instructs Nemotron to retry with `skip_vlm=true`.
- All other plugins are CPU-comfortable at the fixture sizes used (SECOM ≤500 rows, observations 40 rows, Darcy unused in this demo).

---

## UI notes

- **Files panel** (left sidebar): RAG ingestion. Use once for the seed doc.
- **Paperclip**: task-scoped attachments — not used in this demo because all fixtures are pre-staged under `test_docs/` and referenced by absolute path.
- **Cancel** (top-right): clean task-loop halt.
- **Task state badge**: planning → executing → reviewing → completed.

---

## Fixture inventory (all shipped with the repo)

| Fixture | Path | Source / license |
|---|---|---|
| KG seed doc | `test_docs/kg-seed/reflow-soldering-sac305.txt` | hand-authored |
| Univariate time-series | `test_docs/manufacturing/reflow_sensor_series.csv` | UCI SECOM (public), 500 rows |
| Multivariate tabular | `test_docs/manufacturing/sensor_multivariate.csv` | UCI SECOM (public), 500×10 |
| Sandbox observations | `test_docs/manufacturing/sandbox_observations.csv` | Synthetic SAC305-consistent, 40 rows |
| Physics training (Darcy) | `test_docs/physics/darcy_small.npz` | Local FD solver, 100 × 64 × 64 (available for `train`/`predict` mode, not used in this demo) |
| PCB reference | `test_docs/pcb/reference/good_*.jpg` | VisA PCB1 (CC BY-SA 4.0), 10 images |
| PCB defects | `test_docs/pcb/defects/defect_*.jpg` | VisA PCB1 (CC BY-SA 4.0), 5 images |
| SMT lineage graph | `test_docs/manufacturing/smt_lineage_graph.json` | hand-authored PROV-DM, 28 nodes / 50 edges |

To regenerate any fixture: `python test_docs/fetch/<script>.py`.
