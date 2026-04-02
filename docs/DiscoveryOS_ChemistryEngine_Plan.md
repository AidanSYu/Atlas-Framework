# Discovery OS — Scientific Discovery Engine Implementation Plan
**Version:** 2.0
**Status:** Ready for implementation
**Depends on:** `AgentWorkstreams.md` (frontend Golden Path — implement in parallel)
**Last updated:** 2026-03-13
**Proof-of-concept domain:** Small molecule drug discovery (KRAS G12C walkthrough)
**Designed for:** Any scientific research domain (drug discovery, protein engineering, materials science, catalyst design)

---

## Purpose

This document is a complete, agent-ready implementation plan for the **Atlas Discovery OS Scientific Discovery Engine** — a domain-agnostic orchestration layer that transforms a researcher's natural-language goal into a closed-loop computational research workflow: hypothesis → computational campaign → experimental validation → feedback → iteration.

**The engine is domain-agnostic by design.** The proof-of-concept domain is small molecule drug discovery ("find a molecule that binds to protein Y"). The same architecture — plugin registry, async job system, EntityIR intermediate representation, wet lab ingest, and feedback loop — applies unchanged to protein engineering (AlphaFold + stability prediction), materials science (DFT + crystal enumeration), and catalyst design (reaction network tools).

Any agent reading this document should be able to implement any section independently without additional context beyond the existing codebase (`AGENTS.md`, `CLAUDE.md`).

---

## Core Ideology (Read This First)

**Atlas is not a drug discovery tool. Atlas is a domain-agnostic orchestrator of validated scientific tools with persistent memory.**

Every scientific domain has the same problem: excellent specialist tools exist in isolation with no orchestration layer, no persistent memory, and no way for a non-expert to use them. The state-of-the-art tools already exist:
- **Drug discovery**: GNINA (docking), REINVENT4 (generative design), ADMET-AI (property prediction), AiZynthFinder (retrosynthesis)
- **Protein engineering**: AlphaFold2/3 (structure prediction), ESMFold (fast folding), ESM-IF1 (stability), Rosetta (design)
- **Materials science**: M3GNet/CHGNet (property prediction), Materials Project API (DFT data), VASP (ab-initio calculation), PyMatGen (analysis)
- **Catalyst design**: xTB (semiempirical DFT), reaction network enumeration, NEB calculators

What does not exist is a system that:

1. Connects them into a coherent workflow without requiring a domain PhD
2. Maintains persistent memory across sessions so a researcher never starts from zero
3. Translates researcher intent into tool execution without the researcher knowing what GNINA or CHGNet is

**The LLM's only job is routing and planning — never generating domain-specific code.**
Pre-written, hardened plugins execute all computation deterministically. The LLM decides which plugin to call and with what parameters. It never writes RDKit code. It never computes a docking score. It never generates DFT input files. It routes.

---

## Architecture Overview

The architecture is identical regardless of domain. Only the plugins and EntityIR subclass change.

```
Researcher Goal (natural language)
            │
┌───────────▼──────────────┐
│   Coordinator (HITL)     │  ← EXISTS (coordinator.py)
│   DeepSeek reasoning     │  Extracts: domain, target/goal,
│   Up to 5 turns          │  constraints, success criteria
│   Writes session_memory  │  session_memory["domain"] set here
└───────────┬──────────────┘
            │ session_memory.json  ← includes "domain" field
┌───────────▼──────────────┐
│   Pipeline Planner       │  ← NEW (pipeline_planner.py)
│   DeepSeek routing only  │  Reads domain → selects from
│   NO domain-specific     │  domain-scoped plugin registry
│   code generation        │  Returns typed PipelinePlan JSON
└───────────┬──────────────┘
            │ PipelinePlan (JSON)
            │
   ┌─────────┴──────────┬──────────────────┐
   ▼                    ▼                  ▼
Plugin A             Plugin B          Plugin C
(domain-specific)  (domain-specific)  (domain-specific)
   │                    │                  │
   └─────────┬──────────┘                  │
             ▼                             │
          Plugin D                         │
             │                            │
             └──────────────┬─────────────┘
                            ▼
                     Plugin N  ← Async Celery task (long-running)
                            │
                            ▼
                     Output Compiler
                     ├── JSON entity hit cards
                     ├── Order / synthesis CSV
                     └── Session report PDF
                            │
            ════════ RESEARCHER REVIEWS ════════
                            │ orders compounds / initiates experiments
                            │ results arrive (days to weeks)
                            ▼
                     Experimental Data Ingest (schema-validated)
                            │ measurement + assay metadata
                            ▼
                     Knowledge Graph Update
                            │ ground truth nodes
                            ▼
                     Next session reads history
                     "Best prior hit: 50nM. Scaffold X
                      → off-target. Exclude from constraints."
```

### Domain-Specific Instantiations (Proof-of-Concept → Expansion)

| Domain | EntityIR subclass | Example plugins | Wet lab output |
|--------|------------------|-----------------|----------------|
| Small molecule drug discovery | `MoleculeIR` | standardize, rdkit_filters, gnina_dock, reinvent4_generate, aizynthfinder | IC50, EC50, Kd |
| Protein engineering | `ProteinIR` | alphafold2_fold, esm_fold, esm_stability, rosetta_design | Tm, ΔΔG, binding affinity |
| Materials science | `MaterialIR` | matgl_predict, mp_lookup, pymatgen_filters, vasp_dispatch | Band gap, conductivity, XRD |
| Catalyst design | `CatalystIR` | xtb_optimize, neb_barrier, reaction_enumerate | Yield, selectivity, TON |

---

## Critical Architecture Decision: The Entity IR (Domain-Agnostic Base + Domain-Specific Subclasses)

**Every scientific entity in the system must flow through a single typed Intermediate Representation (IR) object.**

This is non-negotiable. Tools in any domain output different data formats. GNINA outputs 3D poses, AlphaFold outputs PDB files, MatGL outputs tensors. Without a master record holding all representations simultaneously, the pipeline silently loses provenance and spatial information as data moves between stages.

**The IR system is two-level:**
1. `EntityIR` — domain-agnostic base class. All pipelines, regardless of domain, pass `list[EntityIR]` between plugins.
2. Domain-specific subclasses (`MoleculeIR`, `ProteinIR`, `MaterialIR`) — extend `EntityIR` with domain-relevant fields.

Plugins are typed to their subclass: `GNINADockPlugin` works on `list[MoleculeIR]`, `AlphaFold2Plugin` works on `list[ProteinIR]`. The base `BasePlugin.run()` accepts `list[EntityIR]` and casts internally.

---

### The Base: `EntityIR`

**File to create:** `src/backend/app/services/plugins/entity_ir.py`

```python
"""
Entity Intermediate Representation — domain-agnostic base.

Every scientific entity in any Discovery OS pipeline is represented as an EntityIR subclass.
Plugins read from and write to specific fields.
The LLM never sees raw domain data — only summaries derived from IR.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class AssayResult:
    """
    Universal experimental result — works for IC50, Tm, band gap, yield, etc.
    The unit and assay_type fields encode the domain-specific interpretation.
    """
    assay_id: str
    assay_type: str           # "biochemical_IC50" | "cellular_EC50" | "protein_Tm" | "XRD_band_gap" | etc.
    target_name: str          # what was measured against
    value: Optional[float]    # numeric result
    unit: str                 # "nM" | "°C" | "eV" | "%" | etc.
    censored: bool            # True if value = detection limit
    passed: bool              # True if value meets session success_threshold
    r_squared: Optional[float]
    curve_quality: str        # "excellent" | "acceptable" | "poor" | "failed"
    date: datetime
    operator: str


@dataclass
class EntityIR:
    """
    Domain-agnostic base for all scientific entities.
    Do not instantiate directly — use MoleculeIR, ProteinIR, MaterialIR, etc.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    entity_id: str             # e.g. "ATLAS-00042" — assigned on ingestion
    domain: str                # "small_molecule" | "protein" | "material" | "catalyst"
    source: str                # where the entity came from
    display_name: str          # human-readable label for UI

    # ── Provenance ────────────────────────────────────────────────────────────
    pipeline_run_id: str = ""
    session_id: str = ""
    epoch_id: str = ""
    plugin_history: list[str] = field(default_factory=list)  # ordered plugin names
    created_at: datetime = field(default_factory=datetime.utcnow)

    # ── Wet lab ground truth (populated after experimental results returned) ──
    assay_results: list[AssayResult] = field(default_factory=list)
    confirmed_hit: Optional[bool] = None  # True if any assay passes success criteria

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
```

---

### The Small Molecule Domain Subclass: `MoleculeIR`

**File to create:** `src/backend/app/services/plugins/molecule_ir.py`

```python
"""
MoleculeIR — small molecule drug discovery domain subclass of EntityIR.

Every molecule in the small_molecule pipeline is represented as a MoleculeIR.
Plugins read from and write to specific fields. No plugin receives raw SMILES
strings or raw docking scores — only MoleculeIR objects.

The LLM never sees raw chemistry data. It only sees summaries derived from IR.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from .entity_ir import EntityIR, AssayResult


@dataclass
class RDKitProps:
    mw: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    qed: float


@dataclass
class ADMETResult:
    # hERG
    herg_ic50_um: Optional[float]
    herg_risk: str            # "LOW" | "MEDIUM" | "HIGH"
    # Permeability
    caco2_cm_s: Optional[float]
    caco2_permeability: str   # "LOW" | "MEDIUM" | "HIGH"
    # Toxicity
    dili_risk: str            # "LOW" | "MEDIUM" | "HIGH"
    ames_mutagenicity: bool
    # Metabolism
    cyp3a4_inhibition: str    # "LOW" | "MEDIUM" | "HIGH"
    cyp2d6_inhibition: str
    # Distribution
    ppb_percent: Optional[float]
    vd_l_kg: Optional[float]
    # Solubility
    aqueous_solubility_mg_ml: Optional[float]
    # Half-life
    half_life_hours: Optional[float]
    # Full 41-endpoint dict for archival
    raw_endpoints: dict = field(default_factory=dict)


@dataclass
class SynthesisStep:
    smarts: str               # reaction SMARTS
    reactants: list[str]      # SMILES of reactants
    product: str              # SMILES of product
    catalog_ids: list[str]    # purchasable reactant catalog numbers


@dataclass
class SynthesisRoute:
    n_steps: int
    steps: list[SynthesisStep]
    building_blocks: list[str]   # catalog IDs of purchasable starting materials
    feasibility_score: float      # AiZynthFinder score 0-1
    solved: bool


@dataclass
class MoleculeIR(EntityIR):
    """Small molecule drug discovery domain IR. Extends EntityIR."""
    # ── Chemistry Identity ────────────────────────────────────────────────────
    # entity_id, source, display_name, domain inherited from EntityIR
    smiles: str = ""           # canonical SMILES (RDKit-standardized)
    inchikey: str = ""         # unique structure hash — used for deduplication

    # ── 2D graph (in-memory, not serialized to disk) ───────────────────────────
    # NOTE: rdkit_mol is populated by standardize plugin, cleared before JSON serialization
    rdkit_mol: object = None   # RDKit Mol object — type: Optional[Chem.Mol]

    # ── 3D geometry ───────────────────────────────────────────────────────────
    conformer_sdf: Optional[str] = None       # SDF string of best 3D conformer (ETKDG)
    docked_pose_sdf: Optional[str] = None     # GNINA output pose SDF
    binding_pocket_ref: Optional[str] = None  # PDB file path or AlphaFold accession

    # ── Computed properties (populated by plugins ONLY — never by LLM) ────────
    rdkit_props: Optional[RDKitProps] = None
    docking_score: Optional[float] = None     # kcal/mol (GNINA CNNaffinity score)
    docking_score_vina: Optional[float] = None  # Vina score if Vina used as pre-filter
    admet: Optional[ADMETResult] = None
    sa_score: Optional[float] = None          # RDKit SA Score: 1 (easy) – 10 (hard)
    rascore: Optional[float] = None           # RAscore: 0–1, higher = more synthesizable

    # ── Filter flags ─────────────────────────────────────────────────────────
    pains_alerts: list[str] = field(default_factory=list)
    brenk_alerts: list[str] = field(default_factory=list)
    passes_lipinski: Optional[bool] = None
    passes_veber: Optional[bool] = None
    passes_user_constraints: Optional[bool] = None
    user_constraint_failures: list[str] = field(default_factory=list)

    # ── Source catalog (for ordering) ─────────────────────────────────────────
    catalog_id: Optional[str] = None          # e.g. "EN300-12345" (Enamine)
    catalog_source: Optional[str] = None      # "Enamine REAL" | "Mcule" | "ZINC22"
    price_usd: Optional[float] = None
    lead_time_weeks: Optional[int] = None
    in_stock: Optional[bool] = None

    # ── Retrosynthesis ────────────────────────────────────────────────────────
    aizynthfinder_routes: list[SynthesisRoute] = field(default_factory=list)
    synthesis_feasible: Optional[bool] = None  # True if at least one route solved

    # ── Wet lab ground truth (small molecule specific) ────────────────────────
    # assay_results, confirmed_hit inherited from EntityIR
    best_ic50_nm: Optional[float] = None       # denormalized for quick querying

    # ── Provenance (inherited from EntityIR) ──────────────────────────────────
    # pipeline_run_id, session_id, epoch_id, plugin_history, created_at — all inherited

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage. Clears non-serializable rdkit_mol."""
        import dataclasses
        d = dataclasses.asdict(self)
        d.pop("rdkit_mol", None)  # never serialize the RDKit object
        return d

    @property
    def overall_score(self) -> Optional[float]:
        """
        Composite score for ranking. Combines docking score, ADMET, and synthesizability.
        Returns None if any required component is missing.
        """
        if self.docking_score is None or self.admet is None or self.sa_score is None:
            return None
        # Normalize docking score: more negative = better, clip at -12 to 0
        dock_norm = max(0.0, min(1.0, -self.docking_score / 12.0))
        # ADMET: penalize HIGH risk flags
        admet_penalty = sum([
            0.3 if self.admet.herg_risk == "HIGH" else 0.1 if self.admet.herg_risk == "MEDIUM" else 0.0,
            0.2 if self.admet.dili_risk == "HIGH" else 0.1 if self.admet.dili_risk == "MEDIUM" else 0.0,
        ])
        admet_score = max(0.0, 1.0 - admet_penalty)
        # Synthesizability: SA Score 1-6 maps to 1.0-0.0
        sa_norm = max(0.0, 1.0 - (self.sa_score - 1.0) / 5.0)
        return round(0.5 * dock_norm + 0.3 * admet_score + 0.2 * sa_norm, 3)
```

**Serialization rule:** `MoleculeIR.to_dict()` must be called before storing to SQLite or returning via API. The `rdkit_mol` field is never serialized — it is always re-computed from `smiles` when needed.

---

### Other Domain Subclasses (Reference — Implement When Domain Is Added)

These are not built in the current implementation. They document what expanding to a new domain looks like.

**`ProteinIR`** — for protein engineering and structure prediction pipelines:

```python
# src/backend/app/services/plugins/protein_ir.py
@dataclass
class ProteinIR(EntityIR):
    """Protein engineering domain IR. Extends EntityIR."""
    sequence: str = ""                           # amino acid sequence (FASTA)
    pdb_path: Optional[str] = None               # path to 3D structure file
    af2_plddt: Optional[float] = None            # AlphaFold2 mean pLDDT confidence
    af3_iptm: Optional[float] = None             # AlphaFold3 ipTM (protein-ligand co-fold)
    melting_temp_celsius: Optional[float] = None # from wet lab or stability predictor
    mutations: list[str] = field(default_factory=list)  # e.g. ["K526R", "D527N"]
    wild_type_sequence: Optional[str] = None
    stability_ddg: Optional[float] = None        # predicted ΔΔG of mutation vs WT
    # assay_results, confirmed_hit, provenance inherited from EntityIR
```

**`MaterialIR`** — for materials science and solid-state chemistry pipelines:

```python
# src/backend/app/services/plugins/material_ir.py
@dataclass
class MaterialIR(EntityIR):
    """Materials science domain IR. Extends EntityIR."""
    formula: str = ""                            # e.g. "LiFePO4", "MoS2"
    crystal_structure_cif: Optional[str] = None  # CIF file content
    space_group: Optional[str] = None
    lattice_params: Optional[dict] = None        # a, b, c, α, β, γ
    # DFT-computed or ML-predicted properties (populated by plugins ONLY)
    band_gap_ev: Optional[float] = None
    formation_energy_ev_per_atom: Optional[float] = None
    bulk_modulus_gpa: Optional[float] = None
    predicted_conductivity: Optional[float] = None
    above_hull_ev_per_atom: Optional[float] = None  # thermodynamic stability
    synthesis_conditions: Optional[dict] = None  # temperature, atmosphere, precursors
    # assay_results, confirmed_hit, provenance inherited from EntityIR
```

**Adding a new domain** requires: (1) create `{domain}_ir.py` subclass, (2) add domain-scoped plugins to registry, (3) add domain key to coordinator's session_memory schema. The rest of the system — Pipeline Planner, async jobs, wet lab ingest, feedback loop — requires no changes.

---

## Phase 0 — Architecture Pivot (Do This Before Anything Else)

### What to change in the existing Executor

The current `executor.py` asks MiniMax to generate Python scripts from scratch. This is the structural dead end. The change:

**Before (broken):**
```
plan_task (MiniMax) → generate_script (MiniMax writes RDKit code) → subprocess.run()
```

**After (correct):**
```
plan_task (DeepSeek) → select_plugin + config (DeepSeek) → plugin.run(molecules, config)
```

The Executor's `generate_script` node is replaced by a `dispatch_plugin` node that calls a registered plugin by name. The LLM output is a JSON object like:
```json
{ "plugin": "gnina_dock", "config": { "top_k": 1000, "exhaustiveness": 8 } }
```

This JSON is validated against a schema before execution. If it fails validation, raise immediately — do not retry.

---

## Phase 1 — Core Tool Plugins

**Location:** `src/backend/app/services/plugins/`

All plugins share this interface (defined in `base.py`). The interface is domain-agnostic — it operates on `EntityIR` base objects. Domain-specific plugins cast to their subclass internally.

```python
# src/backend/app/services/plugins/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from .entity_ir import EntityIR


@dataclass
class PluginResult:
    entities: list[EntityIR]           # updated EntityIR list (MoleculeIR, ProteinIR, etc.)
    summary: str                       # human-readable summary for FINDINGS.md
    metadata: dict                     # plugin-specific output metadata
    failed_ids: list[str]             # entity_ids that failed (non-fatal)
    errors: list[str]                 # error messages for failed_ids


class BasePlugin(ABC):
    name: str                          # unique plugin identifier
    domain: str                        # "small_molecule" | "protein" | "material" | "catalyst"
    version: str
    requires_gpu: bool = False
    estimated_seconds_per_entity: float = 0.01

    @abstractmethod
    def run(self, entities: list[EntityIR], config: dict) -> PluginResult:
        """
        Execute the plugin on a list of EntityIR objects.
        Implementations should cast to the domain-specific subclass:
            molecules = [cast(MoleculeIR, e) for e in entities]
        """
        pass

    def validate_config(self, config: dict) -> None:
        """Raise ValueError if config is invalid. Called before run()."""
        pass
```

---

### Plugin 1: `standardize.py`

**Purpose:** Convert raw SMILES strings into standardized `MoleculeIR` objects. Always the first plugin in any pipeline.

**File:** `src/backend/app/services/plugins/standardize.py`

**Dependencies:** `rdkit`, `chembl_structure_pipeline` (`pip install chembl-structure-pipeline`)

**Implementation requirements:**
- Input: `config["smiles_list"]` — a `list[str]` of raw SMILES from any source
- For each SMILES:
  1. Parse with `Chem.MolFromSmiles(smi)`. If None: log failure, skip.
  2. Standardize with `chembl_structure_pipeline.standardize_smiles(smi)`. Handles: charge neutralization, counterion removal, tautomer normalization.
  3. Generate canonical SMILES: `Chem.MolToSmiles(mol, canonical=True)`
  4. Generate InChIKey: `Chem.InchiInfo.MolToInchiKey(mol)` or `Chem.inchi.MolToInchiKey(mol)`
  5. Assign compound_id: `f"ATLAS-{uuid4().hex[:8].upper()}"`
  6. Deduplicate by InChIKey — if InChIKey already exists in `config["existing_inchikeys"]` (set), skip and add to `failed_ids` with reason "duplicate"
  7. Populate `MoleculeIR`: `smiles`, `inchikey`, `compound_id`, `rdkit_mol`, `source`, `pipeline_run_id`, `session_id`, `epoch_id`
- Returns: `PluginResult` with populated MoleculeIR list

**Performance:** Must handle 500K SMILES in < 5 minutes on CPU. Use `multiprocessing.Pool` with `chunksize=1000`.

**Exit gate:**
- `"invalid_smiles_string"` → logged in `failed_ids`, not raised as exception
- Duplicate InChIKey → in `failed_ids` with reason "duplicate"
- 500K SMILES processed in < 5 minutes (benchmark in tests)

---

### Plugin 2: `rdkit_filters.py`

**Purpose:** Compute physicochemical properties and apply structural alert filters. The first hard filter in the cascade.

**File:** `src/backend/app/services/plugins/rdkit_filters.py`

**Dependencies:** `rdkit` only

**Implementation requirements:**
- For each MoleculeIR with `rdkit_mol` populated:
  1. Compute `RDKitProps`:
     - `MW`: `Descriptors.MolWt(mol)`
     - `LogP`: `Descriptors.MolLogP(mol)`
     - `TPSA`: `Descriptors.TPSA(mol)`
     - `HBD`: `Descriptors.NumHDonors(mol)`
     - `HBA`: `Descriptors.NumHAcceptors(mol)`
     - `RotBonds`: `Descriptors.NumRotatableBonds(mol)`
     - `QED`: `QED.qed(mol)`
  2. Apply Lipinski Ro5: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10. Set `passes_lipinski`.
  3. Apply Veber rules: RotBonds ≤ 10, TPSA ≤ 140. Combined with Lipinski for `passes_veber`.
  4. PAINS alerts: use `FilterCatalog` with `FilterCatalogParams.PAINS_A`, `PAINS_B`, `PAINS_C`. Store alert names in `pains_alerts`.
  5. Brenk alerts: `FilterCatalogParams.BRENK`. Store in `brenk_alerts`.
  6. Apply user constraints from `config["user_constraints"]` (list of `{property, operator, value}`):
     - Properties: any key in RDKitProps
     - Operators: `"<"`, `"<="`, `">"`, `">="`, `"="`
     - Set `passes_user_constraints = True/False`
     - Populate `user_constraint_failures` with names of failed constraints

**Config:**
```json
{
  "apply_lipinski": true,
  "apply_veber": true,
  "apply_pains": true,
  "apply_brenk": true,
  "reject_any_pains": false,
  "reject_any_brenk": true,
  "user_constraints": [
    {"property": "mw", "operator": "<=", "value": 500},
    {"property": "logp", "operator": "<=", "value": 5}
  ]
}
```

**Note on PAINS:** Set `reject_any_pains = false` by default. PAINS alerts are flags, not automatic rejections — many known drugs trigger PAINS. Brenk alerts (toxic/reactive groups) should default to rejection (`reject_any_brenk = true`).

**Performance:** < 1ms per molecule. 500K molecules in < 10 minutes on single CPU core. Use multiprocessing.

**Exit gate:**
- Ethanol (CCO): MW=46, LogP=-0.0014, passes all filters
- A known PAINS compound (e.g., curcumin) gets non-empty `pains_alerts`
- A compound with MW=600 fails Lipinski with `passes_lipinski = False`
- User constraint `{"property": "mw", "operator": "<=", "value": 400}` on MW=450 compound → `passes_user_constraints = False`, `user_constraint_failures = ["mw <= 400"]`

---

### Plugin 3: `sa_score.py`

**Purpose:** Compute synthetic accessibility and retrosynthesis accessibility scores as fast pre-filters before expensive retrosynthesis planning.

**File:** `src/backend/app/services/plugins/sa_score.py`

**Dependencies:** `rdkit` (SA_Score is in `rdkit.Contrib`), `RAscore` (`pip install RAscore`)

**Implementation requirements:**
- SA Score:
  - `from rdkit.Contrib.SA_Score import sascorer`
  - `score = sascorer.calculateScore(mol)`
  - Range: 1 (trivially synthesizable) to 10 (nearly impossible)
  - Set `molecule.sa_score`
- RAscore:
  - `from RAscore import RAscore_NN`
  - Initialize model once (singleton pattern — expensive to reload)
  - `score = model.predict([smiles])[0]`
  - Range: 0–1, higher = more synthesizable (AiZynthFinder can find a route)
  - Set `molecule.rascore`
- Set `molecule.synthesis_feasible = (sa_score <= 6.0 and rascore >= 0.4)`

**Performance:** RAscore model init is slow (~3s). Initialize as a module-level singleton, not per-call.

**Exit gate:**
- Aspirin (CC(=O)Oc1ccccc1C(=O)O): SA Score ≈ 1.6, RAscore > 0.8
- A highly complex natural product: SA Score > 6, RAscore < 0.3
- 10K molecules processed in < 60 seconds

---

### Plugin 4: `admet_ai.py`

**Purpose:** Predict 41 ADMET endpoints using ADMET-AI (Chemprop-RDKit, #1 on TDC ADMET leaderboard). Fully local, no API calls.

**File:** `src/backend/app/services/plugins/admet_ai.py`

**Dependencies:** `admet_ai` (`pip install admet_ai`)

**Implementation requirements:**
- Initialize `ADMETModel` once as a module-level singleton
- Batch all molecules together for efficiency: `predictions = model.predict(smiles_list)`
- Parse predictions into `ADMETResult` dataclass per molecule
- Risk classification rules (apply after prediction):
  - hERG risk: < 1µM = HIGH, 1-10µM = MEDIUM, > 10µM = LOW
  - DILI risk: probability > 0.7 = HIGH, 0.4-0.7 = MEDIUM, < 0.4 = LOW
  - Caco2 permeability: > 1e-5 cm/s = HIGH (good), 1e-6 to 1e-5 = MEDIUM, < 1e-6 = LOW (poor)
- Store full 41-endpoint raw dict in `ADMETResult.raw_endpoints` for archival

**Config:**
```json
{
  "endpoints": "all",
  "herg_threshold_high_um": 1.0,
  "herg_threshold_medium_um": 10.0,
  "dili_threshold_high": 0.7,
  "dili_threshold_medium": 0.4
}
```

**Exit gate:**
- Known hERG blocker (e.g., terfenadine): hERG risk = HIGH
- Aspirin: DILI risk = LOW (aspirin has low DILI liability)
- 10K molecules processed in < 5 minutes on CPU

---

### Plugin 5: `structure_prep.py`

**Purpose:** Obtain and prepare a protein structure for docking. Handles AlphaFold2 fetching, protonation, and binding pocket identification.

**File:** `src/backend/app/services/plugins/structure_prep.py`

**Dependencies:** `requests`, `pdbfixer` (`pip install pdbfixer`), `fpocket` (external binary — optional)

**Implementation requirements:**

Input (via `config`):
- `config["uniprot_id"]` OR `config["pdb_id"]` OR `config["pdb_file_path"]`
- `config["pocket_residues"]` (optional): list of residue numbers defining the binding pocket
- `config["session_id"]`: for saving output files

Processing:
1. **Structure acquisition:**
   - If `pdb_id` provided: download from RCSB (`https://files.rcsb.org/download/{pdb_id}.pdb`)
   - If `uniprot_id` provided: download AlphaFold2 structure from `https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb`
   - If `pdb_file_path` provided: read directly
   - Save raw PDB to `data/discovery/{session_id}/protein/raw_{name}.pdb`

2. **Structure preparation (pdbfixer):**
   - Add missing atoms: `fixer.findMissingAtoms()`, `fixer.addMissingAtoms()`
   - Add missing residues: `fixer.findMissingResidues()`, `fixer.addMissingResidues()`
   - Remove heteroatoms (waters, ligands): `fixer.removeHeterogens(keepWater=False)`
   - Add hydrogens at pH 7.4: `fixer.addMissingHydrogens(7.4)`
   - Save prepared PDB to `data/discovery/{session_id}/protein/prepared_{name}.pdb`

3. **Pocket definition:**
   - If `pocket_residues` provided: compute bounding box from Cα atoms of those residues. Add 5Å padding. Return `(center_x, center_y, center_z, box_x, box_y, box_z)`.
   - If no pocket specified: run `fpocket` binary if available. Parse top-ranked pocket. If fpocket not available: return a warning in `metadata["warnings"]` and set `metadata["requires_manual_pocket"]` = True.

4. **Output:**
   - `metadata["prepared_pdb_path"]`: path to prepared PDB
   - `metadata["pocket_center"]`: `[x, y, z]`
   - `metadata["pocket_box_size"]`: `[x, y, z]`
   - `metadata["structure_source"]`: "alphafold2" | "rcsb_pdb" | "user_upload"
   - `metadata["warnings"]`: list of any non-fatal issues

**AlphaFold2 caveat (important):** Add a warning to `metadata["warnings"]` if structure source is "alphafold2":
> "AlphaFold2 structures are static predictions. Binding pockets may be closed or collapsed compared to experimental structures. If an experimental PDB with a bound ligand is available for this target, use it instead."

**Exit gate:**
- Given UniProt ID `P00533` (EGFR): downloads AF2 structure, runs pdbfixer, returns prepared PDB path + pocket coordinates
- Given a user-uploaded PDB with pocket_residues: returns correct bounding box
- Warning present in metadata when AlphaFold2 source is used

---

### Plugin 6: `gnina_dock.py`

**Purpose:** Dock molecules against a prepared protein structure using GNINA 1.3. The primary virtual screening engine.

**File:** `src/backend/app/services/plugins/gnina_dock.py`

**Dependencies:** `gnina` binary (install via `pip install gnina` or from GitHub `gnina/gnina`), `rdkit` (for conformer generation)

**IMPORTANT: This plugin is always dispatched as an async Celery task. It never runs synchronously in a FastAPI request handler. See Phase 2.**

**Implementation requirements:**

Pre-processing (per molecule, before submitting to GNINA):
1. Generate 3D conformer if `molecule.conformer_sdf` is None:
   - `AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())`
   - `AllChem.MMFFOptimizeMolecule(mol)`
   - Write to temp SDF file
2. Write batch SDF of all conformers to temp directory

GNINA invocation:
```bash
gnina \
  --receptor {prepared_pdb_path} \
  --ligand {batch_sdf_path} \
  --center_x {cx} --center_y {cy} --center_z {cz} \
  --size_x {sx} --size_y {sy} --size_z {sz} \
  --exhaustiveness {config.get("exhaustiveness", 8)} \
  --num_modes {config.get("num_modes", 1)} \
  --out {output_sdf_path} \
  --cnn_scoring {config.get("cnn_scoring", "rescore")} \
  --cpu {config.get("cpu_threads", 4)} \
  --gpu_platform 0
```

Post-processing:
- Parse output SDF: extract `minimizedAffinity` (Vina score) and `CNNaffinity` (GNINA CNN score) from SDF REMARKS
- Set `molecule.docking_score = cnn_affinity` (primary score)
- Set `molecule.docking_score_vina = minimized_affinity` (secondary)
- Set `molecule.docked_pose_sdf = pose_sdf_string`
- Add `molecule.binding_pocket_ref = prepared_pdb_path`

**Config:**
```json
{
  "prepared_pdb_path": "/path/to/protein_prepared.pdb",
  "pocket_center": [10.5, 22.3, -4.1],
  "pocket_box_size": [20.0, 20.0, 20.0],
  "exhaustiveness": 8,
  "num_modes": 1,
  "top_k": 1000,
  "cnn_scoring": "rescore"
}
```

**For ultra-large libraries (> 100K compounds):** Use Vina-GPU 2.1 as a pre-filter first (same interface, faster, less accurate), then rerun top 10K with GNINA for CNN rescoring. Configure via `config["prefilter_with_vina"] = true`.

**Failure handling:** GNINA occasionally fails on unusual molecular geometries. Catch per-molecule failures, log in `failed_ids`, continue with rest of batch.

**Exit gate:**
- Given a prepared EGFR PDB and 10 test SMILES: returns 10 `MoleculeIR` with `docking_score` and `docked_pose_sdf` populated
- A molecule with invalid 3D geometry logs failure in `failed_ids` without crashing
- Top-K filtering returns exactly `config["top_k"]` molecules sorted by docking score

---

### Plugin 7: `reinvent4_generate.py`

**Purpose:** Generate new candidate molecules using REINVENT4's reinforcement learning loop, scored by GNINA docking.

**File:** `src/backend/app/services/plugins/reinvent4_generate.py`

**Dependencies:** `REINVENT4` (`pip install reinvent4`), `DockStream` (for GNINA integration)

**IMPORTANT: This plugin always runs as an async Celery task. Expected runtime: 2-8 hours.**

**Implementation requirements:**
1. Write a REINVENT4 TOML configuration file to `data/discovery/{session_id}/reinvent4/config.toml`:
   - Prior: `"reinvent"` (SMILES RNN) mode
   - Scoring function: composite of GNINA score (weight 0.6) + ADMET-AI hERG penalty (weight 0.2) + SA Score (weight 0.2)
   - Seed SMILES: `config["seed_smiles"]` (top docking hits from prior stage)
   - Forbidden substructures: from `session_memory.json["constraints"]["forbidden_substructures"]`
   - Required substructures: from `session_memory.json["constraints"]["required_substructures"]` (e.g., covalent warheads)
   - N steps: `config.get("n_steps", 500)`

2. Run REINVENT4: `subprocess.run(["reinvent", "-f", toml_path], timeout=28800)` (8-hour timeout)

3. Parse output SMILES CSV from REINVENT4 results folder

4. Pipe generated SMILES through: `standardize` → `rdkit_filters` → `admet_ai` (inline, not as separate jobs)

5. Return top `config["top_k"]` candidates by `overall_score`

**Config:**
```json
{
  "seed_smiles": ["CCOc1ccc...", "..."],
  "n_steps": 500,
  "top_k": 200,
  "scoring_weights": {
    "docking": 0.6,
    "herg_penalty": 0.2,
    "sa_score": 0.2
  },
  "prepared_pdb_path": "/path/to/protein_prepared.pdb",
  "pocket_center": [10.5, 22.3, -4.1],
  "pocket_box_size": [20.0, 20.0, 20.0]
}
```

**Exit gate:**
- Given 5 seed SMILES and n_steps=10 (test mode): completes without error and returns at least 1 new unique SMILES
- Generated SMILES all pass `standardize` without errors
- Config validation rejects n_steps < 1 or empty seed_smiles

---

### Plugin 8: `aizynthfinder_routes.py`

**Purpose:** Plan retrosynthetic routes for the final hit list using AiZynthFinder.

**File:** `src/backend/app/services/plugins/aizynthfinder_routes.py`

**Dependencies:** `aizynthfinder` (`pip install aizynthfinder`)

**IMPORTANT: Run only on final 50-200 candidates. Per-molecule runtime: 5-30 seconds. Dispatch as Celery task.**

**Implementation requirements:**
1. Initialize `AiZynthFinder` with stock file (Enamine building blocks) and policy network (default USPTO model from `aizynthfinder` package)
2. For each molecule:
   - `finder.target_smiles = molecule.smiles`
   - `finder.tree_search()`
   - `finder.build_routes()`
   - Extract routes: `trees = finder.routes`
   - Parse each route into `SynthesisRoute` dataclass
   - Set `molecule.aizynthfinder_routes`
   - Set `molecule.synthesis_feasible = any(r.solved for r in routes)`
3. Log unsolved molecules in `failed_ids` (not an error — just flagged)

**Exit gate:**
- Aspirin: at least 1 solved route with purchasable building blocks
- A complex natural product: `synthesis_feasible = False`, logged in `failed_ids`
- 50 molecules processed in < 30 minutes

---

## Phase 2 — Async Job Infrastructure

### Why This Is Required

GNINA docking 10K compounds takes 30-60 minutes. REINVENT4 generation takes hours. The existing `subprocess.run()` with 5-minute timeout in `executor.py` cannot handle this. This phase replaces that with a proper async job system.

### Stack

**Celery + Redis.** Both run locally. Redis via Docker (`docker run -d -p 6379:6379 redis:alpine`) or native binary.

**Worker configuration:**
- 1 GPU worker: GNINA docking and REINVENT4 (cannot run simultaneously on < 8GB VRAM — use Celery priority queue with `exclusive_gpu` sentinel)
- 4 CPU workers: ADMET-AI, AiZynthFinder, SA Score, standardization

### Files to Create

**`src/backend/app/core/celery_app.py`**
```python
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "atlas_discovery",
    broker=settings.REDIS_URL,          # default: "redis://localhost:6379/0"
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.docking_tasks",
        "app.tasks.generation_tasks",
        "app.tasks.retrosynthesis_tasks",
        "app.tasks.admet_tasks",
    ]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86400,               # results expire after 24 hours
    task_track_started=True,
    task_acks_late=True,                # don't ack until task completes (fault tolerance)
    worker_prefetch_multiplier=1,       # don't prefetch (GPU tasks are long)
)
```

**`src/backend/app/tasks/docking_tasks.py`**
```python
from app.core.celery_app import celery_app
from app.services.plugins.gnina_dock import GNINADockPlugin
from app.services.plugins.molecule_ir import MoleculeIR

@celery_app.task(bind=True, max_retries=1, queue="gpu")
def run_gnina_docking(self, molecule_dicts: list[dict], config: dict) -> dict:
    """
    Async GNINA docking task.
    molecule_dicts: list of MoleculeIR.to_dict() serialized molecules
    Returns: PluginResult as dict
    """
    molecules = [MoleculeIR(**d) for d in molecule_dicts]
    plugin = GNINADockPlugin()

    # Report progress via Celery's update_state
    self.update_state(state="PROGRESS", meta={"progress": 0, "total": len(molecules)})

    result = plugin.run(molecules, config)

    return {
        "molecules": [m.to_dict() for m in result.molecules],
        "summary": result.summary,
        "metadata": result.metadata,
        "failed_ids": result.failed_ids,
        "errors": result.errors,
    }
```

### API Endpoints to Add to `routes.py`

```
POST /api/discovery/{session_id}/jobs/submit
  Body: { "plugin": str, "molecules": list[dict], "config": dict }
  Returns: { "job_id": str, "task_type": str, "estimated_minutes": int }

GET  /api/discovery/{session_id}/jobs/{job_id}/status
  Returns: { "state": "PENDING"|"STARTED"|"PROGRESS"|"SUCCESS"|"FAILURE",
             "progress": int,        # 0-100
             "total": int,
             "eta_seconds": int,
             "log_tail": str }        # last 500 chars of task log

GET  /api/discovery/{session_id}/jobs/{job_id}/results
  Returns: { "molecules": list[dict], "summary": str, "metadata": dict }
  Only available when state = "SUCCESS"

DELETE /api/discovery/{session_id}/jobs/{job_id}
  Cancels the Celery task via celery_app.control.revoke(job_id, terminate=True)
```

**Frontend integration:** `JobsQueue.tsx` (AgentWorkstreams B5) polls `/jobs/{id}/status` every 5 seconds for PENDING/STARTED/PROGRESS tasks. When state = SUCCESS, call `discoveryStore.onJobComplete(jobId, results)` to update the session state.

---

## Phase 3 — Pipeline Planner

**File:** `src/backend/app/services/agents/pipeline_planner.py`

The Pipeline Planner is a LangGraph node that receives `session_memory.json` and returns a typed `PipelinePlan` JSON. It uses DeepSeek (`orchestrate_constrained`) for reasoning.

### PipelinePlan Schema

```python
@dataclass
class PipelineStage:
    stage_id: int
    name: str
    plugin: str               # must match a registered plugin name
    config: dict              # passed directly to plugin.run()
    depends_on: list[int]     # stage_ids that must complete first
    async_task: bool          # True for gnina_dock, reinvent4_generate, aizynthfinder
    estimated_minutes: int    # shown in UI

@dataclass
class PipelinePlan:
    plan_id: str
    session_id: str
    epoch_id: str
    stages: list[PipelineStage]
    created_by_model: str     # "deepseek-reasoner"
    reasoning: str            # DeepSeek's chain-of-thought (displayed in ExecutionPipeline)
```

### DeepSeek Prompt Template

The prompt is domain-aware. The `{available_plugins}` list is injected by calling `get_available_plugins(session_memory["domain"])` before the prompt is sent.

```
You are a scientific research pipeline planner for the domain: {session_memory["domain"]}.
Given a research goal, select a sequence of computational tools to execute.

Domain: {session_memory["domain"]}

Available plugins for this domain:
{available_plugins}
(These are the ONLY valid plugin names. Do not invent plugin names.)

Research goal: {session_memory["objective"]}
Target / subject: {session_memory["target"]}
Constraints: {session_memory["property_constraints"]}
Success threshold: {session_memory["success_threshold"]}
Available input library / dataset: {session_memory["library_source"]}
Prior session findings: {prior_session_summary}

Return a JSON PipelinePlan. Universal rules (apply regardless of domain):
1. An ingestion/standardization plugin (if one exists for the domain) must always be first.
2. Fast, cheap filters must run before slow, expensive compute (filter first to save resources).
3. Long-running jobs (marked [ASYNC]) must be dispatched as Celery tasks, not run inline.
4. Synthesis/route-planning plugins run last, on the final top-K hits only.
5. If the goal requires a capability not in the available plugin list, emit a capability_gap.
6. Never generate Python code. Return only the JSON plan.
7. Do not use plugin names that are not in the available plugins list for this domain.
```

**Domain-specific rule injection:** Each domain's plugin descriptions include their own ordering constraints (e.g., for `small_molecule`: "rdkit_filters + sa_score must run before gnina_dock"). These are injected into `{available_plugins}` as annotations on each plugin name, not hardcoded into the base prompt.

### Capability Gap Detection

If the researcher's goal requires something no plugin can provide (regardless of domain), the planner emits a `capability_gap` field instead of a stage:

```json
{
  "capability_gap": {
    "required_function": "QSAR activity predictor for GPR84",
    "domain": "small_molecule",
    "input_schema": { "smiles": "string" },
    "output_schema": { "activity_probability": "float 0-1" },
    "standard_reference": "ChEMBL assay CHEMBL123456",
    "resolution_options": ["local_script", "api_endpoint", "skip"]
  }
}
```

Examples of capability gaps across domains:
- **small_molecule**: "Need a custom QSAR model for GPR84 — no public training data"
- **protein**: "Need a co-folding tool for RNA-protein complexes — AF3 only handles protein-ligand"
- **material**: "Need a DFT calculation for magnetic properties — MatGL doesn't predict spin states"

This fires the `CapabilityGapArtifact` UI component (AgentWorkstreams C3). The pipeline halts until the researcher resolves the gap or chooses to skip.

---

## Phase 4 — Structured Experimental Data Ingestion

**This is scientifically critical and domain-agnostic.** Raw experimental values without assay metadata poison the knowledge graph regardless of domain — whether it is IC50 values from a biochemical assay, Tm values from a thermal shift assay, or band gap measurements from UV-Vis spectroscopy. The `Assay`, `Measurement`, and `DoseResponseFit` schema below uses `assay_type` as a free string, allowing any domain to define its measurement type.

This phase adds a validation firewall: no experimental data enters the knowledge graph without structured metadata describing how the measurement was made.

### New SQLite Schema

**Add to `src/backend/app/core/database.py`:**

```python
class Assay(Base):
    """
    Assay / experiment definition. Must be created BEFORE any measurements are accepted.
    This is the scientific metadata that makes measurements interpretable.
    Domain-agnostic: assay_type is a free string (biochemical_IC50, protein_Tm, XRD_band_gap, etc.)
    """
    __tablename__ = "assays"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String, ForeignKey("discovery_sessions.id"), nullable=False)
    domain = Column(String, nullable=False)            # "small_molecule" | "protein" | "material"
    target_name = Column(String, nullable=False)      # e.g., "EGFR kinase domain", "WT KRAS G12C", "LiFePO4"
    assay_type = Column(String, nullable=False)        # "biochemical_IC50" | "cellular_EC50" | "protein_Tm" | "XRD_band_gap" | etc.
    detection_method = Column(String, nullable=False)  # "FP" | "HTRF" | "SPR" | "DSF" | "XRD" | "UV-Vis"
    concentration_unit = Column(String, nullable=False, default="nM")
    upper_limit_of_detection = Column(Float, nullable=False)  # e.g., 10000 nM
    lower_limit_of_detection = Column(Float, nullable=False)  # e.g., 0.1 nM
    positive_control_smiles = Column(String, nullable=True)
    positive_control_expected_value = Column(Float, nullable=True)
    positive_control_tolerance_fold = Column(Float, default=3.0)  # ± 3-fold acceptable
    operator = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    protocol_document_id = Column(String, ForeignKey("documents.id"), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())


class Measurement(Base):
    """
    Individual experimental data point.
    Always linked to an Assay — cannot exist without one.
    Domain-agnostic: entity_id references EntityIR.entity_id regardless of domain.
    """
    __tablename__ = "measurements"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    entity_id = Column(String, nullable=False)         # EntityIR.entity_id (ATLAS-00042, etc.)
    entity_key = Column(String, nullable=False, index=True)  # domain-specific unique key: InChIKey (small_molecule) | sequence_hash (protein) | formula+spacegroup (material)
    assay_id = Column(String, ForeignKey("assays.id"), nullable=False)
    concentration = Column(Float, nullable=False)
    response = Column(Float, nullable=False)
    replicate_id = Column(Integer, nullable=False)
    plate_id = Column(String, nullable=True)
    well_position = Column(String, nullable=True)      # e.g., "B04"
    batch_effect_flag = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())


class DoseResponseFit(Base):
    """
    4PL curve fit results for an entity-assay pair.
    Used for dose-response assays (IC50, EC50, Kd). Not used for single-point measurements
    (e.g., material band gap) — those are stored as Measurement records directly.
    """
    __tablename__ = "dose_response_fits"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    entity_id = Column(String, nullable=False, index=True)
    entity_key = Column(String, nullable=False)  # domain-specific unique key (InChIKey, sequence_hash, etc.)
    assay_id = Column(String, ForeignKey("assays.id"), nullable=False)
    ic50 = Column(Float, nullable=True)                # None if fit failed
    ic50_censored = Column(Boolean, default=False)     # True if IC50 > upper LOD
    hill_slope = Column(Float, nullable=True)
    emax = Column(Float, nullable=True)
    emin = Column(Float, nullable=True)
    r_squared = Column(Float, nullable=True)
    curve_quality = Column(String, nullable=True)      # "excellent" | "acceptable" | "poor" | "failed"
    n_replicates = Column(Integer, nullable=False)
    n_concentrations = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())
```

### Ingest API Endpoint

**`POST /api/discovery/{session_id}/wetlab/upload`**

Request: `multipart/form-data` with:
- `file`: CSV or SDF file
- `assay_id`: string — **REQUIRED. Reject with 422 if missing.**
- `file_type`: `"dose_response_csv"` | `"single_point_csv"` | `"sdf_with_activity"`

**Validation pipeline (in order — fail fast):**

1. **Assay exists check:** Query `Assay` table by `assay_id`. Return 422 if not found: `"assay_id {x} not found. Create the assay definition first at POST /api/discovery/{session_id}/wetlab/assay"`

2. **CSV column validation:** Required columns for `dose_response_csv`: `compound_id` (or `smiles`), `concentration`, `response`, `replicate`. Return 422 with missing column names if any are absent.

3. **Concentration range check:** All concentrations must be between `assay.lower_limit_of_detection` and `assay.upper_limit_of_detection`. Warn (don't reject) if any values are at the detection limit — set `Measurement.batch_effect_flag = True` for those rows.

4. **Positive control check:** If `assay.positive_control_smiles` is set, find positive control rows in CSV. Compute observed mean activity. If outside `expected ± tolerance_fold`, reject entire file: `"Positive control out of range. Expected {expected} nM ± {fold}-fold, observed {observed} nM. Check assay conditions before uploading."`

5. **Z' factor check** (for HTS plates): Compute Z' = 1 - (3*(σ_pos + σ_neg) / |µ_pos - µ_neg|). If Z' < 0.5, reject: `"Plate quality failed QC (Z' = {z_prime:.2f} < 0.5)."`

6. **4PL curve fitting** (for dose_response_csv): Use `scipy.optimize.curve_fit` with 4PL model:
   - `y = Emin + (Emax - Emin) / (1 + (IC50 / x)^HillSlope)`
   - If fit converges: create `DoseResponseFit` record
   - If IC50 > `upper_limit_of_detection`: set `ic50_censored = True`, IC50 = None
   - Compute R² — classify curve quality: > 0.95 = excellent, 0.85-0.95 = acceptable, 0.7-0.85 = poor, < 0.7 = failed

7. **Knowledge graph update:** For each compound with a successful fit, create/update a `Node` with `label = inchikey`, `properties = {ic50_nM, target, passed, assay_type}`. Add `BINDS_TO` edge to target node.

**Response:**
```json
{
  "status": "accepted",
  "n_compounds": 12,
  "n_successful_fits": 10,
  "n_censored": 2,
  "n_failed_fits": 0,
  "z_prime": 0.74,
  "confirmed_hits": ["ATLAS-00042", "ATLAS-00038"],
  "graph_nodes_updated": 12
}
```

---

## Phase 5 — Output Compiler

**File:** `src/backend/app/services/output_compiler.py`

Called at the end of every pipeline run to compile the final deliverable.

### Output 1: JSON Hit Cards

```python
def compile_hit_cards(molecules: list[MoleculeIR], session_id: str, run_id: str) -> list[dict]:
    """
    Compile final hit cards from ranked MoleculeIR list.
    Returns list of hit card dicts sorted by overall_score descending.
    """
```

Each hit card contains:
```json
{
  "rank": 1,
  "compound_id": "ATLAS-00042",
  "smiles": "CCOc1ccc(NC(=O)c2ccc(Cl)cc2)cc1",
  "source": "REINVENT4 cycle 2",
  "overall_score": 0.87,
  "docking_score_kcal_mol": -9.2,
  "docking_software": "GNINA_1.3",
  "target_protein": "EGFR_AF2_prepared",
  "catalog_id": "EN300-12345",
  "catalog_source": "Enamine REAL",
  "price_usd": 190,
  "lead_time_weeks": 3,
  "rdkit_props": { "mw": 303.7, "logp": 3.2, "tpsa": 58.6, "hbd": 1, "hba": 3 },
  "admet_flags": {
    "herg_risk": "LOW",
    "caco2_permeability": "HIGH",
    "dili_risk": "LOW",
    "cyp3a4_inhibition": "MEDIUM"
  },
  "sa_score": 2.4,
  "rascore": 0.87,
  "pains_alerts": [],
  "brenk_alerts": [],
  "synthesis_routes": [
    { "n_steps": 3, "solved": true, "building_blocks": ["EN300-AA1", "EN300-BB2"] }
  ],
  "tanimoto_to_nearest_known_active": 0.31,
  "pose_sdf": "...",
  "pipeline_run_id": "run-uuid"
}
```

### Output 2: Enamine Order CSV

```python
def compile_enamine_order(molecules: list[MoleculeIR], project_id: str) -> str:
    """Returns CSV string ready for upload to Enamine ordering portal."""
```

CSV format:
```csv
SMILES,catalog_id,quantity_mg,purity_requirement,project_id,notes
CCOc1ccc...,EN300-12345,5,>95%,ATLAS-KRAS-001,Rank 1 GNINA -9.2 kcal/mol
```

Only include molecules where `catalog_id` is not None. If `catalog_id` is None (custom synthesis needed), exclude from Enamine CSV and include in a separate "requires synthesis" list.

### Output 3: Session Report

```python
def compile_session_report(session_id: str, run_id: str, molecules: list[MoleculeIR]) -> bytes:
    """
    Generate a PDF session report using reportlab.
    Returns PDF bytes.
    """
```

Report sections:
1. Header: project name, target, date, researcher, pipeline stages executed
2. Screening funnel: bar chart of compounds at each stage (total → after filters → after docking → final hits)
3. Top 20 hit cards: 2D structure image (RDKit `Draw.MolToImage`), compound ID, rank, score, key ADMET flags
4. ADMET heatmap: rows = top 20 compounds, columns = key endpoints (hERG, DILI, Caco2, CYP3A4), color-coded
5. Synthesis feasibility: pie chart of solved/unsolved routes
6. Appendix: full hit card JSON

**Dependencies:** `reportlab` (`pip install reportlab`) for PDF generation

---

## Phase 6 — Feedback Loop and Cross-Session Memory

### Closing the Loop

When wet lab results arrive and pass Phase 4 validation:

1. **Update MoleculeIR:** Append `AssayResult` to `molecule.assay_results`. Set `confirmed_hit = (ic50 < session_memory["success_threshold_nm"])`.

2. **Update Knowledge Graph:** In `src/backend/app/services/graph.py`, add method:
   ```python
   def upsert_compound_activity_node(self, inchikey: str, assay_result: AssayResult) -> Node:
       """
       Create or update a Node for a compound with its confirmed activity.
       Uses InChIKey as the structure-based identifier (survives SMILES changes).
       """
   ```
   Node schema:
   - `label = inchikey`
   - `properties.smiles = molecule.smiles`
   - `properties.confirmed_ic50_nM = assay_result.value`
   - `properties.target = assay_result.target_name`
   - `properties.confirmed_hit = assay_result.passed`
   - `properties.assay_type = assay_result.assay_type`
   - `properties.date = assay_result.date.isoformat()`

   Edges: `BINDS_TO` → target Node; `PART_OF` → scaffold cluster Node (by Murcko scaffold); `GENERATED_IN_SESSION` → session Node

3. **Cross-session knowledge transfer:** When a new session starts on the same target (matched by `biological_target` string), `scan_corpus` in `coordinator.py` queries the knowledge graph for historical results:
   ```python
   historical_hits = graph.query(
       "MATCH (n:compound)-[r:BINDS_TO]->(t:target {name: $target})
        WHERE n.confirmed_ic50_nM IS NOT NULL
        RETURN n ORDER BY n.confirmed_ic50_nM ASC LIMIT 10",
       target=session_memory["biological_target"]
   )
   ```
   This context is injected into the Coordinator's opening prompt: "Prior sessions found X confirmed hits against this target. Best: {SMILES}, IC50 = {value} nM. Known liabilities: {scaffold} → hERG risk."

4. **REINVENT4 prior update:** For new generation cycles, add confirmed active SMILES to REINVENT4's `scoring_function.activity_scorer.known_actives` list. This biases the RL loop toward confirmed active scaffolds in the next cycle.

---

## Implementation Priority

Build in this order. Each group is a coherent deliverable. **Groups 1–6 implement the `small_molecule` domain as proof-of-concept.** The domain-agnostic foundation (Group 0) must be built first so that future domains slot in without restructuring.

### Group 0 — Domain-Agnostic Foundation (Build First, Touch Never Again)

| # | File | Task |
|---|------|------|
| 0a | `plugins/entity_ir.py` | `EntityIR` base class + `AssayResult`. All domains depend on this. |
| 0b | `plugins/base.py` | `BasePlugin` ABC + `PluginResult`. Domain-agnostic. All plugins implement this. |
| 0c | `plugins/__init__.py` | Domain-scoped `PLUGIN_REGISTRY` + `get_plugin(domain, name)`. |

**Deliverable after Group 0:** The plugin architecture is locked. Any domain can be added by creating an IR subclass and registering plugins. No structural code changes required for future domains.

### Group 1 — Small Molecule Foundation (Proof-of-Concept Domain)

| # | File | Task |
|---|------|------|
| 1 | `plugins/molecule_ir.py` | `MoleculeIR(EntityIR)` subclass. Small molecule domain IR. |
| 2 | `plugins/standardize.py` | SMILES → MoleculeIR. Always the first plugin in small_molecule pipelines. |
| 3 | `plugins/rdkit_filters.py` | Property computation + PAINS/Brenk/Lipinski. |
| 4 | `plugins/sa_score.py` | SA Score + RAscore. |

**Deliverable after Group 1:** Can ingest any SMILES list, standardize, compute physicochemical properties, filter by Lipinski/PAINS/user constraints, and score synthesizability. No GPU required. No external APIs.

### Group 2 — ADMET and Async Infrastructure

| # | File | Task |
|---|------|------|
| 5 | `plugins/admet_ai.py` | 41-endpoint ADMET prediction (small_molecule domain). |
| 6 | `core/celery_app.py` | Celery + Redis configuration. **Domain-agnostic — all future domains use this.** |
| 7 | `tasks/docking_tasks.py` | Async docking task wrapper. |
| 8 | `routes.py` additions | Job submit/status/results/cancel endpoints. **Domain-agnostic.** |

**Deliverable after Group 2:** Full filter cascade (standardize → Lipinski → PAINS → SA Score → ADMET-AI) on any compound list. Async job infrastructure ready — works for any domain's long-running tasks.

### Group 3 — Docking Engine

| # | File | Task |
|---|------|------|
| 9 | `plugins/structure_prep.py` | AlphaFold2 fetch + pdbfixer + pocket detection. |
| 10 | `plugins/gnina_dock.py` | GNINA docking plugin. |
| 11 | `tasks/docking_tasks.py` | Full implementation with progress reporting. |

**Deliverable after Group 3:** End-to-end virtual screening. Researcher provides target → Atlas fetches structure → screens compounds → returns docking scores. Complete hit-finding campaign.

### Group 4 — Planning and Output

| # | File | Task |
|---|------|------|
| 12 | `agents/pipeline_planner.py` | Domain-aware DeepSeek routing. Reads `session_memory["domain"]`, injects available plugins. **Works for all domains.** |
| 13 | `services/output_compiler.py` | Hit cards, order CSV, session report PDF. **Domain-agnostic structure.** |

**Deliverable after Group 4:** Fully automated pipeline from natural-language goal to order CSV. The researcher clicks "start", reviews the hit list, and places an order.

### Group 5 — Lead Optimization and Synthesis Planning

| # | File | Task |
|---|------|------|
| 14 | `plugins/reinvent4_generate.py` | REINVENT4 generative design (small_molecule domain). |
| 15 | `plugins/aizynthfinder_routes.py` | AiZynthFinder retrosynthesis (small_molecule domain). |

**Deliverable after Group 5:** Full hit-to-lead optimization. From initial virtual screen hits, generates novel optimized analogs and provides synthesis routes.

### Group 6 — Feedback Loop (The Moat, All Domains)

| # | File | Task |
|---|------|------|
| 16 | `core/database.py` additions | `Assay`, `Measurement`, `DoseResponseFit` tables — domain field included, works for any domain. |
| 17 | `routes.py` — experimental data endpoints | Upload, validate, fit, graph update. |
| 18 | `services/graph.py` additions | `upsert_entity_result_node()` — generic (replaces compound-specific method). |
| 19 | `agents/coordinator.py` update | Query historical graph data at session start (any domain). |

**Deliverable after Group 6:** Closed loop. Experimental results flow back into the knowledge graph. Each new session is informed by all prior experimental data. The system gets better with every experiment — for any domain.

---

## Plugin Registration

**File:** `src/backend/app/services/plugins/__init__.py`

The registry is domain-scoped. The Pipeline Planner reads `session_memory["domain"]` to select the correct registry. New domains are added by creating a new key — no existing code changes required.

```python
from .base import BasePlugin
from .standardize import StandardizePlugin
from .rdkit_filters import RDKitFiltersPlugin
from .sa_score import SaScorePlugin
from .admet_ai import ADMETAIPlugin
from .structure_prep import StructurePrepPlugin
from .gnina_dock import GNINADockPlugin
from .reinvent4_generate import Reinvent4GeneratePlugin
from .aizynthfinder_routes import AiZynthFinderPlugin

# Domain-scoped registry.
# To add a new domain: add a new top-level key and import its plugins.
# Existing domains and plugins are never modified.
PLUGIN_REGISTRY: dict[str, dict[str, type[BasePlugin]]] = {
    "small_molecule": {
        "standardize": StandardizePlugin,
        "rdkit_filters": RDKitFiltersPlugin,
        "sa_score": SaScorePlugin,
        "admet_ai": ADMETAIPlugin,
        "structure_prep": StructurePrepPlugin,
        "gnina_dock": GNINADockPlugin,
        "reinvent4_generate": Reinvent4GeneratePlugin,
        "aizynthfinder_routes": AiZynthFinderPlugin,
    },
    # Future domains — not yet implemented. Shown here as the expansion pattern.
    # "protein": {
    #     "alphafold2_fold": AlphaFold2Plugin,
    #     "esm_fold": ESMFoldPlugin,
    #     "esm_stability": ESMStabilityPlugin,
    #     "rosetta_design": RosettaDesignPlugin,
    # },
    # "material": {
    #     "matgl_predict": MatGLPlugin,
    #     "mp_lookup": MaterialsProjectPlugin,
    #     "pymatgen_filters": PyMatGenFiltersPlugin,
    #     "vasp_dispatch": VASPDispatchPlugin,
    # },
}


def get_plugin(domain: str, name: str) -> BasePlugin:
    domain_registry = PLUGIN_REGISTRY.get(domain)
    if domain_registry is None:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(PLUGIN_REGISTRY.keys())}")
    if name not in domain_registry:
        raise ValueError(
            f"Unknown plugin '{name}' for domain '{domain}'. "
            f"Available: {list(domain_registry.keys())}"
        )
    return domain_registry[name]()


def get_available_plugins(domain: str) -> list[str]:
    """Returns plugin names available for a given domain. Used by Pipeline Planner prompt."""
    return list(PLUGIN_REGISTRY.get(domain, {}).keys())
```

The Pipeline Planner validates plugin names against `get_available_plugins(domain)` before returning a plan. Unknown plugin names in the plan are rejected as capability gaps.

---

## New Python Dependencies to Add to `requirements.txt`

### Domain-Agnostic (All Domains)
```
# Async job infrastructure — required for any domain with long-running tools
celery[redis]>=5.3.0
redis>=5.0.0

# Data processing — required for experimental data ingest
scipy>=1.12.0                # 4PL curve fitting

# Reporting
reportlab>=4.0.0             # PDF session reports
```

### Small Molecule Domain (Proof-of-Concept)
```
rdkit>=2024.3.1              # core chemistry (likely already present)
chembl-structure-pipeline>=1.2.0  # SMILES standardization
RAscore>=1.0.2               # retrosynthetic accessibility
admet_ai>=1.2.0              # 41-endpoint ADMET prediction
aizynthfinder>=4.2.0         # retrosynthesis planning
pdbfixer>=1.9                # protein structure preparation for docking
```

GNINA and REINVENT4 are installed as separate binaries/packages:
- GNINA: `pip install gnina` or clone `gnina/gnina` and build
- REINVENT4: `pip install reinvent4` + `pip install DockStream`

### Future Domain Dependencies (Install When Domain Is Added)
```
# Protein engineering domain
# fair-esm>=2.0.0            # ESM-2 embeddings + ESM-IF1 stability
# openmm>=8.0.0              # molecular dynamics (pdbfixer dependency, already installed)

# Materials science domain
# matgl>=1.1.0               # M3GNet / CHGNet property prediction
# pymatgen>=2024.1.1         # crystal structure analysis + enumeration
# smact>=2.4.0               # charge-balanced composition screening
# mp-api>=0.41.0             # Materials Project REST API client
```

---

## Key Scientific Design Rules

These rules encode the scientific rigor decisions made during architecture design. Any agent implementing this system must follow them. Rules 1–3 are domain-agnostic and apply to all future domains.

1. **LLM never writes domain-specific code.** The LLM only routes (selects plugin + config from the registry). All computation runs in pre-written, hardened plugins. This applies to chemistry, protein folding, DFT, and any future domain.

2. **EntityIR is the universal intermediate.** No raw domain data (SMILES strings, PDB files, CIF files) is passed directly between pipeline stages. Every entity in every pipeline is an `EntityIR` subclass. Plugins receive and return `list[EntityIR]`.

3. **PAINS are flags, not hard rejections.** `reject_any_pains = false` by default. Many known drugs trigger PAINS alerts. Only Brenk alerts (toxic/reactive) default to hard rejection.

4. **AlphaFold2 structures always carry a warning.** AF2 structures may have closed or collapsed binding pockets. Agents must surface this warning to the researcher.

5. **Wet lab data cannot enter the knowledge graph without an assay definition.** Uploading `compound_id, ic50_nM` alone is rejected. The assay type, detection method, controls, and detection limits are mandatory.

6. **IC50 at the upper detection limit is censored.** If IC50 = `upper_limit_of_detection`, it is flagged as `ic50_censored = True` and stored as `None`. "IC50 > 10 µM" is not the same as "IC50 = 10 µM".

7. **Deduplication is by InChIKey, not SMILES.** The same molecule can have many valid SMILES representations. InChIKey is the canonical structure identifier.

8. **AiZynthFinder runs last, on top hits only.** It is computationally expensive (5-30s per molecule). Apply to the final 50-200 candidates, never the full screening library.

9. **REINVENT4 generation is always followed by re-screening.** Generated molecules are not trusted — they must pass the same filter cascade (standardize → rdkit_filters → admet_ai → gnina_dock) before appearing in hit lists.

10. **Positive controls must pass before plate data is accepted.** Z' < 0.5 or positive control out of range = entire plate rejected. Batch effects cannot be retrospectively corrected.

---

## Relationship to Existing Code

| Existing file | Relationship |
|---|---|
| `agents/coordinator.py` | Unchanged for now. Phase 6 adds historical graph query at session start. `session_memory["domain"]` field added to extracted JSON. |
| `agents/executor.py` | `generate_script` node replaced by `dispatch_plugin` node. The loop structure (plan → act → observe) stays the same; only the action layer changes. `get_plugin(domain, name)` replaces direct script generation. |
| `discovery_llm.py` | Unchanged. Pipeline Planner uses `orchestrate_constrained()` (DeepSeek). |
| `discovery_session.py` | `session_memory.json` schema extended with `domain`, `library_source`, `success_threshold`, `target`. Field `biological_target` renamed to `target` (domain-agnostic). |
| `graph.py` | Extended with `upsert_entity_result_node()` in Phase 6. Replaces compound-specific method. |
| `database.py` | Extended with `Assay` (adds `domain` field), `Measurement` (entity_id/entity_key instead of compound_id/inchikey), `DoseResponseFit` in Phase 4. |
| `routes.py` | New endpoints added in Phases 2, 4, 5. Existing endpoints unchanged. |
| `AgentWorkstreams.md` | Frontend Golden Path — runs in parallel. `ExperimentalFeedbackForm.tsx` (Agent D5) calls Phase 4 ingest. `JobsQueue.tsx` (Agent B5) polls Phase 2 job status. |

---

---

## Domain Expansion Guide

This section documents exactly what is required to add a new research domain to Atlas. The proof-of-concept domain (`small_molecule`) proves the pattern. Each new domain follows identical steps.

### What Changes Per Domain

| Component | What to do |
|---|---|
| `{domain}_ir.py` | Create `{Domain}IR(EntityIR)` subclass with domain-specific fields |
| `plugins/{name}.py` | Write one plugin per tool (follows `BasePlugin` ABC) |
| `plugins/__init__.py` | Add domain key to `PLUGIN_REGISTRY` with its plugins |
| `session_memory.json` | Add domain-specific constraint keys (coordinator extracts these) |
| `coordinator.py` prompt | Add domain-specific HITL questions when `domain == "{domain}"` |

**That is the complete list.** Everything else — Celery jobs, Pipeline Planner routing, wet lab ingest, knowledge graph update, feedback loop, frontend job queue — requires no changes.

### AlphaFold / Protein Engineering Domain (Expansion Example)

**Files to create:**
1. `plugins/protein_ir.py` — `ProteinIR(EntityIR)` (schema defined in IR section above)
2. `plugins/alphafold2_fold.py` — Downloads AF2 structure from EBI AlphaFold API given UniProt ID. Populates `ProteinIR.pdb_path` and `af2_plddt`.
3. `plugins/esm_fold.py` — Local ESMFold inference for sequences AF2 doesn't cover. 5-50x faster than AF2. Populates `pdb_path`.
4. `plugins/esm_stability.py` — ESM-IF1 or ESM-2 inverse folding: predicts ΔΔG of mutations. Populates `stability_ddg`.
5. `plugins/rosetta_design.py` — Given a target backbone, designs sequences with RosettaFastRelax. Generates new `ProteinIR` entities.
6. `plugins/alphafold3_cofold.py` — Protein + ligand co-folding (AF3 API or local). Populates `af3_iptm`. **Async Celery task.**

**Registry addition:**
```python
"protein": {
    "alphafold2_fold": AlphaFold2Plugin,
    "esm_fold": ESMFoldPlugin,
    "esm_stability": ESMStabilityPlugin,
    "rosetta_design": RosettaDesignPlugin,
    "alphafold3_cofold": AlphaFold3CofoldPlugin,
},
```

**Coordinator additions:** When `domain == "protein"`, the coordinator asks: What is the wild-type protein sequence or UniProt ID? Is this stability engineering or binding design? What is the success criterion (ΔTm > 5°C, Kd < 10 nM)? Which positions are allowed to mutate?

**Wet lab ingest additions:** `assay_type = "protein_Tm"` uses `DoseResponseFit` table for thermal unfolding curves (DSF). `assay_type = "protein_Kd"` uses SPR or ITC data. The `entity_key` column stores a SHA256 hash of the sequence.

---

### Materials Science Domain (Expansion Example)

**Files to create:**
1. `plugins/material_ir.py` — `MaterialIR(EntityIR)` (schema defined in IR section above)
2. `plugins/matgl_predict.py` — M3GNet or CHGNet inference: crystal structure → formation energy, band gap, bulk modulus. Local, no GPU needed for inference. Populates `formation_energy_ev_per_atom`, `band_gap_ev`.
3. `plugins/mp_lookup.py` — Query Materials Project API for known DFT properties on a formula. No compute required.
4. `plugins/pymatgen_filters.py` — Filter by thermodynamic stability (above convex hull), charge neutrality (SMACT), electronegativity rules.
5. `plugins/vasp_dispatch.py` — Generate VASP INCAR/POSCAR/KPOINTS input files, dispatch to external HPC cluster via SSH + Slurm. Poll for completion. **Always async.**
6. `plugins/crystalformer_generate.py` — Generative crystal structure design (analogous to REINVENT4 for materials). **Async Celery task.**

**Registry addition:**
```python
"material": {
    "matgl_predict": MatGLPlugin,
    "mp_lookup": MaterialsProjectPlugin,
    "pymatgen_filters": PyMatGenFiltersPlugin,
    "vasp_dispatch": VASPDispatchPlugin,
    "crystalformer_generate": CrystalFormerPlugin,
},
```

**Key difference from small_molecule:** `vasp_dispatch` doesn't run computation locally — it generates input files and dispatches to a cluster. The plugin's `run()` submits the Slurm job and returns immediately with a `job_id`. The Celery task polls the cluster until completion and fetches output files. This is the same async pattern as GNINA docking, just with an external cluster instead of local GPU.

---

## End-to-End Walkthrough (Small Molecule Domain — Proof of Concept)

A researcher opens Atlas and types: *"I want to find a KRAS G12C covalent inhibitor. Orally bioavailable. No hERG liability. MW < 500."*

```
Step 1: Coordinator (existing, extended with domain field)
  DeepSeek HITL extracts:
    - domain: "small_molecule"           ← NEW field — routes to small_molecule plugin registry
    - target: KRAS G12C (UniProt P01116)
    - covalent_warhead: required (acrylamide or vinyl sulfone)
    - property_constraints: MW <= 500, hERG_risk != HIGH
    - success_threshold: 1000 nM (1 µM — user refines this)
  Writes: session_memory.json, SESSION_CONTEXT.md

Step 2: Pipeline Planner (Phase 3)
  DeepSeek reads session_memory.json (domain = "small_molecule")
  Calls get_available_plugins("small_molecule") → injects into prompt
  Returns PipelinePlan:
    Stage 1: structure_prep (UniProt P01116, switch-II pocket)
    Stage 2: standardize (ZINC22 covalent fragment library, ~50K)
    Stage 3: rdkit_filters (MW<=500, covalent_warhead_filter=acrylamide|vinyl_sulfone)
    Stage 4: admet_ai (hERG threshold: reject HIGH)
    Stage 5: sa_score (reject sa_score > 6)
    Stage 6: gnina_dock (top_k=1000) [ASYNC]
    Stage 7: reinvent4_generate (seeds=top 50 from stage 6) [ASYNC]
    Stage 8: aizynthfinder_routes (top 50 from stage 7) [ASYNC]

Step 3: Execution
  Stages 1-5 run synchronously (~15 minutes, 50K → ~8K compounds)
  Stages 6-8 dispatched as Celery tasks
  UI shows JobsQueue with progress bars
  ~6 hours later: all jobs complete

Step 4: Output Compiler (Phase 5)
  JSON hit cards for top 50 compounds
  Enamine order CSV (30 compounds available from REAL catalog)
  AiZynthFinder routes for remaining 20 (custom synthesis needed)
  Session report PDF

Step 5: Researcher reviews
  Approves 10 compounds → Enamine order placed
  3 weeks later: receives DMSO stocks

Step 6: Wet Lab (off-Atlas)
  Biochemical IC50 assay run
  KRAS GDP-bound protein, competitive binding
  8-point dose-response, in duplicate

Step 7: Wet Lab Ingest (Phase 4)
  Researcher creates Assay definition:
    target: KRAS G12C, type: biochemical_IC50, unit: nM,
    upper_LOD: 10000, positive_control: AMG-510 (Sotorasib), expected: 0.09 nM
  Uploads plate reader CSV
  Validation: ✓ positive control 0.07 nM (within 3-fold), ✓ Z' = 0.81
  4PL fitting: 3 compounds pass IC50 < 1000 nM
  Knowledge graph updated: ATLAS-00042 (IC50 = 43 nM, confirmed_hit = True)

Step 8: Next iteration
  Researcher starts new session on KRAS G12C
  Coordinator reads graph history:
    "Prior session: 10 compounds tested. Best hit: ATLAS-00042, IC50 = 43 nM.
     Scaffold: benzothiophene-acrylamide. Two compounds showed hERG liability —
     naphthalene ring system excluded from new constraints."
  Pipeline Planner biases next screen toward ATLAS-00042 analogs
  REINVENT4 uses ATLAS-00042 as seed SMILES
```

This is the complete closed loop.
