# AGENTS.md - Atlas 2.0 (ContAInuum)

> This file provides comprehensive guidance to AI coding agents working on the Atlas project. Expect the reader to know nothing about the project.

## Project Overview

**Atlas 2.0** (codename: ContAInuum) is an AI-native knowledge management desktop application that builds a continuous knowledge layer beneath an AI model. It is a **standalone Windows desktop application** powered by a Multi-Agent LangGraph Architecture.

**Core Philosophy**: "The AI does not know things. It queries a living knowledge substrate."

### Key Features

- **Local-First Architecture** - All data stays on the user's computer. Zero cloud dependencies.
- **Agentic RAG (Swarm)** - Dynamic query routing via a Meta-Router to specialized agents (Librarian, Navigator, Cortex).
- **Multi-Turn Reflection Loops** - Agents plan, explore graphs, retrieve iteratively, and self-critique.
- **Knowledge Graph** - Entities and relationships stored in SQLite with Rustworkx for high-performance graph operations.
- **Constrained Generation** - Uses GBNF grammars with llama-cpp-python for reliable JSON output from small local models.
- **Hybrid RAG** - Combines vector search (Qdrant), knowledge graph traversal, and BM25 text retrieval.

---

## Technology Stack

### Three-Layer Architecture

```
Atlas (Desktop Application)
├── Frontend Layer (Next.js 14, React, TypeScript, Tailwind CSS)
│   └── Port: 3001 (dev) | Static export (production)
├── Tauri Shell (Rust)
│   └── Manages sidecar processes, window, native APIs
└── Backend Layer (Python/FastAPI)
    ├── FastAPI server (port 8000)
    ├── SQLite (embedded, no external server)
    ├── Qdrant vector store (embedded, path mode)
    ├── Rustworkx knowledge graph engine
    └── LangGraph multi-agent swarm
```

### Frontend Stack

| Component | Technology |
|-----------|------------|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript 5.3 |
| Styling | Tailwind CSS 3.4 |
| UI Components | Custom + Framer Motion |
| State Management | Zustand |
| Rich Text | Tiptap |
| Graph Visualization | @xyflow/react, react-force-graph-2d |
| Build Output | Static export to `out/` |

### Backend Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI 0.109 |
| Database | SQLite (SQLAlchemy 2.0) |
| Vector Store | Qdrant Client 1.12+ (embedded mode) |
| Graph Engine | Rustworkx 0.14+ |
| LLM Runtime | llama-cpp-python 0.2.83 |
| Embeddings | sentence-transformers (Nomic Embed Text) |
| NER | GLiNER (fast entity extraction) |
| Agent Framework | LangGraph 0.2+ |
| Document Parsing | Docling, pdfplumber, PyPDF |
| Reranking | FlashRank |
| Chunking | semantic-text-splitter (Rust-based) |

### Desktop Shell Stack

| Component | Technology |
|-----------|------------|
| Framework | Tauri 1.5 |
| Language | Rust (Edition 2021) |
| Build Tool | cargo tauri |
| Output | Windows MSI/EXE installer |

---

## Project Structure

```
Project Root
├── config/                  # Configuration files
│   ├── .env                 # Environment variables (gitignored)
│   ├── .env.example         # Template for env vars
│   ├── .aider.conf.yml      # Aider AI assistant config
│   └── .aiderignore         # Files to ignore in Aider
├── installers/              # Compiled binaries (.exe/.msi)
├── models/                  # AI models (gitignored)
│   ├── *.gguf              # LLM model files
│   ├── nomic-embed-text-v1.5/  # Embedding model
│   └── gliner_small-v2.1/  # NER model
├── scripts/                 # PowerShell automation scripts
│   ├── build/
│   │   └── build-backend.ps1    # PyInstaller build with caching
│   ├── dev/
│   │   ├── run_backend.ps1      # Start backend dev server
│   │   └── launch_hybrid_aider.ps1
│   └── setup/
│       └── setup_project.ps1    # Initial project setup
├── src/
│   ├── backend/            # Python FastAPI backend
│   │   ├── app/
│   │   │   ├── api/
│   │   │   │   └── routes.py        # API endpoints
│   │   │   ├── core/
│   │   │   │   ├── config.py        # Settings & env vars
│   │   │   │   ├── database.py      # SQLAlchemy models
│   │   │   │   ├── memory.py        # Working memory
│   │   │   │   └── qdrant_store.py  # Vector store
│   │   │   └── services/
│   │   │       ├── agents/          # LangGraph agents
│   │   │       │   ├── coordinator.py      # Discovery OS — HITL session bootstrapper
│   │   │       │   ├── executor.py         # Discovery OS — script-generation sandbox
│   │   │       │   ├── pipeline_planner.py # Deterministic plugin pipeline orchestrator
│   │   │       │   ├── discovery_chat.py  # Unified chat orchestrator (all phases)
│   │   │       │   ├── discovery_graph.py  # Navigator (deep graph walk)
│   │   │       │   ├── experts/            # MoE experts
│   │   │       │   ├── librarian.py
│   │   │       │   ├── meta_router.py
│   │   │       │   └── supervisor.py
│   │   │       ├── plugins/               # Chemistry plugin registry
│   │   │       │   ├── __init__.py        # BasePlugin ABC + PluginManager singleton + built-in tools
│   │   │       │   ├── base.py            # BasePlugin abstract class definition
│   │   │       │   ├── admet_predict.py   # ADMET property prediction (hERG, DILI, Caco2, CYP3A4)
│   │   │       │   ├── sa_scorer.py       # Synthetic accessibility scoring (SA 1-10)
│   │   │       │   ├── standardize.py     # SMILES canonicalization + InChIKey + dedup (RDKit)
│   │   │       │   ├── properties.py      # RDKit molecular property predictor (MW, LogP, TPSA, HBD, HBA)
│   │   │       │   ├── retrosynthesis.py  # AiZynthFinder ONNX retrosynthesis (plan_synthesis)
│   │   │       │   ├── spectrum.py        # NMR .jdx verifier via nmrglue (verify_spectrum)
│   │   │       │   ├── strategy.py        # Synthesis route scoring + viability (evaluate_strategy)
│   │   │       │   └── toxicity.py        # SMARTS/PAINS structural alert screening
│   │   │       ├── candidate_generation.py # LLM+RAG candidate generation + RDKit screening (SSE)
│   │   │       ├── chat.py
│   │   │       ├── discovery_llm.py        # Isolated LLM service for Discovery OS
│   │   │       ├── discovery_session.py    # Session CRUD + SessionMemoryService
│   │   │       ├── document.py
│   │   │       ├── domain_tools.py         # Molecular/entity rendering + capability gap storage
│   │   │       ├── graph.py
│   │   │       ├── ingest.py
│   │   │       ├── llm.py
│   │   │       ├── retrieval.py
│   │   │       ├── spectroscopy.py         # Mock retrosynthesis route planning (SSE streaming)
│   │   │       ├── stage_context.py        # contextvars bridge — injects epoch state into LLM prompts
│   │   │       ├── swarm.py
│   │   │       └── synthesis_memory.py     # SynthesisAttempt graph node + Qdrant embedding (compound moat)
│   │   ├── data/           # Runtime data (uploads, drafts)
│   │   ├── models/         # AI model storage
│   │   ├── requirements.txt
│   │   ├── requirements-dev.txt
│   │   └── run_server.py   # Entry point for PyInstaller
│   ├── frontend/           # Next.js frontend
│   │   ├── app/            # App router pages
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx
│   │   │   └── project/
│   │   ├── components/     # React components
│   │   │   ├── chat/
│   │   │   │   ├── ChatShell.tsx          # Unified chat shell (all modes including coordinator)
│   │   │   │   ├── CommandSurface.tsx     # Input area + routing badge + mode override
│   │   │   │   ├── ConversationView.tsx   # Message rendering + generative UI + follow-up pills
│   │   │   │   ├── RunProgressDisplay.tsx # Streaming telemetry card
│   │   │   │   ├── RunErrorDisplay.tsx    # Error taxonomy + retry
│   │   │   │   ├── RunAuditPanel.tsx      # Slide-over run history + event timeline
│   │   │   │   └── index.ts              # Re-exports
│   │   │   ├── discovery/
│   │   │   │   ├── DiscoveryChat.tsx      # Unified discovery chat (setup→Q&A→plan→execute→analyze)
│   │   │   │   ├── PlanCard.tsx           # Execution plan card (accept/reject)
│   │   │   │   ├── ToolCallCard.tsx       # Inline tool execution display
│   │   │   │   ├── ResultsSummary.tsx     # AI analysis + recommendations
│   │   │   │   └── index.ts              # Re-exports
│   │   │   ├── AppMenuBar.tsx             # Native-style application menu bar
│   │   │   ├── BioassayFeedbackForm.tsx   # Stage 7: experimental result submission
│   │   │   ├── CandidateArtifact.tsx      # Stage 4: candidate card with approve/reject
│   │   │   ├── CapabilityGapArtifact.tsx  # Capability gap display card
│   │   │   ├── ChatHistoryPanel.tsx       # Chat history side panel
│   │   │   ├── DiscoveryWorkbench.tsx     # Terminal-style execution progress UI
│   │   │   ├── DiscoveryWorkspaceTab.tsx  # Main discovery workspace (panels + session mgmt)
│   │   │   ├── DocumentTabs.tsx           # Multi-document tab navigation
│   │   │   ├── EntityHotspot.tsx          # Inline entity highlighting + hover card
│   │   │   ├── EpochNavigator.tsx         # Epoch tree browser (branching research graph)
│   │   │   ├── ExecutionPipeline.tsx      # Pipeline execution progress display
│   │   │   ├── JobsQueue.tsx              # Async jobs queue display
│   │   │   ├── MissionControl.tsx         # Stage 1 PRIME modal (domain + constraints setup)
│   │   │   ├── PluginManager.tsx          # Plugin registry viewer
│   │   │   ├── ScriptApprovalModal.tsx    # Executor script review: approve/edit/reject
│   │   │   ├── SpectroscopyArtifact.tsx   # Stage 6: NMR/XRD peak comparison
│   │   │   ├── StructureCompletionSuggestion.tsx # Structure auto-complete suggestions
│   │   │   ├── TextViewer.tsx             # .txt/.md/.csv/.json viewer
│   │   │   ├── WindowControls.tsx         # Custom window chrome controls
│   │   │   ├── WorkspaceTabs.tsx          # Main workspace tab bar
│   │   │   ├── canvas/     # Graph canvas nodes
│   │   │   └── generative/ # Generative UI components (CitationCard, ComparisonTable, MetricCard)
│   │   ├── hooks/
│   │   │   ├── useGoldenPathPipeline.ts   # Orchestrates Stage 2→3→4 (generate + screen + surface)
│   │   │   ├── usePanelResize.ts          # Panel resize drag logic
│   │   │   ├── useRunManager.ts           # Streaming orchestration (SSE → runStore)
│   │   │   └── useStructureCompletion.ts  # SMILES structure auto-complete suggestions
│   │   ├── stores/
│   │   │   ├── chatStore.ts       # Per-mode message arrays (Zustand + localStorage persist)
│   │   │   ├── discoveryStore.ts  # Discovery OS Golden Path state (Epochs, stages, artifacts)
│   │   │   ├── discoveryConversationStore.ts # Per-session chat messages (Zustand + localStorage)
│   │   │   ├── runStore.ts        # Run audit history (Zustand + IndexedDB, 500 max)
│   │   │   └── toastStore.ts      # Global notifications
│   │   ├── lib/            # Utilities & API client
│   │   │   ├── api.ts              # Backend API client (all endpoints + types)
│   │   │   ├── discovery-types.ts  # Discovery OS Golden Path type definitions
│   │   │   ├── stream-adapter.ts   # Unified SSE parser + NormalizedEvent types
│   │   │   └── truncate-payload.ts # SSE payload size limiter (prevents context blowup)
│   │   ├── public/         # Static assets
│   │   ├── package.json
│   │   └── tsconfig.json
│   └── tauri/              # Rust desktop wrapper
│       ├── src/
│       │   └── main.rs     # Tauri main entry
│       ├── Cargo.toml
│       ├── tauri.conf.json
│       └── resources/      # Bundled backend (gitignored)
├── tests/                  # Test scripts and benchmarks
├── docs/                   # Architecture plans and design docs
│   └── DiscoveryOS_ChemistryEngine_Plan.md  # EntityIR + Chemistry Engine design
├── .venv/                  # Python virtual environment
├── package.json            # Root npm scripts
├── CLAUDE.md               # Claude Code specific guidance
└── README.md               # User-facing documentation
```

---

## Build and Development Commands

### Prerequisites

- **Windows 10/11** (x64)
- **Node.js 18+**
- **Python 3.12**
- **Rust** (for Tauri modifications)

### Initial Setup

```powershell
# Run setup script (installs dependencies)
.\scripts\setup\setup_project.ps1

# Or manual setup:
cd src/frontend
npm install
cd ../backend
python -m venv ../../.venv
.\..\..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Development Commands

```powershell
# Full Tauri development (recommended)
npm run tauri:dev

# Backend only (Terminal 1)
.\scripts\dev\run_backend.ps1
# Or manually:
cd src/backend
python run_server.py

# Frontend only (Terminal 2)
cd src/frontend
npm run dev
# Opens at http://localhost:3001
```

### Production Build

```powershell
# Build complete application (backend + frontend + Tauri installer)
npm run tauri:build

# Build only backend with caching
npm run build:backend

# Force rebuild backend (ignore cache)
powershell -ExecutionPolicy Bypass -File ./scripts/build/build-backend.ps1 -Force
```

**Output**: Installers are placed in `installers/` directory as `.msi` and `.exe` files.

---

## Code Style Guidelines

### Python Backend

- **Style**: PEP 8 compliant
- **Type Hints**: Use explicit type annotations for function signatures
- **Docstrings**: Google-style docstrings for modules and functions
- **Imports**: Group as (1) stdlib, (2) third-party, (3) local app imports
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Error Handling**: Use explicit exception handling with logging

Example:
```python
"""Module description."""
from typing import Optional
import logging

from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)


def process_item(item_id: str) -> Optional[dict]:
    """Process a single item by ID.
    
    Args:
        item_id: Unique identifier for the item.
        
    Returns:
        Processed item data or None if not found.
        
    Raises:
        HTTPException: If processing fails.
    """
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Failed to process {item_id}: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")
```

### TypeScript Frontend

- **Style**: ESLint + Prettier (configured in package.json)
- **Components**: Functional components with explicit return types
- **Props**: Interface definitions for all component props
- **State**: Use Zustand for global state, useState for local state
- **Naming**: `camelCase` for variables/functions, `PascalCase` for components/interfaces

Example:
```typescript
interface ChatMessageProps {
  message: Message;
  onDelete?: (id: string) => void;
}

export function ChatMessage({ message, onDelete }: ChatMessageProps): JSX.Element {
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    // Component JSX
  );
}
```

### Frontend Design Guidelines

When writing frontend code, reject "AI slop" aesthetics:

1. **Design Direction**: Commit to a visual direction (retro-futuristic, editorial, brutalist, luxury)
2. **Typography**: Use distinctive font pairings (display + body), avoid generic fonts
3. **Layout**: Use asymmetry, intentional overlapping, generous negative space
4. **Motion**: CSS-only animations for hover/load states, staggered reveals
5. **Color**: Use CSS variables for themes, add depth via textures/gradients

---

## Testing Strategy

### Testing Philosophy

This project uses **Aider-First TDD**:
1. Claude scaffolds the feature
2. Aider (DeepSeek R1 + MiniMax) writes and verifies tests

### Test Commands

```powershell
# Backend tests with pytest
cd src/backend
pytest

# Frontend linting
cd src/frontend
npm run lint

# Type checking
npx tsc --noEmit
```

### Manual Verification

- **UI**: Upload test PDFs, verify entity extraction in `nodes` table
- **Swarm**: Test `DEEP_DISCOVERY` queries in Chat Interface
- **Database**: Inspect SQLite with `sqlite3 atlas.db`
- **Build**: Verify installer output in `installers/` directory

---

## Database Schema

### SQLite Tables (SQLAlchemy Models in `src/backend/app/core/database.py`)

```
Project                    # Research project scope
├── id (PK)
├── name
├── description
└── created_at/updated_at

Document                   # Uploaded files
├── id (PK)
├── filename
├── file_hash (indexed)
├── project_id (FK)
└── upload_date

DocumentChunk              # Text chunks for retrieval
├── id (PK)
├── document_id (FK)
├── chunk_index
├── text_content
└── embedding

Node                       # Knowledge graph entities
├── id (PK)
├── label (indexed)        # e.g., "chemical", "concept"
├── properties (JSON)
├── document_id (FK)
└── project_id (FK)

Edge                       # Knowledge graph relationships
├── id (PK)
├── source_id (FK)
├── target_id (FK)
├── edge_type
└── properties (JSON)
```

**Indexes**: `idx_nodes_label`, `idx_edges_source_target`, `idx_documents_file_hash`

**Foreign Keys**: Enabled (`PRAGMA foreign_keys=ON`)

---

## Agent Architecture

### Multi-Agent Swarm (LangGraph)

```
User Query
    ↓
Meta-Router (Intent Classification)
    ├─> Simple Query → Librarian Agent (Fast Vector Search)
    ├─> Deep Discovery → Navigator 2.0 (Graph walk + reflection)
    └─> Broad Research → Cortex MoE (Mixture of Experts)
           ↳ Supervisor → Hypothesis/Retrieval/Writer Experts
           ↳ Grounding Auditor → Validates citations
```

### Key Agent Files

| Agent | File | Purpose |
|-------|------|---------|
| Meta-Router | `services/agents/meta_router.py` | Classifies query intent |
| Librarian | `services/agents/librarian.py` | Direct vector search |
| Navigator | `services/agents/discovery_graph.py` | Deep graph exploration |
| Supervisor | `services/agents/supervisor.py` | Coordinates MoE |
| Experts | `services/agents/experts/` | Specialized sub-agents |
| Coordinator | `services/agents/coordinator.py` | HITL session bootstrapping |
| Executor | `services/agents/executor.py` | Script-generation sandbox |
| Pipeline Planner | `services/agents/pipeline_planner.py` | Deterministic plugin pipeline orchestrator |
| Discovery Chat | `services/agents/discovery_chat.py` | Unified chat orchestrator (coordinator + Q&A + plan + execute + analyze) |

---

## Discovery OS Architecture

Discovery OS is a **scientific research automation system**, not a chatbot. It operates via a 3-phase pipeline that gives researchers full transparency and control over autonomous chemistry/science workflows.

### Philosophy

> The agent writes **Python scripts**, not text answers. Deterministic tools (RDKit, pandas, NumPy) replace LLM hallucination. Every artifact is readable and editable by the researcher.

### Phase Flow

```
1. MissionControl (Frontend)
   └─> POST /api/discovery/initialize
       Creates SQLite row + disk folder: data/discovery/{session_id}/

2. Coordinator Agent (HITL — up to 5 turns)
   └─> POST /api/discovery/{session_id}/coordinator/chat  (SSE)
       - Scans corpus via RetrievalService
       - Reads .md files from session folder (CONSTRAINTS.md, HYPOTHESES.md, etc.)
       - LLM (DeepSeek) identifies missing goals via interrupt() + user Q&A
       - Writes SESSION_CONTEXT.md + session_memory.json on completion

3. Executor Agent (Script Sandbox Loop)
   └─> POST /api/discovery/{session_id}/executor/start  (SSE)
       - Reads session_memory.json for goals
       - plan_task → generate_script → [human approval] → execute_script → loop
       - Each iteration appends results to FINDINGS.md
       - Loops until all goals satisfied or max_iterations reached

4. Pipeline Planner (Deterministic Plugin Pipeline)
   └─> POST /api/discovery/{session_id}/pipeline/run  (SSE)
       - Rule-based plugin selection (no LLM) based on session goals + constraints
       - Canonical order: standardize → predict → toxicity → SA score → ADMET → strategy
       - Batches single-SMILES plugins, streams stage progress
       - Saves pipeline_results.json + updates FINDINGS.md
```

### LLM Role Split (Critical)

| Role | Model | Purpose |
|------|-------|---------|
| Orchestration | **DeepSeek** (`orchestrate_constrained`) | Coordinator reasoning, goal extraction, planning |
| Tool Calls | **MiniMax** (`generate_constrained`) | Script generation, structured JSON output |

**Rule**: MiniMax must NEVER be used for coordinator-level reasoning — it returns arrays instead of objects. Always use `orchestrate_constrained` (DeepSeek) for the Coordinator and `generate_constrained` (MiniMax) for Executor planning/scripting.

### Living .md Knowledge Substrate

Session folder `data/discovery/{session_id}/` is the agent's working memory. Knowledge accumulates in files across iterations:

| File | Written by | Purpose |
|------|-----------|---------|
| `SESSION_CONTEXT.md` | Coordinator | Perpetual living context — goals, domain, corpus summary |
| `FINDINGS.md` | Executor | Auto-appended after every successful script run |
| `CONSTRAINTS.md` | User (optional) | Domain constraints — agents auto-read before each run |
| `HYPOTHESES.md` | User or agent | Guides iteration direction |
| `RESEARCH_NOTES.md` | User (optional) | Background knowledge |
| `session_memory.json` | Coordinator | Machine-readable state for multi-agent coordination |
| `generated/` | Executor | Scripts (`.py`), results (`.csv`), logs (`execution_log.txt`), plots (`.png`) |

Both `_read_session_notes()` and `_read_key_artifacts()` are injected into every agent prompt at the start of each run — this is how the "living knowledge substrate" works.

### Executor Script Loop (Detail)

```
plan_task (MiniMax)
    ├─ Reads: session_memory.json, all .md files, execution_log.txt tail, latest CSV
    └─> generate_script (MiniMax)
            ├─ Writes Python script to generated/{filename}.py
            └─> [await_approval] interrupt() — user approves / edits / rejects
                    └─> execute_script — subprocess.run() with 5-min timeout
                            ├─ Appends to execution_log.txt
                            ├─ Appends to FINDINGS.md
                            └─> plan_task (loop) OR END
```

Resume commands passed to `Command(resume=...)`:
- `"approve"` — execute as-is
- `"reject"` — stop session
- `"edit:<new_code>"` — replace script code then execute

### Coordinator Agent (Detail)

LangGraph HITL using `interrupt()` + `Command(resume=user_answer)`:

```
scan_corpus
    ├─ Queries RetrievalService (vector + graph)
    ├─ Reads all .md files from session folder
    └─> analyze_and_ask (loop, max 5 turns)
            ├─ DeepSeek extracts goals, identifies missing context
            ├─ interrupt() → surfaces question + options to frontend
            ├─ On resume: merges new goals, checks 5 required criteria:
            │     1. Research domain
            │     2. Primary objective / biological target
            │     3. Property constraints (MW, LogP, TPSA)
            │     4. Forbidden substructures / exclusion criteria
            │     5. Success criteria and evaluation metrics
            └─> END → _finalize_coordinator() writes session_memory.json + SESSION_CONTEXT.md
```

Thread IDs avoid collision: `f"coordinator-{session_id}"` and `f"executor-{session_id}"`.

### Plugin System (`services/plugins/`)

Chemistry plugins are deterministic (no LLM) and follow a `BasePlugin` ABC defined in `__init__.py`. The `PluginManager` singleton auto-discovers and registers plugins at startup.

| Plugin name | File | Purpose |
|-------------|------|---------|
| `standardize` | `standardize.py` | SMILES canonicalization, InChIKey generation, deduplication (RDKit) |
| `predict_properties` | `properties.py` | RDKit molecular properties: MW, LogP, TPSA, HBD, HBA, RotBonds |
| `check_toxicity` | `toxicity.py` | SMARTS/PAINS structural alert screening (curated Brenk set + RDKit FilterCatalog) |
| `sa_scorer` | `sa_scorer.py` | Synthetic accessibility scoring (SA 1-10, feasibility threshold ≤ 6.0) |
| `admet_predict` | `admet_predict.py` | ADMET prediction (hERG, DILI, Caco2, CYP3A4) — `admet_ai` or mock heuristics |
| `plan_synthesis` | `retrosynthesis.py` | AiZynthFinder ONNX retrosynthesis; falls back to heuristic if model absent |
| `verify_spectrum` | `spectrum.py` | NMR .jdx analysis via nmrglue + scipy peak detection; compares to RDKit hydrogen count |
| `evaluate_strategy` | `strategy.py` | Scores retrosynthesis routes by step count, complexity, and estimated viability |
| `search_literature` | `__init__.py` | Bridges to existing hybrid RAG retrieval service |

All plugins extend `BasePlugin` (defined in `base.py`) and gracefully degrade when optional dependencies (RDKit, `admet_ai`, nmrglue, AiZynthFinder) are unavailable.

### Pipeline Planner (`agents/pipeline_planner.py`)

Deterministic, rule-based orchestrator that replaces LLM-based script generation for standard chemistry workflows. Selects and orders plugins based on session goals and constraints — no LLM involved in pipeline construction.

Canonical plugin order:
```
standardize → predict_properties → check_toxicity → sa_scorer → admet_predict → strategy
```

SSE events emitted during pipeline execution:
- `pipeline_planning` — selected stages + molecule count
- `pipeline_stage_start` / `pipeline_stage_complete` — per-stage progress
- `pipeline_complete` — final summary + artifact list

### Discovery LLM Service (`discovery_llm.py`)

`DiscoveryLLMService` is **completely isolated** from the global `LLMService` used for chat/retrieval. It maintains its own API clients and configuration so global model changes don't affect active discovery sessions.

```python
# Two entry points:
llm_service.orchestrate_constrained(prompt, schema)   # → DeepSeek (coordinator)
llm_service.generate_constrained(prompt, schema)       # → MiniMax via LiteLLM (executor)
```

Config keys (in `config/.env`):
```bash
DEEPSEEK_API_KEY=xxx        # Required — coordinator reasoning
MINIMAX_API_KEY=xxx         # Required — executor script generation
DISCOVERY_ORCHESTRATION_MODEL=deepseek-reasoner
DISCOVERY_TOOL_MODEL=MiniMax-M2.5
```

### Discovery Session Service (`discovery_session.py`)

| Class | Purpose |
|-------|---------|
| `DiscoverySessionService` | CRUD for sessions — initialize, list, get files, update goals |
| `SessionMemoryService` | Read/write `session_memory.json` + `SESSION_CONTEXT.md` |
| `SessionMemoryData` | Pydantic model — shared state for multi-agent coordination |

### Candidate Generation Service (`candidate_generation.py`)

Two async SSE generators that cover the Stage 2→3 pipeline:

- **`generate_candidates(session_id, mock=False)`** — queries corpus via RAG + LLM to propose SMILES strings. `mock=True` returns a fixed set of known molecules (aspirin, ibuprofen, etc.) for frontend testing without LLM tokens.
- **`screen_candidates(candidates, constraints, mock=False)`** — deterministic RDKit property calculations (MW, LogP, TPSA, HBD, HBA, QED) filtered against session constraints. Emits per-candidate SSE events with pass/fail per property.

### Domain Tools (`domain_tools.py`)

Stateless helpers for domain-agnostic rendering and persistence:
- **Molecular / entity rendering**: RDKit SVG generation with fallback for non-chemistry domains
- **Capability gap storage**: Creates a `CapabilityGap` SQLAlchemy node in the session folder so gaps survive restarts and are surfaced in the coordinator's next scan

### Stage Context Bridge (`stage_context.py`)

Uses `contextvars.ContextVar` to inject the current epoch/artifact state into every `generate_chat()` call without threading issues. The frontend bundles `stage_context` in the request body; the route handler calls `set_stage_preamble(stage_context)` which is then transparently prepended as a system message for that request only. No cross-request leakage.

Stage labels (1=PRIME, 2=GENERATE, 3=SCREEN, 4=SURFACE, 5=SYNTHESIS_PLAN, 6=SPECTROSCOPY, 7=FEEDBACK) are injected so the LLM always knows where in the Golden Path it is.

### Synthesis Memory (`synthesis_memory.py`)

Creates the "compound moat" — every experiment cycle deposits long-lived memory that accumulates across sessions:

1. Writes a `SynthesisAttempt` node + `ATTEMPTED_VIA` edge to the SQLite knowledge graph (linked to the molecule node via InChIKey)
2. Embeds a human-readable summary into Qdrant so future `search_literature` calls automatically surface prior experiments for structurally similar molecules
3. Provides a Tanimoto-similarity query (`find_similar_attempts`) to surface past work on related scaffolds

All DB operations run in `asyncio.run_in_executor()` to avoid blocking the event loop.

### Discovery API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/discovery/schema` | Get domain schemas (property constraints, operators) |
| `POST /api/discovery/initialize` | Create session (SQLite row + disk folder) |
| `GET /api/discovery/sessions` | List all sessions |
| `GET /api/discovery/{id}/files` | List session files (root + generated/) |
| `GET /api/discovery/{id}/files/{path}` | Serve individual session file |
| `GET /api/discovery/{id}/memory` | Get session memory (session_memory.json) |
| `POST /api/discovery/{id}/chat` | SSE — **unified chat** (coordinator + Q&A + plan + execute + analyze) |
| `POST /api/discovery/{id}/coordinator/chat` | SSE — coordinator HITL chat (legacy, still functional) |
| `POST /api/discovery/{id}/executor/start` | SSE — start/resume executor script sandbox (legacy) |
| `POST /api/discovery/{id}/pipeline/run` | SSE — deterministic plugin pipeline execution (legacy) |
| `POST /api/discovery/parse-brainstorm` | Parse free-text brainstorm into structured goals |
| `POST /api/discovery/capability-gap` | Log a capability gap to the session |
| `POST /api/discovery/generate-candidates` | SSE — LLM+RAG candidate generation (Stage 2) |
| `POST /api/discovery/screen` | SSE — RDKit property screening + constraint filtering (Stage 3) |
| `POST /api/discovery/feedback` | Submit bioassay results → knowledge graph node (Stage 7) |
| `POST /api/discovery/stream` | SSE — ReAct tool-calling loop (legacy discovery graph) |
| `POST /api/discovery/run` | Non-streaming discovery (legacy) |
| `POST /api/discovery/upload-spectrum` | Upload .jdx NMR spectrum file |
| `GET /api/discovery/plugins` | List all registered plugins with metadata |
| `POST /api/discovery/plugins/{name}/unload` | Unload a plugin from the registry |

### Security Notes

- Script filenames are sanitized with `os.path.basename()` and `_is_path_relative_to()` checks to prevent directory traversal before any file write or execution.
- Scripts are AST-validated before execution to catch syntax errors early.
- Scripts execute via `subprocess.run()` with a configurable timeout (`DISCOVERY_SCRIPT_TIMEOUT`, default 300s) and run in the `generated/` folder (not the backend root).

---

## Configuration

### Environment Variables (`config/.env`)

```bash
# Required API Keys for Aider
DEEPSEEK_API_KEY=xxx
MINIMAX_API_KEY=xxx

# Optional: Cloud LLM APIs
OPENAI_API_KEY=xxx
ANTHROPIC_API_KEY=xxx

# Backend Configuration (dev overrides)
API_PORT=8000
DATABASE_URL=sqlite:///./atlas.db
```

### Application Settings (`src/backend/app/core/config.py`)

Key configuration options:
- `ENABLE_NAVIGATOR_REFLECTION`: Multi-turn reasoning for Navigator
- `MAX_REFLECTION_ITERATIONS`: Cap on reflection loops (default: 3)
- `ENABLE_RERANKING`: FlashRank reranking for RAG
- `USE_DOCLING`: VLM-based document parsing
- `LLM_CONTEXT_SIZE`: Token context window (default: 8192)
- `DISCOVERY_SCRIPT_TIMEOUT`: Seconds before executor script subprocess is killed (default: 300)
- `DISCOVERY_ORCHESTRATION_PROVIDER` / `DISCOVERY_ORCHESTRATION_MODEL`: DeepSeek config for coordinator
- `DISCOVERY_TOOL_PROVIDER` / `DISCOVERY_TOOL_MODEL`: MiniMax config for executor

---

## Security Considerations

### Data Privacy

- **Local-First**: All data stored locally in user's AppData
- **No Cloud Dependencies**: Models run locally via llama.cpp
- **Optional API Keys**: Cloud LLMs only used if keys provided

### CSP (Content Security Policy)

Configured in `src/tauri/tauri.conf.json`:
```json
"csp": "default-src 'self'; connect-src 'self' http://127.0.0.1:8000 ..."
```

### CORS

Backend only accepts connections from:
- `tauri://localhost`
- `https://tauri.localhost`
- `http://localhost:3000`
- `http://localhost:3001`

---

## Common Development Workflows

### Adding a New API Endpoint

1. Add Pydantic models in `src/backend/app/api/routes.py`
2. Implement route handler with service layer call
3. Update frontend API client in `src/frontend/lib/api.ts`
4. Test with Aider: `aider tests/backend/test_new_endpoint.py`

### Modifying Database Schema

1. Update models in `src/backend/app/core/database.py`
2. Add indexes for frequently queried fields
3. **Important**: SQLAlchemy doesn't auto-migrate
4. Dev: Stop app, delete `atlas.db`, restart (auto-recreates)
5. Production: Manual migration scripts required

### Adding a New Agent

1. Create agent file in `src/backend/app/services/agents/`
2. Register in `routes.py` service initialization
3. Connect to Meta-Router for query routing
4. Generate tests via Aider

### Working with Knowledge Graphs

- **Node creation**: `src/backend/app/services/ingest.py` (GLiNER extraction)
- **Graph queries**: `src/backend/app/services/graph.py` (Rustworkx)
- **Performance**: Use `joinedload()` to prevent N+1 queries

---

## Troubleshooting

### Port Already in Use

```powershell
# Find process
netstat -ano | findstr :8000

# Kill process
taskkill /PID <PID> /F
```

### Backend Won't Start

1. Check Python venv is activated
2. Verify `MODELS_DIR` env var is set
3. Check logs in terminal for import errors
4. Ensure port 8000 is free

### Frontend Won't Connect

1. Verify backend is running on port 8000
2. Check CORS settings in `app/main.py`
3. Verify CSP in `tauri.conf.json` allows localhost:8000
4. Check browser console for errors

### Tauri Black Screen

1. Ensure frontend dev server started on port 3001
2. Check `beforeDevCommand` in `tauri.conf.json`
3. Verify `devPath` points to correct URL

---

## Key File Reference

### Backend Critical Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI initialization, startup/shutdown |
| `app/api/routes.py` | All API endpoints, service initialization |
| `app/core/config.py` | Settings, environment variables |
| `app/core/database.py` | SQLAlchemy models, SQLite setup |
| `app/services/llm.py` | LLM service with llama.cpp (chat/RAG) |
| `app/services/retrieval.py` | Hybrid RAG (vector + graph + text) |
| `app/services/swarm.py` | Two-brain swarm orchestration |
| `app/services/discovery_llm.py` | Isolated LLM service for Discovery OS (DeepSeek + MiniMax) |
| `app/services/discovery_session.py` | Session CRUD, SessionMemoryService, living .md substrate |
| `app/services/candidate_generation.py` | LLM+RAG candidate generation + RDKit screening SSE generators |
| `app/services/domain_tools.py` | Molecular/entity rendering + capability gap storage |
| `app/services/stage_context.py` | contextvars bridge — epoch state injection into LLM prompts |
| `app/services/synthesis_memory.py` | SynthesisAttempt graph node + Qdrant embedding (compound moat) |
| `app/services/agents/discovery_chat.py` | **Unified chat orchestrator** (coordinator + Q&A + plan + execute + analyze) |
| `app/services/agents/coordinator.py` | Coordinator HITL graph (LangGraph + interrupt()) |
| `app/services/agents/executor.py` | Executor script-generation sandbox (LangGraph) |
| `app/services/agents/pipeline_planner.py` | Deterministic plugin pipeline orchestrator |
| `app/services/plugins/base.py` | BasePlugin abstract class definition |
| `app/services/plugins/__init__.py` | PluginManager singleton + search_literature built-in |
| `run_server.py` | PyInstaller entry point |

### Frontend Critical Files

| File | Purpose |
|------|---------|
| `app/page.tsx` | Projects dashboard |
| `app/project/workspace-page.tsx` | Main 3-pane workspace (Library \| Stage \| Context) |
| `components/chat/ChatShell.tsx` | Unified chat shell (all modes including coordinator) |
| `components/chat/ConversationView.tsx` | Message rendering + generative UI + follow-up pills |
| `components/chat/RunProgressDisplay.tsx` | Streaming telemetry display |
| `components/chat/RunAuditPanel.tsx` | Run history slide-over drawer |
| `components/MissionControl.tsx` | Stage 1 PRIME modal — domain + property constraints setup |
| `components/DiscoveryWorkspaceTab.tsx` | Discovery workspace — chat-first layout (60/40 split) |
| `components/discovery/DiscoveryChat.tsx` | Unified discovery chat (setup, Q&A, plan, tools, analysis) |
| `components/discovery/PlanCard.tsx` | Execution plan display with accept/reject |
| `components/discovery/ToolCallCard.tsx` | Inline tool execution display |
| `components/discovery/ResultsSummary.tsx` | AI analysis + recommendation cards |
| `components/DiscoveryWorkbench.tsx` | Terminal-style execution progress display (legacy) |
| `components/ExecutionPipeline.tsx` | Pipeline stage progress display |
| `components/EpochNavigator.tsx` | Epoch tree browser (non-linear research branching) |
| `components/ScriptApprovalModal.tsx` | Executor script approve / edit / reject UI |
| `components/CandidateArtifact.tsx` | Stage 4: candidate card with approve/reject |
| `components/SpectroscopyArtifact.tsx` | Stage 6: NMR/XRD peak comparison display |
| `components/BioassayFeedbackForm.tsx` | Stage 7: experimental result submission |
| `components/PluginManager.tsx` | Plugin registry viewer |
| `hooks/useRunManager.ts` | Streaming orchestration (SSE → runStore) |
| `hooks/useGoldenPathPipeline.ts` | Stage 2→3→4 pipeline orchestration hook |
| `stores/discoveryStore.ts` | Discovery OS state (Epochs, stages, artifacts) |
| `stores/discoveryConversationStore.ts` | Per-session chat messages (Zustand + localStorage) |
| `stores/runStore.ts` | Run audit history (Zustand + IndexedDB) |
| `lib/api.ts` | Backend API client (all endpoints + types) |
| `lib/discovery-types.ts` | Discovery OS Golden Path type definitions |
| `lib/stream-adapter.ts` | Unified SSE parser, NormalizedEvent types |
| `lib/truncate-payload.ts` | SSE payload size limiter |

### Build & Config Critical Files

| File | Purpose |
|------|---------|
| `scripts/build/build-backend.ps1` | PyInstaller build with caching |
| `src/tauri/tauri.conf.json` | Tauri window, security, bundle config |
| `src/tauri/Cargo.toml` | Rust dependencies |
| `src/tauri/src/main.rs` | Tauri main entry, sidecar management |

---

## Agent Ecosystem (Development Workflow)

This project operates within a 3-tier AI agent hierarchy:

### Level 1: Antigravity (The Planner)
- **Role**: Project Manager & Orchestrator
- **Responsibility**: Breaks complex goals into step-by-step Runbooks
- **Interaction**: Receives plans, executes one step at a time

### Level 2: Claude (The Architect)
- **Role**: Lead Engineer & System Architect (YOU)
- **Responsibility**: Converts plans into concrete code and structures
- **Interaction**: Executes plans and delegates to Aider

### Level 3: Aider (The Builder/QA)
- **Role**: Junior Dev, QA, Debugger
- **Stack**: DeepSeek R1 (Planner) + MiniMax 2.5 (Coder)
- **Responsibility**: TDD loops, linting fixes, test writing
- **Command Format**: `aider <file> tests/<test_file> --message "Use R1 to plan: <Goal>. Then MiniMax to: <Task>."`

---

## License

MIT License - See `LICENSE` file for details.

---

**Last Updated**: March 2026
**Version**: 2.3.0
**Status**: Discovery OS Active — Full Golden Path (Stage 1–7) + Synthesis Memory + Stage Context Bridge
