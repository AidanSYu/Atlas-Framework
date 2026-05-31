# CLAUDE.md — Atlas Framework guidance

This file guides Claude Code when working within the Atlas repo. It pairs general engineering discipline (**Working principles**, below) with Atlas-specific architecture and operations: the system orbits a single local Orchestrator and an always-on knowledge substrate.

## Working principles

Behavioral guidelines to reduce common LLM coding mistakes. These apply alongside the Atlas-specific guidance that follows.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Atlas Framework essentials

- **Single Orchestrator**: All reasoning happens inside `llama-cpp-python` running `nvidia_Orchestrator-8B-IQ2_M.gguf`. The loop ingests the user prompt, manifest catalog, and hybrid RAG hits, emits JSON tool calls, and appends observations until final synthesis.
- **Always-on substrate**: SQLite + Qdrant + Rustworkx persistently power retrieval. These components are treated like system hardware (non-optional). They remain active for every request and are surfaced through `CoreToolRegistry` handlers, not the plugin directory.
- **Universal Plugin Protocol**: Optional capabilities live under `src/backend/plugins/`, each with `manifest.json` plus `wrapper.py`. The Orchestrator merges plugin schemas with core tools so it can call any tool uniformly. Adding a plugin should not require touching the Orchestrator loop.
- **Offline-first**: No external agents, no cloud dependencies; everything runs locally. Plugin shells must avoid remote calls unless explicitly allowed by the user.

## Workflow reminders

1. Read `src/backend/app/atlas_plugin_system/` to understand how the orchestrator loads schemas and routes JSON calls.
2. When you add features, decide if they belong in:
   - `CoreToolRegistry` (if foundational retrieval/graph behavior), or
   - `src/backend/plugins/` (if optional and implements the manifest/wrapper contract).
3. Keep documentation consistent: describe the orchestrator + plugin story, mention the always-on substrate, and highlight `nvidia_Orchestrator-8B-IQ2_M.gguf`.
4. Avoid references to LangGraph, agents, or swarms; they have been purged and are no longer part of the architecture.

## Run commands

```powershell
cd src/backend
python run_server.py
```

```powershell
cd src/frontend
npm run dev
```

```powershell
npm run tauri:dev
```

## Model setup

Ensure `.env` or `config/.env` points to:

```env
MODELS_DIR=C:/path/to/models
QDRANT_STORAGE_PATH=C:/path/to/qdrant_storage
ATLAS_PLUGIN_DIR=C:/path/to/ContAInuumAtlas/src/backend/plugins
DATABASE_PATH=C:/path/to/atlas.db
```

Place `nvidia_Orchestrator-8B-IQ2_M.gguf` in `MODELS_DIR` along with `nomic-embed-text-v1.5` and `gliner_small-v2.1`. The orchestrator will pick it up automatically.

## Testing and delegation

- After you implement a change, validate via `python run_server.py` and ping `/api/framework/run`.
- For UI work, run `npm run dev` inside `src/frontend` and confirm the interface mentions the Atlas Framework story.
- Delegate repetitive tasks using Aider commands when appropriate.
