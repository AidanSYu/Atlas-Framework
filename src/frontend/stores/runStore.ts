/**
 * Run Store — IndexedDB-backed persistence for run audit history.
 *
 * Each "run" is a complete lifecycle record of a single query execution,
 * including the full normalized event stream, tool invocations, and
 * final result or error.
 *
 * Retention: last 500 runs per project, auto-pruned on write.
 */
import { create } from 'zustand';
import type { NormalizedEvent, FailureCategory } from '@/lib/stream-adapter';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type RunStatus =
  | 'queued'
  | 'routing'
  | 'awaiting_override'
  | 'running'
  | 'awaiting_input'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface ToolInvocation {
  tool: string;
  input: Record<string, any>;
  output: Record<string, any> | null;
  startedAt: number;
  completedAt: number | null;
  status: 'running' | 'completed' | 'failed';
}

export interface Run {
  id: string;
  mode: 'librarian' | 'cortex' | 'moe';
  intent: string;
  query: string;
  projectId: string;
  status: RunStatus;
  startedAt: number;
  completedAt: number | null;
  events: NormalizedEvent[];
  toolInvocations: ToolInvocation[];
  result: any | null;
  error: { message: string; category: FailureCategory } | null;
}

// ---------------------------------------------------------------------------
// Valid state transitions
// ---------------------------------------------------------------------------

const VALID_TRANSITIONS: Record<RunStatus, RunStatus[]> = {
  queued: ['routing', 'cancelled'],
  routing: ['awaiting_override', 'running', 'failed', 'cancelled'],
  awaiting_override: ['routing', 'running', 'cancelled'],
  running: ['awaiting_input', 'completed', 'failed', 'cancelled'],
  awaiting_input: ['running', 'cancelled'],
  completed: [],
  failed: ['queued'],
  cancelled: ['queued'],
};

export function canTransition(from: RunStatus, to: RunStatus): boolean {
  return VALID_TRANSITIONS[from]?.includes(to) ?? false;
}

// ---------------------------------------------------------------------------
// IndexedDB helpers
// ---------------------------------------------------------------------------

const DB_NAME = 'atlas-runs';
const DB_VERSION = 1;
const STORE_NAME = 'runs';
const MAX_RUNS_PER_PROJECT = 500;

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        store.createIndex('projectId', 'projectId', { unique: false });
        store.createIndex('startedAt', 'startedAt', { unique: false });
        store.createIndex('projectId_startedAt', ['projectId', 'startedAt'], { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function persistRun(run: Run): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    store.put(run);
    await txComplete(tx);
    db.close();
  } catch (err) {
    console.error('Failed to persist run:', err);
  }
}

async function loadRunsForProject(projectId: string): Promise<Run[]> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('projectId');
    const request = index.getAll(projectId);
    const runs = await reqResult<Run[]>(request);
    db.close();
    return runs.sort((a, b) => b.startedAt - a.startedAt);
  } catch (err) {
    console.error('Failed to load runs:', err);
    return [];
  }
}

async function pruneOldRuns(projectId: string): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('projectId_startedAt');
    const range = IDBKeyRange.bound([projectId, 0], [projectId, Infinity]);
    const request = index.getAllKeys(range);
    const keys = await reqResult<IDBValidKey[]>(request);

    if (keys.length > MAX_RUNS_PER_PROJECT) {
      const toDelete = keys.slice(0, keys.length - MAX_RUNS_PER_PROJECT);
      for (const key of toDelete) {
        store.delete(key);
      }
    }
    await txComplete(tx);
    db.close();
  } catch (err) {
    console.error('Failed to prune runs:', err);
  }
}

async function clearRunsForProject(projectId: string): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const index = store.index('projectId');
    const request = index.getAllKeys(projectId);
    const keys = await reqResult<IDBValidKey[]>(request);
    for (const key of keys) {
      store.delete(key);
    }
    await txComplete(tx);
    db.close();
  } catch (err) {
    console.error('Failed to clear runs:', err);
  }
}

function reqResult<T>(request: IDBRequest): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result as T);
    request.onerror = () => reject(request.error);
  });
}

function txComplete(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ---------------------------------------------------------------------------
// Zustand store
// ---------------------------------------------------------------------------

interface RunStoreState {
  currentRun: Run | null;
  runHistory: Run[];
  isLoadingHistory: boolean;

  createRun: (params: { query: string; mode: Run['mode']; intent: string; projectId: string }) => Run;
  updateRunStatus: (runId: string, status: RunStatus) => void;
  appendEvent: (runId: string, event: NormalizedEvent) => void;
  addToolInvocation: (runId: string, invocation: ToolInvocation) => void;
  updateToolInvocation: (runId: string, tool: string, update: Partial<ToolInvocation>) => void;
  completeRun: (runId: string, result: any) => void;
  failRun: (runId: string, message: string, category: FailureCategory) => void;
  cancelRun: (runId: string) => void;

  loadHistory: (projectId: string) => Promise<void>;
  clearHistory: (projectId: string) => Promise<void>;
  persistCurrentRun: () => Promise<void>;
}

export const useRunStore = create<RunStoreState>()((set, get) => ({
  currentRun: null,
  runHistory: [],
  isLoadingHistory: false,

  createRun: ({ query, mode, intent, projectId }) => {
    const run: Run = {
      id: crypto.randomUUID(),
      mode,
      intent,
      query,
      projectId,
      status: 'queued',
      startedAt: Date.now(),
      completedAt: null,
      events: [],
      toolInvocations: [],
      result: null,
      error: null,
    };
    set({ currentRun: run });
    return run;
  },

  updateRunStatus: (runId, status) => {
    set((state) => {
      if (!state.currentRun || state.currentRun.id !== runId) return state;
      if (!canTransition(state.currentRun.status, status)) {
        console.warn(`Invalid run transition: ${state.currentRun.status} -> ${status}`);
        return state;
      }
      return { currentRun: { ...state.currentRun, status } };
    });
  },

  appendEvent: (runId, event) => {
    set((state) => {
      if (!state.currentRun || state.currentRun.id !== runId) return state;
      return {
        currentRun: {
          ...state.currentRun,
          events: [...state.currentRun.events, event],
        },
      };
    });
  },

  addToolInvocation: (runId, invocation) => {
    set((state) => {
      if (!state.currentRun || state.currentRun.id !== runId) return state;
      return {
        currentRun: {
          ...state.currentRun,
          toolInvocations: [...state.currentRun.toolInvocations, invocation],
        },
      };
    });
  },

  updateToolInvocation: (runId, tool, update) => {
    set((state) => {
      if (!state.currentRun || state.currentRun.id !== runId) return state;
      const invocations = [...state.currentRun.toolInvocations];
      const idx = invocations.findLastIndex((t) => t.tool === tool && t.status === 'running');
      if (idx >= 0) {
        invocations[idx] = { ...invocations[idx], ...update };
      }
      return { currentRun: { ...state.currentRun, toolInvocations: invocations } };
    });
  },

  completeRun: (runId, result) => {
    set((state) => {
      if (!state.currentRun || state.currentRun.id !== runId) return state;
      const completed: Run = {
        ...state.currentRun,
        status: 'completed',
        completedAt: Date.now(),
        result,
      };
      return {
        currentRun: completed,
        runHistory: [completed, ...state.runHistory],
      };
    });
    get().persistCurrentRun();
  },

  failRun: (runId, message, category) => {
    set((state) => {
      if (!state.currentRun || state.currentRun.id !== runId) return state;
      const failed: Run = {
        ...state.currentRun,
        status: 'failed',
        completedAt: Date.now(),
        error: { message, category },
      };
      return {
        currentRun: failed,
        runHistory: [failed, ...state.runHistory],
      };
    });
    get().persistCurrentRun();
  },

  cancelRun: (runId) => {
    set((state) => {
      if (!state.currentRun || state.currentRun.id !== runId) return state;
      const cancelled: Run = {
        ...state.currentRun,
        status: 'cancelled',
        completedAt: Date.now(),
      };
      return {
        currentRun: cancelled,
        runHistory: [cancelled, ...state.runHistory],
      };
    });
    get().persistCurrentRun();
  },

  loadHistory: async (projectId) => {
    set({ isLoadingHistory: true });
    const runs = await loadRunsForProject(projectId);
    set({ runHistory: runs, isLoadingHistory: false });
  },

  clearHistory: async (projectId) => {
    await clearRunsForProject(projectId);
    set({ runHistory: [] });
  },

  persistCurrentRun: async () => {
    const run = get().currentRun;
    if (!run) return;
    await persistRun(run);
    await pruneOldRuns(run.projectId);
  },
}));
