/**
 * useRunManager — Orchestration hook for the run lifecycle.
 *
 * Manages the full state machine for a single query execution: routing,
 * streaming, tool tracking, completion, failure, cancellation, and retry.
 *
 * Phase 3: Now owns streaming dispatch via streamSSE. ChatShell delegates
 * all query execution here.
 */
import { useCallback, useRef } from 'react';
import { useRunStore, type Run, type RunStatus, type ToolInvocation } from '@/stores/runStore';
import { useDiscoveryStore } from '@/stores/discoveryStore';
import { streamSSE, type NormalizedEvent, type FailureCategory } from '@/lib/stream-adapter';
import { api, getApiBase } from '@/lib/api';
import type {
  StageContextBundle,
  StageArtifactSummary,
  TruncatedToolInvocation,
  GoldenPathStage,
  CandidateArtifact,
} from '@/lib/discovery-types';

export type ChatMode = 'librarian' | 'cortex' | 'moe';

const INTENT_TO_MODE: Record<string, ChatMode> = {
  SIMPLE: 'librarian',
  DEEP_DISCOVERY: 'cortex',
  BROAD_RESEARCH: 'cortex',
  MULTI_STEP: 'moe',
};

export function intentToMode(intent: string): ChatMode {
  return INTENT_TO_MODE[intent] || 'cortex';
}

// ---------------------------------------------------------------------------
// Cross-Store Bridge: getStageContext()
// ---------------------------------------------------------------------------

const MAX_PREVIEW_CHARS = 500;

function truncateJson(value: any): string {
  try {
    const str = typeof value === 'string' ? value : JSON.stringify(value);
    return str.length > MAX_PREVIEW_CHARS
      ? str.slice(0, MAX_PREVIEW_CHARS) + '…'
      : str;
  } catch {
    return String(value).slice(0, MAX_PREVIEW_CHARS);
  }
}

function buildArtifactSummary(
  artifact: ReturnType<typeof useDiscoveryStore.getState>['activeStageArtifact'],
): StageArtifactSummary | null {
  if (!artifact) return null;
  switch (artifact.type) {
    case 'hit_grid':
      return {
        type: 'hit_grid',
        label: `Hit Grid — ${artifact.hits.length} candidates`,
        candidateCount: artifact.hits.length,
      };
    case 'corpus_viewer':
      return { type: 'corpus_viewer', label: `Corpus viewer: ${artifact.documentId}` };
    case 'synthesis_plan_tree':
      return { type: 'synthesis_plan_tree', label: `Synthesis plan for hit ${artifact.hitId}` };
    case 'spectroscopy_validation':
      return { type: 'spectroscopy_validation', label: `Spectroscopy validation for hit ${artifact.hitId}` };
    case 'knowledge_graph':
      return { type: 'knowledge_graph', label: 'Knowledge graph' };
    case 'capability_gap':
      return { type: 'capability_gap', label: `Capability gap: ${artifact.gap.requiredFunction}` };
    default:
      return null;
  }
}

/**
 * Read the current Discovery OS state and assemble a StageContextBundle.
 *
 * Returns `null` when no discovery session is active, which means the
 * chat should behave exactly as it does today (no stage_context in the
 * request body).
 */
export function getStageContext(): StageContextBundle | null {
  const disco = useDiscoveryStore.getState();
  const epoch = disco.activeEpochId ? disco.epochs.get(disco.activeEpochId) ?? null : null;

  if (!epoch) return null;

  // Grab the last 10 completed tool invocations from the run store
  const runState = useRunStore.getState();
  const recentToolInvocations: TruncatedToolInvocation[] = (
    runState.currentRun?.toolInvocations ?? []
  )
    .filter((t) => t.status === 'completed' || t.status === 'failed')
    .slice(-10)
    .map((t) => ({
      tool: t.tool,
      inputPreview: truncateJson(t.input),
      outputPreview: truncateJson(t.output),
      status: t.status as 'completed' | 'failed',
    }));

  // Determine focused candidate (first approved, or first pending)
  const focused: CandidateArtifact | null =
    epoch.candidates.find((c) => c.status === 'approved') ??
    epoch.candidates.find((c) => c.status === 'pending') ??
    null;

  const artifact = disco.activeStageArtifact;

  return {
    activeEpochId: epoch.id,
    activeStage: epoch.currentStage as GoldenPathStage,
    targetParams: epoch.targetParams,
    activeArtifact: buildArtifactSummary(artifact),
    focusedCandidateId: focused?.id ?? null,
    focusedCandidate: focused,
    recentToolInvocations,
  };
}

// ---------------------------------------------------------------------------
// Streaming endpoint configuration
// ---------------------------------------------------------------------------

interface StreamEndpoint {
  url: string;
  body: Record<string, any>;
}

function getStreamEndpoint(
  mode: ChatMode,
  query: string,
  projectId: string,
  sessionId?: string,
  spectrumFilePath?: string,
  isMoeHypotheses?: boolean,
): StreamEndpoint {
  const base = getApiBase();
  const stageContext = getStageContext();
  const commonBody: Record<string, any> = {
    query,
    project_id: projectId,
    session_id: sessionId,
  };
  if (stageContext) commonBody.stage_context = stageContext;

  switch (mode) {
    case 'cortex':
      return { url: `${base}/api/swarm/stream`, body: commonBody };
    case 'moe':
      return isMoeHypotheses
        ? { url: `${base}/api/moe/hypotheses`, body: commonBody }
        : { url: `${base}/api/moe/stream`, body: commonBody };
    default:
      return { url: `${base}/api/swarm/stream`, body: commonBody };
  }
}

// ---------------------------------------------------------------------------
// Progress callback type — bridges NormalizedEvent to legacy StreamProgress
// ---------------------------------------------------------------------------

export type OnProgressCallback = (event: NormalizedEvent) => void;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface SubmitOptions {
  sessionId?: string;
  spectrumFilePath?: string;
  isMoeHypotheses?: boolean;
  onProgress?: OnProgressCallback;
}

export interface SubmitResult {
  result: any;
  cancelled: boolean;
}

export interface RunManagerAPI {
  currentRun: Run | null;
  isRunning: boolean;
  abortSignal: AbortSignal | null;

  startRun: (query: string, mode: ChatMode, intent?: string) => Run;
  handleEvent: (event: NormalizedEvent) => void;
  cancelCurrentRun: () => void;
  retryRun: (run: Run) => Run;

  /** Assemble current Discovery OS state for the backend. Returns null when no session is active. */
  getStageContext: () => StageContextBundle | null;

  submitQuery: (
    query: string,
    mode: ChatMode,
    options?: SubmitOptions,
  ) => Promise<SubmitResult>;

  submitLibrarian: (
    query: string,
    projectId: string,
    signal: AbortSignal,
  ) => Promise<any>;
}

export function useRunManager(projectId: string): RunManagerAPI {
  const abortRef = useRef<AbortController | null>(null);

  const {
    currentRun,
    createRun,
    updateRunStatus,
    appendEvent,
    addToolInvocation,
    updateToolInvocation,
    completeRun,
    failRun,
    cancelRun,
  } = useRunStore();

  const isRunning = currentRun !== null &&
    !['completed', 'failed', 'cancelled', 'awaiting_input'].includes(currentRun.status);

  const startRun = useCallback((query: string, mode: ChatMode, intent?: string): Run => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    const run = createRun({
      query,
      mode,
      intent: intent || mode.toUpperCase(),
      projectId,
    });

    updateRunStatus(run.id, 'routing');
    return run;
  }, [projectId, createRun, updateRunStatus]);

  const handleEvent = useCallback((event: NormalizedEvent) => {
    const run = useRunStore.getState().currentRun;
    if (!run) return;

    appendEvent(run.id, event);

    switch (event.type) {
      case 'routing':
        updateRunStatus(run.id, 'running');
        break;

      case 'tool_call':
        addToolInvocation(run.id, {
          tool: event.tool,
          input: event.input,
          output: null,
          startedAt: Date.now(),
          completedAt: null,
          status: 'running',
        });
        break;

      case 'tool_result':
        updateToolInvocation(run.id, event.tool, {
          output: event.output,
          completedAt: Date.now(),
          status: 'completed',
        });
        break;

      case 'hypotheses':
        updateRunStatus(run.id, 'awaiting_input');
        break;

      case 'coordinator_question':
        updateRunStatus(run.id, 'awaiting_input');
        break;

      case 'coordinator_complete':
        completeRun(run.id, {
          extractedGoals: event.extractedGoals,
          summary: event.summary,
          corpusEntities: event.corpusEntities,
          corpusSummary: event.corpusSummary,
        });
        break;

      case 'complete':
        completeRun(run.id, event.result);
        break;

      case 'error':
        failRun(run.id, event.message, event.category);
        break;

      case 'cancelled':
        cancelRun(run.id);
        break;
    }
  }, [appendEvent, updateRunStatus, addToolInvocation, updateToolInvocation, completeRun, failRun, cancelRun]);

  const cancelCurrentRun = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    const run = useRunStore.getState().currentRun;
    if (run && !['completed', 'failed', 'cancelled'].includes(run.status)) {
      cancelRun(run.id);
    }
  }, [cancelRun]);

  const retryRun = useCallback((run: Run): Run => {
    return startRun(run.query, run.mode, run.intent);
  }, [startRun]);

  // -------------------------------------------------------------------------
  // submitQuery — streaming dispatch for cortex, moe, discovery
  // -------------------------------------------------------------------------

  const submitQuery = useCallback(async (
    query: string,
    mode: ChatMode,
    options?: SubmitOptions,
  ): Promise<SubmitResult> => {
    const run = startRun(query, mode);
    const signal = abortRef.current!.signal;

    const endpoint = getStreamEndpoint(
      mode,
      query,
      projectId,
      options?.sessionId,
      options?.spectrumFilePath,
      options?.isMoeHypotheses,
    );

    return new Promise<SubmitResult>((resolve, reject) => {
      let finalResult: any = null;
      let wasCancelled = false;

      const onEvent = (event: NormalizedEvent) => {
        handleEvent(event);
        options?.onProgress?.(event);

        switch (event.type) {
          case 'complete':
            finalResult = event.result;
            resolve({ result: finalResult, cancelled: false });
            break;
          case 'cancelled':
            wasCancelled = true;
            resolve({ result: null, cancelled: true });
            break;
          case 'error':
            reject(new Error(event.message));
            break;
          case 'hypotheses':
            resolve({ result: event, cancelled: false });
            break;
          case 'coordinator_question':
            resolve({ result: event, cancelled: false });
            break;
          case 'coordinator_complete':
            finalResult = event;
            resolve({ result: event, cancelled: false });
            break;
        }
      };

      streamSSE(endpoint.url, endpoint.body, onEvent, { signal, timeout: 300_000 })
        .then(() => {
          if (!finalResult && !wasCancelled) {
            const currentState = useRunStore.getState().currentRun;
            if (currentState && !['completed', 'failed', 'cancelled'].includes(currentState.status)) {
              resolve({ result: null, cancelled: false });
            }
          }
        })
        .catch((err) => {
          if (err?.name !== 'AbortError') {
            reject(err);
          }
        });
    });
  }, [projectId, startRun, handleEvent]);

  // -------------------------------------------------------------------------
  // submitLibrarian — non-streaming librarian (uses api.chat)
  // -------------------------------------------------------------------------

  const submitLibrarian = useCallback(async (
    query: string,
    librarianProjectId: string,
    signal: AbortSignal,
  ): Promise<any> => {
    const run = startRun(query, 'librarian', 'SIMPLE');
    try {
      const response = await api.chat(query, librarianProjectId, signal, getStageContext());
      completeRun(run.id, response);
      return response;
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        cancelRun(run.id);
        throw err;
      }
      failRun(run.id, err?.message || 'Unknown error', 'backend_runtime');
      throw err;
    }
  }, [startRun, completeRun, cancelRun, failRun]);

  return {
    currentRun,
    isRunning,
    abortSignal: abortRef.current?.signal ?? null,
    startRun,
    handleEvent,
    cancelCurrentRun,
    retryRun,
    getStageContext,
    submitQuery,
    submitLibrarian,
  };
}
