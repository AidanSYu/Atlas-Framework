/**
 * useRunManager - lightweight run hook for grounded chat and task execution.
 *
 * Librarian/Cortex stay on the grounded chat path, while heavier execution
 * modes can still use the framework orchestrator when needed.
 */
import { useCallback, useRef } from 'react';
import { useRunStore, type Run } from '@/stores/runStore';
import { useDiscoveryStore } from '@/stores/discoveryStore';
import { api } from '@/lib/api';
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
  BROAD_RESEARCH: 'moe',
  MULTI_STEP: 'moe',
};

export function intentToMode(intent: string): ChatMode {
  return INTENT_TO_MODE[intent] || 'cortex';
}

const MAX_PREVIEW_CHARS = 500;

function truncateJson(value: any): string {
  try {
    const str = typeof value === 'string' ? value : JSON.stringify(value);
    return str.length > MAX_PREVIEW_CHARS ? str.slice(0, MAX_PREVIEW_CHARS) + '...' : str;
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
        label: `Hit Grid - ${artifact.hits.length} candidates`,
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

export function getStageContext(): StageContextBundle | null {
  const discovery = useDiscoveryStore.getState();
  const epoch = discovery.activeEpochId ? discovery.epochs.get(discovery.activeEpochId) ?? null : null;
  if (!epoch) return null;

  const runState = useRunStore.getState();
  const recentToolInvocations: TruncatedToolInvocation[] = (
    runState.currentRun?.toolInvocations ?? []
  )
    .filter((tool) => tool.status === 'completed' || tool.status === 'failed')
    .slice(-10)
    .map((tool) => ({
      tool: tool.tool,
      inputPreview: truncateJson(tool.input),
      outputPreview: truncateJson(tool.output),
      status: tool.status as 'completed' | 'failed',
    }));

  const focused: CandidateArtifact | null =
    epoch.candidates.find((candidate) => candidate.status === 'approved') ??
    epoch.candidates.find((candidate) => candidate.status === 'pending') ??
    null;

  return {
    activeEpochId: epoch.id,
    activeStage: epoch.currentStage as GoldenPathStage,
    targetParams: epoch.targetParams,
    activeArtifact: buildArtifactSummary(discovery.activeStageArtifact),
    focusedCandidateId: focused?.id ?? null,
    focusedCandidate: focused,
    recentToolInvocations,
  };
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
  cancelCurrentRun: () => void;
  retryRun: (run: Run) => Run;
  getStageContext: () => StageContextBundle | null;
  submitQuery: (query: string, mode: ChatMode) => Promise<SubmitResult>;
  submitLibrarian: (query: string) => Promise<any>;
}

function buildModeInstructions(mode: ChatMode): string {
  switch (mode) {
    case 'librarian':
      return 'Prefer grounded retrieval first. Keep the answer concise and evidence-led.';
    case 'moe':
      return 'Synthesize across retrieved evidence and optional tools before answering.';
    default:
      return 'Use multiple tool steps when helpful, especially graph traversal and grounded retrieval.';
  }
}

function buildConversation(mode: ChatMode, stageContext: StageContextBundle | null): Array<{ role: string; content: string }> {
  const conversation: Array<{ role: string; content: string }> = [
    { role: 'system', content: buildModeInstructions(mode) },
  ];

  if (stageContext) {
    conversation.push({
      role: 'system',
      content: `Current research stage context:\n${JSON.stringify(stageContext)}`,
    });
  }

  return conversation;
}

export function useRunManager(projectId: string): RunManagerAPI {
  const abortRef = useRef<AbortController | null>(null);

  const currentRun = useRunStore((s) => s.currentRun);
  const createRun = useRunStore((s) => s.createRun);
  const updateRunStatus = useRunStore((s) => s.updateRunStatus);
  const appendEvent = useRunStore((s) => s.appendEvent);
  const addToolInvocation = useRunStore((s) => s.addToolInvocation);
  const updateToolInvocation = useRunStore((s) => s.updateToolInvocation);
  const completeRun = useRunStore((s) => s.completeRun);
  const failRun = useRunStore((s) => s.failRun);
  const cancelRun = useRunStore((s) => s.cancelRun);

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
    appendEvent(run.id, { type: 'routing', mode, intent: intent || mode.toUpperCase() });
    return run;
  }, [appendEvent, createRun, projectId, updateRunStatus]);

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

  const submitQuery = useCallback(async (query: string, mode: ChatMode): Promise<SubmitResult> => {
    const run = startRun(query, mode);
    const signal = abortRef.current?.signal ?? null;
    updateRunStatus(run.id, 'running');

    try {
      if (mode === 'librarian' || mode === 'cortex') {
        appendEvent(run.id, {
          type: 'progress',
          node: 'retrieval',
          message: 'Searching the project corpus and graph context',
        });

        const result = await api.chat(
          query,
          projectId,
          signal ?? undefined,
        );

        appendEvent(run.id, { type: 'thinking', content: result.reasoning });
        appendEvent(run.id, { type: 'evidence', count: result.citations?.length ?? 0 });
        if (result.relationships && result.relationships.length > 0) {
          appendEvent(run.id, { type: 'graph_analysis', data: { relationships: result.relationships } });
        }

        completeRun(run.id, result);
        return { result, cancelled: false };
      }

      let finalResult: any = null;
      let fatalError: { message: string; category: any } | null = null;
      let wasCancelled = false;

      await api.streamFramework(
        {
          prompt: query,
          project_id: projectId,
          conversation: buildConversation(mode, getStageContext()),
        },
        (event) => {
          switch (event.type) {
            case 'run_start':
              appendEvent(run.id, {
                type: 'progress',
                node: 'orchestrator',
                message: event.model
                  ? `Orchestrator ready: ${event.model} (${event.availableTools.length} tools)`
                  : 'Orchestrator initialising',
              });
              break;
            case 'iteration_start':
              appendEvent(run.id, {
                type: 'progress',
                node: 'iteration',
                message: `Iteration ${event.iteration}`,
              });
              break;
            case 'chunk':
              appendEvent(run.id, { type: 'chunk', content: event.content });
              break;
            case 'thinking':
              appendEvent(run.id, { type: 'thinking', content: event.content });
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
              appendEvent(run.id, { type: 'tool_call', tool: event.tool, input: event.input });
              break;
            case 'tool_result':
              updateToolInvocation(run.id, event.tool, {
                output: event.output,
                completedAt: Date.now(),
                status: (event.output as any)?.error ? 'failed' : 'completed',
              });
              appendEvent(run.id, { type: 'tool_result', tool: event.tool, output: event.output });
              break;
            case 'error': {
              const detail = [event.where, event.message].filter(Boolean).join(': ');
              const message = event.traceback ? `${detail}\n\n${event.traceback}` : detail;
              appendEvent(run.id, { type: 'error', message, category: event.category });
              if (!event.nonFatal) {
                fatalError = { message, category: event.category };
              }
              break;
            }
            case 'complete':
              finalResult = event.result;
              break;
            case 'cancelled':
              wasCancelled = true;
              break;
            default:
              break;
          }
        },
        signal ?? undefined,
      );

      if (wasCancelled || signal?.aborted) {
        cancelRun(run.id);
        return { result: null, cancelled: true };
      }
      if (fatalError) {
        failRun(run.id, (fatalError as any).message, (fatalError as any).category);
        return { result: null, cancelled: false };
      }
      if (!finalResult) {
        failRun(run.id, 'Orchestration ended without a final answer', 'backend_runtime');
        return { result: null, cancelled: false };
      }

      completeRun(run.id, finalResult);
      return { result: finalResult, cancelled: false };
    } catch (error: any) {
      if (signal?.aborted || error?.name === 'AbortError') {
        cancelRun(run.id);
        return { result: null, cancelled: true };
      }
      failRun(run.id, error?.message || 'Framework query failed', 'backend_runtime');
      throw error;
    }
  }, [
    addToolInvocation,
    appendEvent,
    cancelRun,
    completeRun,
    failRun,
    projectId,
    startRun,
    updateRunStatus,
    updateToolInvocation,
  ]);

  const submitLibrarian = useCallback(async (query: string): Promise<any> => {
    const response = await submitQuery(query, 'librarian');
    return response.result;
  }, [submitQuery]);

  return {
    currentRun,
    isRunning,
    abortSignal: abortRef.current?.signal ?? null,
    startRun,
    cancelCurrentRun,
    retryRun,
    getStageContext,
    submitQuery,
    submitLibrarian,
  };
}
