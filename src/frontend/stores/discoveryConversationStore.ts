import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type DiscoveryMessageType =
  | 'text'
  | 'thinking'
  | 'question'
  | 'plan'
  | 'tool_start'
  | 'tool_complete'
  | 'analysis'
  | 'recommendation'
  | 'error'
  | 'session_update';

export interface PlanStage {
  stageId: number;
  plugin: string;
  description: string;
  estimatedSeconds: number;
}

export interface PlanData {
  planId: string;
  summary: string;
  reasoning: string;
  moleculeNotes: string;
  moleculeCount: number;
  iteration: number;
  estimatedTotalSeconds: number;
  warnings: string[];
  isDemoData: boolean;
  stages: PlanStage[];
  status: 'proposed' | 'accepted' | 'rejected';
}

export interface ToolData {
  stageId: number;
  plugin: string;
  description: string;
  thinking?: string;
  summary?: string;
  stats?: Record<string, any>;
  totalStages: number;
  candidatesSoFar?: number;
  error?: boolean;
  status: 'running' | 'complete' | 'error';
}

export interface AnalysisData {
  keyFindings: string[];
  topCandidates: Array<{ smiles: string; reasoning: string; composite_score?: number }>;
  concerns: string[];
  recommendations: Array<{ action: string; description: string; priority: string }>;
  missingCapabilities: string[];
}

/** Groups all messages belonging to one pipeline execution run. */
export interface PipelineRun {
  runId: string;
  iteration: number;
  startedAt: number;
  completedAt?: number;
  messageIds: string[];  // IDs of tool_start/tool_complete/analysis messages
  candidateCount: number;
  stagesCompleted: number;
  collapsed: boolean;
}

export interface DiscoveryMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  type: DiscoveryMessageType;
  content: string;
  timestamp: number;
  runId?: string;
  plan?: PlanData;
  tool?: ToolData;
  analysis?: AnalysisData;
  questionData?: {
    question: string;
    options: string[];
    context?: string;
    turn: number;
    goalsSoFar: string[];
  };
}

export interface SessionConversation {
  sessionId: string;
  messages: DiscoveryMessage[];
  stage: string;
  isStreaming: boolean;
  pendingPlan: PlanData | null;
  candidates: any[];
  pipelineRuns: PipelineRun[];
  activeRunId: string | null;
}

interface DiscoveryConversationState {
  conversations: Record<string, SessionConversation>;

  getOrCreate: (sessionId: string) => SessionConversation;
  addMessage: (sessionId: string, msg: DiscoveryMessage) => void;
  updateMessage: (sessionId: string, messageId: string, updates: Partial<DiscoveryMessage>) => void;
  setStage: (sessionId: string, stage: string) => void;
  setStreaming: (sessionId: string, streaming: boolean) => void;
  setPendingPlan: (sessionId: string, plan: PlanData | null) => void;
  setCandidates: (sessionId: string, candidates: any[]) => void;
  appendThinking: (sessionId: string, content: string) => void;
  startPipelineRun: (sessionId: string, iteration: number) => string;
  addMessageToRun: (sessionId: string, runId: string, messageId: string) => void;
  completePipelineRun: (sessionId: string, runId: string, stagesCompleted: number, candidateCount: number) => void;
  toggleRunCollapsed: (sessionId: string, runId: string) => void;
  collapseOldRuns: (sessionId: string) => void;
}

function createEmptyConversation(sessionId: string): SessionConversation {
  return {
    sessionId,
    messages: [],
    stage: 'setup',
    isStreaming: false,
    pendingPlan: null,
    candidates: [],
    pipelineRuns: [],
    activeRunId: null,
  };
}

export const useDiscoveryConversation = create<DiscoveryConversationState>()(
  persist(
    (set, get) => ({
      conversations: {},

      getOrCreate: (sessionId: string) => {
        const state = get();
        return state.conversations[sessionId] || createEmptyConversation(sessionId);
      },

      addMessage: (sessionId, msg) => set((state) => {
        const conv = state.conversations[sessionId] || createEmptyConversation(sessionId);
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: { ...conv, messages: [...conv.messages, msg] },
          },
        };
      }),

      updateMessage: (sessionId, messageId, updates) => set((state) => {
        const conv = state.conversations[sessionId];
        if (!conv) return state;
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: {
              ...conv,
              messages: conv.messages.map((m) => {
                if (m.id !== messageId) return m;
                // Shallow-merge the `tool` sub-object so tool_complete preserves
                // the `thinking` field written by tool_start.
                const mergedTool = (updates.tool && m.tool)
                  ? { ...m.tool, ...updates.tool }
                  : (updates.tool ?? m.tool);
                return { ...m, ...updates, tool: mergedTool };
              }),
            },
          },
        };
      }),

      setStage: (sessionId, stage) => set((state) => {
        const conv = state.conversations[sessionId] || createEmptyConversation(sessionId);
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: { ...conv, stage },
          },
        };
      }),

      setStreaming: (sessionId, streaming) => set((state) => {
        const conv = state.conversations[sessionId] || createEmptyConversation(sessionId);
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: { ...conv, isStreaming: streaming },
          },
        };
      }),

      setPendingPlan: (sessionId, plan) => set((state) => {
        const conv = state.conversations[sessionId] || createEmptyConversation(sessionId);
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: { ...conv, pendingPlan: plan },
          },
        };
      }),

      setCandidates: (sessionId, candidates) => set((state) => {
        const conv = state.conversations[sessionId] || createEmptyConversation(sessionId);
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: { ...conv, candidates },
          },
        };
      }),

      appendThinking: (sessionId, content) => set((state) => {
        const conv = state.conversations[sessionId] || createEmptyConversation(sessionId);
        const msgs = [...conv.messages];
        const last = msgs[msgs.length - 1];
        if (last?.type === 'thinking' && last.role === 'assistant') {
          msgs[msgs.length - 1] = { ...last, content: last.content + '\n' + content };
        } else {
          msgs.push({
            id: crypto.randomUUID(),
            role: 'assistant',
            type: 'thinking',
            content,
            timestamp: Date.now(),
          });
        }
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: { ...conv, messages: msgs },
          },
        };
      }),

      startPipelineRun: (sessionId, iteration) => {
        const runId = crypto.randomUUID();
        set((state) => {
          const conv = state.conversations[sessionId] || createEmptyConversation(sessionId);
          const run: PipelineRun = {
            runId,
            iteration,
            startedAt: Date.now(),
            messageIds: [],
            candidateCount: 0,
            stagesCompleted: 0,
            collapsed: false,
          };
          return {
            conversations: {
              ...state.conversations,
              [sessionId]: {
                ...conv,
                pipelineRuns: [...conv.pipelineRuns, run],
                activeRunId: runId,
              },
            },
          };
        });
        return runId;
      },

      addMessageToRun: (sessionId, runId, messageId) => set((state) => {
        const conv = state.conversations[sessionId];
        if (!conv) return state;
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: {
              ...conv,
              pipelineRuns: conv.pipelineRuns.map((r) =>
                r.runId === runId
                  ? { ...r, messageIds: [...r.messageIds, messageId] }
                  : r,
              ),
            },
          },
        };
      }),

      completePipelineRun: (sessionId, runId, stagesCompleted, candidateCount) => set((state) => {
        const conv = state.conversations[sessionId];
        if (!conv) return state;
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: {
              ...conv,
              activeRunId: null,
              pipelineRuns: conv.pipelineRuns.map((r) =>
                r.runId === runId
                  ? { ...r, completedAt: Date.now(), stagesCompleted, candidateCount }
                  : r,
              ),
            },
          },
        };
      }),

      toggleRunCollapsed: (sessionId, runId) => set((state) => {
        const conv = state.conversations[sessionId];
        if (!conv) return state;
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: {
              ...conv,
              pipelineRuns: conv.pipelineRuns.map((r) =>
                r.runId === runId ? { ...r, collapsed: !r.collapsed } : r,
              ),
            },
          },
        };
      }),

      collapseOldRuns: (sessionId) => set((state) => {
        const conv = state.conversations[sessionId];
        if (!conv || conv.pipelineRuns.length <= 1) return state;
        return {
          conversations: {
            ...state.conversations,
            [sessionId]: {
              ...conv,
              pipelineRuns: conv.pipelineRuns.map((r, i) => ({
                ...r,
                collapsed: i < conv.pipelineRuns.length - 1,
              })),
            },
          },
        };
      }),
    }),
    {
      name: 'atlas-discovery-conversations',
      version: 1,
      partialize: (state) => ({
        conversations: Object.fromEntries(
          Object.entries(state.conversations).map(([k, v]) => [
            k,
            { ...v, isStreaming: false },
          ]),
        ),
      }),
    },
  ),
);
