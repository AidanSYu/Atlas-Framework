/**
 * Threaded Chat Store — Cursor-style
 *
 * Instead of 4 hardcoded agent histories, we now track an array of
 * ChatThread objects. Each thread holds its own messages and the agent
 * mode used for the *next* submission. Previous chats are listed in the
 * sidebar; users can start a new chat or resume an old one.
 */
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ChatMode } from '@/hooks/useRunManager';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface FollowUpSuggestion {
  label: string;
  query: string;
}

export interface FollowUpSuggestions {
  depth: FollowUpSuggestion;
  breadth: FollowUpSuggestion;
  opposition: FollowUpSuggestion;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /** Which agent produced this message (only set on assistant messages) */
  agent?: ChatMode;
  citations?: Array<{
    source: string;
    page: number;
    doc_id?: string;
    relevance?: number;
    text?: string;
  }>;
  brainActivity?: {
    brain: string;
    trace: string[];
    evidence: Array<{
      source: string;
      page: number;
      excerpt: string;
      relevance: number;
    }>;
    confidenceScore?: number;
    iterations?: number;
    contradictions?: Array<{
      claim_a: string;
      claim_b: string;
      severity: 'HIGH' | 'LOW';
      resolution?: string;
    }>;
    candidates?: Array<{
      smiles: string;
      properties?: Record<string, any>;
      toxicity?: Record<string, any>;
    }>;
  };
  librarianMetadata?: {
    reasoning?: string;
    relationships?: Array<{
      source: string;
      type: string;
      target: string;
      context?: string;
    }>;
    contextSources?: any;
  };
  runId?: string;
  errorInfo?: {
    category: string;
    message: string;
    retryable: boolean;
  };
  followUps?: FollowUpSuggestions;
  timestamp: number;
}

export interface ChatThread {
  id: string;
  projectId: string;
  title: string;
  messages: ChatMessage[];
  /** The agent mode selected for this thread's next message */
  chatMode: ChatMode;
  sessionId: string;
  createdAt: number;
  updatedAt: number;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface ChatState {
  threads: ChatThread[];
  activeThreadId: string | null;
  activeProjectId: string | null;
  currentInput: string;

  // Pending question (pre-filled from other views)
  pendingQuestion: string | null;
  // Phase 4: Pending hypotheses for MoE user-in-the-loop interaction
  pendingHypotheses: any[] | null;

  // --- Actions ---
  createThread: (projectId: string) => ChatThread;
  switchThread: (threadId: string) => void;
  deleteThread: (threadId: string) => void;
  addMessage: (msg: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  setCurrentInput: (input: string) => void;
  setChatMode: (mode: ChatMode) => void;
  clearCurrentChat: () => void;
  setActiveProject: (projectId: string | null) => void;
  setPendingQuestion: (question: string | null) => void;
  setPendingHypotheses: (hypotheses: any[] | null) => void;

  // Derived / convenience
  getActiveThread: () => ChatThread | null;
  getProjectThreads: (projectId: string) => ChatThread[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const generateId = () => Math.random().toString(36).substring(2, 15);

function makeWelcomeMessage(mode: ChatMode): ChatMessage {
  const messages: Record<ChatMode, string> = {
    librarian:
      "I'm your **Librarian**. I specialize in document analysis and can answer questions about your uploaded research papers.",
    cortex:
      "I'm **Cortex**, your research analysis agent. I can cross-reference documents, identify patterns, and discover connections across your research.",
    moe:
      "I'm the **Mixture of Experts (MoE) Supervisor**. I manage a team of specialized agents. Ask a complex research question, and I'll orchestrate the team.",
  };
  return {
    id: `welcome-${generateId()}`,
    role: 'assistant',
    content: messages[mode],
    agent: mode,
    timestamp: Date.now(),
  };
}

function createNewThread(projectId: string, mode: ChatMode = 'librarian'): ChatThread {
  return {
    id: generateId(),
    projectId,
    title: 'New Chat',
    messages: [makeWelcomeMessage(mode)],
    chatMode: mode,
    sessionId: crypto.randomUUID(),
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      threads: [],
      activeThreadId: null,
      activeProjectId: null,
      currentInput: '',
      pendingQuestion: null,
      pendingHypotheses: null,

      createThread: (projectId: string) => {
        const thread = createNewThread(projectId);
        set((state) => ({
          threads: [thread, ...state.threads],
          activeThreadId: thread.id,
          currentInput: '',
          pendingHypotheses: null,
        }));
        return thread;
      },

      switchThread: (threadId: string) => {
        set({ activeThreadId: threadId, currentInput: '', pendingHypotheses: null });
      },

      deleteThread: (threadId: string) => {
        set((state) => {
          const next = state.threads.filter((t) => t.id !== threadId);
          const wasActive = state.activeThreadId === threadId;
          return {
            threads: next,
            activeThreadId: wasActive ? (next[0]?.id ?? null) : state.activeThreadId,
            pendingHypotheses: wasActive ? null : state.pendingHypotheses,
          };
        });
      },

      addMessage: (msg) => {
        set((state) => {
          const threadId = state.activeThreadId;
          if (!threadId) return state;
          const newMsg: ChatMessage = { ...msg, id: generateId(), timestamp: Date.now() };

          const threads = state.threads.map((t) => {
            if (t.id !== threadId) return t;
            // Auto-title based on first user message
            let title = t.title;
            if (title === 'New Chat' && newMsg.role === 'user') {
              title = newMsg.content.slice(0, 50) + (newMsg.content.length > 50 ? '...' : '');
            }
            return { ...t, messages: [...t.messages, newMsg], title, updatedAt: Date.now() };
          });
          return { threads };
        });
      },

      setCurrentInput: (input) => set({ currentInput: input }),

      setChatMode: (mode) => {
        set((state) => {
          const threadId = state.activeThreadId;
          if (!threadId) return state;
          return {
            threads: state.threads.map((t) =>
              t.id === threadId ? { ...t, chatMode: mode } : t
            ),
          };
        });
      },

      clearCurrentChat: () => {
        set((state) => {
          const threadId = state.activeThreadId;
          if (!threadId) return state;
          const thread = state.threads.find((t) => t.id === threadId);
          if (!thread) return state;
          return {
            threads: state.threads.map((t) =>
              t.id === threadId
                ? { ...t, messages: [makeWelcomeMessage(t.chatMode)], title: 'New Chat', updatedAt: Date.now() }
                : t
            ),
            pendingHypotheses: null,
          };
        });
      },

      setActiveProject: (projectId) => {
        const current = get().activeProjectId;
        if (projectId === current) return;
        // On project change, switch to the most recent thread of that project or create one
        const projectThreads = get().threads.filter((t) => t.projectId === projectId);
        if (projectId && projectThreads.length === 0) {
          const thread = createNewThread(projectId);
          set({
            activeProjectId: projectId,
            threads: [thread, ...get().threads],
            activeThreadId: thread.id,
            currentInput: '',
            pendingQuestion: null,
            pendingHypotheses: null,
          });
        } else {
          set({
            activeProjectId: projectId,
            activeThreadId: projectThreads[0]?.id ?? null,
            currentInput: '',
            pendingQuestion: null,
            pendingHypotheses: null,
          });
        }
      },

      setPendingQuestion: (question) => set({ pendingQuestion: question }),
      setPendingHypotheses: (hypotheses) => set({ pendingHypotheses: hypotheses }),

      getActiveThread: () => {
        const state = get();
        return state.threads.find((t) => t.id === state.activeThreadId) ?? null;
      },

      getProjectThreads: (projectId: string) => {
        return get().threads.filter((t) => t.projectId === projectId);
      },
    }),
    {
      name: 'atlas-chat-storage',
      version: 4, // Bump version to force migration (wipes old data)
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        threads: state.threads,
        activeThreadId: state.activeThreadId,
        activeProjectId: state.activeProjectId,
      }),
    }
  )
);
