'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  BookOpen,
  Brain,
  ChevronRight,
  FileText,
  Loader2,
  Search,
  Trash2,
  AlertCircle,
  Zap,
  PanelRightOpen,
  PanelRightClose,
} from 'lucide-react';

import { useChatStore } from '@/stores/chatStore';
import { useRunStore } from '@/stores/runStore';
import { useRunManager, type ChatMode } from '@/hooks/useRunManager';
import type { ModelRegistryResponse, ModelStatusResponse } from '@/lib/api';
import { ConversationView } from './ConversationView';
import { RunProgressDisplay, type StreamProgress } from './RunProgressDisplay';
import { CommandSurface } from './CommandSurface';

const CHAT_BRAINS: Array<{
  mode: ChatMode;
  label: string;
  icon: typeof BookOpen;
  accent: string;
}> = [
  { mode: 'librarian', label: 'Librarian', icon: BookOpen, accent: 'text-accent' },
  { mode: 'cortex', label: 'Cortex', icon: Brain, accent: 'text-info' },
];

function extractCitations(result: any): Array<{ source: string; page: number; relevance?: number; text?: string }> {
  if (Array.isArray(result?.citations) && result.citations.length > 0) {
    return result.citations.map((citation: any) => ({
      source: citation.source || 'Unknown',
      page: citation.page ?? 1,
      relevance: citation.relevance,
      text: citation.text,
    }));
  }

  const citations: Array<{ source: string; page: number; relevance?: number; text?: string }> = [];
  for (const step of result?.trace || []) {
    const toolResults = step?.tool_results || [];
    for (const toolResult of toolResults) {
      const evidence = toolResult?.evidence;
      if (!Array.isArray(evidence)) continue;
      for (const item of evidence) {
        citations.push({
          source: item.source || 'Unknown',
          page: item.page ?? 1,
          text: item.excerpt,
        });
      }
    }
  }
  return citations;
}

function buildBrainTrace(result: any): string[] {
  if (!Array.isArray(result?.trace)) {
    return [];
  }

  const trace: string[] = [];
  for (const step of result?.trace || []) {
    if (step.thinking) trace.push(step.thinking);
    for (const toolCall of step?.tool_calls || []) {
      trace.push(`Tool: ${toolCall.name}`);
    }
  }
  return trace;
}

/* ================================================================
   Context Panel — right sidebar showing run info and sources
   Inspired by Cowork's progress/context panels
   ================================================================ */

function ContextPanel({
  isRunning,
  currentRun,
  streamProgress,
  messages,
  chatMode,
}: {
  isRunning: boolean;
  currentRun: any;
  streamProgress: StreamProgress | null;
  messages: any[];
  chatMode: ChatMode;
}) {
  // Collect all citations from messages
  const allCitations = useMemo(() => {
    const citations: Array<{ source: string; page: number; text?: string }> = [];
    const seen = new Set<string>();
    for (const msg of messages) {
      if (msg.citations) {
        for (const c of msg.citations) {
          const key = `${c.source}:${c.page}`;
          if (!seen.has(key)) {
            seen.add(key);
            citations.push(c);
          }
        }
      }
    }
    return citations;
  }, [messages]);

  // Collect tools used in current run
  const toolsUsed = useMemo(() => {
    if (!currentRun?.toolInvocations) return [];
    return currentRun.toolInvocations.map((t: any) => ({
      name: t.tool,
      status: t.status,
    }));
  }, [currentRun]);

  return (
    <div className="flex h-full w-[260px] shrink-0 flex-col overflow-hidden border-l border-border bg-card">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-2.5">
        <Zap className="h-3.5 w-3.5 text-accent" />
        <span className="text-xs font-semibold text-foreground">Context</span>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Progress section */}
        {isRunning && streamProgress && (
          <div className="border-b border-border px-3 py-3">
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              <Loader2 className="h-3 w-3 animate-spin text-accent" />
              Progress
            </div>
            <div className="space-y-1.5">
              {streamProgress.thinkingSteps.slice(-5).map((step, i) => (
                <div key={i} className="flex items-start gap-1.5 text-[11px] text-muted-foreground">
                  <ChevronRight className="h-3 w-3 mt-0.5 shrink-0 text-accent/50" />
                  <span className="leading-tight">{step}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tools used */}
        {toolsUsed.length > 0 && (
          <div className="border-b border-border px-3 py-3">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              Tools Used
            </div>
            <div className="space-y-1">
              {toolsUsed.map((tool: any, i: number) => (
                <div key={i} className="flex items-center gap-2 text-[11px]">
                  <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                    tool.status === 'completed' ? 'bg-accent' :
                    tool.status === 'failed' ? 'bg-destructive' :
                    'bg-warning animate-pulse'
                  }`} />
                  <span className="text-foreground/80 truncate">{tool.name}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Referenced sources */}
        {allCitations.length > 0 && (
          <div className="border-b border-border px-3 py-3">
            <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              <FileText className="h-3 w-3" />
              Sources ({allCitations.length})
            </div>
            <div className="space-y-1.5">
              {allCitations.slice(0, 10).map((citation, i) => (
                <div key={i} className="rounded border border-border/50 bg-background/50 px-2 py-1.5">
                  <div className="text-[11px] font-medium text-foreground/80 truncate">{citation.source}</div>
                  <div className="text-[10px] text-muted-foreground/60">p. {citation.page}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Session info */}
        <div className="px-3 py-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
            Session
          </div>
          <div className="space-y-2 text-[11px] text-muted-foreground">
            <div className="flex items-center justify-between">
              <span>Messages</span>
              <span className="text-foreground/70">{messages.filter((m: any) => m.role === 'user').length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Sources cited</span>
              <span className="text-foreground/70">{allCitations.length}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Mode</span>
              <span className="text-accent/70 text-[10px] capitalize">{chatMode}</span>
            </div>
          </div>
        </div>

        {/* Empty state when nothing to show */}
        {!isRunning && toolsUsed.length === 0 && allCitations.length === 0 && messages.length <= 1 && (
          <div className="flex flex-col items-center px-4 py-8 text-center">
            <Search className="h-5 w-5 text-muted-foreground/20 mb-2" />
            <p className="text-[11px] text-muted-foreground/60">
              Sources and graph context used in this session will appear here
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ================================================================
   ChatShell — main chat interface, no mode selector
   ================================================================ */

interface ChatShellProps {
  onCitationClick: (filename: string, page: number, docId?: string) => void;
  projectId?: string;
  autoSubmitQuery?: string | null;
  onAutoSubmitConsumed?: () => void;
  onOpenRunHistory?: () => void;
  onViewRunDetails?: (runId: string) => void;
  lockedMode?: ChatMode;
  modelRegistry?: ModelRegistryResponse | null;
  modelStatus?: ModelStatusResponse | null;
  onLoadModel?: (modelName: string) => Promise<void>;
  isModelSwitching?: boolean;
}

export default function ChatShell({
  onCitationClick,
  projectId,
  autoSubmitQuery,
  onAutoSubmitConsumed,
  onViewRunDetails,
  lockedMode,
  modelRegistry,
  modelStatus,
  onLoadModel,
  isModelSwitching,
}: ChatShellProps) {
  const runManager = useRunManager(projectId || '');

  const globalInput = useChatStore((s) => s.currentInput);
  const pendingQuestion = useChatStore((s) => s.pendingQuestion);
  const threads = useChatStore((s) => s.threads);
  const activeThreadId = useChatStore((s) => s.activeThreadId);
  const addMessage = useChatStore((s) => s.addMessage);
  const setGlobalInput = useChatStore((s) => s.setCurrentInput);
  const clearCurrentChat = useChatStore((s) => s.clearCurrentChat);
  const setChatMode = useChatStore((s) => s.setChatMode);
  const setPendingQuestion = useChatStore((s) => s.setPendingQuestion);

  const [localInput, setLocalInput] = useState(globalInput || '');
  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) ?? null,
    [activeThreadId, threads]
  );
  // Librarian is the default grounded mode for new chat threads.
  const effectiveChatMode: ChatMode = lockedMode || activeThread?.chatMode || 'librarian';
  const currentMessages = activeThread?.messages ?? [];

  const [streamProgress, setStreamProgress] = useState<StreamProgress | null>(null);
  const [contextPanelOpen, setContextPanelOpen] = useState(true);

  useEffect(() => {
    if (projectId) {
      useRunStore.getState().loadHistory(projectId);
    }
  }, [projectId]);

  useEffect(() => {
    if (pendingQuestion) {
      setLocalInput(pendingQuestion);
      setGlobalInput(pendingQuestion);
      setPendingQuestion(null);
    }
  }, [pendingQuestion, setGlobalInput, setPendingQuestion]);

  useEffect(() => {
    if (autoSubmitQuery && !runManager.isRunning && projectId) {
      setLocalInput(autoSubmitQuery);
      setGlobalInput(autoSubmitQuery);
      onAutoSubmitConsumed?.();
      const timer = setTimeout(() => {
        void handleSubmitWithContent(autoSubmitQuery);
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [autoSubmitQuery, onAutoSubmitConsumed, projectId, runManager.isRunning, setGlobalInput]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmitWithContent = useCallback(async (userContent: string) => {
    if (!userContent.trim() || runManager.isRunning || !projectId) return;

    const content = userContent.trim();
    addMessage({ role: 'user', content });
    setStreamProgress({
      currentNode: 'retrieval',
      message: effectiveChatMode === 'cortex'
        ? 'Cortex is reviewing your corpus and graph context...'
        : 'Librarian is searching your project library...',
      thinkingSteps: [
        effectiveChatMode === 'cortex'
          ? 'Reviewing relevant passages and graph relationships'
          : 'Finding grounded evidence in your project corpus',
      ],
      evidenceFound: 0,
    });

    try {
      const { result, cancelled } = await runManager.submitQuery(content, effectiveChatMode);
      if (cancelled || !result) return;

      const citations = extractCitations(result);
      const trace = buildBrainTrace(result);
      addMessage({
        role: 'assistant',
        content: result.answer || 'No answer generated.',
        agent: effectiveChatMode,
        citations,
        brainActivity: trace.length > 0 ? {
          brain: effectiveChatMode === 'cortex' ? 'Cortex' : 'Librarian',
          trace,
          evidence: citations.map((citation) => ({
            source: citation.source,
            page: citation.page,
            excerpt: citation.text || '',
            relevance: citation.relevance ?? 1,
          })),
          iterations: result.iterations,
        } : undefined,
        librarianMetadata: result.reasoning || result.relationships ? {
          reasoning: result.reasoning,
          relationships: (result.relationships || []).map((relationship: any) => ({
            source: relationship.source,
            type: relationship.type,
            target: relationship.target,
            context: relationship.properties?.context,
          })),
          contextSources: result.context_sources,
        } : undefined,
        runId: runManager.currentRun?.id,
      });
    } catch (error: any) {
      const failedRun = runManager.currentRun;
      const errorMsg = error?.message || 'An unexpected error occurred.';
      addMessage({
        role: 'assistant',
        content: errorMsg,
        runId: failedRun?.id,
        errorInfo: {
          category: failedRun?.error?.category || 'backend_runtime',
          message: errorMsg,
          retryable: true,
        },
      });
    } finally {
      setStreamProgress(null);
    }
  }, [addMessage, effectiveChatMode, projectId, runManager]);

  const handleSubmit = useCallback((overrideContent?: string) => {
    const content = overrideContent ?? localInput;
    if (!content.trim()) return;
    if (!overrideContent) {
      setLocalInput('');
      setGlobalInput('');
    }
    void handleSubmitWithContent(content);
  }, [handleSubmitWithContent, localInput, setGlobalInput]);

  const handleRetry = useCallback((query: string) => {
    void handleSubmitWithContent(query);
  }, [handleSubmitWithContent]);

  const handleQuickQuery = useCallback((query: string) => {
    setLocalInput(query);
    setGlobalInput(query);
    setTimeout(() => void handleSubmitWithContent(query), 50);
  }, [handleSubmitWithContent, setGlobalInput]);

  const clearChat = useCallback(() => {
    if (runManager.isRunning) {
      runManager.cancelCurrentRun();
    }
    clearCurrentChat();
  }, [clearCurrentChat, runManager]);

  const isLoading = runManager.isRunning;
  const startTime = runManager.currentRun?.startedAt ?? null;

  if (!projectId) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-6 text-center text-muted-foreground">
        <AlertCircle className="mb-2 h-8 w-8 opacity-20" />
        <p className="text-sm">No Project Selected</p>
      </div>
    );
  }

  return (
    <div className="flex h-full max-h-full w-full overflow-hidden">
      {/* Main chat column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Minimal top bar — clear + context panel toggle.
            Note: the Librarian/Cortex mode toggle was removed in the single-orchestrator
            transition. The backend now ignores `mode` entirely. */}
        <div className="shrink-0 flex items-center justify-between border-b border-border/50 bg-card/30 px-3 py-1.5">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
              <Zap className="h-3 w-3 text-accent/60" />
              <span>Atlas grounded chat</span>
              {isLoading && (
                <span className="flex items-center gap-1 text-accent">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span className="text-[10px]">running</span>
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setContextPanelOpen(!contextPanelOpen)}
              className="rounded p-1 text-muted-foreground transition-colors hover:bg-surface hover:text-foreground"
              title={contextPanelOpen ? 'Hide context panel' : 'Show context panel'}
            >
              {contextPanelOpen ? <PanelRightClose className="h-3.5 w-3.5" /> : <PanelRightOpen className="h-3.5 w-3.5" />}
            </button>
            <button
              onClick={clearChat}
              className="rounded p-1 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
              title="Clear chat"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4">
          <ConversationView
            messages={currentMessages}
            onCitationClick={onCitationClick}
            chatMode={effectiveChatMode}
            isLoading={isLoading}
            onRetry={handleRetry}
            onQuickQuery={handleQuickQuery}
            onViewRunDetails={onViewRunDetails}
            onFollowUpClick={handleQuickQuery}
          />

          <RunProgressDisplay
            streamProgress={streamProgress}
            streamingText=""
            isLoading={isLoading}
            startTime={startTime}
          />
        </div>

        {/* Input */}
        <CommandSurface
          value={localInput}
          onChange={(value) => {
            setLocalInput(value);
            setGlobalInput(value);
          }}
          onSubmit={() => handleSubmit()}
          onCancel={() => runManager.cancelCurrentRun()}
          onLoadModel={onLoadModel}
          isLoading={isLoading}
          disabled={!projectId}
          chatMode={effectiveChatMode}
          modelRegistry={modelRegistry}
          modelStatus={modelStatus}
          isModelSwitching={isModelSwitching}
          streamingText=""
          onStopWithPartial={() => runManager.cancelCurrentRun()}
        />
      </div>

      {/* Context panel — right sidebar */}
      {contextPanelOpen && (
        <ContextPanel
          isRunning={isLoading}
          currentRun={runManager.currentRun}
          streamProgress={streamProgress}
          messages={currentMessages}
          chatMode={effectiveChatMode}
        />
      )}
    </div>
  );
}
