'use client';

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Send, Loader2, FlaskConical, Square } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { getApiBase } from '@/lib/api';
import { streamSSE, type NormalizedEvent } from '@/lib/stream-adapter';
import {
  useDiscoveryConversation,
  type DiscoveryMessage,
  type PlanData,
  type ToolData,
  type AnalysisData,
} from '@/stores/discoveryConversationStore';
import { PlanCard } from './PlanCard';
import { PipelineRunBlock } from './PipelineRunBlock';
import { ResultsSummary } from './ResultsSummary';

interface DiscoveryChatProps {
  sessionId: string;
  projectId: string;
  onCandidatesUpdate?: (candidates: any[]) => void;
  onStageChange?: (stage: string) => void;
}

export function DiscoveryChat({
  sessionId,
  projectId,
  onCandidatesUpdate,
  onStageChange,
}: DiscoveryChatProps) {
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const bootstrapTriggered = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Track the active pipeline run while it's executing
  const currentRunIdRef = useRef<string | null>(null);

  const conv = useDiscoveryConversation((s) => s.getOrCreate(sessionId));
  const addMessage = useDiscoveryConversation((s) => s.addMessage);
  const updateMessage = useDiscoveryConversation((s) => s.updateMessage);
  const setStreaming = useDiscoveryConversation((s) => s.setStreaming);
  const setStage = useDiscoveryConversation((s) => s.setStage);
  const setPendingPlan = useDiscoveryConversation((s) => s.setPendingPlan);
  const setCandidates = useDiscoveryConversation((s) => s.setCandidates);
  const appendThinking = useDiscoveryConversation((s) => s.appendThinking);
  const startPipelineRun = useDiscoveryConversation((s) => s.startPipelineRun);
  const addMessageToRun = useDiscoveryConversation((s) => s.addMessageToRun);
  const completePipelineRun = useDiscoveryConversation((s) => s.completePipelineRun);
  const toggleRunCollapsed = useDiscoveryConversation((s) => s.toggleRunCollapsed);
  const collapseOldRuns = useDiscoveryConversation((s) => s.collapseOldRuns);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conv.messages]);

  const handleSSEEvent = useCallback((event: NormalizedEvent) => {
    switch (event.type) {
      case 'session_update':
        setStage(sessionId, event.stage);
        onStageChange?.(event.stage);
        break;

      case 'thinking':
      case 'coordinator_thinking':
        appendThinking(sessionId, event.content);
        break;

      case 'message':
        addMessage(sessionId, {
          id: crypto.randomUUID(),
          role: 'assistant',
          type: 'text',
          content: event.content,
          timestamp: Date.now(),
        });
        break;

      case 'coordinator_question': {
        let questionContent = '';
        if (event.context) questionContent += `*${event.context}*\n\n`;
        questionContent += event.question;
        if (event.goalsSoFar?.length > 0) {
          questionContent += `\n\n**Goals so far:** ${event.goalsSoFar.map((g: string) => `\`${g}\``).join(', ')}`;
        }
        addMessage(sessionId, {
          id: crypto.randomUUID(),
          role: 'assistant',
          type: 'question',
          content: questionContent,
          timestamp: Date.now(),
          questionData: {
            question: event.question,
            options: event.options,
            context: event.context,
            turn: event.turn,
            goalsSoFar: event.goalsSoFar,
          },
        });
        break;
      }

      case 'coordinator_complete': {
        addMessage(sessionId, {
          id: crypto.randomUUID(),
          role: 'assistant',
          type: 'text',
          content: `**Session configured!** ${event.summary}\n\n**Extracted Goals:**\n${(event.extractedGoals || []).map((g: string) => `- ${g}`).join('\n')}`,
          timestamp: Date.now(),
        });
        setStage(sessionId, 'ready');
        onStageChange?.('ready');
        break;
      }

      case 'plan_proposed': {
        const planData: PlanData = {
          planId: event.planId,
          summary: event.summary,
          reasoning: event.reasoning,
          moleculeNotes: event.moleculeNotes,
          moleculeCount: event.moleculeCount,
          iteration: event.iteration,
          estimatedTotalSeconds: event.estimatedTotalSeconds,
          warnings: event.warnings,
          isDemoData: event.isDemoData,
          stages: event.stages,
          status: 'proposed',
        };
        setPendingPlan(sessionId, planData);
        addMessage(sessionId, {
          id: crypto.randomUUID(),
          role: 'assistant',
          type: 'plan',
          content: event.summary,
          timestamp: Date.now(),
          plan: planData,
        });
        break;
      }

      case 'tool_start': {
        const runId = currentRunIdRef.current;
        const msgId = `tool-${event.stageId}-${runId || 'solo'}`;
        const toolData: ToolData = {
          stageId: event.stageId,
          plugin: event.plugin,
          description: event.description,
          thinking: event.thinking,
          totalStages: event.totalStages,
          status: 'running',
        };
        const msg: DiscoveryMessage = {
          id: msgId,
          role: 'assistant',
          type: 'tool_start',
          content: `Running ${event.plugin}...`,
          timestamp: Date.now(),
          runId: runId ?? undefined,
          tool: toolData,
        };
        addMessage(sessionId, msg);
        if (runId) addMessageToRun(sessionId, runId, msgId);
        break;
      }

      case 'tool_complete': {
        const runId = currentRunIdRef.current;
        const msgId = `tool-${event.stageId}-${runId || 'solo'}`;
        const updatedTool: ToolData = {
          stageId: event.stageId,
          plugin: event.plugin,
          description: '',
          summary: event.summary,
          stats: event.stats,
          totalStages: event.totalStages,
          candidatesSoFar: event.candidatesSoFar,
          error: event.error,
          status: event.error ? 'error' : 'complete',
        };
        updateMessage(sessionId, msgId, {
          type: 'tool_complete',
          content: event.summary,
          tool: updatedTool,
        });
        break;
      }

      case 'pipeline_complete': {
        const pEvt = event as any;
        const candidates: any[] = pEvt.candidates || [];
        const stagesCompleted = pEvt.stages_completed ?? pEvt.stagesCompleted ?? 0;

        if (candidates.length > 0) {
          setCandidates(sessionId, candidates);
          onCandidatesUpdate?.(candidates);
        }

        // Complete the active run
        const runId = currentRunIdRef.current;
        if (runId) {
          completePipelineRun(sessionId, runId, stagesCompleted, candidates.length);
          currentRunIdRef.current = null;
        }
        break;
      }

      case 'analysis': {
        const analysisData: AnalysisData = {
          keyFindings: event.keyFindings || [],
          topCandidates: event.topCandidates || [],
          concerns: event.concerns || [],
          recommendations: event.recommendations || [],
          missingCapabilities: event.missingCapabilities || [],
        };
        // Tag with the last completed run (currentRunId is null after pipeline_complete)
        const latestRun = conv.pipelineRuns[conv.pipelineRuns.length - 1];
        const runId = latestRun?.runId ?? null;
        const msgId = crypto.randomUUID();
        const msg: DiscoveryMessage = {
          id: msgId,
          role: 'assistant',
          type: 'analysis',
          content: 'Analysis complete',
          timestamp: Date.now(),
          runId: runId ?? undefined,
          analysis: analysisData,
        };
        addMessage(sessionId, msg);
        if (runId) addMessageToRun(sessionId, runId, msgId);
        break;
      }

      case 'recommendation':
        // Handled inside analysis; skip duplicate render
        break;

      case 'error':
        addMessage(sessionId, {
          id: crypto.randomUUID(),
          role: 'assistant',
          type: 'error',
          content: event.message,
          timestamp: Date.now(),
        });
        break;

      case 'routing':
      default:
        break;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, addMessage, updateMessage, setStage, setPendingPlan, setCandidates,
      appendThinking, startPipelineRun, addMessageToRun, completePipelineRun,
      onStageChange, onCandidatesUpdate, conv.pipelineRuns]);

  const sendMessage = useCallback(async (
    messageText: string,
    action?: string,
  ) => {
    if (conv.isStreaming) return;

    const trimmed = messageText.trim();
    if (trimmed) {
      addMessage(sessionId, {
        id: crypto.randomUUID(),
        role: 'user',
        type: 'text',
        content: trimmed,
        timestamp: Date.now(),
      });
    }

    setInput('');
    setStreaming(sessionId, true);
    abortRef.current = new AbortController();

    const url = `${getApiBase()}/api/discovery/${sessionId}/chat`;
    const body: Record<string, any> = {
      message: trimmed || null,
      project_id: projectId,
    };
    if (action) body.action = action;

    try {
      await streamSSE(url, body, handleSSEEvent, {
        signal: abortRef.current.signal,
        timeout: 600_000,
      });
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        addMessage(sessionId, {
          id: crypto.randomUUID(),
          role: 'assistant',
          type: 'error',
          content: `Connection error: ${err?.message || 'Unknown error'}. Try sending again.`,
          timestamp: Date.now(),
        });
      }
    } finally {
      setStreaming(sessionId, false);
    }
  }, [sessionId, projectId, conv.isStreaming, addMessage, setStreaming, handleSSEEvent]);

  const handleAcceptPlan = useCallback(() => {
    if (conv.pendingPlan) {
      const planMsgIdx = conv.messages.findIndex(
        (m) => m.type === 'plan' && m.plan?.planId === conv.pendingPlan?.planId,
      );
      if (planMsgIdx >= 0) {
        updateMessage(sessionId, conv.messages[planMsgIdx].id, {
          plan: { ...conv.messages[planMsgIdx].plan!, status: 'accepted' },
        });
      }
      // Start a new pipeline run group and collapse older ones
      const iteration = conv.pipelineRuns.length + 1;
      const runId = startPipelineRun(sessionId, iteration);
      currentRunIdRef.current = runId;
      collapseOldRuns(sessionId);
      setPendingPlan(sessionId, null);
    }
    sendMessage('', 'accept_plan');
  }, [sessionId, conv.pendingPlan, conv.messages, conv.pipelineRuns.length,
      sendMessage, updateMessage, setPendingPlan, startPipelineRun, collapseOldRuns]);

  const handleRejectPlan = useCallback(() => {
    if (conv.pendingPlan) {
      const planMsgIdx = conv.messages.findIndex(
        (m) => m.type === 'plan' && m.plan?.planId === conv.pendingPlan?.planId,
      );
      if (planMsgIdx >= 0) {
        updateMessage(sessionId, conv.messages[planMsgIdx].id, {
          plan: { ...conv.messages[planMsgIdx].plan!, status: 'rejected' },
        });
      }
      setPendingPlan(sessionId, null);
    }
    sendMessage('', 'reject_plan');
  }, [sessionId, conv.pendingPlan, conv.messages, sendMessage, updateMessage, setPendingPlan]);

  const handleSubmit = useCallback(() => {
    if (!input.trim() && !conv.isStreaming) return;
    sendMessage(input);
  }, [input, conv.isStreaming, sendMessage]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(sessionId, false);
  }, [sessionId, setStreaming]);

  const handleTextareaInput = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 128) + 'px';
  }, []);

  useEffect(() => {
    if (!bootstrapTriggered.current) {
      bootstrapTriggered.current = true;
      if (conv.messages.length === 0) {
        sendMessage('');
      }
    }
  }, [sendMessage, conv.messages.length]);

  // Build a set of message IDs that belong to a pipeline run (rendered inside PipelineRunBlock)
  const runMessageIds = useMemo(() => {
    const ids = new Set<string>();
    for (const run of conv.pipelineRuns) {
      for (const id of run.messageIds) ids.add(id);
    }
    return ids;
  }, [conv.pipelineRuns]);

  // Track which runs have been "anchored" in render order
  const renderedRunIds = new Set<string>();

  // Map messages to their run
  const msgRunMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const run of conv.pipelineRuns) {
      for (const id of run.messageIds) map.set(id, run.runId);
    }
    return map;
  }, [conv.pipelineRuns]);

  // Build run→messages lookup
  const runMessages = useMemo(() => {
    const map = new Map<string, DiscoveryMessage[]>();
    for (const run of conv.pipelineRuns) {
      const msgs = run.messageIds
        .map((id) => conv.messages.find((m) => m.id === id))
        .filter((m): m is DiscoveryMessage => m != null);
      map.set(run.runId, msgs);
    }
    return map;
  }, [conv.pipelineRuns, conv.messages]);

  const renderMessage = (msg: DiscoveryMessage) => {
    // Messages that belong to a pipeline run are rendered inside PipelineRunBlock
    if (runMessageIds.has(msg.id)) {
      const runId = msgRunMap.get(msg.id);
      if (!runId || renderedRunIds.has(runId)) return null;
      renderedRunIds.add(runId);
      const run = conv.pipelineRuns.find((r) => r.runId === runId)!;
      const msgs = runMessages.get(runId) || [];
      return (
        <div key={`run-${runId}`} className="w-full">
          <PipelineRunBlock
            run={run}
            messages={msgs}
            onToggle={() => toggleRunCollapsed(sessionId, runId)}
          />
        </div>
      );
    }

    if (msg.role === 'user') {
      return (
        <div key={msg.id} className="flex justify-end">
          <div className="max-w-[85%] rounded-2xl bg-emerald-600 text-white px-4 py-3 text-sm rounded-br-sm">
            {msg.content}
          </div>
        </div>
      );
    }

    switch (msg.type) {
      case 'thinking':
        return (
          <div key={msg.id} className="flex justify-start">
            <div className="max-w-[90%] rounded-2xl bg-emerald-500/8 text-emerald-500/70 border border-emerald-500/15 px-4 py-2.5 text-[11px] font-mono rounded-bl-sm whitespace-pre-wrap leading-relaxed">
              {msg.content}
            </div>
          </div>
        );

      case 'plan':
        if (msg.plan) {
          return (
            <div key={msg.id} className="w-full">
              <PlanCard
                plan={msg.plan}
                onAccept={handleAcceptPlan}
                onReject={handleRejectPlan}
                disabled={conv.isStreaming}
              />
            </div>
          );
        }
        break;

      // Standalone tool cards (outside a run group — should be rare)
      case 'tool_start':
      case 'tool_complete':
        // These are handled inside PipelineRunBlock; only reach here if untagged
        return null;

      case 'analysis':
        // If analysis is tagged to a run, it's rendered in PipelineRunBlock
        if (msg.analysis && !msg.runId) {
          return (
            <div key={msg.id} className="w-full">
              <ResultsSummary analysis={msg.analysis} />
            </div>
          );
        }
        return null;

      case 'error':
        return (
          <div key={msg.id} className="flex justify-start">
            <div className="max-w-[85%] rounded-2xl bg-red-500/10 text-red-400 border border-red-500/20 px-4 py-3 text-sm rounded-bl-sm">
              {msg.content}
            </div>
          </div>
        );

      default:
        return (
          <div key={msg.id} className="flex justify-start">
            <div className="max-w-[85%] rounded-2xl bg-card border border-border text-foreground px-4 py-3 text-sm rounded-bl-sm prose prose-sm prose-invert max-w-none [&>p]:mb-2 [&>p:last-child]:mb-0 [&>ul]:mb-2 [&>ol]:mb-2 [&_code]:text-emerald-400 [&_code]:bg-emerald-500/10 [&_code]:px-1 [&_code]:rounded [&_strong]:text-foreground">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            </div>
          </div>
        );
    }

    return null;
  };

  const stageBadge = (() => {
    switch (conv.stage) {
      case 'setup':     return { label: 'Setup',    color: 'bg-amber-500/10 text-amber-500 border-amber-500/20' };
      case 'ready':     return { label: 'Ready',    color: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' };
      case 'executing': return { label: 'Running',  color: 'bg-orange-500/10 text-orange-500 border-orange-500/20' };
      case 'complete':  return { label: 'Complete', color: 'bg-blue-500/10 text-blue-500 border-blue-500/20' };
      default:          return { label: 'Discovery',color: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' };
    }
  })();

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex h-11 shrink-0 items-center gap-2 border-b border-border bg-surface/30 px-3">
        <FlaskConical className="h-3.5 w-3.5 text-emerald-500" />
        <span className="text-xs font-medium text-foreground/80">Discovery</span>
        {conv.pipelineRuns.length > 0 && (
          <span className="text-[10px] text-muted-foreground/40">
            {conv.pipelineRuns.length} run{conv.pipelineRuns.length !== 1 ? 's' : ''}
          </span>
        )}
        <span className={`ml-auto rounded-full border px-2 py-0.5 text-[10px] font-medium ${stageBadge.color}`}>
          {stageBadge.label}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {conv.messages.map(renderMessage)}
        {conv.isStreaming && conv.messages[conv.messages.length - 1]?.type !== 'thinking' && (
          <div className="flex justify-start">
            <div className="rounded-2xl bg-emerald-500/8 border border-emerald-500/15 px-4 py-2.5 rounded-bl-sm">
              <Loader2 className="h-4 w-4 animate-spin text-emerald-500" />
            </div>
          </div>
        )}
        <div ref={scrollRef} />
      </div>

      <div className="shrink-0 border-t border-border p-3">
        <div className="relative flex items-end rounded-xl border border-border bg-surface shadow-sm focus-within:border-emerald-500/50 transition-colors">
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleTextareaInput}
            onKeyDown={handleKeyDown}
            placeholder={
              conv.stage === 'setup'    ? 'Answer the question above...' :
              conv.stage === 'ready'    ? 'Ask a question or say "run" to start...' :
                                          'Ask about results or say "run" for another iteration...'
            }
            disabled={conv.isStreaming}
            rows={1}
            className="flex-1 resize-none bg-transparent px-4 py-3 text-sm outline-none disabled:opacity-50 max-h-32"
            style={{ minHeight: '44px' }}
          />
          {conv.isStreaming ? (
            <button
              onClick={handleStop}
              className="m-2 rounded-lg bg-red-500/80 p-2 text-white transition-colors hover:bg-red-600 shrink-0"
              title="Stop generation"
            >
              <Square className="h-4 w-4" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!input.trim()}
              className="m-2 rounded-lg bg-emerald-600 p-2 text-white transition-colors hover:bg-emerald-700 disabled:opacity-40 shrink-0"
            >
              <Send className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
