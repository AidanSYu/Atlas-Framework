'use client';

import React, { useEffect, useCallback, useState, useRef } from 'react';
import { useChatStore } from '@/stores/chatStore';
import { useRunStore } from '@/stores/runStore';
import { useDiscoveryStore } from '@/stores/discoveryStore';
import type { CandidateArtifact, PredictedProperty } from '@/lib/discovery-types';
import { useRunManager, type ChatMode } from '@/hooks/useRunManager';
import { api, type GroundingEvent, type SpectrumUploadResponse } from '@/lib/api';
import type { NormalizedEvent } from '@/lib/stream-adapter';
import {
  Brain,
  BookOpen,
  Trash2,
  AlertCircle,
  Network,
  Beaker,
  ShieldCheck,
  ChevronDown,
  Loader2,
} from 'lucide-react';
import { AgentWorkbench } from '@/components/AgentWorkbench';
import { DiscoveryWorkbench } from '@/components/DiscoveryWorkbench';
import { ConversationView } from './ConversationView';
import { RunProgressDisplay, type StreamProgress } from './RunProgressDisplay';
import { CommandSurface } from './CommandSurface';



function mapDiscoveryCandidatesToArtifacts(candidates: Array<{ smiles: string; properties?: Record<string, any>; toxicity?: any }>): CandidateArtifact[] {
  return candidates.map((c, i) => {
    const properties: PredictedProperty[] = [];
    if (c.properties && typeof c.properties === 'object') {
      for (const [name, value] of Object.entries(c.properties)) {
        if (value === undefined) continue;
        properties.push({
          name,
          value: value as number | string | boolean,
          passesConstraint: null,
          model: 'discovery',
        });
      }
    }
    return {
      id: crypto.randomUUID(),
      rank: i + 1,
      score: 0.8,
      renderType: 'molecule_2d' as const,
      renderData: c.smiles,
      properties,
      sourceReasoning: 'From Discovery chat run.',
      sourceDocumentIds: [],
      status: 'pending' as const,
    };
  });
}

// ---------------------------------------------------------------------------
// Agent dropdown config
// ---------------------------------------------------------------------------

const AGENT_OPTIONS: { mode: ChatMode; label: string; Icon: typeof BookOpen; color: string }[] = [
  { mode: 'librarian', label: 'Librarian', Icon: BookOpen, color: 'text-primary' },
  { mode: 'cortex', label: 'Cortex', Icon: Brain, color: 'text-accent' },
  { mode: 'moe', label: 'MoE', Icon: Network, color: 'text-blue-500' },
  { mode: 'discovery', label: 'Discovery', Icon: Beaker, color: 'text-orange-500' },
];

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ChatShellProps {
  onCitationClick: (filename: string, page: number, docId?: string) => void;
  projectId?: string;
  autoSubmitQuery?: string | null;
  onAutoSubmitConsumed?: () => void;
  onOpenRunHistory?: () => void;
  onViewRunDetails?: (runId: string) => void;
  /** Lock to a specific mode (hides agent dropdown, disables auto-routing) */
  lockedMode?: ChatMode;
  /** Override session ID (e.g. for coordinator embedded in DiscoveryWorkspaceTab) */
  coordinatorSessionId?: string;
  /** Callback when coordinator finishes bootstrapping */
  onCoordinatorComplete?: (goals: string[]) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ChatShell({
  onCitationClick,
  projectId,
  autoSubmitQuery,
  onAutoSubmitConsumed,
  onOpenRunHistory,
  onViewRunDetails,
  lockedMode,
  coordinatorSessionId,
  onCoordinatorComplete,
}: ChatShellProps) {
  const runManager = useRunManager(projectId || '');

  const {
    currentInput: globalInput,
    pendingQuestion,
    pendingHypotheses,
    addMessage,
    setCurrentInput: setGlobalInput,
    setChatMode,
    clearCurrentChat,
    setActiveProject,
    setPendingQuestion,
    setPendingHypotheses,
    getActiveThread,
  } = useChatStore();

  const [localInput, setLocalInput] = useState(globalInput || '');

  const activeThread = useChatStore((s) => s.threads.find((t) => t.id === s.activeThreadId) ?? null);
  const chatMode: ChatMode = activeThread?.chatMode ?? 'librarian';
  const sessionId = activeThread?.sessionId ?? '';

  const discoveryBackendSessionId = useDiscoveryStore((s) => s.sessionId);

  // Phase 4: Coordinator mode
  const isCoordinatorMode = lockedMode === 'coordinator';
  const effectiveChatMode: ChatMode = lockedMode || chatMode;
  const effectiveSessionIdForCoordinator = coordinatorSessionId || sessionId;

  // We need an isolated message state for the Coordinator so it doesn't pollute or show Librarian chat history
  const [coordinatorMessages, setCoordinatorMessages] = useState<any[]>([]);
  const currentMessages = isCoordinatorMode ? coordinatorMessages : (activeThread?.messages ?? []);

  // Coordinator question state (multiple-choice options from HITL interrupt)
  const [coordinatorQuestion, setCoordinatorQuestion] = useState<{
    question: string;
    options: string[];
    context?: string;
    turn: number;
    goalsSoFar: string[];
  } | null>(null);

  // Auto-trigger coordinator on mount
  const coordinatorTriggered = useRef(false);

  const [agentDropdownOpen, setAgentDropdownOpen] = useState(false);
  const agentDropdownRef = useRef<HTMLDivElement>(null);

  // Close agent dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (agentDropdownRef.current && !agentDropdownRef.current.contains(e.target as Node)) {
        setAgentDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  useEffect(() => {
    setActiveProject(projectId || null);
  }, [projectId, setActiveProject]);

  useEffect(() => {
    if (projectId) {
      useRunStore.getState().loadHistory(projectId);
    }
  }, [projectId]);

  // Legacy bridge state
  const [streamProgress, setStreamProgress] = useState<StreamProgress | null>(null);
  const [groundingMap, setGroundingMap] = useState<Map<string, GroundingEvent>>(new Map());
  const [streamingText, setStreamingText] = useState<string>('');
  const [spectrumFile, setSpectrumFile] = useState<SpectrumUploadResponse | null>(null);
  const [isUploadingSpectrum, setIsUploadingSpectrum] = useState(false);



  useEffect(() => {
    if (pendingQuestion) {
      setLocalInput(pendingQuestion);
      setGlobalInput(pendingQuestion);
      setPendingQuestion(null);
    }
  }, [pendingQuestion, setPendingQuestion, setGlobalInput]);



  // -------------------------------------------------------------------------
  // Auto-submit from OmniBar
  // -------------------------------------------------------------------------

  useEffect(() => {
    if (autoSubmitQuery && !runManager.isRunning && projectId) {
      setLocalInput(autoSubmitQuery);
      setGlobalInput(autoSubmitQuery);
      onAutoSubmitConsumed?.();
      const timer = setTimeout(() => {
        handleSubmitWithContent(autoSubmitQuery);
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [autoSubmitQuery, projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // Auto-trigger coordinator on mount (empty message starts corpus scan)
  // -------------------------------------------------------------------------

  useEffect(() => {
    if (!isCoordinatorMode || !projectId || coordinatorTriggered.current) return;

    // BUGFIX: Clear any stale global run state before starting coordinator
    runManager.cancelCurrentRun();

    coordinatorTriggered.current = true;

    // BUGFIX: Immediate execution instead of setTimeout to avoid race conditions
    const bootstrapCoordinator = async () => {
      try {
        await handleSubmitWithContent('');
      } catch (error) {
        console.error('Coordinator bootstrap failed:', error);
        coordinatorTriggered.current = false; // Allow retry on error
      }
    };

    bootstrapCoordinator();
  }, [isCoordinatorMode, projectId, runManager]); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // onProgress bridge
  // -------------------------------------------------------------------------

  const buildOnProgress = useCallback((): ((event: NormalizedEvent) => void) => {
    return (event: NormalizedEvent) => {
      switch (event.type) {
        case 'routing':
          setStreamProgress((prev) => ({
            ...(prev || { currentNode: 'router', message: 'Routing...', thinkingSteps: [], evidenceFound: 0 }),
            routing: { brain: event.mode, intent: event.intent },
            message: `Routing to ${event.mode} (${event.intent})...`,
            thinkingSteps: prev ? [...prev.thinkingSteps, `Request routed to ${event.mode}`] : [`Request routed to ${event.mode}`],
          }));
          break;

        case 'progress':
          setStreamProgress((prev) => prev ? { ...prev, currentNode: event.node, message: event.message } : null);
          break;

        case 'thinking':
          setStreamProgress((prev) => prev ? { ...prev, thinkingSteps: [...prev.thinkingSteps, event.content] } : null);
          break;

        case 'coordinator_thinking':
          // BUGFIX: Handle coordinator progress events to show "Scanning corpus..." etc.
          setStreamProgress((prev) => ({
            ...(prev || { currentNode: 'coordinator', message: '', thinkingSteps: [], evidenceFound: 0 }),
            currentNode: 'coordinator',
            message: event.content || 'Processing...',
            thinkingSteps: prev ? [...prev.thinkingSteps, event.content] : [event.content],
          }));
          break;

        case 'graph_analysis':
          setStreamProgress((prev) => prev ? { ...prev, graphData: event.data, thinkingSteps: [...prev.thinkingSteps, 'Graph structure analyzed.'] } : null);
          break;

        case 'evidence':
          setStreamProgress((prev) => prev ? { ...prev, evidenceFound: prev.evidenceFound + event.count } : null);
          break;

        case 'grounding': {
          const ge: GroundingEvent = { claim: event.claim, status: event.status as any, confidence: event.confidence };
          setGroundingMap((prev) => {
            const next = new Map(prev);
            next.set(event.claim, ge);
            return next;
          });
          break;
        }

        case 'chunk':
          setStreamingText((prev) => prev + event.content);
          break;

        case 'tool_call':
          setStreamProgress((prev) => prev ? {
            ...prev,
            thinkingSteps: [...prev.thinkingSteps, `Calling **${event.tool}**(${JSON.stringify(event.input).slice(0, 80)}...)`],
            message: `Executing ${event.tool}...`,
            activeTool: { name: event.tool, input: event.input },
            currentNode: 'execute',
          } : null);
          break;

        case 'tool_result':
          setStreamProgress((prev) => {
            if (!prev) return null;
            const output = event.output;
            let summary = JSON.stringify(output).slice(0, 100);
            const updatedCandidates = [...(prev.candidates || [])];

            if (event.tool === 'predict_properties' && output.valid) {
              summary = `MW: ${output.MolWt}, LogP: ${output.LogP}, QED: ${output.QED}`;
              const idx = updatedCandidates.findIndex((c: any) => c.smiles === output.smiles);
              if (idx >= 0) updatedCandidates[idx] = { ...updatedCandidates[idx], properties: output };
              else updatedCandidates.push({ smiles: output.smiles, properties: output });
            } else if (event.tool === 'check_toxicity' && output.valid) {
              summary = output.clean ? 'Clean (no alerts)' : `${output.alert_count} alert(s)`;
              const idx = updatedCandidates.findIndex((c: any) => c.smiles === output.smiles);
              if (idx >= 0) updatedCandidates[idx] = { ...updatedCandidates[idx], toxicity: output };
              else updatedCandidates.push({ smiles: output.smiles, toxicity: output });
            } else if (event.tool === 'verify_spectrum' && output.valid) {
              summary = `Match: ${output.match_score != null ? (output.match_score * 100).toFixed(0) + '%' : 'N/A'}, ${output.peak_count} peaks observed`;
            } else if (event.tool === 'search_literature') {
              summary = `Found ${output.total_results} passages`;
            }

            return {
              ...prev,
              thinkingSteps: [...prev.thinkingSteps, `Result: ${summary}`],
              message: 'Reasoning about results...',
              currentNode: 'think',
              activeTool: undefined,
              toolResults: [...(prev.toolResults || []), { tool: event.tool, output }],
              candidates: updatedCandidates,
            };
          });
          break;

        case 'complete':
          setStreamingText('');
          break;

        case 'hypotheses':
          break;
      }
    };
  }, []);

  // -------------------------------------------------------------------------
  // handleSubmitWithContent
  // -------------------------------------------------------------------------

  const handleSubmitWithContent = useCallback(async (userContent: string, selectedHypothesis?: boolean) => {
    // Coordinator mode: always allow submission (user is answering HITL questions).
    // The run may still be "running" from the previous SSE stream — that's fine,
    // startRun() will abort the old stream and create a new one.
    const isCoordinatorBootstrap = isCoordinatorMode && !userContent.trim();
    if (!isCoordinatorMode) {
      if ((!userContent.trim()) || (runManager.isRunning) || !projectId) return;
    } else {
      if (!isCoordinatorBootstrap && !userContent.trim()) return;
      if (!projectId) return;
    }

    const content = userContent.trim();
    const effectiveMode = isCoordinatorMode ? 'coordinator' as ChatMode : chatMode;

    const effectiveSessionId = effectiveMode === 'coordinator'
      ? effectiveSessionIdForCoordinator
      : effectiveMode === 'discovery' && discoveryBackendSessionId
        ? discoveryBackendSessionId
        : sessionId;

    // Only add user message if there's actual content (skip for coordinator initial trigger)
    if (content) {
      if (isCoordinatorMode) {
        setCoordinatorMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', content, timestamp: Date.now() }]);
      } else {
        addMessage({ role: 'user', content });
      }
    }
    setStreamProgress(null);
    setStreamingText('');

    const onProgress = buildOnProgress();

    try {
      if (effectiveMode === 'librarian') {
        setStreamProgress({ currentNode: 'librarian', message: 'Searching document library...', thinkingSteps: [], evidenceFound: 0 });
        const abortController = new AbortController();
        try {
          const response = await runManager.submitLibrarian(content, projectId, abortController.signal);
          addMessage({
            role: 'assistant',
            content: response.answer,
            agent: 'librarian',
            citations: response.citations,
            librarianMetadata: { reasoning: response.reasoning, relationships: response.relationships, contextSources: response.context_sources },
            runId: runManager.currentRun?.id,
            followUps: response.follow_ups,
          });
        } catch (error: any) {
          if (error?.name === 'AbortError') return;
          throw error;
        }
      } else if (effectiveMode === 'moe' && !selectedHypothesis) {
        setStreamProgress({ currentNode: 'supervisor', message: 'Analyzing query and planning approach...', thinkingSteps: [], evidenceFound: 0 });
        try {
          const { result, cancelled } = await runManager.submitQuery(content, 'moe', {
            sessionId: effectiveSessionId,
            isMoeHypotheses: true,
            onProgress,
          });
          if (!cancelled && result?.items) {
            setPendingHypotheses(result.items);
            addMessage({
              role: 'assistant',
              content: 'I have analyzed your query and generated several distinct research hypotheses. Please select the one you would like me to investigate.',
              agent: 'moe',
              brainActivity: { brain: 'MoE Supervisor', trace: ['Generated research hypotheses.'], evidence: [] },
              runId: runManager.currentRun?.id,
            });
          }
        } catch (error: any) {
          if (error?.name === 'AbortError' || error?.message?.includes('abort')) return;
          console.error('MoE Hypotheses Stream Error:', error);
          setStreamProgress(null);
          addMessage({ role: 'assistant', content: 'Failed to generate hypotheses interactively. Proceeding with standard research...', agent: 'moe' });
          handleSubmitWithContent(content, true);
          return;
        }
      } else if (effectiveMode === 'moe' && selectedHypothesis) {
        setPendingHypotheses(null);
        setStreamProgress({ currentNode: 'supervisor', message: 'Executing MoE plan on selected hypothesis...', thinkingSteps: [], evidenceFound: 0 });
        try {
          const { result, cancelled } = await runManager.submitQuery(content, 'moe', {
            sessionId: effectiveSessionId,
            onProgress,
          });
          if (!cancelled && result) {
            addMessage({
              role: 'assistant',
              content: result.final_answer || result.hypothesis || 'No answer generated.',
              agent: 'moe',
              citations: result.evidence?.map((e: any) => ({ source: e.source, page: e.page, relevance: e.relevance, text: e.text || e.excerpt })),
              brainActivity: { brain: 'MoE Supervisor', trace: result.reasoning_trace || [], evidence: result.evidence || [], confidenceScore: result.confidence_score },
              runId: runManager.currentRun?.id,
            });
          }
        } catch (error: any) {
          if (error?.name === 'AbortError' || error?.message?.includes('abort')) return;
          console.error('MoE Execute Stream Error:', error);
          setStreamProgress({ currentNode: 'fallback', message: 'Processing MoE (non-streaming)...', thinkingSteps: [], evidenceFound: 0 });
          const result = await api.runMoE(content, projectId);
          addMessage({
            role: 'assistant',
            content: result.hypothesis || result.final_answer || 'No analysis generated.',
            agent: 'moe',
            citations: result.evidence?.map((e: any) => ({ source: e.source, page: e.page, relevance: e.relevance, text: e.excerpt || e.text })),
            brainActivity: { brain: 'MoE Supervisor', trace: result.reasoning_trace || [], evidence: result.evidence || [], confidenceScore: result.confidence_score },
            runId: runManager.currentRun?.id,
          });
        }
      } else if (effectiveMode === 'cortex') {
        setStreamProgress({ currentNode: 'router', message: 'Analyzing query...', thinkingSteps: [], evidenceFound: 0 });
        try {
          const { result, cancelled } = await runManager.submitQuery(content, 'cortex', {
            sessionId: effectiveSessionId,
            onProgress,
          });
          if (!cancelled && result) {
            addMessage({
              role: 'assistant',
              content: result.hypothesis || 'No analysis generated.',
              agent: 'cortex',
              citations: result.evidence?.map((e: any) => ({ source: e.source, page: e.page, relevance: e.relevance, text: e.excerpt })),
              brainActivity: {
                brain: result.brain_used,
                trace: result.reasoning_trace || [],
                evidence: result.evidence || [],
                confidenceScore: result.confidence_score,
                iterations: result.iterations,
                contradictions: result.contradictions,
              },
              runId: runManager.currentRun?.id,
            });
          }
        } catch (error: any) {
          if (error?.name === 'AbortError' || error?.message?.includes('abort')) return;
          setStreamProgress({ currentNode: 'fallback', message: 'Processing (non-streaming)...', thinkingSteps: [], evidenceFound: 0 });
          const result = await api.runSwarm(content, projectId);
          addMessage({
            role: 'assistant',
            content: result.hypothesis || 'No analysis generated.',
            agent: 'cortex',
            citations: result.evidence?.map((e: any) => ({ source: e.source, page: e.page, relevance: e.relevance, text: e.excerpt })),
            brainActivity: { brain: result.brain_used, trace: result.reasoning_trace || [], evidence: result.evidence || [], confidenceScore: result.confidence_score, iterations: result.iterations, contradictions: result.contradictions },
            runId: runManager.currentRun?.id,
          });
        }
      } else if (effectiveMode === 'discovery') {
        setStreamProgress({ currentNode: 'router', message: 'Routing to Discovery OS...', thinkingSteps: [], evidenceFound: 0, toolResults: [], candidates: [] });
        try {
          const { result, cancelled } = await runManager.submitQuery(content, 'discovery', {
            sessionId: effectiveSessionId,
            spectrumFilePath: spectrumFile?.file_path,
            onProgress,
          });
          setSpectrumFile(null);
          if (!cancelled && result) {
            addMessage({
              role: 'assistant',
              content: result.hypothesis || 'No analysis generated.',
              agent: 'discovery',
              citations: result.evidence?.map((e: any) => ({ source: e.source, page: e.page, relevance: e.relevance, text: e.excerpt })),
              brainActivity: { brain: result.brain_used, trace: result.reasoning_trace || [], evidence: result.evidence || [], confidenceScore: result.confidence_score, iterations: result.iterations, candidates: result.candidates },
              runId: runManager.currentRun?.id,
            });
            if (result.candidates?.length && projectId) {
              const disco = useDiscoveryStore.getState();
              const activeEpochId = disco.activeEpochId;
              if (activeEpochId) {
                const artifacts = mapDiscoveryCandidatesToArtifacts(result.candidates);
                disco.setCandidatesForEpoch(activeEpochId, artifacts);
              }
            }
          }
        } catch (error: any) {
          if (error?.name === 'AbortError' || error?.message?.includes('abort')) return;
          setStreamProgress({ currentNode: 'fallback', message: 'Processing (non-streaming)...', thinkingSteps: [], evidenceFound: 0 });
          const result = await api.runDiscovery(content, projectId);
          addMessage({
            role: 'assistant',
            content: result.hypothesis || 'No analysis generated.',
            agent: 'discovery',
            brainActivity: { brain: result.brain_used, trace: result.reasoning_trace || [], evidence: result.evidence || [], confidenceScore: result.confidence_score, iterations: result.iterations, candidates: result.candidates },
            runId: runManager.currentRun?.id,
          });
          if (result.candidates?.length && projectId) {
            const disco = useDiscoveryStore.getState();
            const activeEpochId = disco.activeEpochId;
            if (activeEpochId) {
              const artifacts = mapDiscoveryCandidatesToArtifacts(result.candidates);
              disco.setCandidatesForEpoch(activeEpochId, artifacts);
            }
          }
        }
      } else if (effectiveMode === 'coordinator') {
        // Phase 4: Coordinator HITL flow
        setStreamProgress({
          currentNode: 'coordinator',
          message: content ? 'Processing your answer...' : 'Starting coordinator...',
          thinkingSteps: [],
          evidenceFound: 0,
        });
        try {
          const { result, cancelled } = await runManager.submitQuery(
            content,  // empty string triggers initial corpus scan
            'coordinator',
            {
              sessionId: effectiveSessionId,
              onProgress: (event: NormalizedEvent) => {
                onProgress(event);

                if (event.type === 'coordinator_thinking') {
                  // Stream thinking steps as chat messages so user sees live activity
                  setCoordinatorMessages(prev => {
                    // Merge consecutive thinking into one message to avoid spam
                    const last = prev[prev.length - 1];
                    if (last?.role === 'assistant' && last?.isThinking) {
                      const updated = [...prev];
                      updated[updated.length - 1] = {
                        ...last,
                        content: last.content + '\n' + event.content,
                      };
                      return updated;
                    }
                    return [...prev, {
                      id: crypto.randomUUID(),
                      role: 'assistant',
                      content: event.content,
                      agent: 'coordinator' as any,
                      timestamp: Date.now(),
                      isThinking: true,
                    }];
                  });
                } else if (event.type === 'coordinator_question') {
                  setCoordinatorQuestion(event);
                  // Build a rich question message with context
                  let questionContent = '';
                  if (event.context) {
                    questionContent += `*${event.context}*\n\n`;
                  }
                  questionContent += event.question;
                  if (event.goalsSoFar?.length > 0) {
                    questionContent += `\n\n**Goals so far:** ${event.goalsSoFar.map((g: string) => `\`${g}\``).join(', ')}`;
                  }
                  setCoordinatorMessages(prev => [...prev, {
                    id: crypto.randomUUID(),
                    role: 'assistant',
                    content: questionContent,
                    agent: 'coordinator' as any,
                    timestamp: Date.now()
                  }]);
                } else if (event.type === 'coordinator_complete') {
                  setCoordinatorQuestion(null);
                  onCoordinatorComplete?.(event.extractedGoals);

                  let completionMsg = `**Session configured!** ${event.summary}\n\n`;

                  if (event.extractedGoals && event.extractedGoals.length > 0) {
                    completionMsg += `**Extracted Goals:**\n${event.extractedGoals.map((g: string) => `- ${g}`).join('\n')}\n\n`;
                  }

                  if (event.corpusEntities && event.corpusEntities.length > 0) {
                    completionMsg += `**Key Entities Found:** ${event.corpusEntities.slice(0, 10).join(', ')}${event.corpusEntities.length > 10 ? `, +${event.corpusEntities.length - 10} more` : ''}\n\n`;
                  }

                  if (event.corpusSummary) {
                    completionMsg += `**Files Written:**\n- \`SESSION_CONTEXT.md\` — living session context\n- \`CONSTRAINTS.md\` — extracted constraints\n- \`HYPOTHESES.md\` — working hypotheses\n- \`RESEARCH_NOTES.md\` — corpus summary\n- \`FINDINGS.md\` — execution log (empty)\n`;
                  }

                  setCoordinatorMessages(prev => [...prev, {
                    id: crypto.randomUUID(),
                    role: 'assistant',
                    content: completionMsg,
                    agent: 'coordinator' as any,
                    timestamp: Date.now()
                  }]);
                }
              },
            },
          );
          if (cancelled) return;
        } catch (error: any) {
          if (error?.name === 'AbortError' || error?.message?.includes('abort')) return;
          console.error('Coordinator error:', error);
          const errMsg = error?.message || 'Unknown error';
          setCoordinatorMessages(prev => [...prev, {
            id: crypto.randomUUID(),
            role: 'assistant',
            content:
              `Coordinator hit an error but your session is still safe.\n\n` +
              `Details: ${errMsg}\n\n` +
              `Try clicking send again to resume. If this repeats, verify API keys in \`config/.env\` and then retry.`,
            agent: 'coordinator' as any,
            timestamp: Date.now()
          }]);
        }
      }
    } catch (error: any) {
      if (error?.name === 'AbortError' || error?.message?.includes('abort')) return;
      console.error('Chat error:', error);
      const failedRun = runManager.currentRun;
      const category = failedRun?.error?.category || 'backend_runtime';
      const errorMsg = error?.message || 'An unexpected error occurred.';
      addMessage({
        role: 'assistant',
        content: errorMsg,
        runId: failedRun?.id,
        errorInfo: {
          category,
          message: errorMsg,
          retryable: category !== 'backend_validation',
        },
      });
    } finally {
      setStreamProgress(null);
    }
  }, [globalInput, runManager, projectId, chatMode, addMessage, setGlobalInput, sessionId, spectrumFile, setPendingHypotheses, buildOnProgress, setChatMode, discoveryBackendSessionId, isCoordinatorMode, effectiveSessionIdForCoordinator, onCoordinatorComplete]);

  // -------------------------------------------------------------------------
  // handleSubmit
  // -------------------------------------------------------------------------

  const handleSubmit = useCallback((overrideContent?: string) => {
    const content = overrideContent ?? localInput;
    if (!content.trim() && !isCoordinatorMode) return;

    // Use setTimeout to ensure state updates properly in fast successions
    setTimeout(() => {
      if (!overrideContent) {
        setLocalInput('');
        setGlobalInput('');
      }
      handleSubmitWithContent(content, !!overrideContent);
    }, 0);
  }, [localInput, handleSubmitWithContent, isCoordinatorMode, setGlobalInput]);

  const handleSpectrumUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !projectId) return;
    if (!file.name.toLowerCase().endsWith('.jdx')) {
      alert('Only .jdx (JCAMP-DX) NMR files are supported.');
      return;
    }
    setIsUploadingSpectrum(true);
    try {
      const result = await api.uploadSpectrum(file, projectId);
      setSpectrumFile(result);
    } catch (err: any) {
      console.error('Spectrum upload failed:', err);
      alert(`Spectrum upload failed: ${err.message}`);
    } finally {
      setIsUploadingSpectrum(false);
    }
  }, [projectId]);

  const handleStopWithPartial = useCallback((partialText: string) => {
    runManager.cancelCurrentRun();
    setStreamProgress(null);
    setStreamingText('');
    if (partialText.trim()) {
      addMessage({ role: 'assistant', content: partialText + '\n\n*(Generation stopped by user)*' });
    }
  }, [addMessage, runManager]);

  const handleRetry = useCallback((query: string) => {
    handleSubmitWithContent(query);
  }, [handleSubmitWithContent]);

  const handleQuickQuery = useCallback((query: string) => {
    setLocalInput(query);
    setGlobalInput(query);
    setTimeout(() => handleSubmitWithContent(query), 50);
  }, [setGlobalInput, handleSubmitWithContent]);



  const clearChat = () => {
    if (runManager.isRunning) {
      runManager.cancelCurrentRun();
    }
    clearCurrentChat();
  };

  // -------------------------------------------------------------------------
  // Derived state
  // -------------------------------------------------------------------------

  const isLoading = isCoordinatorMode
    ? false  // Coordinator mode: NEVER disable the input — user must always be able to type/click
    : runManager.isRunning;
  const startTime = runManager.currentRun?.startedAt ?? null;

  const currentAgent = AGENT_OPTIONS.find((a) => a.mode === chatMode) ?? AGENT_OPTIONS[0];

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  if (!projectId) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-6 text-center text-muted-foreground">
        <AlertCircle className="mb-2 h-8 w-8 opacity-20" />
        <p className="text-sm">No Project Selected</p>
      </div>
    );
  }

  return (
    <div className="flex h-full max-h-full w-full flex-col overflow-hidden">
      {/* Agent Dropdown Header — hidden in coordinator mode (parent provides its own header) */}
      {!isCoordinatorMode && (
        <div className="shrink-0 border-b border-border/50 bg-card/50 px-4 py-2">
          <div className="flex items-center justify-between">
            {/* Agent Selector Dropdown */}
            <div ref={agentDropdownRef} className="relative">
              <button
                onClick={() => setAgentDropdownOpen((o) => !o)}
                className="flex items-center gap-2 rounded-lg border border-border bg-surface/50 px-3 py-1.5 text-xs font-medium transition-colors hover:bg-surface"
              >
                <currentAgent.Icon className={`h-3.5 w-3.5 ${currentAgent.color}`} />
                <span className="text-foreground">{currentAgent.label}</span>
                <ChevronDown className="h-3 w-3 text-muted-foreground" />
              </button>
              {agentDropdownOpen && (
                <div className="absolute left-0 top-full z-50 mt-1 min-w-[180px] rounded-lg border border-border bg-card py-1 shadow-2xl">
                  {AGENT_OPTIONS.map(({ mode, label, Icon, color }) => (
                    <button
                      key={mode}
                      onClick={() => {
                        setChatMode(mode);
                        setAgentDropdownOpen(false);
                      }}
                      className={`flex w-full items-center gap-2 px-3 py-2 text-left text-xs transition-colors hover:bg-primary/10 ${chatMode === mode ? 'bg-primary/5 font-medium text-foreground' : 'text-muted-foreground'}`}
                    >
                      <Icon className={`h-3.5 w-3.5 ${color}`} />
                      {label}
                      {chatMode === mode && <span className="ml-auto text-[10px] text-primary">●</span>}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button
              onClick={clearChat}
              className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
              title="Clear chat"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <div className="flex flex-col flex-1 min-w-0">
          {/* Messages area */}
          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 space-y-5">
            <ConversationView
              messages={currentMessages}
              onCitationClick={onCitationClick}
              groundingMap={groundingMap}
              chatMode={effectiveChatMode}
              isLoading={isLoading}
              onRetry={handleRetry}
              onQuickQuery={handleQuickQuery}
              onViewRunDetails={onViewRunDetails}
              onFollowUpClick={handleQuickQuery}
            />

            {chatMode !== 'moe' && (
              <RunProgressDisplay
                streamProgress={streamProgress}
                streamingText={streamingText}
                isLoading={isLoading}
                startTime={startTime}
              />
            )}
          </div>

          {/* Pending Hypotheses (MoE) */}
          {chatMode === 'moe' && pendingHypotheses && pendingHypotheses.length > 0 && !isLoading && (
            <div className="shrink-0 border-t border-border bg-card/50 p-4">
              <div className="mx-auto max-w-3xl space-y-3">
                <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
                  <Network className="h-4 w-4 text-primary" />
                  Select a research path:
                </h4>
                <div className="grid gap-2 sm:grid-cols-1 md:grid-cols-2">
                  {pendingHypotheses.map((hypothesis: any, idx: number) => (
                    <button
                      key={idx}
                      onClick={() => handleSubmit(hypothesis.text)}
                      className="flex flex-col text-left gap-1 rounded-lg border border-border bg-card p-3 hover:border-primary/50 hover:bg-surface transition-all group"
                    >
                      <div className="text-sm font-medium text-foreground group-hover:text-primary transition-colors">
                        {hypothesis.text}
                      </div>
                      {hypothesis.confidence !== undefined && (
                        <div className="text-xs text-muted-foreground flex items-center gap-1.5">
                          <ShieldCheck className="h-3 w-3 text-success/80" />
                          Confidence: {(hypothesis.confidence * 100).toFixed(0)}%
                        </div>
                      )}
                      {hypothesis.reasoning && (
                        <div className="mt-1 text-xs text-muted-foreground line-clamp-2">
                          {hypothesis.reasoning}
                        </div>
                      )}
                    </button>
                  ))}
                </div>
                <div className="text-xs flex justify-end">
                  <button onClick={() => setPendingHypotheses(null)} className="text-muted-foreground hover:text-foreground underline transition-colors">
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Coordinator thinking indicator (inline, shows current step) */}
          {isCoordinatorMode && !coordinatorQuestion && runManager.isRunning && (
            <div className="flex gap-3 px-1">
              <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-emerald-500/30 bg-gradient-to-br from-emerald-500/20 to-emerald-500/10">
                <Loader2 className="h-4 w-4 animate-spin text-emerald-500" />
              </div>
              <div className="rounded-2xl rounded-bl-sm border border-emerald-500/20 bg-card px-4 py-3 text-xs text-emerald-400">
                {streamProgress?.message || 'Coordinator is thinking...'}
                <span className="ml-1 inline-block h-3 w-1 animate-pulse bg-emerald-500/50 rounded-sm" />
              </div>
            </div>
          )}

          {/* Coordinator multiple-choice options (Phase 4 HITL) */}
          {isCoordinatorMode && coordinatorQuestion && (
            <div className="shrink-0 border-t border-border bg-card/50 p-3">
              <div className="space-y-2">
                {coordinatorQuestion.context && (
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    {coordinatorQuestion.context}
                  </p>
                )}
                <div className="flex flex-col gap-1.5">
                  {coordinatorQuestion.options.map((option, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setCoordinatorQuestion(null);
                        handleSubmitWithContent(option);
                      }}
                      className="text-left rounded-lg border border-border bg-card px-3 py-2 text-xs
                                 hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all"
                    >
                      {option}
                    </button>
                  ))}
                </div>
                <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                  <span>Turn {coordinatorQuestion.turn} | {coordinatorQuestion.goalsSoFar.length} goals extracted</span>
                </div>
              </div>
            </div>
          )}

          {/* Input */}
          <CommandSurface
            value={localInput}
            onChange={(val) => {
              setLocalInput(val);
              setGlobalInput(val); // Sync to global so switching tabs preserves it
            }}
            onSubmit={() => handleSubmit()}
            onCancel={() => runManager.cancelCurrentRun()}
            isLoading={isLoading}
            disabled={!projectId}
            chatMode={effectiveChatMode}
            spectrumFile={isCoordinatorMode ? null : spectrumFile}
            onSpectrumUpload={handleSpectrumUpload}
            onSpectrumRemove={() => setSpectrumFile(null)}
            isUploadingSpectrum={isCoordinatorMode ? false : isUploadingSpectrum}
            streamingText={streamingText}
            onStopWithPartial={handleStopWithPartial}
          />
        </div>

        {/* Side Panels (hidden in coordinator mode — parent provides its own panels) */}
        {!isCoordinatorMode && chatMode === 'moe' && (
          <div className="w-[450px] shrink-0 border-l border-border bg-card/50">
            <AgentWorkbench streamProgress={streamProgress as any} streamingText={streamingText} isLoading={isLoading} />
          </div>
        )}
        {!isCoordinatorMode && chatMode === 'discovery' && (
          <div className="w-[500px] shrink-0 border-l border-border bg-card/50">
            <DiscoveryWorkbench
              streamProgress={streamProgress as any}
              isLoading={isLoading}
              finalCandidates={currentMessages[currentMessages.length - 1]?.brainActivity?.candidates}
            />
          </div>
        )}
      </div>
    </div>
  );
}
