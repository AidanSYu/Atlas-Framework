'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  api,
  TaskInfo,
  TaskEvent,
  TaskState,
} from '@/lib/api';
import { ThemeToggle } from '@/components/ThemeToggle';
import { toastError } from '@/stores/toastStore';
import {
  Send,
  Loader2,
  XCircle,
  AlertTriangle,
  ChevronRight,
  ChevronDown,
  Wrench,
  Sparkles,
  Target,
  MessageSquare,
  HelpCircle,
  Package,
  Square,
  Paperclip,
  X,
  FileText,
  FlaskConical,
  Brain,
} from 'lucide-react';

interface TaskChatProps {
  projectId: string;
  taskId?: string | null;
  onTaskCreated?: (task: TaskInfo) => void;
}

/**
 * Claude Code-inspired chat: user prompt → streaming trace of tool calls,
 * supervisor reviews, and final answer. One task per component instance.
 */
export function TaskChat({ projectId, taskId: initialTaskId, onTaskCreated }: TaskChatProps) {
  const [taskId, setTaskId] = useState<string | null>(initialTaskId || null);
  const [task, setTask] = useState<TaskInfo | null>(null);
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedCalls, setExpandedCalls] = useState<Record<string, boolean>>({});
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const stickToBottomRef = useRef(true);
  const unsubRef = useRef<(() => void) | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const isNearTraceBottom = useCallback((el: HTMLDivElement) => {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 96;
  }, []);

  const scrollTraceToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, []);

  const handleTraceScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottomRef.current = isNearTraceBottom(el);
  }, [isNearTraceBottom]);

  // Keep `task` fresh whenever we see a STATE_TRANSITION
  useEffect(() => {
    if (!taskId) return;
    const lastTransition = [...events].reverse().find((e) => e.event_type === 'STATE_TRANSITION');
    if (!lastTransition) return;
    const nextState = lastTransition.payload?.to_state as TaskState | undefined;
    if (nextState && task && nextState !== task.state) {
      setTask({ ...task, state: nextState });
    }
  }, [events, taskId, task]);

  // Auto-scroll only while the user is already reading the live edge. If they
  // scroll up during token streaming, leave the viewport alone.
  useEffect(() => {
    if (stickToBottomRef.current) {
      scrollTraceToBottom();
    }
  }, [events.length, scrollTraceToBottom]);

  // Subscribe to SSE when we have a task id
  useEffect(() => {
    if (!taskId) return;
    // Reset view on task change
    stickToBottomRef.current = true;
    setEvents([]);
    api
      .getTask(taskId)
      .then((t) => setTask(t))
      .catch((err) => {
        // 404 right after creation is expected — task row may not have been
        // committed yet. Surface anything else; silent failures hide bugs.
        if (err?.status !== 404) {
          toastError(`Failed to load task: ${err?.message ?? String(err)}`);
        }
      });

    const unsub = api.subscribeToTask(
      taskId,
      (evt) => {
        setEvents((prev) => {
          if (prev.some((p) => p.event_id === evt.event_id)) return prev;
          return [...prev, evt].sort((a, b) => a.sequence - b.sequence);
        });
        if (
          evt.event_type === 'STATE_TRANSITION' ||
          evt.event_type === 'SYSTEM_CIRCUIT_BREAKER' ||
          evt.event_type === 'USER_CANCELLED' ||
          evt.event_type === 'FINAL_ANSWER'
        ) {
          api
            .getTask(evt.task_id)
            .then(setTask)
            .catch((err) => {
              if (err?.status !== 404) {
                toastError(`Failed to refresh task: ${err?.message ?? String(err)}`);
              }
            });
        }
      },
      {
        fromSequence: -1,
        onError: (evt) => {
          // EventSource error event has no message; use type/name where available.
          const detail = (evt as ErrorEvent)?.message ?? evt?.type ?? 'connection closed';
          toastError(`Task stream lost: ${detail}`);
        },
      }
    );
    unsubRef.current = unsub;
    return () => {
      unsub();
      unsubRef.current = null;
    };
  }, [taskId]);

  const createFreshTask = useCallback(async (parentTaskId?: string | null): Promise<TaskInfo> => {
    const created = await api.createTask({
      project_id: projectId,
      parent_task_id: parentTaskId || undefined,
    });
    setTaskId(created.id);
    setTask(created);
    onTaskCreated?.(created);
    return created;
  }, [projectId, onTaskCreated]);

  const ensureWritableTask = useCallback(async (): Promise<TaskInfo> => {
    if (!taskId || !task) return createFreshTask();
    // COMPLETED is soft-terminal: a follow-up reopens the same task on the
    // backend (FSM: COMPLETED → PLANNING). Only hard-terminal outcomes
    // (cancelled / failed) spawn a fresh task — those imply broken state.
    if (task.state === 'cancelled' || task.state === 'failed') return createFreshTask(task.id);
    return task;
  }, [createFreshTask, taskId, task]);

  const handleSend = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    stickToBottomRef.current = true;
    try {
      const t = await ensureWritableTask();

      // Upload any pending attachments now that we have a task id; we want the
      // returned absolute paths to include in the start/respond request so the
      // executor surfaces them to the orchestrator.
      const uploadedPaths: string[] = [];
      for (const file of pendingFiles) {
        try {
          const res = await api.uploadTaskAttachment(t.id, file);
          uploadedPaths.push(res.path);
        } catch (err: any) {
          throw new Error(`Failed to upload "${file.name}": ${err?.message || err}`);
        }
      }

      setInputValue('');
      setPendingFiles([]);
      if (t.state === 'suspended') {
        await api.respondToTask(t.id, text, uploadedPaths);
      } else {
        await api.startTask(t.id, text, uploadedPaths);
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to send');
    } finally {
      setBusy(false);
    }
  }, [inputValue, busy, ensureWritableTask, pendingFiles]);

  const handleAddFiles = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return;
    const incoming = Array.from(files);
    setPendingFiles((prev) => [...prev, ...incoming]);
  }, []);

  const handleRemovePendingFile = useCallback((index: number) => {
    setPendingFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleCancel = useCallback(async () => {
    if (!taskId) return;
    try {
      await api.cancelTask(taskId);
    } catch (err: any) {
      setError(err?.message || 'Failed to cancel');
    }
  }, [taskId]);

  // Pair tool-call intents with their results by call_id.
  const callResults = useMemo(() => {
    const map: Record<string, TaskEvent> = {};
    for (const evt of events) {
      if (evt.event_type === 'TOOL_EXECUTION_RESULT') {
        const cid = evt.payload?.call_id;
        if (cid) map[cid] = evt;
      }
    }
    return map;
  }, [events]);

  // ---------------------------------------------------------------------
  // Streaming coalescer — fold ORCHESTRATOR_STREAM_DELTA events into virtual
  // "in-flight" thinking/response blocks per iteration. Once the canonical
  // ORCHESTRATOR_THINKING / ORCHESTRATOR_RESPONSE event arrives for that
  // iteration, the structured event wins and the in-flight buffer is hidden.
  // The result: tokens appear typing-live during generation; the final block
  // is the replay-truth.
  // ---------------------------------------------------------------------
  const liveBlocks = useMemo(() => {
    // Map<iteration, {thinking: string, response: string,
    //                 thinkingFinal: bool, responseFinal: bool,
    //                 firstDeltaEventId: string|null}>
    const blocks = new Map<number, {
      thinking: string;
      response: string;
      thinkingFinal: boolean;
      responseFinal: boolean;
      firstThinkingEventId: string | null;
      firstResponseEventId: string | null;
    }>();
    const get = (iter: number) => {
      let b = blocks.get(iter);
      if (!b) {
        b = {
          thinking: '',
          response: '',
          thinkingFinal: false,
          responseFinal: false,
          firstThinkingEventId: null,
          firstResponseEventId: null,
        };
        blocks.set(iter, b);
      }
      return b;
    };
    for (const evt of events) {
      if (evt.event_type === 'ORCHESTRATOR_STREAM_DELTA') {
        const iter = evt.payload?.iteration ?? 0;
        const phase = evt.payload?.phase;
        const delta = evt.payload?.delta || '';
        const b = get(iter);
        if (phase === 'thinking' && !b.thinkingFinal) {
          b.thinking += delta;
          if (!b.firstThinkingEventId) b.firstThinkingEventId = evt.event_id;
        } else if (phase === 'response' && !b.responseFinal) {
          b.response += delta;
          if (!b.firstResponseEventId) b.firstResponseEventId = evt.event_id;
        }
      } else if (evt.event_type === 'ORCHESTRATOR_THINKING') {
        const iter = evt.payload?.iteration ?? 0;
        const b = get(iter);
        b.thinking = evt.payload?.content || '';
        b.thinkingFinal = true;
      } else if (evt.event_type === 'ORCHESTRATOR_RESPONSE') {
        const iter = evt.payload?.iteration ?? 0;
        const b = get(iter);
        b.response = evt.payload?.content || '';
        b.responseFinal = true;
      }
    }
    return blocks;
  }, [events]);

  const renderedEvents = useMemo(() => {
    // Render strategy:
    // - The FIRST delta of each (iteration, phase) is the anchor — it renders
    //   the live-streaming content from liveBlocks (with cursor while open,
    //   without once the canonical event lands).
    // - All subsequent deltas are skipped.
    // - Canonical THINKING / RESPONSE events are SKIPPED when an anchor
    //   exists, because their content is already shown at the anchor's
    //   timeline position via liveBlocks. (Otherwise the canonical event,
    //   which is sequenced AFTER all deltas of the iteration, would render
    //   AFTER the response stream and the order would flip — thinking ends
    //   up below the response prose. Anchoring at the delta position
    //   preserves natural temporal order.)
    // - Legacy / replayed tasks with no delta events still see the canonical
    //   events render normally (no anchor in liveBlocks).
    return events.filter((e) => {
      if (e.event_type === 'ORCHESTRATOR_STREAM_DELTA') {
        const iter = e.payload?.iteration ?? 0;
        const phase = e.payload?.phase;
        const b = liveBlocks.get(iter);
        if (!b) return false;
        return (phase === 'thinking' && b.firstThinkingEventId === e.event_id)
            || (phase === 'response' && b.firstResponseEventId === e.event_id);
      }
      if (e.event_type === 'ORCHESTRATOR_THINKING') {
        const iter = e.payload?.iteration ?? 0;
        const b = liveBlocks.get(iter);
        if (b && b.firstThinkingEventId) return false;
        return shouldRenderEvent(e);
      }
      if (e.event_type === 'ORCHESTRATOR_RESPONSE') {
        const iter = e.payload?.iteration ?? 0;
        const b = liveBlocks.get(iter);
        if (b && b.firstResponseEventId) return false;
        return shouldRenderEvent(e);
      }
      return shouldRenderEvent(e);
    });
  }, [events, liveBlocks]);

  // Detect whether streaming captured a response block — used by FINAL_ANSWER
  // to decide whether to render the grey body box (no streaming → render it
  // as the canonical answer view; streaming present → orange block already
  // shows the body, only the Next-step chip is added).
  const hasStreamedResponse = useMemo(() => {
    let found = false;
    liveBlocks.forEach((b) => {
      if (b.response.trim()) found = true;
    });
    return found;
  }, [liveBlocks]);

  const stateMeta = getStateMeta(task?.state);
  const isTerminal = isTerminalState(task?.state);
  const isSuspended = task?.state === 'suspended';
  const isRunning = task?.state === 'planning' || task?.state === 'executing' || task?.state === 'reviewing';
  const canCompose = !isRunning && !busy;

  // Track when the most recent rendered event arrived. While `isRunning` we
  // show a "thinking" indicator with elapsed seconds so the user knows we're
  // not hung — the orchestrator's GGUF model can take 30-90s on a cold load.
  const lastEventTs = useMemo(() => {
    const t = renderedEvents[renderedEvents.length - 1]?.timestamp;
    return t ? new Date(t).getTime() : null;
  }, [renderedEvents]);
  const [nowTs, setNowTs] = useState(() => Date.now());
  useEffect(() => {
    if (!isRunning) return;
    const id = window.setInterval(() => setNowTs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [isRunning]);
  const elapsedSinceLastEvent = lastEventTs ? Math.max(0, Math.floor((nowTs - lastEventTs) / 1000)) : null;
  const showThinkingIndicator = isRunning && (elapsedSinceLastEvent === null || elapsedSinceLastEvent >= 2);

  return (
    <div className="flex h-full flex-col bg-background">
      {/* Header / status */}
      <div className="flex items-center justify-between border-b border-border/40 bg-card/30 px-4 py-2">
        <div className="flex items-center gap-2">
          <div className={`flex h-6 w-6 items-center justify-center rounded ${stateMeta.bg}`}>
            <stateMeta.icon className={`h-3.5 w-3.5 ${stateMeta.fg}`} />
          </div>
          <div>
            <p className="text-[13px] font-medium text-foreground">
              {task?.title || task?.initial_prompt?.slice(0, 60) || 'New Task'}
            </p>
            <p className={`text-[11px] ${stateMeta.fg}`}>{stateMeta.label}</p>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {isRunning && (
            <button
              onClick={handleCancel}
              className="inline-flex h-7 items-center gap-1.5 rounded border border-destructive/30 bg-destructive/5 px-2.5 text-[11px] font-medium text-destructive hover:bg-destructive/10"
            >
              <Square className="h-3 w-3" />
              Cancel
            </button>
          )}
          <ThemeToggle />
        </div>
      </div>

      {/* Event trace */}
      <div
        ref={scrollRef}
        onScroll={handleTraceScroll}
        className="flex-1 overflow-y-auto px-5 py-5 space-y-4"
      >
        {renderedEvents.length === 0 && !busy && (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 border border-accent/20">
              <Sparkles className="h-5 w-5 text-accent" />
            </div>
            <div>
              <p className="text-[13px] font-medium text-foreground">Ask, chat, or hand me a task</p>
              <p className="max-w-sm text-[11px] text-muted-foreground">
                Simple messages get simple replies. When tools are useful, Atlas will show the calls here.
              </p>
            </div>
          </div>
        )}

        {renderedEvents.map((evt) => {
          // Thinking is the model showing its work — default expanded so the
          // audience sees reasoning. Everything else defaults collapsed.
          const defaultExpanded =
            evt.event_type === 'ORCHESTRATOR_THINKING' ||
            (evt.event_type === 'ORCHESTRATOR_STREAM_DELTA' && evt.payload?.phase === 'thinking');
          return (
            <TaskEventRow
              key={evt.event_id}
              event={evt}
              callResults={callResults}
              liveBlocks={liveBlocks}
              hasStreamedResponse={hasStreamedResponse}
              expanded={expandedCalls[evt.event_id] ?? defaultExpanded}
              onToggle={() =>
                setExpandedCalls((prev) => ({
                  ...prev,
                  [evt.event_id]: !(prev[evt.event_id] ?? defaultExpanded),
                }))
              }
              onUseNextStep={(text) => setInputValue(text)}
            />
          );
        })}

        {showThinkingIndicator && (
          <div className="flex items-center gap-2 rounded-md border border-accent/20 bg-accent/5 px-3 py-2 text-[12px] text-foreground/80">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />
            <span>
              Orchestrator is {task?.state === 'planning' ? 'planning' : task?.state === 'reviewing' ? 'reviewing' : 'thinking'}
              {elapsedSinceLastEvent !== null && elapsedSinceLastEvent >= 5
                ? ` — ${elapsedSinceLastEvent}s elapsed`
                : '...'}
              {elapsedSinceLastEvent !== null && elapsedSinceLastEvent >= 15 && (
                <span className="ml-2 text-muted-foreground">
                  (first request can take 30–90s while the GGUF model loads)
                </span>
              )}
            </span>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-border/40 bg-card/20 p-3">
        {error && (
          <div className="mb-2 rounded border border-destructive/30 bg-destructive/5 px-2.5 py-1.5 text-[11px] text-destructive">
            {error}
          </div>
        )}
        {isSuspended && (
          <div className="mb-2 flex items-start gap-2 rounded border border-amber-500/30 bg-amber-500/5 px-2.5 py-1.5 text-[11px] text-amber-500">
            <HelpCircle className="mt-0.5 h-3 w-3 shrink-0" />
            <span>The supervisor is waiting for your response below.</span>
          </div>
        )}
        {pendingFiles.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {pendingFiles.map((file, idx) => (
              <div
                key={`${file.name}-${idx}`}
                className="inline-flex items-center gap-1.5 rounded border border-border/50 bg-surface px-2 py-1 text-[11px] text-foreground"
              >
                <FileText className="h-3 w-3 text-muted-foreground" />
                <span className="max-w-[180px] truncate font-mono">{file.name}</span>
                <span className="text-muted-foreground/60">
                  {formatBytes(file.size)}
                </span>
                <button
                  onClick={() => handleRemovePendingFile(idx)}
                  className="ml-0.5 text-muted-foreground hover:text-foreground"
                  aria-label={`Remove ${file.name}`}
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => {
            handleAddFiles(e.target.files);
            // allow re-selecting the same file after removal
            if (e.target) e.target.value = '';
          }}
        />
        <div className="flex gap-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={!canCompose}
            className="inline-flex h-[52px] w-[52px] shrink-0 items-center justify-center rounded border border-border/50 bg-background text-muted-foreground transition-colors hover:border-accent/50 hover:text-foreground disabled:opacity-40"
            aria-label="Attach file"
            title="Attach file"
          >
            <Paperclip className="h-4 w-4" />
          </button>
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && canCompose) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={
              isTerminal
                ? 'Send a new message...'
                : isSuspended
                ? 'Respond to the supervisor...'
                : isRunning
                ? 'Task is running. Cancel first to send a new prompt.'
                : 'Message Atlas...'
            }
            disabled={!canCompose}
            rows={2}
            className="flex-1 resize-none rounded border border-border/50 bg-background px-3 py-2 text-[13px] text-foreground outline-none placeholder:text-muted-foreground/60 focus:border-accent/50 focus:ring-1 focus:ring-accent/25 disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || !canCompose}
            className="inline-flex h-[52px] w-[52px] shrink-0 items-center justify-center rounded bg-accent text-white transition-all hover:bg-accent/90 disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Event row — dispatches by event_type
// ---------------------------------------------------------------------------

interface LiveBlock {
  thinking: string;
  response: string;
  thinkingFinal: boolean;
  responseFinal: boolean;
  firstThinkingEventId: string | null;
  firstResponseEventId: string | null;
}

function TaskEventRow({
  event,
  callResults,
  liveBlocks,
  hasStreamedResponse,
  expanded,
  onToggle,
  onUseNextStep,
}: {
  event: TaskEvent;
  callResults: Record<string, TaskEvent>;
  liveBlocks?: Map<number, LiveBlock>;
  hasStreamedResponse?: boolean;
  expanded: boolean;
  onToggle: () => void;
  onUseNextStep?: (text: string) => void;
}) {
  switch (event.event_type) {
    case 'USER_PROMPT':
    case 'USER_RESPONSE': {
      const attachments = (event.payload?.attachments || []) as string[];
      // Lab-instrument framing: cite the request as a labeled input, not as
      // a saturated speech bubble. Borrowed from how observability tools render
      // a request payload — a left-bordered quoted line above the response.
      return (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              Request
            </span>
            <div className="h-px flex-1 bg-border/40" />
          </div>
          <div className="border-l-2 border-accent/40 bg-card/30 pl-3 pr-2 py-1.5 text-[13px] leading-relaxed text-foreground">
            {event.payload?.content}
          </div>
          {attachments.length > 0 && (
            <div className="flex flex-wrap gap-1 pl-3">
              {attachments.map((path, i) => (
                <div
                  key={`${path}-${i}`}
                  className="inline-flex items-center gap-1 rounded border border-border/40 bg-surface px-1.5 py-0.5 text-[10px] font-mono text-foreground/80"
                  title={path}
                >
                  <FileText className="h-2.5 w-2.5 text-muted-foreground" />
                  {basename(path)}
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    case 'MANIFEST_SCOPED': {
      const selected: string[] = event.payload?.selected_tools || [];
      const reasoning: string = event.payload?.scoping_reasoning || '';
      if (selected.length === 0) {
        return (
          <div className="flex justify-start">
            <div className="max-w-[76%] rounded-xl rounded-bl-sm border border-border/35 bg-card/35 px-3.5 py-2 text-[12px] leading-relaxed text-muted-foreground">
              No tools needed for this turn.
            </div>
          </div>
        );
      }
      return (
        <div className="max-w-[88%] rounded-md border border-border/40 bg-card/35 px-3 py-2">
          <div className="mb-1.5 flex items-center gap-1.5">
            <Package className="h-3 w-3 text-muted-foreground" />
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Tools selected
            </span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {selected.map((t) => (
              <span
                key={t}
                className="rounded bg-surface px-1.5 py-0.5 font-mono text-[10px] text-foreground/80"
              >
                {t}
              </span>
            ))}
          </div>
          {reasoning && (
            <p className="mt-1.5 text-[11px] text-muted-foreground leading-relaxed">{reasoning}</p>
          )}
        </div>
      );
    }

    case 'SUPERVISOR_BRIEF': {
      const goal = event.payload?.goal_statement || '';
      const dod = event.payload?.definition_of_done || '';
      return (
        <div className="max-w-[88%] rounded-md border border-accent/20 bg-accent/5 px-3 py-2">
          <div className="mb-1 flex items-center gap-1.5">
            <Target className="h-3 w-3 text-accent" />
            <span className="text-[11px] font-medium uppercase tracking-wider text-accent">
              Goal Brief
            </span>
          </div>
          <p className="text-[12px] text-foreground leading-relaxed">{goal}</p>
          {dod && (
            <p className="mt-1.5 text-[11px] text-muted-foreground leading-relaxed">
              <span className="font-medium">Done when:</span> {dod}
            </p>
          )}
        </div>
      );
    }

    case 'GOAL_BRIEF_REVISION': {
      const amendment = event.payload?.amendment || '';
      return (
        <div className="max-w-[88%] rounded-md border border-amber-500/20 bg-amber-500/5 px-3 py-2">
          <div className="mb-1 flex items-center gap-1.5">
            <Sparkles className="h-3 w-3 text-amber-500" />
            <span className="text-[11px] font-medium uppercase tracking-wider text-amber-500">
              Supervisor amendment
            </span>
          </div>
          <p className="text-[12px] text-foreground leading-relaxed">{amendment}</p>
        </div>
      );
    }

    case 'ORCHESTRATOR_STREAM_DELTA': {
      // Anchor for an iteration's thinking/response block. Renders the live
      // content from liveBlocks; when the canonical event lands, the block's
      // `*Final` flag flips and we render the finalized style (collapsible
      // thinking / response without cursor). When the canonical event NEVER
      // lands for response (= iteration produced no tool calls and the prose
      // IS the final answer), the cursor keeps pulsing — that's the
      // "loop never closes" autonomous-lab feel.
      const iter: number = event.payload?.iteration || 0;
      const phase: string = event.payload?.phase || 'response';
      const block = liveBlocks?.get(iter);
      if (!block) return null;
      const isThinking = phase === 'thinking';
      const content = isThinking ? block.thinking : block.response;
      if (!content.trim()) return null;
      if (isThinking) {
        const liveFirstLine = content.split('\n').find((l) => l.trim()) || content;
        const livePreview = liveFirstLine.length > 120
          ? liveFirstLine.slice(0, 117).trimEnd() + '...'
          : liveFirstLine;
        const liveHasCollapsedPreview = content.trim().includes('\n') || content.trim().length > 120;
        if (block.thinkingFinal) {
          // Final thinking — collapsible (mirrors the ORCHESTRATOR_THINKING
          // legacy render below).
          const firstLine = content.split('\n').find((l) => l.trim()) || content;
          const preview = firstLine.length > 120 ? firstLine.slice(0, 117).trimEnd() + '…' : firstLine;
          return (
            <div className="rounded border-l-2 border-l-muted-foreground/30 bg-card/20">
              <button
                onClick={onToggle}
                className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left hover:bg-card/40"
              >
                {expanded ? (
                  <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
                ) : (
                  <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
                )}
                <Brain className="h-3 w-3 shrink-0 text-muted-foreground" />
                <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground shrink-0">
                  Reasoning{iter > 0 ? ` · iter ${iter}` : ''}
                </span>
                {!expanded && (content.trim().includes('\n') || content.trim().length > 120) && (
                  <span className="text-[11px] italic text-muted-foreground/60 truncate min-w-0">
                    {preview}
                  </span>
                )}
              </button>
              {expanded && (
                <div className="border-t border-border/20 px-3 py-2">
                  <p className="text-[12px] italic leading-relaxed text-foreground/70 whitespace-pre-wrap">
                    {content}
                  </p>
                </div>
              )}
            </div>
          );
        }
        // Live thinking — typing cursor.
        return (
          <div className="rounded border-l-2 border-l-muted-foreground/30 bg-card/20">
            <button
              onClick={onToggle}
              className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left hover:bg-card/40"
            >
              {expanded ? (
                <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
              )}
              <Brain className="h-3 w-3 shrink-0 text-muted-foreground animate-pulse" />
              <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground shrink-0">
                Reasoning · iter {iter}
              </span>
              <span className="text-[10px] text-accent">●</span>
              {!expanded && liveHasCollapsedPreview && (
                <span className="text-[11px] italic text-muted-foreground/60 truncate min-w-0">
                  {livePreview}
                </span>
              )}
            </button>
            {expanded && (
              <div className="border-t border-border/20 px-3 py-2">
                <p className="text-[12px] italic leading-relaxed text-foreground/70 whitespace-pre-wrap">
                  {content}
                  <span className="ml-0.5 inline-block h-3 w-1.5 animate-pulse bg-muted-foreground/50 align-baseline" />
                </p>
              </div>
            )}
          </div>
        );
      }
      // Response block — orange left border. Cursor shows while streaming OR
      // when the canonical RESPONSE event never lands (= this iteration's
      // prose became the final answer, no tool calls followed).
      const showCursor = !block.responseFinal;
      return (
        <div className="rounded bg-card/30 border-l-2 border-l-accent/40 px-3 py-2.5">
          <p className="text-[13px] leading-relaxed text-foreground whitespace-pre-wrap">
            {content}
            {showCursor && (
              <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-accent/60 align-baseline" />
            )}
          </p>
        </div>
      );
    }

    case 'ORCHESTRATOR_RESPONSE': {
      const content: string = event.payload?.content || '';
      if (!content.trim()) return null;
      // Final response prose — the model's "I'll do X then Y" plan. Rendered
      // prominently because this is the user-facing narration, the Claude
      // moment where the assistant says what it's about to do.
      return (
        <div className="rounded bg-card/30 border-l-2 border-l-accent/40 px-3 py-2.5">
          <p className="text-[13px] leading-relaxed text-foreground whitespace-pre-wrap">
            {content}
          </p>
        </div>
      );
    }

    case 'ORCHESTRATOR_THINKING': {
      const content: string = event.payload?.content || '';
      const iteration: number = event.payload?.iteration || 0;
      if (!content.trim()) return null;
      // Visible reasoning. Muted styling — supportive, not louder than the
      // tool cards. Collapsible because long chains stack up fast and the
      // audience usually wants the gist, not every iteration's monologue.
      // The first line of the content serves as the preview when collapsed.
      const firstLine = content.split('\n').find((l) => l.trim()) || content;
      const preview = firstLine.length > 120 ? firstLine.slice(0, 117).trimEnd() + '…' : firstLine;
      return (
        <div className="rounded border-l-2 border-l-muted-foreground/30 bg-card/20">
          <button
            onClick={onToggle}
            className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left hover:bg-card/40"
          >
            {expanded ? (
              <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
            )}
            <Brain className="h-3 w-3 shrink-0 text-muted-foreground" />
            <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground shrink-0">
              Reasoning{iteration > 0 ? ` · iter ${iteration}` : ''}
            </span>
            {!expanded && (content.trim().includes('\n') || content.trim().length > 120) && (
              <span className="text-[11px] italic text-muted-foreground/60 truncate min-w-0">
                {preview}
              </span>
            )}
          </button>
          {expanded && (
            <div className="border-t border-border/20 px-3 py-2">
              <p className="text-[12px] italic leading-relaxed text-foreground/70 whitespace-pre-wrap">
                {content}
              </p>
            </div>
          )}
        </div>
      );
    }

    case 'TOOL_CALL_INTENT': {
      const toolName = event.payload?.tool_name || 'unknown';
      const args = event.payload?.arguments || {};
      // intent prose is now surfaced as its own ORCHESTRATOR_RESPONSE block
      // above the tool cards — no longer duplicated inside the card.
      const callId = event.payload?.call_id;
      const result = callId ? callResults[callId] : undefined;
      const resultStatus = result?.payload?.status as string | undefined;
      const resultOutput = result?.payload?.output;
      const resultSummary = resultOutput?.summary as string | undefined;
      const elapsed = result?.payload?.execution_time_ms;
      const chips = resultStatus === 'success' && resultOutput
        ? extractToolHeadline(toolName, resultOutput)
        : null;
      const physical = isPhysicalTool(toolName);
      const instrumentLabel = instrumentLabelFor(toolName, resultOutput);
      const statusColor =
        resultStatus === 'success'
          ? 'text-emerald-500'
          : resultStatus?.startsWith('error')
          ? 'text-destructive'
          : resultStatus === 'requires_human'
          ? 'text-amber-500'
          : 'text-muted-foreground';
      const cardClasses = physical
        ? 'rounded border-l-2 border-l-info border-y border-r border-y-border/40 border-r-border/40 bg-info/[0.03]'
        : 'rounded border border-border/40 bg-card/40';
      const ToolIcon = physical ? FlaskConical : Wrench;
      return (
        <div className={cardClasses}>
          <button
            onClick={onToggle}
            className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left hover:bg-card/60"
          >
            <div className="flex items-center gap-2 min-w-0">
              {expanded ? (
                <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
              )}
              <ToolIcon className={`h-3 w-3 shrink-0 ${physical ? 'text-info' : statusColor}`} />
              <div className="flex flex-col min-w-0">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-mono text-[12px] text-foreground truncate">{toolName}</span>
                  {physical && (
                    <span className="shrink-0 rounded bg-info/10 px-1.5 py-px text-[9px] font-medium uppercase tracking-wider text-info">
                      instrument
                    </span>
                  )}
                </div>
                {physical && instrumentLabel && (
                  <span className="text-[10px] leading-tight text-muted-foreground/80 truncate">
                    {instrumentLabel}
                  </span>
                )}
              </div>
              {!result && (
                <Loader2 className="h-3 w-3 animate-spin shrink-0 text-muted-foreground" />
              )}
            </div>
            <div className="flex items-center gap-1.5 shrink-0 flex-wrap justify-end">
              {chips && chips.length > 0 && chips.map((c, i) => (
                <span
                  key={i}
                  className="inline-flex items-center gap-1 rounded bg-surface px-1.5 py-0.5 font-mono text-[10px] text-foreground/80"
                >
                  {c.label && <span className="text-muted-foreground/70">{c.label}</span>}
                  <span>{c.value}</span>
                </span>
              ))}
              {typeof elapsed === 'number' && (
                <span className="font-mono text-[10px] text-muted-foreground/60">
                  {elapsed}ms
                </span>
              )}
              {(!chips || chips.length === 0) && resultStatus && (
                <span className={`font-mono text-[10px] uppercase ${statusColor}`}>
                  {resultStatus}
                </span>
              )}
            </div>
          </button>
          {expanded && (
            <div className="border-t border-border/30 px-3 py-2 space-y-2">
              <div>
                <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                  Arguments
                </p>
                <pre className="overflow-x-auto rounded bg-background/60 px-2 py-1.5 font-mono text-[11px] text-foreground/80">
                  {JSON.stringify(args, null, 2)}
                </pre>
              </div>
              {resultSummary && (
                <div>
                  <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                    Result
                  </p>
                  <pre className="overflow-x-auto rounded bg-background/60 px-2 py-1.5 font-mono text-[11px] text-foreground/80 whitespace-pre-wrap">
                    {resultSummary}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      );
    }

    case 'TOOL_YIELD':
      return (
        <div className="max-w-[88%] rounded-md border border-info/20 bg-info/5 px-3 py-2">
          <div className="mb-1 flex items-center gap-1.5">
            <MessageSquare className="h-3 w-3 text-info" />
            <span className="text-[11px] font-medium uppercase tracking-wider text-info">
              Yielded to supervisor
            </span>
          </div>
          <p className="text-[12px] text-foreground/90 leading-relaxed">
            {event.payload?.reason || '(no reason given)'}
          </p>
        </div>
      );

    case 'SUPERVISOR_REVIEW': {
      const verdict = event.payload?.verdict;
      const reasoning = event.payload?.reasoning || '';
      const reviewClasses =
        verdict === 'approve'
          ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-500'
          : verdict === 'revise'
          ? 'border-amber-500/20 bg-amber-500/5 text-amber-500'
          : verdict === 'rescope'
          ? 'border-info/20 bg-info/5 text-info'
          : 'border-destructive/20 bg-destructive/5 text-destructive';
      return (
        <div className={`max-w-[88%] rounded-md border px-3 py-2 ${reviewClasses}`}>
          <div className="mb-1 flex items-center gap-1.5">
            <span className="text-[11px] font-medium uppercase tracking-wider">
              Supervisor: {verdict}
            </span>
          </div>
          <p className="text-[12px] text-foreground/90 leading-relaxed">
            {event.payload?.user_question || reasoning}
          </p>
        </div>
      );
    }

    case 'FINAL_ANSWER': {
      const answer: string = event.payload?.answer || '';
      const nextStep = parseNextStep(answer);
      // When streaming captured the answer prose, the orange response block
      // above is THE answer view — don't render a duplicate grey box. Just
      // surface the Next-step chip (if any) so the loop has a continuation
      // affordance. Legacy / replayed tasks with no streaming render the
      // full answer box as a fallback.
      const answerBody = nextStep
        ? answer.replace(/(?:^|\n)\s*next\s*step\s*:[^\n]+(?:\n(?!\s*\n)[^\n]+)*/i, '').trim()
        : answer;
      const showBody = !hasStreamedResponse;
      return (
        <div className="flex flex-col gap-2 items-start">
          {showBody && answerBody && (
            <div className="max-w-[76%] rounded-xl rounded-bl-sm border border-border/40 bg-card/45 px-4 py-3">
              <div className="mb-2 flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                  Answer
                </span>
              </div>
              <div className="prose prose-sm max-w-none text-[13px] text-foreground leading-relaxed whitespace-pre-wrap">
                {answerBody}
              </div>
            </div>
          )}
          {nextStep && (
            <button
              type="button"
              onClick={() => onUseNextStep?.(nextStep)}
              className="group inline-flex max-w-[76%] items-start gap-2 rounded-xl border border-accent/30 bg-accent/5 px-3.5 py-2.5 text-left transition-colors hover:bg-accent/10 hover:border-accent/50"
            >
              <ChevronRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent transition-transform group-hover:translate-x-0.5" />
              <div className="flex flex-col gap-0.5 min-w-0">
                <span className="text-[10px] font-medium uppercase tracking-wider text-accent">
                  Next step — click to continue
                </span>
                <span className="text-[12px] leading-relaxed text-foreground">
                  {nextStep}
                </span>
              </div>
            </button>
          )}
        </div>
      );
    }

    case 'SYSTEM_CIRCUIT_BREAKER': {
      const reason = event.payload?.reason || 'unknown';
      const context = (event.payload?.context ?? {}) as Record<string, any>;
      const errorMessage = typeof context.error === 'string' ? context.error : null;
      const phase = typeof context.phase === 'string' ? context.phase : null;
      // Fatal-style breakers are real failures; loop/error-threshold are softer.
      const isFatal = reason === 'fatal_wrapper_crash';
      return (
        <div
          className={[
            'max-w-[88%] rounded-md px-3 py-2',
            isFatal
              ? 'border border-destructive/40 bg-destructive/10'
              : 'border border-amber-500/30 bg-amber-500/5',
          ].join(' ')}
        >
          <div className="mb-1 flex items-center gap-1.5">
            <AlertTriangle className={`h-3 w-3 ${isFatal ? 'text-destructive' : 'text-amber-500'}`} />
            <span
              className={`text-[11px] font-medium uppercase tracking-wider ${
                isFatal ? 'text-destructive' : 'text-amber-500'
              }`}
            >
              {isFatal ? 'Execution failed' : `Circuit breaker: ${reason}`}
            </span>
          </div>
          {phase && (
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Phase: <span className="font-mono">{phase}</span>
            </p>
          )}
          {errorMessage && (
            <pre className="mt-1 max-h-48 overflow-y-auto whitespace-pre-wrap break-words rounded bg-background/40 px-2 py-1.5 font-mono text-[11px] leading-snug text-foreground">
              {errorMessage}
            </pre>
          )}
          {!errorMessage && !phase && (
            <p className="text-[11px] text-muted-foreground">
              Reason: <span className="font-mono">{reason}</span>
            </p>
          )}
        </div>
      );
    }

    case 'USER_CANCELLED':
      return (
        <div className="text-center text-[11px] text-muted-foreground">
          Task cancelled{event.payload?.reason ? ` — ${event.payload.reason}` : ''}
        </div>
      );

    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function basename(p: string): string {
  // handle both Windows and POSIX paths without importing a library
  const s = p.replace(/\\/g, '/');
  const i = s.lastIndexOf('/');
  return i >= 0 ? s.slice(i + 1) : s;
}

function shouldRenderEvent(event: TaskEvent): boolean {
  const type = event.event_type;
  // STATE_TRANSITION and TOOL_EXECUTION_RESULT are consumed by other rows; skip them.
  // MANIFEST_SCOPED and SUPERVISOR_BRIEF were always filler under the single-
  // orchestrator architecture (the deterministic supervisor passes the full
  // catalog through and uses the prompt verbatim as the goal). They added noise
  // without information and were suppressed on the write path — still filter
  // here so historical events from before the rip-out don't reappear in replays.
  if (
    type === 'STATE_TRANSITION' ||
    type === 'TOOL_EXECUTION_RESULT' ||
    type === 'ARTIFACT_WRITTEN' ||
    type === 'LOG_COMPACTED' ||
    type === 'SYSTEM_PLUGIN_VERSION_DRIFT' ||
    type === 'MANIFEST_SCOPED' ||
    type === 'SUPERVISOR_BRIEF'
  ) {
    return false;
  }
  // Drop SUPERVISOR_REVIEW rubber-stamps from the single-orchestrator pipeline.
  // Under that architecture the supervisor always approves and the reasoning
  // is literally "FINAL_ANSWER from the orchestrator is taken as final" —
  // pure noise that makes the UI look like a chatbot ceremony rather than
  // a research instrument. When we re-introduce a real reviewer model, drop
  // this filter and these rows will reappear with actual content.
  if (type === 'SUPERVISOR_REVIEW') {
    const reasoning: string = event.payload?.reasoning || '';
    if (
      event.payload?.verdict === 'approve' &&
      /single[-\s_]?orchestrator|taken as final/i.test(reasoning)
    ) {
      return false;
    }
  }
  return true;
}

function isTerminalState(state: TaskState | undefined): boolean {
  return state === 'completed' || state === 'cancelled' || state === 'failed';
}

function getStateMeta(state: TaskState | undefined) {
  switch (state) {
    case 'idle':
      return { label: 'Ready', icon: Sparkles, bg: 'bg-muted-foreground/10', fg: 'text-muted-foreground' };
    case 'initializing':
      return { label: 'Initializing...', icon: Loader2, bg: 'bg-accent/10', fg: 'text-accent' };
    case 'planning':
      return { label: 'Planning...', icon: Target, bg: 'bg-accent/10', fg: 'text-accent' };
    case 'executing':
      return { label: 'Executing...', icon: Wrench, bg: 'bg-accent/10', fg: 'text-accent' };
    case 'reviewing':
      return { label: 'Reviewing...', icon: MessageSquare, bg: 'bg-accent/10', fg: 'text-accent' };
    case 'suspended':
      return { label: 'Waiting for you', icon: HelpCircle, bg: 'bg-amber-500/10', fg: 'text-amber-500' };
    case 'completed':
      // "Completed" is misleading — research loops are never done. The header
      // surfaces "Standing by" instead; the icon is the calmer Sparkles, not
      // a victorious checkmark.
      return { label: 'Standing by', icon: Sparkles, bg: 'bg-muted-foreground/10', fg: 'text-muted-foreground' };
    case 'cancelled':
      return { label: 'Cancelled', icon: XCircle, bg: 'bg-muted-foreground/10', fg: 'text-muted-foreground' };
    case 'failed':
      return { label: 'Failed', icon: XCircle, bg: 'bg-destructive/10', fg: 'text-destructive' };
    default:
      return { label: 'New task', icon: Sparkles, bg: 'bg-accent/10', fg: 'text-accent' };
  }
}

// ---------------------------------------------------------------------------
// Tool result headline — turn a plugin's raw output into a 2-3 chip summary
// so the collapsed tool card carries actual information, not just "SUCCESS".
// Unknown tools fall through to null; the card uses its default status pill.
// ---------------------------------------------------------------------------

interface ToolChip {
  label: string;
  value: string;
}

function extractToolHeadline(toolName: string, output: any): ToolChip[] | null {
  if (!output || typeof output !== 'object') return null;
  const fmt = (v: any, suffix = ''): string => {
    if (v == null) return '—';
    if (typeof v === 'number') return `${v}${suffix}`;
    return `${String(v)}${suffix}`;
  };
  switch (toolName) {
    case 'standardize_smiles': {
      const mols = output.molecules || [];
      const first = mols[0] || {};
      return [
        { label: 'mols', value: fmt(mols.length) },
        ...(first.inchikey ? [{ label: 'InChIKey', value: String(first.inchikey).slice(0, 14) + '…' }] : []),
      ];
    }
    case 'check_toxicity': {
      const r = output.toxicity_results?.[0] || output;
      return [
        { label: 'PAINS', value: fmt(r.pains_hits ?? 0) },
        { label: 'alerts', value: fmt(r.alert_count ?? 0) },
        { label: '', value: r.clean ? 'clean' : 'flagged' },
      ];
    }
    case 'predict_admet': {
      const p = output.predictions?.[0] || output;
      const flags: string[] = [];
      if (p.herg) flags.push(`hERG ${p.herg}`);
      if (p.dili) flags.push(`DILI ${p.dili}`);
      if (p.caco2) flags.push(`Caco-2 ${p.caco2}`);
      return flags.slice(0, 3).map((f) => ({ label: '', value: f }));
    }
    case 'score_synthesizability': {
      const s = output.scores?.[0] || output;
      const score = s.sa_score ?? s.score;
      return [
        { label: 'SA', value: typeof score === 'number' ? score.toFixed(2) : fmt(score) },
        ...(s.feasible !== undefined ? [{ label: '', value: s.feasible ? 'feasible' : 'hard' }] : []),
      ];
    }
    case 'liquid_handler':
      return [
        { label: 'batch', value: fmt(output.batch_id) },
        { label: 'ETA', value: fmt(output.estimated_completion_min, ' min') },
      ];
    case 'hplc_qc':
      return [
        { label: 'purity', value: fmt(output.purity_percent, '%') },
        { label: 'peaks', value: fmt(output.peak_count) },
        { label: '', value: output.passed_qc ? 'pass' : 'fail' },
      ];
    case 'nmr_qc':
      return [
        { label: 'shifts', value: fmt((output.chemical_shifts_ppm || []).length) },
        { label: '', value: output.structure_confirmed ? 'confirmed' : 'mismatch' },
      ];
    case 'plate_reader':
      return [
        { label: 'IC50', value: fmt(output.ic50_nm, ' nM') },
        { label: '', value: output.hit_classification || '—' },
      ];
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// Parse "Next step:" prose from a final answer so we can render a clickable
// continuation chip. The model is prompt-instructed to always include one.
// ---------------------------------------------------------------------------

export function parseNextStep(answer: string | undefined | null): string | null {
  if (!answer) return null;
  const m = answer.match(/(?:^|\n)\s*next\s*step\s*:\s*([^\n]+(?:\n(?!\s*\n)[^\n]+)*)/i);
  return m ? m[1].trim() : null;
}

// ---------------------------------------------------------------------------
// Tool classification — physical instruments get distinct treatment so the
// audience sees "this is hitting real lab hardware" vs "this is a model
// running on the same box."
// ---------------------------------------------------------------------------

const PHYSICAL_TOOL_NAMES = new Set([
  'liquid_handler',
  'hplc_qc',
  'plate_reader',
  'nmr_qc',
]);

function isPhysicalTool(name: string): boolean {
  return PHYSICAL_TOOL_NAMES.has(name);
}

function instrumentLabelFor(toolName: string, output: any): string | null {
  if (!isPhysicalTool(toolName)) return null;
  if (output && typeof output === 'object' && typeof output.instrument === 'string') {
    return output.instrument;
  }
  // Sensible defaults if instrument field is absent for any reason
  const fallback: Record<string, string> = {
    liquid_handler: 'Hamilton STARlet',
    hplc_qc: 'Agilent 1290 Infinity II',
    plate_reader: 'BMG CLARIOstar',
    nmr_qc: 'Bruker AVANCE NEO 400 MHz',
  };
  return fallback[toolName] || null;
}
