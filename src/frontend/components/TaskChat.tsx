'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  api,
  TaskInfo,
  TaskEvent,
  TaskEventType,
  TaskState,
} from '@/lib/api';
import {
  Send,
  Loader2,
  CheckCircle2,
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
  const unsubRef = useRef<(() => void) | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  // Auto-scroll to bottom on new event
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events.length]);

  // Subscribe to SSE when we have a task id
  useEffect(() => {
    if (!taskId) return;
    // Reset view on task change
    setEvents([]);
    api
      .getTask(taskId)
      .then((t) => setTask(t))
      .catch(() => {/* ignore; task may not be finalized yet */});

    const unsub = api.subscribeToTask(
      taskId,
      (evt) => {
        setEvents((prev) => {
          if (prev.some((p) => p.event_id === evt.event_id)) return prev;
          return [...prev, evt].sort((a, b) => a.sequence - b.sequence);
        });
      },
      {
        fromSequence: -1,
        onError: () => {
          // SSE errors are common on dev reload; don't spam.
        },
      }
    );
    unsubRef.current = unsub;
    return () => {
      unsub();
      unsubRef.current = null;
    };
  }, [taskId]);

  const ensureTask = useCallback(async (): Promise<TaskInfo> => {
    if (taskId && task) return task;
    const created = await api.createTask({ project_id: projectId });
    setTaskId(created.id);
    setTask(created);
    onTaskCreated?.(created);
    return created;
  }, [projectId, taskId, task, onTaskCreated]);

  const handleSend = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || busy) return;
    setBusy(true);
    setError(null);
    try {
      const t = await ensureTask();

      // Upload any pending attachments now that we have a task id; we want the
      // returned absolute paths to include in the start/respond request so the
      // executor surfaces them to Nemotron.
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
  }, [inputValue, busy, ensureTask, pendingFiles]);

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

  const renderedEvents = useMemo(() => {
    return events.filter((e) => shouldRenderEvent(e.event_type));
  }, [events]);

  const stateMeta = getStateMeta(task?.state);
  const isTerminal = task?.state === 'completed' || task?.state === 'cancelled' || task?.state === 'failed';
  const isSuspended = task?.state === 'suspended';
  const isRunning = task?.state === 'planning' || task?.state === 'executing' || task?.state === 'reviewing';

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
        {isRunning && (
          <button
            onClick={handleCancel}
            className="inline-flex h-7 items-center gap-1.5 rounded border border-destructive/30 bg-destructive/5 px-2.5 text-[11px] font-medium text-destructive hover:bg-destructive/10"
          >
            <Square className="h-3 w-3" />
            Cancel
          </button>
        )}
      </div>

      {/* Event trace */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {renderedEvents.length === 0 && !busy && (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/10 border border-accent/20">
              <Sparkles className="h-5 w-5 text-accent" />
            </div>
            <div>
              <p className="text-[13px] font-medium text-foreground">Ready when you are</p>
              <p className="max-w-sm text-[11px] text-muted-foreground">
                Describe what you want Atlas to do. DeepSeek will scope the tools, Nemotron will
                run them, and the supervisor will review the result.
              </p>
            </div>
          </div>
        )}

        {renderedEvents.map((evt) => (
          <TaskEventRow
            key={evt.event_id}
            event={evt}
            callResults={callResults}
            expanded={expandedCalls[evt.event_id] ?? false}
            onToggle={() =>
              setExpandedCalls((prev) => ({ ...prev, [evt.event_id]: !prev[evt.event_id] }))
            }
          />
        ))}

        {busy && !isTerminal && (
          <div className="flex items-center gap-2 text-[12px] text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>Working...</span>
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
            disabled={isTerminal || isRunning || busy}
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
              if (e.key === 'Enter' && !e.shiftKey && !busy && !isTerminal) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={
              isTerminal
                ? 'Task is finished. Start a new task to continue.'
                : isSuspended
                ? 'Respond to the supervisor...'
                : isRunning
                ? 'Task is running. Cancel first to send a new prompt.'
                : 'What would you like to accomplish?'
            }
            disabled={isTerminal || isRunning || busy}
            rows={2}
            className="flex-1 resize-none rounded border border-border/50 bg-background px-3 py-2 text-[13px] text-foreground outline-none placeholder:text-muted-foreground/60 focus:border-accent/50 focus:ring-1 focus:ring-accent/25 disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || busy || isTerminal || isRunning}
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

function TaskEventRow({
  event,
  callResults,
  expanded,
  onToggle,
}: {
  event: TaskEvent;
  callResults: Record<string, TaskEvent>;
  expanded: boolean;
  onToggle: () => void;
}) {
  switch (event.event_type) {
    case 'USER_PROMPT':
    case 'USER_RESPONSE': {
      const attachments = (event.payload?.attachments || []) as string[];
      return (
        <div className="flex justify-end">
          <div className="flex max-w-[80%] flex-col items-end gap-1">
            <div className="rounded-2xl rounded-br-sm bg-accent/90 px-3.5 py-2 text-[13px] text-white">
              {event.payload?.content}
            </div>
            {attachments.length > 0 && (
              <div className="flex flex-wrap justify-end gap-1">
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
        </div>
      );
    }

    case 'MANIFEST_SCOPED': {
      const selected: string[] = event.payload?.selected_tools || [];
      const reasoning: string = event.payload?.scoping_reasoning || '';
      return (
        <div className="rounded border border-border/40 bg-card/40 px-3 py-2">
          <div className="mb-1.5 flex items-center gap-1.5">
            <Package className="h-3 w-3 text-muted-foreground" />
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Tools selected
            </span>
          </div>
          <div className="flex flex-wrap gap-1">
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
        <div className="rounded border border-accent/20 bg-accent/5 px-3 py-2">
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
        <div className="rounded border border-amber-500/20 bg-amber-500/5 px-3 py-2">
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

    case 'TOOL_CALL_INTENT': {
      const toolName = event.payload?.tool_name || 'unknown';
      const args = event.payload?.arguments || {};
      const callId = event.payload?.call_id;
      const result = callId ? callResults[callId] : undefined;
      const resultStatus = result?.payload?.status as string | undefined;
      const resultSummary = result?.payload?.output?.summary as string | undefined;
      const elapsed = result?.payload?.execution_time_ms;
      const statusColor =
        resultStatus === 'success'
          ? 'text-emerald-500'
          : resultStatus?.startsWith('error')
          ? 'text-destructive'
          : resultStatus === 'requires_human'
          ? 'text-amber-500'
          : 'text-muted-foreground';
      return (
        <div className="rounded border border-border/40 bg-card/40">
          <button
            onClick={onToggle}
            className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-card/60"
          >
            <div className="flex items-center gap-2 min-w-0">
              {expanded ? (
                <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
              )}
              <Wrench className={`h-3 w-3 shrink-0 ${statusColor}`} />
              <span className="font-mono text-[12px] text-foreground truncate">{toolName}</span>
              {!result && (
                <Loader2 className="h-3 w-3 animate-spin shrink-0 text-muted-foreground" />
              )}
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {typeof elapsed === 'number' && (
                <span className="font-mono text-[10px] text-muted-foreground/70">
                  {elapsed}ms
                </span>
              )}
              {resultStatus && (
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
        <div className="rounded border border-sky-500/20 bg-sky-500/5 px-3 py-2">
          <div className="mb-1 flex items-center gap-1.5">
            <MessageSquare className="h-3 w-3 text-sky-500" />
            <span className="text-[11px] font-medium uppercase tracking-wider text-sky-500">
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
      const color =
        verdict === 'approve'
          ? 'emerald-500'
          : verdict === 'revise'
          ? 'amber-500'
          : verdict === 'rescope'
          ? 'sky-500'
          : 'destructive';
      return (
        <div className={`rounded border border-${color}/20 bg-${color}/5 px-3 py-2`}>
          <div className="mb-1 flex items-center gap-1.5">
            <span className={`text-[11px] font-medium uppercase tracking-wider text-${color}`}>
              Supervisor: {verdict}
            </span>
          </div>
          <p className="text-[12px] text-foreground/90 leading-relaxed">{reasoning}</p>
        </div>
      );
    }

    case 'FINAL_ANSWER':
      return (
        <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-4 py-3">
          <div className="mb-2 flex items-center gap-1.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
            <span className="text-[11px] font-medium uppercase tracking-wider text-emerald-500">
              Answer
            </span>
          </div>
          <div className="prose prose-sm max-w-none text-[13px] text-foreground leading-relaxed whitespace-pre-wrap">
            {event.payload?.answer}
          </div>
        </div>
      );

    case 'SYSTEM_CIRCUIT_BREAKER':
      return (
        <div className="rounded border border-amber-500/30 bg-amber-500/5 px-3 py-2">
          <div className="mb-1 flex items-center gap-1.5">
            <AlertTriangle className="h-3 w-3 text-amber-500" />
            <span className="text-[11px] font-medium uppercase tracking-wider text-amber-500">
              Circuit breaker: {event.payload?.reason}
            </span>
          </div>
        </div>
      );

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

function shouldRenderEvent(type: TaskEventType): boolean {
  // STATE_TRANSITION and TOOL_EXECUTION_RESULT are consumed by other rows; skip them.
  return (
    type !== 'STATE_TRANSITION' &&
    type !== 'TOOL_EXECUTION_RESULT' &&
    type !== 'ARTIFACT_WRITTEN' &&
    type !== 'LOG_COMPACTED' &&
    type !== 'SYSTEM_PLUGIN_VERSION_DRIFT'
  );
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
      return { label: 'Completed', icon: CheckCircle2, bg: 'bg-emerald-500/10', fg: 'text-emerald-500' };
    case 'cancelled':
      return { label: 'Cancelled', icon: XCircle, bg: 'bg-muted-foreground/10', fg: 'text-muted-foreground' };
    case 'failed':
      return { label: 'Failed', icon: XCircle, bg: 'bg-destructive/10', fg: 'text-destructive' };
    default:
      return { label: 'New task', icon: Sparkles, bg: 'bg-accent/10', fg: 'text-accent' };
  }
}
