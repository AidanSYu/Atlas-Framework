'use client';

import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Clock,
  Brain,
  BookOpen,
  Network,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronRight,
  Zap,
  Search,
  Wrench,
  MessageSquare,
  Shield,
  Ban,
  Loader2,
  History,
  ArrowRight,
} from 'lucide-react';
import { useRunStore, type Run, type RunStatus } from '@/stores/runStore';
import type { NormalizedEvent } from '@/lib/stream-adapter';
import type { ChatMode } from '@/hooks/useRunManager';

// ---------------------------------------------------------------------------
// Mode metadata
// ---------------------------------------------------------------------------

const MODE_META: Record<ChatMode, { label: string; icon: typeof BookOpen; color: string; bg: string }> = {
  librarian: { label: 'Librarian', icon: BookOpen, color: 'text-primary', bg: 'bg-primary/10' },
  cortex: { label: 'Cortex', icon: Brain, color: 'text-accent', bg: 'bg-accent/10' },
  moe: { label: 'MoE', icon: Network, color: 'text-blue-500', bg: 'bg-blue-500/10' },
};

const STATUS_META: Record<RunStatus, { label: string; icon: typeof CheckCircle2; color: string }> = {
  queued: { label: 'Queued', icon: Clock, color: 'text-muted-foreground' },
  routing: { label: 'Routing', icon: Loader2, color: 'text-accent' },
  awaiting_override: { label: 'Awaiting Override', icon: Clock, color: 'text-warning' },
  running: { label: 'Running', icon: Loader2, color: 'text-accent' },
  awaiting_input: { label: 'Awaiting Input', icon: Clock, color: 'text-blue-500' },
  completed: { label: 'Completed', icon: CheckCircle2, color: 'text-success' },
  failed: { label: 'Failed', icon: XCircle, color: 'text-destructive' },
  cancelled: { label: 'Cancelled', icon: Ban, color: 'text-warning' },
};

// ---------------------------------------------------------------------------
// Event rendering
// ---------------------------------------------------------------------------

function EventIcon({ type }: { type: string }) {
  switch (type) {
    case 'routing': return <Zap className="h-3 w-3 text-accent" />;
    case 'progress': return <ArrowRight className="h-3 w-3 text-primary" />;
    case 'run_start': return <Zap className="h-3 w-3 text-accent" />;
    case 'iteration_start': return <ArrowRight className="h-3 w-3 text-primary" />;
    case 'thinking': return <Brain className="h-3 w-3 text-purple-400" />;
    case 'tool_call': return <Wrench className="h-3 w-3 text-orange-500" />;
    case 'tool_result': return <CheckCircle2 className="h-3 w-3 text-success" />;
    case 'evidence': return <Search className="h-3 w-3 text-blue-400" />;
    case 'grounding': return <Shield className="h-3 w-3 text-success" />;
    case 'chunk': return <MessageSquare className="h-3 w-3 text-muted-foreground" />;
    case 'complete': return <CheckCircle2 className="h-3 w-3 text-success" />;
    case 'error': return <XCircle className="h-3 w-3 text-destructive" />;
    case 'cancelled': return <Ban className="h-3 w-3 text-warning" />;
    default: return <ArrowRight className="h-3 w-3 text-muted-foreground" />;
  }
}

function formatEventSummary(event: NormalizedEvent): string {
  switch (event.type) {
    case 'routing': return `Routed to ${event.mode} (${event.intent})`;
    case 'progress': return `${event.node}: ${event.message}`;
    case 'thinking': return event.content.length > 80 ? event.content.slice(0, 80) + '...' : event.content;
    case 'tool_call': return `Called ${event.tool}(${JSON.stringify(event.input).slice(0, 60)}...)`;
    case 'tool_result': return `${event.tool} → ${JSON.stringify(event.output).slice(0, 80)}`;
    case 'evidence': return `Found ${event.count} evidence items`;
    case 'grounding': return `"${event.claim.slice(0, 40)}..." → ${event.status} (${(event.confidence * 100).toFixed(0)}%)`;
    case 'chunk': return `Token stream (${event.content.length} chars)`;
    case 'complete': return 'Run completed';
    case 'error': return event.message;
    case 'cancelled': return 'Cancelled by user';
    case 'hypotheses': return `Generated ${event.items.length} hypotheses`;
    case 'graph_analysis': return 'Graph analysis complete';
    case 'run_start': return `Orchestrator ready: ${event.model ?? 'unknown'} (${event.availableTools.length} tools)`;
    case 'iteration_start': return `Iteration ${event.iteration}`;
    default: return JSON.stringify(event).slice(0, 80);
  }
}

// ---------------------------------------------------------------------------
// Run detail view
// ---------------------------------------------------------------------------

function RunDetail({ run }: { run: Run }) {
  const [expandedEvents, setExpandedEvents] = useState<Set<number>>(new Set());

  const meta = MODE_META[run.mode] || MODE_META.librarian;
  const statusMeta = STATUS_META[run.status];
  const StatusIcon = statusMeta.icon;
  const ModeIcon = meta.icon;
  const duration = run.completedAt && run.startedAt
    ? ((run.completedAt - run.startedAt) / 1000).toFixed(1)
    : run.startedAt ? ((Date.now() - run.startedAt) / 1000).toFixed(1) : '—';

  // Filter out chunk events (too noisy) for the timeline
  const timelineEvents = run.events.filter(e => e.type !== 'chunk');

  return (
    <div className="space-y-4">
      {/* Run header */}
      <div className="rounded-xl border border-border/50 bg-surface/30 p-4 space-y-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2.5">
            <div className={`flex h-9 w-9 items-center justify-center rounded-xl ${meta.bg}`}>
              <ModeIcon className={`h-4.5 w-4.5 ${meta.color}`} />
            </div>
            <div>
              <div className="text-sm font-medium text-foreground">{meta.label}</div>
              <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                <StatusIcon className={`h-3 w-3 ${statusMeta.color} ${run.status === 'running' ? 'animate-spin' : ''}`} />
                <span className={statusMeta.color}>{statusMeta.label}</span>
                <span>·</span>
                <Clock className="h-3 w-3" />
                <span>{duration}s</span>
              </div>
            </div>
          </div>
          <span className="rounded-md bg-muted/50 px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
            {run.id.slice(0, 8)}
          </span>
        </div>

        {run.query && (
          <div className="rounded-lg bg-background/50 p-2.5 text-xs text-foreground/80 leading-relaxed">
            "{run.query}"
          </div>
        )}

        <div className="flex gap-4 text-[11px] text-muted-foreground">
          <span>{run.events.length} events</span>
          <span>{run.events.filter(e => e.type === 'evidence').reduce((acc, e) => acc + ((e as any).count || 0), 0)} sources</span>
          <span>{run.events.filter(e => e.type === 'tool_call').length} tool calls</span>
        </div>
      </div>

      {/* Event timeline */}
      <div>
        <h4 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3 px-1">
          <Clock className="h-3 w-3" />
          Event Timeline
        </h4>
        <div className="space-y-0.5">
          {timelineEvents.map((event, i) => {
            const isExpanded = expandedEvents.has(i);
            const hasDetail = ['tool_call', 'tool_result', 'thinking', 'grounding'].includes(event.type);
            return (
              <div key={i}>
                <button
                  onClick={() => {
                    if (!hasDetail) return;
                    setExpandedEvents(prev => {
                      const next = new Set(prev);
                      if (next.has(i)) next.delete(i); else next.add(i);
                      return next;
                    });
                  }}
                  className={`w-full flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors ${
                    hasDetail ? 'hover:bg-surface/50 cursor-pointer' : 'cursor-default'
                  } ${isExpanded ? 'bg-surface/30' : ''}`}
                >
                  <div className="flex h-5 w-5 shrink-0 items-center justify-center">
                    <EventIcon type={event.type} />
                  </div>
                  <span className="flex-1 text-[11px] text-foreground/80 truncate">
                    {formatEventSummary(event)}
                  </span>
                  <span className="shrink-0 rounded bg-muted/30 px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground/60">
                    {event.type}
                  </span>
                  {hasDetail && (
                    <ChevronRight className={`h-3 w-3 text-muted-foreground/40 transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
                  )}
                </button>

                {isExpanded && hasDetail && (
                  <div className="ml-10 mr-2 mb-1 rounded-lg bg-zinc-950/50 p-2.5 font-mono text-[10px] text-muted-foreground overflow-x-auto max-h-[200px] overflow-y-auto">
                    <pre className="whitespace-pre-wrap break-words">
                      {JSON.stringify(event, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export interface RunAuditPanelProps {
  open: boolean;
  onClose: () => void;
  projectId: string;
  selectedRunId?: string | null;
}

export function RunAuditPanel({ open, onClose, projectId, selectedRunId }: RunAuditPanelProps) {
  const runHistory = useRunStore((s) => s.runHistory);
  const currentRun = useRunStore((s) => s.currentRun);
  const [filterMode, setFilterMode] = useState<ChatMode | 'all'>('all');
  const [selectedId, setSelectedId] = useState<string | null>(selectedRunId || null);

  // Sync external selection
  React.useEffect(() => {
    if (selectedRunId) setSelectedId(selectedRunId);
  }, [selectedRunId]);

  // Combine current run (if any) with history
  const allRuns = useMemo(() => {
    const combined = [...runHistory];
    if (currentRun && !combined.find(r => r.id === currentRun.id)) {
      combined.unshift(currentRun);
    }
    return combined.filter(r => r.projectId === projectId);
  }, [runHistory, currentRun, projectId]);

  const filteredRuns = useMemo(() => {
    if (filterMode === 'all') return allRuns;
    return allRuns.filter(r => r.mode === filterMode);
  }, [allRuns, filterMode]);

  const selectedRun = selectedId ? allRuns.find(r => r.id === selectedId) || null : null;

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[150] bg-background/40 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Drawer */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 bottom-0 z-[151] w-full max-w-[480px] border-l border-border/50 bg-card shadow-2xl shadow-black/30 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-border/50 px-5 py-4">
              <div className="flex items-center gap-2.5">
                <History className="h-4.5 w-4.5 text-accent" />
                <h2 className="text-sm font-semibold text-foreground">
                  {selectedRun ? 'Run Details' : 'Run History'}
                </h2>
                {!selectedRun && (
                  <span className="rounded-full bg-muted/50 px-2 py-0.5 text-[10px] text-muted-foreground">
                    {filteredRuns.length} runs
                  </span>
                )}
              </div>
              <button
                onClick={onClose}
                className="rounded-lg p-1.5 text-muted-foreground hover:text-foreground hover:bg-surface transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Back button if in detail view */}
            {selectedRun && (
              <button
                onClick={() => setSelectedId(null)}
                className="flex items-center gap-2 border-b border-border/30 px-5 py-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                <ChevronRight className="h-3 w-3 rotate-180" />
                Back to history
              </button>
            )}

            {/* Filter tabs (only in list view) */}
            {!selectedRun && (
              <div className="flex gap-1 border-b border-border/30 px-4 py-2">
                {(['all', 'librarian', 'cortex', 'moe'] as const).map((mode) => {
                  const isActive = filterMode === mode;
                  return (
                    <button
                      key={mode}
                      onClick={() => setFilterMode(mode)}
                      className={`rounded-lg px-2.5 py-1 text-[11px] font-medium transition-all ${
                        isActive
                          ? 'bg-primary/10 text-primary'
                          : 'text-muted-foreground hover:text-foreground hover:bg-surface'
                      }`}
                    >
                      {mode === 'all' ? 'All' : MODE_META[mode].label}
                    </button>
                  );
                })}
              </div>
            )}

            {/* Content */}
            <div className="flex-1 min-h-0 overflow-y-auto p-4">
              {selectedRun ? (
                <RunDetail run={selectedRun} />
              ) : filteredRuns.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <History className="h-10 w-10 text-muted-foreground/20 mb-3" />
                  <p className="text-sm text-muted-foreground/60">No runs yet</p>
                  <p className="text-xs text-muted-foreground/40 mt-1">
                    Ask a question to create your first run
                  </p>
                </div>
              ) : (
                <div className="space-y-1.5">
                  {filteredRuns.map((run) => {
                    const meta = MODE_META[run.mode] || MODE_META.librarian;
                    const statusMeta = STATUS_META[run.status];
                    const StatusIcon = statusMeta.icon;
                    const ModeIcon = meta.icon;
                    const duration = run.completedAt && run.startedAt
                      ? ((run.completedAt - run.startedAt) / 1000).toFixed(1) + 's'
                      : '...';

                    return (
                      <button
                        key={run.id}
                        onClick={() => setSelectedId(run.id)}
                        className="w-full flex items-start gap-3 rounded-xl border border-border/30 bg-surface/20 p-3 text-left hover:border-border/60 hover:bg-surface/40 transition-all group"
                      >
                        <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${meta.bg}`}>
                          <ModeIcon className={`h-4 w-4 ${meta.color}`} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-0.5">
                            <span className="text-xs font-medium text-foreground truncate">
                              {run.query ? (run.query.length > 50 ? run.query.slice(0, 50) + '...' : run.query) : 'No query'}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                            <StatusIcon className={`h-3 w-3 ${statusMeta.color} ${run.status === 'running' ? 'animate-spin' : ''}`} />
                            <span>{statusMeta.label}</span>
                            <span>·</span>
                            <span>{duration}</span>
                            <span>·</span>
                            <span>{run.events.length} events</span>
                          </div>
                        </div>
                        <ChevronRight className="h-4 w-4 text-muted-foreground/30 group-hover:text-muted-foreground transition-colors mt-1" />
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
