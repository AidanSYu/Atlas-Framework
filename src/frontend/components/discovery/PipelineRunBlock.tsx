'use client';

import React from 'react';
import { ChevronDown, ChevronUp, FlaskConical, CheckCircle2, Loader2 } from 'lucide-react';
import type { DiscoveryMessage, PipelineRun } from '@/stores/discoveryConversationStore';
import { ToolCallCard } from './ToolCallCard';
import { ResultsSummary } from './ResultsSummary';

interface PipelineRunBlockProps {
  run: PipelineRun;
  messages: DiscoveryMessage[];
  onToggle: () => void;
}

function timeAgo(ts: number): string {
  const sec = Math.floor((Date.now() - ts) / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ago`;
  return `${Math.floor(min / 60)}h ago`;
}

export function PipelineRunBlock({ run, messages, onToggle }: PipelineRunBlockProps) {
  const isComplete = run.completedAt != null;
  const isRunning = !isComplete;

  // Split messages by type
  const toolMessages = messages.filter(
    (m) => m.type === 'tool_start' || m.type === 'tool_complete',
  );
  const analysisMsg = messages.find((m) => m.type === 'analysis');

  const summaryLine = isComplete
    ? `${run.stagesCompleted} stages · ${run.candidateCount} candidates`
    : 'running...';

  return (
    <div className="rounded-xl border border-border/40 overflow-hidden bg-surface/20">
      {/* Collapsible header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-3.5 py-2.5 text-left hover:bg-surface/30 transition-colors"
      >
        <div className="flex items-center gap-1.5 flex-1 min-w-0">
          {isRunning
            ? <Loader2 className="h-3.5 w-3.5 text-emerald-400 animate-spin shrink-0" />
            : <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
          }
          <FlaskConical className="h-3 w-3 text-foreground/40 shrink-0" />
          <span className="text-[11px] font-semibold text-foreground/80">
            Pipeline Run #{run.iteration}
          </span>
          <span className="text-[10px] text-muted-foreground/50 truncate">
            — {summaryLine}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-[10px] text-muted-foreground/40">{timeAgo(run.startedAt)}</span>
          {run.collapsed
            ? <ChevronDown className="h-3.5 w-3.5 text-muted-foreground/40" />
            : <ChevronUp className="h-3.5 w-3.5 text-muted-foreground/40" />
          }
        </div>
      </button>

      {/* Expanded content */}
      {!run.collapsed && (
        <div className="px-3 pb-3 pt-1 space-y-2 border-t border-border/20">
          {toolMessages.map((m) => (
            m.tool && <ToolCallCard key={m.id} tool={m.tool} />
          ))}
          {analysisMsg?.analysis && (
            <ResultsSummary analysis={analysisMsg.analysis} />
          )}
        </div>
      )}
    </div>
  );
}
