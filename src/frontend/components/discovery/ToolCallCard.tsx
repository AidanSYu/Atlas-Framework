'use client';

import React, { useState } from 'react';
import { Loader2, CheckCircle2, XCircle, ChevronDown, ChevronUp, Terminal } from 'lucide-react';
import type { ToolData } from '@/stores/discoveryConversationStore';

interface ToolCallCardProps {
  tool: ToolData;
}

const PLUGIN_COLORS: Record<string, { border: string; text: string; bg: string; dot: string }> = {
  standardize_smiles:    { border: 'border-blue-500/25',   text: 'text-blue-400',   bg: 'bg-blue-500/8',   dot: 'bg-blue-400' },
  predict_properties:    { border: 'border-cyan-500/25',   text: 'text-cyan-400',   bg: 'bg-cyan-500/8',   dot: 'bg-cyan-400' },
  check_toxicity:        { border: 'border-red-500/25',    text: 'text-red-400',    bg: 'bg-red-500/8',    dot: 'bg-red-400' },
  score_synthesizability:{ border: 'border-amber-500/25',  text: 'text-amber-400',  bg: 'bg-amber-500/8',  dot: 'bg-amber-400' },
  predict_admet:         { border: 'border-purple-500/25', text: 'text-purple-400', bg: 'bg-purple-500/8', dot: 'bg-purple-400' },
  plan_synthesis:        { border: 'border-green-500/25',  text: 'text-green-400',  bg: 'bg-green-500/8',  dot: 'bg-green-400' },
  evaluate_strategy:     { border: 'border-indigo-500/25', text: 'text-indigo-400', bg: 'bg-indigo-500/8', dot: 'bg-indigo-400' },
  verify_spectrum:       { border: 'border-teal-500/25',   text: 'text-teal-400',   bg: 'bg-teal-500/8',   dot: 'bg-teal-400' },
  propose_candidates:    { border: 'border-emerald-500/25',text: 'text-emerald-400',bg: 'bg-emerald-500/8',dot: 'bg-emerald-400' },
  enumerate_fragments:   { border: 'border-orange-500/25', text: 'text-orange-400', bg: 'bg-orange-500/8', dot: 'bg-orange-400' },
};

const FALLBACK_COLORS = { border: 'border-border/40', text: 'text-muted-foreground', bg: 'bg-surface/40', dot: 'bg-muted-foreground' };

function getColors(plugin: string) {
  return PLUGIN_COLORS[plugin] || FALLBACK_COLORS;
}

function StatsBlock({ stats, plugin }: { stats: Record<string, any>; plugin: string }) {
  if (!stats || Object.keys(stats).length === 0) return null;

  const items: React.ReactNode[] = [];

  if (plugin === 'standardize_smiles') {
    items.push(
      <span key="valid" className="text-emerald-400">{stats.valid} valid</span>,
      stats.dropped > 0 && <span key="drop" className="text-amber-400">{stats.dropped} dropped</span>,
    );
  } else if (plugin === 'predict_properties') {
    items.push(
      <span key="mw">MW {stats.mw_range}</span>,
      <span key="logp">LogP {stats.logp_range}</span>,
      <span key="lipo" className="text-emerald-400">{stats.lipinski_pass}/{stats.total} Lipinski pass</span>,
    );
  } else if (plugin === 'check_toxicity') {
    items.push(
      <span key="clean" className="text-emerald-400">{stats.clean} clean</span>,
      stats.flagged > 0 && <span key="flag" className="text-red-400">{stats.flagged} flagged</span>,
    );
    if (stats.top_alerts?.length > 0) {
      const alertStr = stats.top_alerts.map(([name, count]: [string, number]) => `${name}(${count})`).join(', ');
      items.push(<span key="alerts" className="text-red-400/70">alerts: {alertStr}</span>);
    }
  } else if (plugin === 'score_synthesizability') {
    items.push(
      <span key="feasible" className="text-emerald-400">{stats.feasible} feasible (SA≤6)</span>,
      stats.infeasible > 0 && <span key="inf" className="text-amber-400">{stats.infeasible} infeasible</span>,
      stats.best_sa != null && <span key="best">best SA {stats.best_sa}</span>,
    );
  } else if (plugin === 'predict_admet') {
    items.push(
      stats.low_risk > 0 && <span key="low" className="text-emerald-400">{stats.low_risk} low-risk</span>,
      stats.medium_risk > 0 && <span key="med" className="text-amber-400">{stats.medium_risk} medium-risk</span>,
      stats.high_risk > 0 && <span key="high" className="text-red-400">{stats.high_risk} high-risk</span>,
    );
  }

  const filtered = items.filter(Boolean);
  if (filtered.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-foreground/60 mt-1.5 font-mono">
      {filtered.map((item, i) => (
        <span key={i}>{item}</span>
      ))}
    </div>
  );
}

export function ToolCallCard({ tool }: ToolCallCardProps) {
  const [expanded, setExpanded] = useState(false);
  const colors = getColors(tool.plugin);
  const isRunning = tool.status === 'running';
  const isError = tool.status === 'error' || tool.error;
  const isDone = tool.status === 'complete' && !tool.error;
  const hasDetails = !!(tool.thinking || (tool.stats && Object.keys(tool.stats).length > 0));

  return (
    <div className={[
      'rounded-lg border overflow-hidden transition-all duration-200',
      colors.border, colors.bg,
    ].join(' ')}>
      {/* Header row */}
      <div
        className={['flex items-start gap-2 px-3 py-2', hasDetails ? 'cursor-pointer' : ''].join(' ')}
        onClick={() => hasDetails && setExpanded(!expanded)}
      >
        {/* Status icon */}
        <div className="mt-0.5 shrink-0">
          {isRunning && <Loader2 className="h-3.5 w-3.5 animate-spin text-foreground/50" />}
          {isDone && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />}
          {isError && <XCircle className="h-3.5 w-3.5 text-red-400" />}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={['text-[11px] font-semibold font-mono', colors.text].join(' ')}>
              {tool.plugin}
            </span>
            <span className="text-[10px] text-muted-foreground/50">
              {tool.stageId}/{tool.totalStages}
            </span>
            {isRunning && (
              <span className="text-[10px] text-foreground/40 animate-pulse">running...</span>
            )}
          </div>

          {/* Running: show description */}
          {isRunning && (
            <p className="text-[11px] text-foreground/60 mt-0.5 leading-snug">{tool.description}</p>
          )}

          {/* Done: show summary + stats */}
          {isDone && tool.summary && (
            <p className="text-[11px] text-foreground/75 mt-0.5 leading-snug">{tool.summary}</p>
          )}
          {isDone && tool.stats && (
            <StatsBlock stats={tool.stats} plugin={tool.plugin} />
          )}

          {/* Error */}
          {isError && tool.summary && (
            <p className="text-[11px] text-red-400/80 mt-0.5">{tool.summary}</p>
          )}
        </div>

        {/* Expand toggle */}
        {hasDetails && (
          <div className="shrink-0 mt-0.5">
            {expanded
              ? <ChevronUp className="h-3 w-3 text-muted-foreground/40" />
              : <ChevronDown className="h-3 w-3 text-muted-foreground/40" />
            }
          </div>
        )}
      </div>

      {/* Expandable: thinking text */}
      {expanded && tool.thinking && (
        <div className="px-3 pb-2.5 border-t border-current/10 pt-2">
          <div className="flex items-center gap-1.5 mb-1">
            <Terminal className="h-3 w-3 text-muted-foreground/40" />
            <span className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground/40">
              reasoning
            </span>
          </div>
          <p className="text-[10px] text-foreground/50 font-mono leading-relaxed whitespace-pre-wrap">
            {tool.thinking}
          </p>
        </div>
      )}
    </div>
  );
}
