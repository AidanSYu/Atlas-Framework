'use client';

import React, { useState } from 'react';
import {
  FlaskConical, AlertTriangle, ArrowRight,
  ChevronDown, ChevronUp, Beaker, Microscope, Cpu, RotateCw, Trophy,
} from 'lucide-react';
import type { AnalysisData } from '@/stores/discoveryConversationStore';

interface ResultsSummaryProps {
  analysis: AnalysisData;
}

const ACTION_ICONS: Record<string, React.ReactNode> = {
  wetlab_validation:    <Beaker className="h-3 w-3" />,
  additional_modeling:  <Cpu className="h-3 w-3" />,
  iterate_constraints:  <RotateCw className="h-3 w-3" />,
  manual_review:        <Microscope className="h-3 w-3" />,
};

const PRIORITY_RING: Record<string, string> = {
  high:   'border-red-500/30 bg-red-500/5 text-red-300',
  medium: 'border-amber-500/30 bg-amber-500/5 text-amber-300',
  low:    'border-blue-500/30 bg-blue-500/5 text-blue-300',
};

function ScoreDot({ score }: { score?: number }) {
  if (score == null) return null;
  const pct = Math.round(score * 100);
  const color = pct >= 60 ? 'bg-emerald-400' : pct >= 35 ? 'bg-amber-400' : 'bg-red-400';
  return (
    <span
      title={`Composite score: ${score.toFixed(3)}`}
      className={['inline-block w-1.5 h-1.5 rounded-full shrink-0 mt-1', color].join(' ')}
    />
  );
}

export function ResultsSummary({ analysis }: ResultsSummaryProps) {
  const [showAll, setShowAll] = useState(false);

  const hasCandidates = analysis.topCandidates.length > 0;
  const hasFindings = analysis.keyFindings.length > 0;
  const hasConcerns = analysis.concerns.length > 0;
  const hasRecs = analysis.recommendations.length > 0;

  // Show up to 3 candidates collapsed, all when expanded
  const visibleCandidates = showAll
    ? analysis.topCandidates
    : analysis.topCandidates.slice(0, 3);

  return (
    <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/4 overflow-hidden text-[11px]">
      {/* Header */}
      <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-emerald-500/15 bg-emerald-500/8">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-3.5 w-3.5 text-emerald-400" />
          <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-400">
            Analysis
          </span>
        </div>
        <button
          onClick={() => setShowAll(!showAll)}
          className="flex items-center gap-1 text-[10px] text-emerald-400/50 hover:text-emerald-400 transition-colors"
        >
          {showAll ? 'Less' : 'More'}
          {showAll ? <ChevronUp className="h-2.5 w-2.5" /> : <ChevronDown className="h-2.5 w-2.5" />}
        </button>
      </div>

      <div className="px-3.5 py-3 space-y-3.5">
        {/* Key Findings — tool-attributed, concrete */}
        {hasFindings && (
          <div>
            <p className="text-[9px] font-semibold uppercase tracking-wider text-foreground/35 mb-1.5">
              Findings
            </p>
            <ul className="space-y-1.5">
              {analysis.keyFindings.map((f, i) => (
                <li key={i} className="flex items-start gap-2 leading-snug text-foreground/80">
                  <span className="text-emerald-400/60 mt-0.5 shrink-0 font-mono text-[9px]">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Top Candidates — deterministically ranked, SMILES shown */}
        {hasCandidates && (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <p className="text-[9px] font-semibold uppercase tracking-wider text-foreground/35">
                Top Candidates
              </p>
              <span className="text-[9px] text-muted-foreground/40">
                ranked by QED × safety × SA × ADMET
              </span>
            </div>
            <div className="space-y-1.5">
              {visibleCandidates.map((c, i) => (
                <div
                  key={i}
                  className="rounded-lg bg-surface/30 border border-border/25 px-2.5 py-2"
                >
                  <div className="flex items-start gap-2">
                    <div className="flex items-center gap-1 shrink-0">
                      <Trophy className="h-2.5 w-2.5 text-amber-400/60" />
                      <span className="text-[9px] font-bold text-foreground/40">#{i + 1}</span>
                    </div>
                    <ScoreDot score={c.composite_score} />
                    <div className="min-w-0 flex-1">
                      <code className="text-[10px] font-mono text-foreground/65 break-all leading-tight block">
                        {c.smiles || <span className="text-muted-foreground/40 italic">no SMILES</span>}
                      </code>
                      {c.reasoning && (
                        <p className="text-[10px] text-muted-foreground/70 mt-0.5 leading-snug">{c.reasoning}</p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              {!showAll && analysis.topCandidates.length > 3 && (
                <button
                  onClick={() => setShowAll(true)}
                  className="text-[10px] text-muted-foreground/50 hover:text-foreground/70 transition-colors w-full text-center py-0.5"
                >
                  +{analysis.topCandidates.length - 3} more
                </button>
              )}
            </div>
          </div>
        )}

        {/* Concerns */}
        {hasConcerns && (
          <div>
            <p className="text-[9px] font-semibold uppercase tracking-wider text-foreground/35 mb-1.5">
              Concerns
            </p>
            <ul className="space-y-1">
              {analysis.concerns.map((c, i) => (
                <li key={i} className="flex items-start gap-2 text-amber-400/80 leading-snug">
                  <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" />
                  <span>{c}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Recommendations — shown when expanded */}
        {showAll && hasRecs && (
          <div>
            <p className="text-[9px] font-semibold uppercase tracking-wider text-foreground/35 mb-1.5">
              Next Steps
            </p>
            <div className="space-y-1.5">
              {analysis.recommendations.map((r, i) => (
                <div
                  key={i}
                  className={[
                    'flex items-start gap-2 rounded-lg border px-2.5 py-2',
                    PRIORITY_RING[r.priority] || PRIORITY_RING.medium,
                  ].join(' ')}
                >
                  {ACTION_ICONS[r.action] || <ArrowRight className="h-3 w-3 shrink-0" />}
                  <div className="min-w-0 flex-1">
                    <p className="text-[10px] font-medium capitalize">{r.action.replace(/_/g, ' ')}</p>
                    <p className="text-[10px] opacity-65 mt-0.5">{r.description}</p>
                  </div>
                  <span className="text-[9px] font-semibold uppercase shrink-0 opacity-60">{r.priority}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Missing Capabilities — shown when expanded */}
        {showAll && analysis.missingCapabilities.length > 0 && (
          <div>
            <p className="text-[9px] font-semibold uppercase tracking-wider text-foreground/35 mb-1.5">
              Would Help
            </p>
            <div className="flex flex-wrap gap-1.5">
              {analysis.missingCapabilities.map((cap, i) => (
                <span
                  key={i}
                  className="rounded-full border border-border/30 bg-surface/30 px-2 py-0.5 text-[10px] text-muted-foreground/60"
                >
                  {cap}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
