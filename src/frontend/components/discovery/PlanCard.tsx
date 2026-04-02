'use client';

import React from 'react';
import { Check, X, Clock, FlaskConical, AlertTriangle, ChevronRight, Loader2 } from 'lucide-react';
import type { PlanData } from '@/stores/discoveryConversationStore';

interface PlanCardProps {
  plan: PlanData;
  onAccept: () => void;
  onReject: () => void;
  disabled?: boolean;
}

const PLUGIN_ICONS: Record<string, string> = {
  standardize_smiles: 'Std',
  predict_properties: 'Prop',
  check_toxicity: 'Tox',
  score_synthesizability: 'SA',
  predict_admet: 'ADMET',
  plan_synthesis: 'Syn',
  evaluate_strategy: 'Eval',
  verify_spectrum: 'NMR',
};

export function PlanCard({ plan, onAccept, onReject, disabled }: PlanCardProps) {
  const totalTime = plan.estimatedTotalSeconds;
  const minutes = Math.floor(totalTime / 60);
  const seconds = totalTime % 60;
  const timeStr = minutes > 0 ? `~${minutes}m ${seconds}s` : `~${seconds}s`;

  const isAccepted = plan.status === 'accepted';
  const isRejected = plan.status === 'rejected';
  const isResolved = isAccepted || isRejected;

  return (
    <div className={[
      'rounded-xl border overflow-hidden transition-all duration-300',
      isAccepted ? 'border-emerald-500/40 bg-emerald-500/5' :
      isRejected ? 'border-red-500/30 bg-red-500/5 opacity-60' :
      'border-accent/30 bg-accent/5',
    ].join(' ')}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/30">
        <div className="flex items-center gap-2">
          <FlaskConical className="h-4 w-4 text-accent" />
          <span className="text-xs font-semibold uppercase tracking-wider text-accent">
            Execution Plan
          </span>
          {plan.isDemoData && (
            <span className="rounded-full bg-amber-500/20 px-2 py-0.5 text-[9px] font-medium text-amber-500 border border-amber-500/30">
              DEMO DATA
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <Clock className="h-3 w-3" />
          <span>{timeStr}</span>
          <span className="text-border">|</span>
          <span>{plan.moleculeCount} molecules</span>
          <span className="text-border">|</span>
          <span>Iteration {plan.iteration}</span>
        </div>
      </div>

      <div className="px-4 py-3">
        <p className="text-sm text-foreground/90 mb-3">{plan.summary}</p>

        <div className="flex flex-wrap gap-1.5 mb-3">
          {plan.stages.map((stage) => (
            <div
              key={stage.stageId}
              className="flex items-center gap-1.5 rounded-lg bg-surface/60 border border-border/40 px-2.5 py-1.5"
            >
              <span className="text-[9px] font-bold text-accent/70 bg-accent/10 rounded px-1 py-0.5">
                {PLUGIN_ICONS[stage.plugin] || stage.stageId}
              </span>
              <span className="text-[11px] text-foreground/80">{stage.description}</span>
              <ChevronRight className="h-2.5 w-2.5 text-muted-foreground/40 last:hidden" />
            </div>
          ))}
        </div>

        {plan.warnings.length > 0 && (
          <div className="mb-3 space-y-1">
            {plan.warnings.map((w, i) => (
              <div key={i} className="flex items-start gap-2 text-[11px] text-amber-500/90">
                <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" />
                <span>{w}</span>
              </div>
            ))}
          </div>
        )}

        {plan.moleculeNotes && (
          <p className="text-[11px] text-muted-foreground mb-3 italic">{plan.moleculeNotes}</p>
        )}
      </div>

      {!isResolved && (
        <div className="flex items-center gap-2 border-t border-border/30 px-4 py-3 bg-surface/20">
          <button
            onClick={onAccept}
            disabled={disabled}
            className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-emerald-700 disabled:opacity-50"
          >
            <Check className="h-3.5 w-3.5" />
            Accept & Run
          </button>
          <button
            onClick={onReject}
            disabled={disabled}
            className="flex items-center gap-1.5 rounded-lg border border-border px-4 py-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-surface hover:text-foreground disabled:opacity-50"
          >
            <X className="h-3.5 w-3.5" />
            Reject
          </button>
          <span className="ml-auto text-[10px] text-muted-foreground">
            {plan.reasoning.slice(0, 80)}{plan.reasoning.length > 80 ? '...' : ''}
          </span>
        </div>
      )}

      {isAccepted && (
        <div className="flex items-center gap-2 border-t border-emerald-500/20 px-4 py-2.5 bg-emerald-500/5">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-500" />
          <span className="text-xs font-medium text-emerald-500">Plan accepted — executing...</span>
        </div>
      )}

      {isRejected && (
        <div className="flex items-center gap-2 border-t border-red-500/20 px-4 py-2.5 bg-red-500/5">
          <X className="h-3.5 w-3.5 text-red-400" />
          <span className="text-xs text-red-400">Plan rejected</span>
        </div>
      )}
    </div>
  );
}
