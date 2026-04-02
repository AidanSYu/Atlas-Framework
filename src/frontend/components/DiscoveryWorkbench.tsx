'use client';

import React, { useEffect, useRef, useMemo } from 'react';
import { Terminal, Activity, FileText, Database, ShieldCheck, Loader2, Beaker, Zap, CheckCircle2, AlertTriangle, ChevronRight, Binary, Server, BarChart3 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Run } from '@/stores/runStore';
import { deriveStreamProgress } from './chat/RunProgressDisplay';

interface StreamProgress {
    currentNode: string;
    message: string;
    thinkingSteps: string[];
    evidenceFound: number;
    routing?: { brain: string; intent: string };
    activeTool?: { name: string; input: any };
    toolResults?: { tool: string; output: any }[];
    candidates?: any[];
}

interface DiscoveryWorkbenchProps {
    streamProgress?: StreamProgress | null;
    isLoading?: boolean;
    finalCandidates?: any[];
    run?: Run | null;
}

export function DiscoveryWorkbench({
    streamProgress: streamProgressProp,
    isLoading: isLoadingProp,
    finalCandidates,
    run,
}: DiscoveryWorkbenchProps) {
    const isRunBased = run != null;
    const streamProgress = isRunBased ? deriveStreamProgress(run) : (streamProgressProp ?? null);
    const isLoading = isRunBased
        ? run != null && !['completed', 'failed', 'cancelled', 'queued'].includes(run.status)
        : (isLoadingProp ?? false);
    const terminalEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll the terminal
    useEffect(() => {
        terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [streamProgress?.thinkingSteps, streamProgress?.activeTool, streamProgress?.toolResults]);

    const candidates = streamProgress?.candidates || finalCandidates || [];

    // Executor mode: progress is present, no routing (not a ReAct run), has executor step markers
    const isExecutorMode = streamProgress != null &&
      !streamProgress.routing &&
      (streamProgress.thinkingSteps ?? []).some(s =>
        s.startsWith('Executing:') || s.startsWith('Generating:') ||
        s.startsWith('Artifact') || s.startsWith('Parsed ') || s.startsWith('Complete:')
      );

    const spectrumResult = useMemo(() => {
        const results = streamProgress?.toolResults || [];
        const specRes = results.find(r => r.tool === 'verify_spectrum' && r.output?.valid);
        return specRes?.output ?? null;
    }, [streamProgress?.toolResults]);

    if (!isLoading && !streamProgress && candidates.length === 0) {
        return (
            <div className="flex h-full w-full flex-col items-center justify-center p-8 text-center bg-card/30">
                <div className="rounded-2xl bg-orange-500/10 p-6 mb-4 border border-orange-500/20">
                    <Beaker className="h-10 w-10 text-orange-500/60" />
                </div>
                <p className="font-serif text-lg text-foreground/80">Discovery OS Idle</p>
                <p className="mt-2 text-xs text-muted-foreground w-64 leading-relaxed">
                    Closed-loop chemical synthesis and property reasoning. Send a molecular query to observe deterministic tool execution.
                </p>
            </div>
        );
    }

    return (
        <div className="flex h-full w-full flex-col overflow-hidden bg-card/30 border-l border-border/50">
            <div className="flex shrink-0 items-center justify-between border-b border-border/50 bg-background/50 px-4 py-3">
                <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-orange-500" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        OS Telemetry
                    </span>
                </div>
                {isLoading && (
                    <div className="flex items-center gap-1.5 rounded-full bg-orange-500/10 px-2 py-0.5 text-[10px] font-medium text-orange-500 border border-orange-500/20">
                        <span className="relative flex h-1.5 w-1.5">
                            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-orange-400 opacity-75"></span>
                            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-orange-500"></span>
                        </span>
                        COMPUTING
                    </div>
                )}
            </div>

            <div className="flex flex-1 flex-col overflow-hidden p-4 gap-4">
                {/* Top: LLM semantic state + Deterministic CPU tool state */}
                <div className="shrink-0 grid grid-cols-2 gap-3">
                    <div className="flex flex-col gap-2 rounded-lg border border-border/50 bg-surface/40 p-3">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-1.5">
                                <Server className="h-3.5 w-3.5 text-accent" />
                                <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                                    {isExecutorMode ? 'Task Planner' : 'Semantic Layer'}
                                </span>
                            </div>
                            <span className="text-[10px] text-accent font-mono bg-accent/10 px-1.5 rounded">
                                {isExecutorMode ? 'DeepSeek' : 'GPU'}
                            </span>
                        </div>
                        <div className="text-xs font-medium text-foreground truncate mt-1">
                            {isExecutorMode
                                ? (streamProgress?.currentNode === 'scripting' ? 'Script Generation (MiniMax)'
                                    : streamProgress?.currentNode === 'complete' ? 'Done'
                                    : streamProgress?.currentNode === 'error' ? 'Error'
                                    : 'Planning Next Task')
                                : (streamProgress?.currentNode === 'think' ? 'Reasoning (LLM)' : 'Awaiting Tool')
                            }
                        </div>
                    </div>

                    <div className="flex flex-col gap-2 rounded-lg border border-border/50 bg-surface/40 p-3">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-1.5">
                                <Binary className="h-3.5 w-3.5 text-primary" />
                                <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                                    {isExecutorMode ? 'Script Executor' : 'Deterministic'}
                                </span>
                            </div>
                            <span className="text-[10px] text-primary font-mono bg-primary/10 px-1.5 rounded">
                                {isExecutorMode ? 'Subprocess' : 'CPU (ONNX)'}
                            </span>
                        </div>
                        <div className="text-xs font-medium text-foreground truncate mt-1">
                            {isExecutorMode
                                ? (streamProgress?.currentNode === 'executing'
                                    ? streamProgress.activeTool?.name || 'Running...'
                                    : 'Idle')
                                : (streamProgress?.currentNode === 'execute'
                                    ? streamProgress.activeTool?.name || 'Executing Plugin...'
                                    : 'Idle')
                            }
                        </div>
                    </div>
                </div>

                {/* Middle: Tool Execution Log */}
                <div className="flex-1 flex flex-col min-h-0 rounded-lg border border-border/50 bg-[#0F111A] text-zinc-300 font-mono text-xs shadow-inner overflow-hidden">
                    <div className="flex items-center justify-between shrink-0 px-3 py-2 border-b border-white/10 bg-white/5">
                        <div className="flex items-center gap-2">
                            <Terminal className="h-3.5 w-3.5 text-orange-500" />
                            <span className="text-zinc-400 font-sans text-xs font-medium">
                                {isExecutorMode ? 'Discovery OS Terminal' : 'ReAct Execution Trace'}
                            </span>
                        </div>
                    </div>
                    <div className="flex-1 overflow-y-auto p-3 space-y-3 custom-scrollbar">
                        {streamProgress?.thinkingSteps && streamProgress.thinkingSteps.map((step, idx) => (
                            <div key={idx} className="flex gap-2">
                                <span className="text-zinc-600 select-none">{`>`}</span>
                                <span className={
                                    step.includes('Calling **') ? 'text-primary drop-shadow-[0_0_8px_rgba(56,189,248,0.3)]' :
                                        step.includes('Result:') ? 'text-success drop-shadow-[0_0_8px_rgba(74,222,128,0.3)]' :
                                            step.includes('Thought:') ? 'text-accent' :
                                                'text-zinc-400'
                                }>
                                    {step.replace('Thought: ', '')}
                                </span>
                            </div>
                        ))}
                        {streamProgress?.activeTool && streamProgress.currentNode === 'execute' && (
                            <div className="flex gap-2 text-primary animate-pulse">
                                <span className="select-none text-zinc-600">{`>`}</span>
                                <span>Executing deterministic plugin: {streamProgress.activeTool.name}...</span>
                            </div>
                        )}
                        <div ref={terminalEndRef} />
                    </div>
                </div>

                {/* Spectrum Verification Results */}
                {spectrumResult && (
                    <div className="shrink-0 rounded-lg border border-cyan-500/20 bg-cyan-500/5 overflow-hidden">
                        <div className="flex items-center gap-2 px-3 py-2 border-b border-cyan-500/20 bg-cyan-500/10">
                            <BarChart3 className="h-4 w-4 text-cyan-500" />
                            <span className="text-xs font-semibold text-cyan-500">NMR Spectrum Verification</span>
                            <div className="ml-auto text-[10px] text-cyan-500/80 bg-cyan-500/10 px-1.5 rounded font-mono">
                                {spectrumResult.file}
                            </div>
                        </div>
                        <div className="p-3 space-y-3">
                            {/* Match Score Gauge */}
                            <div className="flex items-center gap-3">
                                <div className="flex-1">
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Match Score</span>
                                        <span className={`text-sm font-bold ${
                                            spectrumResult.match_score == null ? 'text-muted-foreground' :
                                            spectrumResult.match_score >= 0.8 ? 'text-green-400' :
                                            spectrumResult.match_score >= 0.5 ? 'text-yellow-400' : 'text-red-400'
                                        }`}>
                                            {spectrumResult.match_score != null ? `${(spectrumResult.match_score * 100).toFixed(0)}%` : 'N/A'}
                                        </span>
                                    </div>
                                    {spectrumResult.match_score != null && (
                                        <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                                            <div
                                                className={`h-full rounded-full transition-all duration-500 ${
                                                    spectrumResult.match_score >= 0.8 ? 'bg-green-400' :
                                                    spectrumResult.match_score >= 0.5 ? 'bg-yellow-400' : 'bg-red-400'
                                                }`}
                                                style={{ width: `${Math.min(100, spectrumResult.match_score * 100)}%` }}
                                            />
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Peak Count Comparison */}
                            <div className="grid grid-cols-3 gap-2">
                                <div className="rounded-md bg-background/30 border border-border/30 p-2 text-center">
                                    <div className="text-lg font-bold text-foreground">{spectrumResult.peak_count}</div>
                                    <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Observed</div>
                                </div>
                                <div className="rounded-md bg-background/30 border border-border/30 p-2 text-center">
                                    <div className="text-lg font-bold text-foreground">{spectrumResult.expected_h_count ?? '—'}</div>
                                    <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Expected H</div>
                                </div>
                                <div className="rounded-md bg-background/30 border border-border/30 p-2 text-center">
                                    <div className="text-[10px] font-medium text-muted-foreground mt-1">
                                        {spectrumResult.metadata?.nucleus || '—'}
                                    </div>
                                    <div className="text-[9px] text-muted-foreground uppercase tracking-wider">Nucleus</div>
                                </div>
                            </div>

                            {/* Peak Table */}
                            {spectrumResult.observed_peaks && spectrumResult.observed_peaks.length > 0 && (
                                <div className="max-h-[120px] overflow-y-auto rounded border border-border/30 custom-scrollbar">
                                    <table className="w-full text-[10px] border-collapse">
                                        <thead className="sticky top-0 bg-cyan-500/10 text-cyan-500/70">
                                            <tr>
                                                <th className="px-2 py-1 text-left font-medium">#</th>
                                                <th className="px-2 py-1 text-right font-medium">ppm</th>
                                                <th className="px-2 py-1 text-right font-medium">Intensity</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {spectrumResult.observed_peaks.slice(0, 20).map((peak: any, idx: number) => (
                                                <tr key={idx} className="border-b border-border/20 hover:bg-cyan-500/5">
                                                    <td className="px-2 py-1 text-muted-foreground">{idx + 1}</td>
                                                    <td className="px-2 py-1 text-right font-mono text-foreground">{peak.ppm?.toFixed(3)}</td>
                                                    <td className="px-2 py-1 text-right font-mono text-muted-foreground">{peak.intensity?.toFixed(3)}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Bottom: Candidates Database */}
                <div className={`shrink-0 ${spectrumResult ? 'h-1/4' : 'h-2/5'} flex flex-col rounded-lg border border-orange-500/20 bg-orange-500/5 overflow-hidden`}>
                    <div className="flex items-center gap-2 px-3 py-2 border-b border-orange-500/20 bg-orange-500/10 shrink-0">
                        <Database className="h-4 w-4 text-orange-500" />
                        <span className="text-xs font-semibold text-orange-500">
                            Hit List
                        </span>
                        <div className="ml-auto text-[10px] text-orange-500/80 bg-orange-500/10 px-1.5 rounded font-mono">
                            {candidates.length} CANDIDATES
                        </div>
                    </div>
                    <div className="flex-1 overflow-y-auto p-0 min-h-0 custom-scrollbar">
                        {candidates.length === 0 ? (
                            <div className="flex h-full items-center justify-center p-4 text-[11px] text-orange-500/50 italic">
                                Run the pipeline to populate candidates
                            </div>
                        ) : (
                            <table className="w-full text-left border-collapse text-[11px]">
                                <thead>
                                    <tr className="bg-orange-500/5 border-b border-orange-500/10 text-orange-500/70 uppercase sticky top-0">
                                        <th className="px-2 py-2 font-medium">#</th>
                                        <th className="px-2 py-2 font-medium">SMILES</th>
                                        <th className="px-2 py-2 font-medium">MW</th>
                                        <th className="px-2 py-2 font-medium">LogP</th>
                                        <th className="px-2 py-2 font-medium">SA</th>
                                        <th className="px-2 py-2 font-medium">hERG</th>
                                        <th className="px-2 py-2 font-medium">Safety</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {candidates.map((c: any, i: number) => {
                                        const riskColor = (risk: string) => {
                                            if (!risk || risk === 'N/A') return 'text-muted-foreground';
                                            if (risk === 'LOW') return 'text-green-400';
                                            if (risk === 'MEDIUM') return 'text-yellow-400';
                                            return 'text-red-400';
                                        };
                                        return (
                                        <tr key={i} className="border-b border-orange-500/10 hover:bg-orange-500/5 transition-colors">
                                            <td className="px-2 py-1.5 text-muted-foreground">{i + 1}</td>
                                            <td className="px-2 py-1.5 font-mono text-foreground/90 truncate max-w-[120px]" title={c.smiles}>
                                                {c.compound_id || c.smiles?.slice(0, 20)}
                                            </td>
                                            <td className="px-2 py-1.5 text-muted-foreground whitespace-nowrap">
                                                {c.properties?.MolWt != null ? Number(c.properties.MolWt).toFixed(0) : '-'}
                                            </td>
                                            <td className="px-2 py-1.5 text-muted-foreground whitespace-nowrap">
                                                {c.properties?.LogP != null ? Number(c.properties.LogP).toFixed(1) : '-'}
                                            </td>
                                            <td className="px-2 py-1.5 text-muted-foreground whitespace-nowrap">
                                                {c.sa_score != null ? Number(c.sa_score).toFixed(1) : '-'}
                                            </td>
                                            <td className={`px-2 py-1.5 whitespace-nowrap font-medium ${riskColor(c.admet?.herg_risk)}`}>
                                                {c.admet?.herg_risk || '-'}
                                            </td>
                                            <td className="px-2 py-1.5 whitespace-nowrap">
                                                {c.toxicity ? (
                                                    c.toxicity.clean ? (
                                                        <span className="flex items-center gap-1 text-green-400 text-[10px]">
                                                            <CheckCircle2 className="h-3 w-3" /> Pass
                                                        </span>
                                                    ) : (
                                                        <span className="flex items-center gap-1 text-red-400 text-[10px]" title={`${c.toxicity.alert_count} alerts`}>
                                                            <AlertTriangle className="h-3 w-3" /> Fail
                                                        </span>
                                                    )
                                                ) : (
                                                    <span className={`text-[10px] font-medium ${riskColor(c.admet?.overall)}`}>
                                                        {c.admet?.overall || '-'}
                                                    </span>
                                                )}
                                            </td>
                                        </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
