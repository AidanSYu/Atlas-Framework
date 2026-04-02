'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useDiscoveryStore } from '@/stores/discoveryStore';
import { useDiscoveryConversation } from '@/stores/discoveryConversationStore';
import { DiscoveryChat } from './discovery/DiscoveryChat';
import { PluginManager } from './PluginManager';
import {
  Database, FileText, Loader2, X, Eye, CheckCircle2, AlertTriangle,
  Puzzle, ChevronDown, ChevronUp,
} from 'lucide-react';
import { api } from '@/lib/api';
import { usePanelResize } from '@/hooks/usePanelResize';

interface DiscoveryWorkspaceTabProps {
  sessionId: string;
  projectId: string;
}

function DragHandle({
  direction,
  onMouseDown,
  isDragging,
}: {
  direction: 'horizontal' | 'vertical';
  onMouseDown: (e: React.MouseEvent) => void;
  isDragging: boolean;
}) {
  const isH = direction === 'horizontal';
  return (
    <div
      onMouseDown={onMouseDown}
      className={[
        'group relative shrink-0 flex items-center justify-center',
        'transition-colors duration-100 select-none',
        isH
          ? 'w-[5px] cursor-col-resize hover:bg-safety/20 active:bg-safety/30'
          : 'h-[5px] cursor-row-resize hover:bg-safety/20 active:bg-safety/30',
        isDragging ? (isH ? 'bg-safety/30' : 'bg-safety/30') : 'bg-border/60',
      ].join(' ')}
    >
      <div
        className={[
          'rounded-full bg-border group-hover:bg-safety/60 transition-colors duration-100',
          isDragging ? 'bg-safety/60' : '',
          isH ? 'w-[2px] h-8' : 'h-[2px] w-8',
        ].join(' ')}
      />
    </div>
  );
}

type RightTab = 'results' | 'files' | 'plugins';

export function DiscoveryWorkspaceTab({ sessionId, projectId }: DiscoveryWorkspaceTabProps) {
  const session = useDiscoveryStore((s) => s.sessions[sessionId]);
  // Read candidates from the conversation store — populated when pipeline_complete fires
  const storeCandidates = useDiscoveryConversation((s) => s.getOrCreate(sessionId).candidates);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rightTab, setRightTab] = useState<RightTab>('results');
  const [sessionStage, setSessionStage] = useState('setup');

  const { size: rightWidth, handleMouseDown: handleHDrag, isDragging: isHDragging } = usePanelResize({
    initialSize: 420,
    minSize: 280,
    maxSize: 700,
    direction: 'horizontal',
    storageKey: 'discovery-right-panel-width',
  });

  const [previewFile, setPreviewFile] = useState<{
    filename: string;
    content: string;
  } | null>(null);

  const handleViewFile = useCallback(async (filePath: string) => {
    try {
      const file = await api.readSessionFile(sessionId, filePath);
      setPreviewFile({ filename: file.filename, content: file.content });
    } catch (err) {
      console.error('Failed to read session file:', err);
    }
  }, [sessionId]);

  const refreshSessionFiles = useCallback(async () => {
    try {
      const files = await api.getSessionFiles(sessionId);
      useDiscoveryStore.setState((state) => {
        if (state.sessions[sessionId]) {
          return {
            ...state,
            sessions: {
              ...state.sessions,
              [sessionId]: {
                ...state.sessions[sessionId],
                generatedFiles: files.map((f) => f.path),
              },
            },
          };
        }
        return state;
      });
    } catch (err) {
      console.error('Failed to refresh session files:', err);
    }
  }, [sessionId]);

  useEffect(() => {
    async function loadSessionFiles() {
      try {
        setIsLoading(true);
        setError(null);
        const files = await api.getSessionFiles(sessionId);
        useDiscoveryStore.setState((state) => {
          if (state.sessions[sessionId]) {
            return {
              ...state,
              sessions: {
                ...state.sessions,
                [sessionId]: {
                  ...state.sessions[sessionId],
                  generatedFiles: files.map(f => f.path),
                },
              },
            };
          }
          return state;
        });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (!msg.includes('404') && !msg.toLowerCase().includes('not found')) {
          console.error('Failed to load session files:', err);
        }
      } finally {
        setIsLoading(false);
      }
    }
    loadSessionFiles();
  }, [sessionId]);

  const handleCandidatesUpdate = useCallback((_newCandidates: any[]) => {
    // Candidates are stored in discoveryConversationStore; also refresh files
    refreshSessionFiles();
    // Auto-switch to results tab when candidates arrive
    setRightTab('results');
  }, [refreshSessionFiles]);

  const handleStageChange = useCallback((stage: string) => {
    setSessionStage(stage);
    if (stage === 'ready') {
      refreshSessionFiles();
    }
  }, [refreshSessionFiles]);

  // Use candidates from conversation store (single source of truth)
  const candidates = storeCandidates;

  if (!session) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="text-center">
          <p className="text-sm text-muted-foreground">Session not found: {sessionId}</p>
          <p className="text-xs text-muted-foreground mt-2">This session may have been deleted or corrupted.</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-8 w-8 animate-spin text-orange-500" />
          <div className="text-center">
            <p className="text-sm font-medium text-foreground">Loading Discovery Session</p>
            <p className="text-xs text-muted-foreground mt-1">{session.sessionName}</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="text-center">
          <p className="text-sm text-destructive">Error: {error}</p>
          <p className="text-xs text-muted-foreground mt-2">Failed to load session data</p>
        </div>
      </div>
    );
  }

  const riskColor = (risk: string) => {
    if (!risk || risk === 'N/A') return 'text-muted-foreground';
    if (risk === 'LOW') return 'text-green-400';
    if (risk === 'MEDIUM') return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="flex h-full w-full overflow-hidden">
      {/* ========== LEFT: Discovery Chat (primary) ========== */}
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <DiscoveryChat
          sessionId={sessionId}
          projectId={projectId}
          onCandidatesUpdate={handleCandidatesUpdate}
          onStageChange={handleStageChange}
        />
      </div>

      {/* ========== DRAG HANDLE ========== */}
      <DragHandle direction="horizontal" onMouseDown={handleHDrag} isDragging={isHDragging} />

      {/* ========== RIGHT: Context Panel ========== */}
      <div
        className="flex shrink-0 flex-col border-l border-border bg-card/50 overflow-hidden"
        style={{ width: rightWidth }}
      >
        {/* Tab bar */}
        <div className="flex h-10 shrink-0 border-b border-border bg-surface/30">
          {([
            { key: 'results' as RightTab, label: 'Results', icon: Database, count: candidates.length },
            { key: 'files' as RightTab, label: 'Files', icon: FileText, count: session.generatedFiles.length },
            { key: 'plugins' as RightTab, label: 'Plugins', icon: Puzzle },
          ]).map(({ key, label, icon: Icon, count }) => (
            <button
              key={key}
              onClick={() => setRightTab(key)}
              className={[
                'flex items-center gap-1.5 px-3 h-full text-[11px] font-medium transition-colors border-b-2',
                rightTab === key
                  ? 'border-emerald-500 text-foreground'
                  : 'border-transparent text-muted-foreground hover:text-foreground',
              ].join(' ')}
            >
              <Icon className="h-3 w-3" />
              {label}
              {count != null && count > 0 && (
                <span className="text-[9px] bg-surface rounded-full px-1.5 py-0.5">{count}</span>
              )}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="flex-1 overflow-y-auto">
          {/* Results tab */}
          {rightTab === 'results' && (
            <div className="h-full flex flex-col">
              {candidates.length === 0 ? (
                <div className="flex h-full items-center justify-center p-6 text-center">
                  <div>
                    <Database className="h-8 w-8 text-muted-foreground/30 mx-auto mb-2" />
                    <p className="text-xs text-muted-foreground">No candidates yet</p>
                    <p className="text-[10px] text-muted-foreground/60 mt-1">Run the pipeline to populate results</p>
                  </div>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-[11px]">
                    <thead>
                      <tr className="bg-orange-500/5 border-b border-orange-500/10 text-orange-500/70 uppercase sticky top-0 z-10">
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
                      {candidates.map((c: any, i: number) => (
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
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {/* Files tab */}
          {rightTab === 'files' && (
            <div className="p-2">
              {session.generatedFiles.length === 0 ? (
                <p className="text-xs text-muted-foreground italic text-center py-6">
                  No files generated yet. Complete setup to create session files.
                </p>
              ) : (
                <div className="space-y-1">
                  {session.generatedFiles.map((file, idx) => {
                    const isViewable = /\.(md|json|txt|csv|py|log)$/i.test(file);
                    return (
                      <button
                        key={idx}
                        onClick={() => isViewable && handleViewFile(file)}
                        className={[
                          'flex w-full items-center gap-2 rounded px-2 py-1.5 text-xs transition-colors group text-left',
                          isViewable
                            ? 'hover:bg-primary/10 cursor-pointer'
                            : 'hover:bg-surface cursor-default',
                        ].join(' ')}
                      >
                        <FileText className={`h-3 w-3 shrink-0 ${file.endsWith('.md') ? 'text-emerald-500' : 'text-muted-foreground'}`} />
                        <span className="flex-1 truncate font-mono text-foreground/80">{file}</span>
                        {isViewable && (
                          <Eye className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Plugins tab */}
          {rightTab === 'plugins' && (
            <PluginManager projectId={projectId} />
          )}
        </div>
      </div>

      {/* File Preview Overlay */}
      {previewFile && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="relative mx-4 flex max-h-[80vh] w-full max-w-3xl flex-col rounded-xl border border-border bg-card shadow-2xl">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <div className="flex items-center gap-2">
                <FileText className={`h-4 w-4 ${previewFile.filename.endsWith('.md') ? 'text-emerald-500' : 'text-muted-foreground'}`} />
                <span className="text-sm font-medium text-foreground">{previewFile.filename}</span>
              </div>
              <button
                onClick={() => setPreviewFile(null)}
                className="rounded-lg p-1.5 text-muted-foreground hover:bg-surface hover:text-foreground transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              <pre className="whitespace-pre-wrap text-xs font-mono text-foreground/90 leading-relaxed">
                {previewFile.content}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
