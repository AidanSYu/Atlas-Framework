'use client';

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import {
  Upload,
  FileText,
  Trash2,
  Loader2,
  Search,
  ChevronDown,
  ChevronRight,
  FolderOpen,
  Sparkles,
  X,
  CheckCircle2,
  Clock,
  AlertCircle,
  FlaskConical,
  Plus,
} from 'lucide-react';
import { api, FileInfo } from '@/lib/api';
import { toastError, toast, toastSuccess } from '@/stores/toastStore';
import { useDiscoveryStore } from '@/stores/discoveryStore';

interface LibrarySidebarProps {
  onFileSelect: (docId: string, filename: string) => void;
  selectedDocId: string | null;
  projectId: string;
  onIngestionComplete?: () => void;
  onFileDeleted?: (docId: string) => void;
  onOpenDiscoverySession?: (sessionId: string, sessionName: string) => void;
  onStartDiscovery?: () => void;
  /** Increment to trigger a refetch of the file list (e.g. after upload/import from menu) */
  refreshTrigger?: number;
}

export default function LibrarySidebar({
  onFileSelect,
  selectedDocId,
  projectId,
  onIngestionComplete,
  onFileDeleted,
  onOpenDiscoverySession,
  onStartDiscovery,
  refreshTrigger,
}: LibrarySidebarProps) {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [smartGroupsOpen, setSmartGroupsOpen] = useState(true);
  const [discoverySessionsOpen, setDiscoverySessionsOpen] = useState(true);
  const [allDocsOpen, setAllDocsOpen] = useState(true);
  const prevProcessingRef = useRef(false);

  const sessions = useDiscoveryStore((s) => s.sessions);
  const upsertBackendSession = useDiscoveryStore((s) => s.upsertBackendSession);
  // Only show sessions that belong to this project
  const sessionList = useMemo(
    () => Object.values(sessions).filter((s) => s.projectId === projectId),
    [sessions, projectId]
  );

  // Sync persisted sessions from the backend on mount so sessions from previous
  // runs appear even if localStorage was cleared or the store is stale.
  useEffect(() => {
    if (!projectId) return;
    api.listDiscoverySessions(projectId).then((backendSessions) => {
      backendSessions.forEach((s) => {
        upsertBackendSession(s.session_id, s.session_name, s.created_at ?? new Date().toISOString(), projectId);
      });
    }).catch(() => {
      // Backend not ready yet — sessions will appear once it connects
    });
  }, [projectId, upsertBackendSession]);

  const loadFiles = useCallback(
    async (silent = false) => {
      if (!projectId) return;
      try {
        if (!silent) setLoading(true);
        const fileList = await api.listFiles(projectId);
        setFiles(fileList);
      } catch (error) {
        console.error('Failed to load files:', error);
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [projectId]
  );

  useEffect(() => {
    loadFiles();
  }, [loadFiles]);

  useEffect(() => {
    if (refreshTrigger != null && refreshTrigger > 0) loadFiles(true);
  }, [refreshTrigger, loadFiles]);

  const hasProcessing = files.some((f) => f.status === 'processing');

  useEffect(() => {
    if (!hasProcessing) {
      if (prevProcessingRef.current && onIngestionComplete) {
        onIngestionComplete();
      }
      prevProcessingRef.current = false;
      return;
    }
    prevProcessingRef.current = true;
    const interval = window.setInterval(() => loadFiles(true), 3000);
    return () => window.clearInterval(interval);
  }, [hasProcessing, loadFiles, onIngestionComplete]);

  const handleDelete = async (e: React.MouseEvent, fileId: string) => {
    e.stopPropagation();
    try {
      await api.deleteFile(fileId);
      onFileDeleted?.(fileId);
      setFiles((prev) => prev.filter((f) => f.doc_id !== fileId));
    } catch (err) {
      console.error(err);
    }
  };

  const filteredFiles = useMemo(() => {
    if (!searchQuery.trim()) return files;
    const q = searchQuery.toLowerCase();
    return files.filter((f) => f.filename.toLowerCase().includes(q));
  }, [files, searchQuery]);

  // Group files by extension type
  const fileGroups = useMemo(() => {
    const groups: Record<string, FileInfo[]> = {};
    files.forEach((f) => {
      const ext = f.filename.split('.').pop()?.toUpperCase() || 'OTHER';
      if (!groups[ext]) groups[ext] = [];
      groups[ext].push(f);
    });
    return groups;
  }, [files]);

  const stats = useMemo(() => {
    const total = files.length;
    const ready = files.filter((f) => f.status === 'completed' || f.status === 'indexed').length;
    const processing = files.filter((f) => f.status === 'processing').length;
    return { total, ready, processing };
  }, [files]);

  const StatusIcon = ({ status }: { status: string }) => {
    if (status === 'completed' || status === 'indexed') {
      return <CheckCircle2 className="h-3 w-3 text-emerald-400" />;
    }
    if (status === 'processing') {
      return <Loader2 className="h-3 w-3 animate-spin text-amber-400" />;
    }
    return <Clock className="h-3 w-3 text-muted-foreground" />;
  };

  return (
    <div className="flex h-full flex-col">
      {/* Search */}
      <div className="shrink-0 p-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search documents..."
            className="h-8 w-full rounded-lg border border-border bg-background pl-8 pr-8 text-xs text-foreground placeholder:text-muted-foreground focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/25"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Stats bar */}
      {files.length > 0 && (
        <div className="mx-3 mb-2 flex items-center gap-3 rounded-md bg-surface/50 px-2.5 py-1.5">
          <span className="text-[10px] text-muted-foreground">
            <span className="font-semibold text-foreground">{stats.total}</span> docs
          </span>
          <span className="text-[10px] text-emerald-400">
            {stats.ready} ready
          </span>
          {stats.processing > 0 && (
            <span className="text-[10px] text-amber-400 flex items-center gap-1">
              <Loader2 className="h-2.5 w-2.5 animate-spin" />
              {stats.processing} processing
            </span>
          )}
        </div>
      )}

      {/* File Tree */}
      <div className="min-h-0 flex-1 overflow-y-auto px-2">
        {loading ? (
          <div className="flex justify-center py-12">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <span className="text-[10px] text-muted-foreground">Loading library...</span>
            </div>
          </div>
        ) : files.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
            <div className="rounded-full bg-gradient-to-br from-primary/20 to-accent/20 p-4 mb-4">
              <Upload className="h-8 w-8 text-primary" />
            </div>
            <p className="text-sm font-medium text-foreground mb-2">No documents yet</p>
            <p className="text-xs text-muted-foreground max-w-[200px]">
              Use <strong>File → Upload Documents</strong> or <strong>File → Import BibTeX/RIS</strong> in the menu bar to add documents.
            </p>
          </div>
        ) : (
          <>
            {/* Discovery Sessions */}
            <div className="mb-3">
              <button
                onClick={() => setDiscoverySessionsOpen(!discoverySessionsOpen)}
                className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground"
              >
                {discoverySessionsOpen ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                <FlaskConical className="h-3 w-3 text-orange-500" />
                Discovery Sessions
                {onStartDiscovery && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onStartDiscovery();
                    }}
                    className="ml-auto rounded p-0.5 text-orange-500 hover:bg-orange-500/10"
                    title="Start new discovery session"
                  >
                    <Plus className="h-3 w-3" />
                  </button>
                )}
              </button>
              {discoverySessionsOpen && (
                <div className="mt-0.5 space-y-0.5 pl-2">
                  {sessionList.length === 0 ? (
                    <p className="px-2 py-2 text-[10px] text-muted-foreground italic">
                      No sessions yet. Click [+] to start.
                    </p>
                  ) : (
                    sessionList.map((session) => (
                      <div
                        key={session.sessionId}
                        onClick={() => onOpenDiscoverySession?.(session.sessionId, session.sessionName)}
                        className="group flex cursor-pointer items-center gap-2 rounded-lg px-2 py-2 text-muted-foreground transition-all hover:bg-surface hover:text-foreground border border-transparent hover:border-orange-500/30"
                      >
                        <FlaskConical className="h-3 w-3 text-orange-500 shrink-0" />
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-xs font-medium">{session.sessionName}</p>
                          <p className="text-[10px] opacity-60">
                            {new Date(session.createdAt).toLocaleDateString()}
                            <span className="ml-2 inline-flex items-center gap-1">
                              {session.status === 'running' && (
                                <span className="rounded-full bg-orange-500/20 px-1.5 py-0.5 text-[9px] font-medium text-orange-500">
                                  Running
                                </span>
                              )}
                              {session.status === 'complete' && (
                                <CheckCircle2 className="h-2.5 w-2.5 text-emerald-400" />
                              )}
                            </span>
                          </p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Smart Groups */}
            {Object.keys(fileGroups).length > 1 && !searchQuery && (
              <div className="mb-1">
                <button
                  onClick={() => setSmartGroupsOpen(!smartGroupsOpen)}
                  className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground"
                >
                  {smartGroupsOpen ? (
                    <ChevronDown className="h-3 w-3" />
                  ) : (
                    <ChevronRight className="h-3 w-3" />
                  )}
                  <Sparkles className="h-3 w-3 text-primary" />
                  Groups
                </button>
                {smartGroupsOpen && (
                  <div className="mt-0.5 space-y-0.5 pl-2">
                    {Object.entries(fileGroups).map(([ext, groupFiles]) => (
                      <div
                        key={ext}
                        className="flex items-center gap-2 rounded-md px-2 py-1 text-[11px] text-muted-foreground"
                      >
                        <FolderOpen className="h-3 w-3 text-primary/60" />
                        <span>{ext} Files</span>
                        <span className="ml-auto rounded bg-surface px-1.5 py-0.5 text-[10px]">
                          {groupFiles.length}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* All Documents */}
            <div>
              <button
                onClick={() => setAllDocsOpen(!allDocsOpen)}
                className="flex w-full items-center gap-1.5 rounded-md px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground"
              >
                {allDocsOpen ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                All Documents
              </button>
              {allDocsOpen && (
                <div className="mt-0.5 space-y-0.5">
                  {filteredFiles.map((file) => (
                    <div
                      key={file.doc_id}
                      draggable
                      onDragStart={(e) => {
                        e.dataTransfer.setData('application/atlas-document', JSON.stringify({
                          id: file.doc_id,
                          filename: file.filename,
                          status: file.status,
                          pageCount: file.page_count,
                        }));
                      }}
                      onClick={() => onFileSelect(file.doc_id, file.filename)}
                      className={`group flex cursor-move items-center gap-2 rounded-lg px-2 py-2 transition-all ${
                        selectedDocId === file.doc_id
                          ? 'bg-primary/15 text-foreground border border-primary/30'
                          : 'text-muted-foreground hover:bg-surface hover:text-foreground border border-transparent'
                      }`}
                    >
                      <StatusIcon status={file.status} />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-xs font-medium">{file.filename}</p>
                        <p className="text-[10px] opacity-60">
                          {((file.size_bytes || 0) / 1024).toFixed(0)} KB
                        </p>
                      </div>
                      <button
                        onClick={(e) => handleDelete(e, file.doc_id)}
                        className="rounded p-1 opacity-0 transition-all group-hover:opacity-100 hover:bg-destructive/20 hover:text-destructive"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
