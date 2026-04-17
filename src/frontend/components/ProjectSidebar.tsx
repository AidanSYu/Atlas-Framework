'use client';

import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import {
  ChevronDown,
  ChevronRight,
  FileText,
  FlaskConical,
  Loader2,
  MessageSquare,
  Plus,
  Puzzle,
  Search,
  Trash2,
  Upload,
} from 'lucide-react';

import { api, type FileInfo, type TaskInfo, type TaskState } from '@/lib/api';
import type { WorkspaceMode } from '@/lib/workspace-mode';
import { useChatStore } from '@/stores/chatStore';

export interface DiscoverySessionListItem {
  sessionId: string;
  sessionName: string;
  createdAt: string | null;
  status: string;
}

interface ProjectSidebarProps {
  projectId: string;
  selectedDocId: string | null;
  onFileSelect: (docId: string, filename: string) => void;
  onFileDeleted?: (docId: string) => void;
  onIngestionComplete?: () => void;
  onSessionSelect: (threadId: string) => void;
  onNewSession: () => void;
  activeSessionId: string | null;
  refreshTrigger?: number;
  onUploadClick: () => void;
  workspaceMode: WorkspaceMode;
  onWorkspaceModeChange: (mode: WorkspaceMode) => void;
  // Legacy discovery-session props — kept for compatibility; unused now that Tasks
  // use the new two-tier orchestration runtime.
  discoverySessions?: DiscoverySessionListItem[];
  activeExperimentSessionId?: string | null;
  onExperimentSelect?: (sessionId: string) => void;
  onNewExperiment?: () => void;
  // Task runtime
  tasks: TaskInfo[];
  activeTaskId: string | null;
  onTaskSelect: (taskId: string) => void;
  onNewTask: () => void;
}

const MODE_META: Record<
  WorkspaceMode,
  {
    label: string;
    description: string;
    icon: typeof MessageSquare;
    accent: string;
    shortcut: string;
  }
> = {
  chat: {
    label: 'Chat',
    description: 'Use Librarian and Cortex over the project corpus.',
    icon: MessageSquare,
    accent: 'text-accent',
    shortcut: 'Librarian + Cortex',
  },
  experiment: {
    label: 'Task',
    description: 'Run discovery, orchestration, and plugin-backed work.',
    icon: FlaskConical,
    accent: 'text-emerald-400',
    shortcut: 'Discovery + Atlas',
  },
  plugins: {
    label: 'Plugins',
    description: 'Inspect, validate, and operate the local plugin runtime.',
    icon: Puzzle,
    accent: 'text-info',
    shortcut: 'Registry + Proofs',
  },
};

export function ProjectSidebar({
  projectId,
  selectedDocId,
  onFileSelect,
  onFileDeleted,
  onIngestionComplete,
  onSessionSelect,
  onNewSession,
  activeSessionId,
  refreshTrigger,
  onUploadClick,
  workspaceMode,
  onWorkspaceModeChange,
  discoverySessions: _unusedDiscoverySessions,
  activeExperimentSessionId: _unusedActiveExperimentSessionId,
  onExperimentSelect: _unusedOnExperimentSelect,
  onNewExperiment: _unusedOnNewExperiment,
  tasks,
  activeTaskId,
  onTaskSelect,
  onNewTask,
}: ProjectSidebarProps) {
  const [filesOpen, setFilesOpen] = useState(true);
  const [sessionsOpen, setSessionsOpen] = useState(true);
  const [tasksOpen, setTasksOpen] = useState(true);

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <WorkspaceModeSwitch
        activeMode={workspaceMode}
        onModeChange={onWorkspaceModeChange}
      />

      <ModeLead
        mode={workspaceMode}
        onNewSession={onNewSession}
        onNewExperiment={onNewTask}
      />

      {workspaceMode === 'chat' && (
        <>
          <SectionHeader
            label="Files"
            open={filesOpen}
            onToggle={() => setFilesOpen((open) => !open)}
            action={
              <button
                onClick={onUploadClick}
                className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
                title="Upload file"
              >
                <Upload className="h-3 w-3" />
              </button>
            }
          />
          {filesOpen && (
            <div className="min-h-0 flex-1 overflow-y-auto">
              <FileList
                projectId={projectId}
                selectedDocId={selectedDocId}
                onFileSelect={onFileSelect}
                onFileDeleted={onFileDeleted}
                onIngestionComplete={onIngestionComplete}
                refreshTrigger={refreshTrigger}
              />
            </div>
          )}

          <SectionHeader
            label="Chats"
            open={sessionsOpen}
            onToggle={() => setSessionsOpen((open) => !open)}
            action={
              <button
                onClick={onNewSession}
                className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
                title="New chat"
              >
                <Plus className="h-3 w-3" />
              </button>
            }
          />
          {sessionsOpen && (
            <div className="min-h-0 flex-1 overflow-y-auto border-t border-border/50">
              <SessionList
                projectId={projectId}
                activeSessionId={activeSessionId}
                onSelect={onSessionSelect}
              />
            </div>
          )}
        </>
      )}

      {workspaceMode === 'experiment' && (
        <>
          <SectionHeader
            label="Tasks"
            open={tasksOpen}
            onToggle={() => setTasksOpen((open) => !open)}
            action={
              <button
                onClick={onNewTask}
                className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
                title="New task"
              >
                <Plus className="h-3 w-3" />
              </button>
            }
          />
          {tasksOpen && (
            <div className="min-h-0 flex-1 overflow-y-auto">
              <TaskList
                tasks={tasks}
                activeTaskId={activeTaskId}
                onSelect={onTaskSelect}
              />
            </div>
          )}

          <SectionHeader
            label="Files"
            open={filesOpen}
            onToggle={() => setFilesOpen((open) => !open)}
            action={
              <button
                onClick={onUploadClick}
                className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
                title="Upload file"
              >
                <Upload className="h-3 w-3" />
              </button>
            }
          />
          {filesOpen && (
            <div className="min-h-0 max-h-[38%] overflow-y-auto border-t border-border/50">
              <FileList
                projectId={projectId}
                selectedDocId={selectedDocId}
                onFileSelect={onFileSelect}
                onFileDeleted={onFileDeleted}
                onIngestionComplete={onIngestionComplete}
                refreshTrigger={refreshTrigger}
              />
            </div>
          )}
        </>
      )}

      {workspaceMode === 'plugins' && (
        <>
          <SectionHeader
            label="Registry"
            open
            onToggle={() => {}}
          />
          <div className="border-b border-border px-3 py-3">
            <div className="rounded-2xl border border-border/70 bg-background/60 p-3">
              <div className="flex items-center gap-2 text-[11px] font-medium text-foreground">
                <Puzzle className="h-3.5 w-3.5 text-info" />
                Plugin manager
              </div>
              <p className="mt-2 text-[11px] leading-5 text-muted-foreground">
                Atlas plugins live at the framework level. Use the main workspace to verify runtime health,
                run proofs, and operate individual tools.
              </p>
            </div>
          </div>

          <SectionHeader
            label="Files"
            open={filesOpen}
            onToggle={() => setFilesOpen((open) => !open)}
            action={
              <button
                onClick={onUploadClick}
                className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
                title="Upload file"
              >
                <Upload className="h-3 w-3" />
              </button>
            }
          />
          {filesOpen && (
            <div className="min-h-0 flex-1 overflow-y-auto border-t border-border/50">
              <FileList
                projectId={projectId}
                selectedDocId={selectedDocId}
                onFileSelect={onFileSelect}
                onFileDeleted={onFileDeleted}
                onIngestionComplete={onIngestionComplete}
                refreshTrigger={refreshTrigger}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

function WorkspaceModeSwitch({
  activeMode,
  onModeChange,
}: {
  activeMode: WorkspaceMode;
  onModeChange: (mode: WorkspaceMode) => void;
}) {
  return (
    <div className="border-b border-border bg-card px-2 py-2">
      <div className="rounded-2xl border border-border/80 bg-background/60 p-1">
        <div className="grid grid-cols-3 gap-1">
          {(Object.entries(MODE_META) as Array<[WorkspaceMode, typeof MODE_META[WorkspaceMode]]>).map(
            ([mode, meta]) => {
              const Icon = meta.icon;
              const isActive = mode === activeMode;
              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => onModeChange(mode)}
                  className={[
                    'group flex min-w-0 items-center justify-center gap-2 rounded-xl px-2 py-2 text-left transition-all',
                    isActive
                      ? 'bg-surface text-foreground shadow-sm ring-1 ring-border/80'
                      : 'text-muted-foreground hover:bg-surface/60 hover:text-foreground',
                  ].join(' ')}
                  title={meta.label}
                >
                  <Icon className={['h-3.5 w-3.5 shrink-0', isActive ? meta.accent : ''].join(' ')} />
                  <span className="truncate text-[11px] font-medium">{meta.label}</span>
                </button>
              );
            }
          )}
        </div>
      </div>
    </div>
  );
}

function ModeLead({
  mode,
  onNewSession,
  onNewExperiment,
}: {
  mode: WorkspaceMode;
  onNewSession: () => void;
  onNewExperiment: () => void;  // now bound to onNewTask in the new task runtime
}) {
  const meta = MODE_META[mode];
  const Icon = meta.icon;

  return (
    <div className="border-b border-border px-3 py-3">
      <div className="rounded-2xl border border-border/70 bg-background/55 p-3">
        <div className="flex items-center gap-2">
          <Icon className={['h-4 w-4', meta.accent].join(' ')} />
          <div className="min-w-0">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
              {meta.label}
            </div>
            <div className="mt-0.5 text-sm font-medium text-foreground">{meta.shortcut}</div>
          </div>
        </div>
        <p className="mt-2 text-[11px] leading-5 text-muted-foreground">{meta.description}</p>

        {mode === 'chat' && (
          <button
            type="button"
            onClick={onNewSession}
            className="mt-3 inline-flex items-center gap-2 rounded-xl border border-border bg-surface/60 px-3 py-1.5 text-[11px] font-medium text-foreground transition-colors hover:bg-surface"
          >
            <Plus className="h-3 w-3" />
            New chat
          </button>
        )}

        {mode === 'experiment' && (
          <button
            type="button"
            onClick={onNewExperiment}
            className="mt-3 inline-flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-[11px] font-medium text-emerald-300 transition-colors hover:bg-emerald-500/15"
          >
            <Plus className="h-3 w-3" />
            New task
          </button>
        )}
      </div>
    </div>
  );
}

function SectionHeader({
  label,
  open,
  onToggle,
  action,
}: {
  label: string;
  open: boolean;
  onToggle: () => void;
  action?: React.ReactNode;
}) {
  const Chevron = open ? ChevronDown : ChevronRight;

  return (
    <div className="flex h-7 shrink-0 items-center justify-between border-b border-border bg-card px-2">
      <button
        onClick={onToggle}
        className="flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground transition-colors hover:text-foreground"
      >
        <Chevron className="h-3 w-3" />
        {label}
      </button>
      {action}
    </div>
  );
}

function FileList({
  projectId,
  selectedDocId,
  onFileSelect,
  onFileDeleted,
  onIngestionComplete,
  refreshTrigger,
}: {
  projectId: string;
  selectedDocId: string | null;
  onFileSelect: (docId: string, filename: string) => void;
  onFileDeleted?: (docId: string) => void;
  onIngestionComplete?: () => void;
  refreshTrigger?: number;
}) {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const prevProcessingRef = useRef(false);

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
    void loadFiles();
  }, [loadFiles]);

  useEffect(() => {
    if (refreshTrigger != null && refreshTrigger > 0) {
      void loadFiles(true);
    }
  }, [refreshTrigger, loadFiles]);

  const hasProcessing = files.some((file) => file.status === 'processing');
  useEffect(() => {
    if (!hasProcessing) {
      if (prevProcessingRef.current && onIngestionComplete) onIngestionComplete();
      prevProcessingRef.current = false;
      return;
    }

    prevProcessingRef.current = true;
    const interval = window.setInterval(() => {
      void loadFiles(true);
    }, 3000);
    return () => window.clearInterval(interval);
  }, [hasProcessing, loadFiles, onIngestionComplete]);

  const handleDelete = async (docId: string) => {
    setDeletingId(docId);
    try {
      await api.deleteFile(docId);
      setFiles((prev) => prev.filter((file) => file.doc_id !== docId));
      onFileDeleted?.(docId);
    } catch (error) {
      console.error('Delete failed:', error);
    } finally {
      setDeletingId(null);
    }
  };

  const filteredFiles = files.filter((file) =>
    !searchQuery || file.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-6">
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      {files.length > 5 && (
        <div className="relative px-2 py-1.5">
          <Search className="pointer-events-none absolute left-4 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Filter files..."
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            className="h-6 w-full rounded border border-border bg-background pl-7 pr-2 text-xs text-foreground placeholder:text-muted-foreground focus:border-accent focus:outline-none"
          />
        </div>
      )}

      {filteredFiles.length === 0 ? (
        <div className="flex flex-col items-center px-3 py-6 text-center">
          <Upload className="mb-2 h-5 w-5 text-muted-foreground/20" />
          <p className="text-xs text-muted-foreground">
            {files.length === 0 ? 'No files yet - upload to begin' : 'No matches'}
          </p>
        </div>
      ) : (
        filteredFiles.map((file) => {
          const isProcessing = file.status === 'processing' || file.status === 'pending';
          const isClickable = !isProcessing;
          const isSelected = selectedDocId === file.doc_id;

          return (
            <div
              key={file.doc_id}
              onClick={() => isClickable && onFileSelect(file.doc_id, file.filename)}
              className={[
                'group flex items-center gap-2 px-3 py-1.5 text-xs transition-colors',
                isClickable ? 'cursor-pointer hover:bg-surface-hover' : 'cursor-default opacity-50',
                isSelected ? 'border-l-2 border-accent bg-accent/8 text-accent' : 'text-muted-foreground',
              ].join(' ')}
            >
              {isProcessing ? (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-accent" />
              ) : (
                <FileText className={['h-3.5 w-3.5 shrink-0', isSelected ? 'text-accent' : ''].join(' ')} />
              )}
              <span className="min-w-0 flex-1 truncate">{file.filename}</span>
              {isClickable && (
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    void handleDelete(file.doc_id);
                  }}
                  disabled={deletingId === file.doc_id}
                  className="flex h-4 w-4 shrink-0 items-center justify-center rounded opacity-0 transition-opacity hover:bg-destructive/20 hover:text-destructive group-hover:opacity-100"
                >
                  {deletingId === file.doc_id ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    <Trash2 className="h-3 w-3" />
                  )}
                </button>
              )}
            </div>
          );
        })
      )}
    </div>
  );
}

function SessionList({
  projectId,
  activeSessionId,
  onSelect,
}: {
  projectId: string;
  activeSessionId: string | null;
  onSelect: (threadId: string) => void;
}) {
  const storeThreads = useChatStore((state) => state.threads);
  const deleteThread = useChatStore((state) => state.deleteThread);

  const threads = useMemo(
    () => storeThreads.filter((thread) => thread.projectId === projectId),
    [projectId, storeThreads]
  );

  const sortedThreads = useMemo(
    () => [...threads].sort((a, b) => b.updatedAt - a.updatedAt),
    [threads]
  );

  if (threads.length === 0) {
    return (
      <div className="flex flex-col items-center px-3 py-6 text-center">
        <MessageSquare className="mb-2 h-5 w-5 text-muted-foreground/20" />
        <p className="text-xs text-muted-foreground">No chats yet</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      {sortedThreads.map((thread) => {
        const isActive = activeSessionId === thread.id;
        return (
          <div
            key={thread.id}
            onClick={() => onSelect(thread.id)}
            className={[
              'group flex cursor-pointer items-center gap-2 px-3 py-1.5 text-xs transition-colors hover:bg-surface-hover',
              isActive ? 'border-l-2 border-accent bg-accent/8 text-accent' : 'text-muted-foreground',
            ].join(' ')}
          >
            <MessageSquare className="h-3.5 w-3.5 shrink-0" />
            <div className="min-w-0 flex-1">
              <div className="truncate">{thread.title}</div>
              <div className="text-[10px] opacity-60">
                {thread.messages.length - 1} messages · {formatRelativeTime(thread.updatedAt)}
              </div>
            </div>
            <button
              onClick={(event) => {
                event.stopPropagation();
                deleteThread(thread.id);
              }}
              className="flex h-4 w-4 shrink-0 items-center justify-center rounded opacity-0 transition-opacity hover:bg-destructive/20 hover:text-destructive group-hover:opacity-100"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

function DiscoverySessionList({
  sessions,
  activeSessionId,
  onSelect,
}: {
  sessions: DiscoverySessionListItem[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void;
}) {
  if (sessions.length === 0) {
    return (
      <div className="flex flex-col items-center px-3 py-6 text-center">
        <FlaskConical className="mb-2 h-5 w-5 text-muted-foreground/20" />
        <p className="text-xs text-muted-foreground">No tasks yet</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      {sessions.map((session) => {
        const isActive = activeSessionId === session.sessionId;
        const statusTone =
          session.status === 'running'
            ? 'bg-warning'
            : session.status === 'complete'
              ? 'bg-success'
              : 'bg-muted-foreground/50';

        return (
          <button
            key={session.sessionId}
            type="button"
            onClick={() => onSelect(session.sessionId)}
            className={[
              'group flex w-full items-start gap-2 px-3 py-2 text-left text-xs transition-colors hover:bg-surface-hover',
              isActive ? 'border-l-2 border-emerald-400 bg-emerald-500/8 text-emerald-200' : 'text-muted-foreground',
            ].join(' ')}
          >
            <div className="mt-1 flex items-center gap-1.5">
              <span className={['h-1.5 w-1.5 rounded-full', statusTone].join(' ')} />
              <FlaskConical className="h-3.5 w-3.5 shrink-0" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium text-foreground">{session.sessionName}</div>
              <div className="mt-0.5 text-[10px] capitalize text-muted-foreground/75">
                {session.status} · {formatRelativeTime(session.createdAt ?? Date.now())}
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

function TaskList({
  tasks,
  activeTaskId,
  onSelect,
}: {
  tasks: TaskInfo[];
  activeTaskId: string | null;
  onSelect: (taskId: string) => void;
}) {
  if (tasks.length === 0) {
    return (
      <div className="flex flex-col items-center px-3 py-6 text-center">
        <FlaskConical className="mb-2 h-5 w-5 text-muted-foreground/20" />
        <p className="text-xs text-muted-foreground">No tasks yet</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      {tasks.map((task) => {
        const isActive = activeTaskId === task.id;
        const tone = taskStateTone(task.state);
        const label = task.title || task.initial_prompt || 'Untitled task';
        return (
          <button
            key={task.id}
            type="button"
            onClick={() => onSelect(task.id)}
            className={[
              'group flex w-full items-start gap-2 px-3 py-2 text-left text-xs transition-colors hover:bg-surface-hover',
              isActive ? 'border-l-2 border-emerald-400 bg-emerald-500/8 text-emerald-200' : 'text-muted-foreground',
            ].join(' ')}
          >
            <div className="mt-1 flex items-center gap-1.5">
              <span className={['h-1.5 w-1.5 rounded-full', tone.dot].join(' ')} />
              <FlaskConical className="h-3.5 w-3.5 shrink-0" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate font-medium text-foreground">{label}</div>
              <div className="mt-0.5 text-[10px] capitalize text-muted-foreground/75">
                {tone.label} · {formatRelativeTime(task.updated_at || task.created_at)}
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

function taskStateTone(state: TaskState): { label: string; dot: string } {
  switch (state) {
    case 'idle':
      return { label: 'ready', dot: 'bg-muted-foreground/50' };
    case 'initializing':
    case 'planning':
    case 'executing':
    case 'reviewing':
      return { label: state, dot: 'bg-warning' };
    case 'suspended':
      return { label: 'waiting', dot: 'bg-warning' };
    case 'completed':
      return { label: 'completed', dot: 'bg-success' };
    case 'cancelled':
      return { label: 'cancelled', dot: 'bg-muted-foreground/50' };
    case 'failed':
      return { label: 'failed', dot: 'bg-destructive' };
  }
}

function formatRelativeTime(value: number | string): string {
  const timestamp = typeof value === 'number' ? value : new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return 'just now';

  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
