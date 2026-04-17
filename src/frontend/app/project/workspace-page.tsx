'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { useParams, useRouter } from 'next/navigation';
import {
  AlertCircle,
  BookOpen,
  Brain,
  ChevronLeft,
  FlaskConical,
  Loader2,
  MessageSquare,
  Puzzle,
  Settings,
  Sparkles,
} from 'lucide-react';

import { ExperimentWorkspace } from '@/components/ExperimentWorkspace';
import { FrameworkPluginsTab } from '@/components/FrameworkPluginsTab';
import { MissionControl } from '@/components/MissionControl';
import { OmniBar } from '@/components/OmniBar';
import PDFViewer from '@/components/PDFViewer';
import { ProjectSidebar, type DiscoverySessionListItem } from '@/components/ProjectSidebar';
import SettingsModal from '@/components/SettingsModal';
import { StatusBar } from '@/components/StatusBar';
import TextViewer from '@/components/TextViewer';
import { RunAuditPanel } from '@/components/chat/RunAuditPanel';
import ChatShell from '@/components/chat/ChatShell';
import { useAtlasTheme } from '@/hooks/useAtlasTheme';
import { WorkspaceTabs, type WorkspaceTab } from '@/components/WorkspaceTabs';
import type { ChatMode } from '@/hooks/useRunManager';
import { api, type ModelRegistryResponse, type ModelStatusResponse, type ProjectInfo, type TaskInfo } from '@/lib/api';
import type { WorkspaceMode } from '@/lib/workspace-mode';
import { useChatStore } from '@/stores/chatStore';
import { useDiscoveryStore } from '@/stores/discoveryStore';
import { useGraphStore } from '@/stores/graphStore';
import { toastError, toastSuccess } from '@/stores/toastStore';

const WindowControls = dynamic(
  () => import('@/components/WindowControls').then((module) => ({ default: module.WindowControls })),
  { ssr: false }
);

function ChatLanding({
  onNewChat,
  onSwitchToExperiment,
  onSwitchToPlugins,
}: {
  onNewChat: () => void;
  onSwitchToExperiment: () => void;
  onSwitchToPlugins: () => void;
}) {
  return (
    <div className="relative flex h-full min-h-0 items-center justify-center overflow-hidden bg-background">
      <div className="pointer-events-none absolute inset-0 opacity-70">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.06),transparent_40%)]" />
        <div className="absolute left-24 top-24 h-56 w-56 rounded-full bg-accent/8 blur-3xl" />
        <div className="absolute bottom-20 right-20 h-72 w-72 rounded-full bg-info/6 blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto flex w-full max-w-4xl flex-col items-center px-8 text-center">
        <div className="inline-flex items-center gap-2 rounded-full border border-border/70 bg-card/70 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
          <Sparkles className="h-3.5 w-3.5 text-accent" />
          Chat
        </div>

        <h1 className="mt-5 text-4xl font-semibold tracking-tight text-foreground">
          Atlas is ready for grounded conversation.
        </h1>
        <p className="mt-4 max-w-2xl text-sm leading-7 text-muted-foreground">
          Start a chat for document Q&A, switch to Cortex for wider research reasoning, or move into Task mode
          when the work needs planning, plugins, and execution tracking.
        </p>

        <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
          <button
            type="button"
            onClick={onNewChat}
            className="inline-flex items-center gap-2 rounded-2xl border border-accent/30 bg-accent/10 px-4 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-accent/15"
          >
            <MessageSquare className="h-4 w-4 text-accent" />
            New chat
          </button>
          <button
            type="button"
            onClick={onSwitchToExperiment}
            className="inline-flex items-center gap-2 rounded-2xl border border-border/70 bg-card/70 px-4 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-surface/70"
          >
            <FlaskConical className="h-4 w-4 text-emerald-400" />
            Open task
          </button>
          <button
            type="button"
            onClick={onSwitchToPlugins}
            className="inline-flex items-center gap-2 rounded-2xl border border-border/70 bg-card/70 px-4 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-surface/70"
          >
            <Puzzle className="h-4 w-4 text-info" />
            Manage plugins
          </button>
        </div>

        <div className="mt-10 grid w-full gap-4 md:grid-cols-3">
          <div className="rounded-3xl border border-border/70 bg-card/60 p-5 text-left">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-border/70 bg-background/55">
              <BookOpen className="h-4 w-4 text-accent" />
            </div>
            <h2 className="mt-4 text-base font-semibold text-foreground">Librarian</h2>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              Keep answers grounded in the project library with citations and source awareness.
            </p>
          </div>
          <div className="rounded-3xl border border-border/70 bg-card/60 p-5 text-left">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-border/70 bg-background/55">
              <Brain className="h-4 w-4 text-info" />
            </div>
            <h2 className="mt-4 text-base font-semibold text-foreground">Cortex</h2>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              Ask for broader grounded reasoning across files, graph context, and cross-document synthesis.
            </p>
          </div>
          <div className="rounded-3xl border border-border/70 bg-card/60 p-5 text-left">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-border/70 bg-background/55">
              <FlaskConical className="h-4 w-4 text-emerald-400" />
            </div>
            <h2 className="mt-4 text-base font-semibold text-foreground">Task</h2>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              Promote work into a plan-driven cockpit when Atlas needs to run tools and generate artifacts.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function SessionTabContent({
  projectId,
  onCitationClick,
  pendingQuery,
  onQueryConsumed,
  onOpenRunHistory,
  onViewRunDetails,
  modelRegistry,
  modelStatus,
  onLoadModel,
  isModelSwitching,
}: {
  projectId: string;
  onCitationClick: (filename: string, page: number, docId?: string) => void;
  pendingQuery: string | null;
  onQueryConsumed: () => void;
  onOpenRunHistory: () => void;
  onViewRunDetails: (runId: string) => void;
  modelRegistry: ModelRegistryResponse | null;
  modelStatus: ModelStatusResponse | null;
  onLoadModel: (modelName: string) => Promise<void>;
  isModelSwitching: boolean;
}) {
  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="min-h-0 flex-1 overflow-hidden">
        <ChatShell
          onCitationClick={onCitationClick}
          projectId={projectId}
          autoSubmitQuery={pendingQuery}
          onAutoSubmitConsumed={onQueryConsumed}
          onOpenRunHistory={onOpenRunHistory}
          onViewRunDetails={onViewRunDetails}
          modelRegistry={modelRegistry}
          modelStatus={modelStatus}
          onLoadModel={onLoadModel}
          isModelSwitching={isModelSwitching}
        />
      </div>

      <div className="flex h-6 shrink-0 items-center gap-2 border-t border-border bg-card px-3 text-[10px] text-muted-foreground/55">
        <Sparkles className="h-2.5 w-2.5 text-accent/70" />
        <span>Grounded project chat</span>
      </div>
    </div>
  );
}

export default function ProjectWorkspacePage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const projectId = decodeURIComponent(params.id || '');

  const nodeCount = useGraphStore((state) => state.nodes.length);
  const refreshGraph = useGraphStore((state) => state.refreshGraph);

  const [currentProject, setCurrentProject] = useState<ProjectInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [openTabs, setOpenTabs] = useState<WorkspaceTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);
  const [libraryRefreshTrigger, setLibraryRefreshTrigger] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [pendingOmniBarQuery, setPendingOmniBarQuery] = useState<string | null>(null);
  const [runHistoryOpen, setRunHistoryOpen] = useState(false);
  const [auditRunId, setAuditRunId] = useState<string | null>(null);
  const [pdfPage, setPdfPage] = useState(1);
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>('chat');
  const [discoverySessions, setDiscoverySessions] = useState<DiscoverySessionListItem[]>([]);
  const [activeExperimentSessionId, setActiveExperimentSessionId] = useState<string | null>(null);
  const [experimentSetupOpen, setExperimentSetupOpen] = useState(false);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [creatingTask, setCreatingTask] = useState(false);
  const { theme, setTheme, toggleTheme } = useAtlasTheme();
  const [modelRegistry, setModelRegistry] = useState<ModelRegistryResponse | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
  const [isModelSwitching, setIsModelSwitching] = useState(false);

  const uploadInputRef = useRef<HTMLInputElement>(null);
  const initializedProjectIdRef = useRef<string | null>(null);

  useEffect(() => {
    const savedMode = localStorage.getItem('atlas-workspace-mode') as WorkspaceMode | null;
    if (savedMode === 'chat' || savedMode === 'experiment' || savedMode === 'plugins') {
      setWorkspaceMode(savedMode);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('atlas-workspace-mode', workspaceMode);
  }, [workspaceMode]);

  const loadProjectRef = useRef<() => void>();

  useEffect(() => {
    let alive = true;
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let retries = 0;
    const MAX_RETRIES = 3;

    const loadProject = async () => {
      if (!projectId) {
        setLoading(false);
        return;
      }

      try {
        const projects = await api.listProjects();
        if (!alive) return;
        const project = projects.find((entry) => entry.id === projectId);
        if (!project) {
          router.push('/');
          return;
        }
        setCurrentProject(project);
        setLoadError(null);
        setLoading(false);
      } catch (error: any) {
        if (!alive) return;
        retries += 1;
        if (retries >= MAX_RETRIES) {
          setLoadError(
            'Cannot connect to the Atlas backend. Make sure the server is running (cd src/backend && python run_server.py).'
          );
          setLoading(false);
        } else {
          timeoutId = setTimeout(loadProject, 2000);
        }
      }
    };

    loadProjectRef.current = () => {
      retries = 0;
      setLoading(true);
      setLoadError(null);
      void loadProject();
    };

    setLoading(true);
    setLoadError(null);
    void loadProject();

    return () => {
      alive = false;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [projectId, router]);

  const refreshModelRegistry = useCallback(async () => {
    try {
      const registry = await api.getModelRegistry();
      setModelRegistry(registry);
      setModelStatus(registry.active);
    } catch (error) {
      console.error('Failed to load model registry:', error);
    }
  }, []);

  const handleLoadModel = useCallback(async (modelName: string) => {
    const currentModel = modelStatus?.active_model ?? modelRegistry?.active?.active_model ?? null;
    if (!modelName || modelName === currentModel) {
      return;
    }

    setIsModelSwitching(true);
    try {
      const status = await api.loadModel(modelName);
      setModelStatus(status);
      await refreshModelRegistry();
      toastSuccess(`Switched to ${modelName}`);
    } catch (error: any) {
      toastError(error?.message ?? `Failed to load ${modelName}`);
      throw error;
    } finally {
      setIsModelSwitching(false);
    }
  }, [modelRegistry, modelStatus, refreshModelRegistry]);

  useEffect(() => {
    void refreshModelRegistry();
    const interval = window.setInterval(() => {
      void refreshModelRegistry();
    }, 8000);
    return () => window.clearInterval(interval);
  }, [refreshModelRegistry]);

  const refreshDiscoverySessions = useCallback(async () => {
    if (!currentProject) return;

    try {
      const sessions = await api.listDiscoverySessions(currentProject.id);
      const sortedSessions = sessions
        .map((session) => ({
          sessionId: session.session_id,
          sessionName: session.session_name,
          createdAt: session.created_at,
          status: session.status,
        }))
        .sort((a, b) => {
          const aTime = a.createdAt ? new Date(a.createdAt).getTime() : 0;
          const bTime = b.createdAt ? new Date(b.createdAt).getTime() : 0;
          return bTime - aTime;
        });

      setDiscoverySessions(sortedSessions);

      sortedSessions.forEach((session) => {
        useDiscoveryStore.getState().upsertBackendSession(
          session.sessionId,
          session.sessionName,
          session.createdAt ?? new Date().toISOString(),
          currentProject.id
        );
      });

      if (sortedSessions.length > 0) {
        setActiveExperimentSessionId((currentId) => {
          const nextId = currentId && sortedSessions.some((session) => session.sessionId === currentId)
            ? currentId
            : sortedSessions[0].sessionId;

          useDiscoveryStore.getState().setActiveSession(nextId);
          return nextId;
        });
      } else {
        setActiveExperimentSessionId(null);
      }
    } catch (error) {
      console.error('Failed to load discovery sessions:', error);
    }
  }, [currentProject]);

  useEffect(() => {
    if (!currentProject) return;
    void refreshDiscoverySessions();
  }, [currentProject, refreshDiscoverySessions]);

  const openDocumentTab = useCallback((docId: string, filename: string) => {
    const tabId = `doc:${docId}`;
    setOpenTabs((previousTabs) =>
      previousTabs.some((tab) => tab.id === tabId)
        ? previousTabs
        : [...previousTabs, { kind: 'document', id: tabId, docId, filename }]
    );
    setActiveTabId(tabId);
    setWorkspaceMode('chat');
    setPdfPage(1);
  }, []);

  const openSessionTab = useCallback((threadId: string) => {
    const tabId = `session:${threadId}`;
    const thread = useChatStore.getState().threads.find((entry) => entry.id === threadId);
    const title = thread?.title ?? 'Session';

    setOpenTabs((previousTabs) =>
      previousTabs.some((tab) => tab.id === tabId)
        ? previousTabs
        : [...previousTabs, { kind: 'session', id: tabId, threadId, title }]
    );
    setActiveTabId(tabId);
    setWorkspaceMode('chat');
    useChatStore.getState().switchThread(threadId);
  }, []);

  const handleNewSession = useCallback(() => {
    if (!currentProject) return;
    const thread = useChatStore.getState().createThread(currentProject.id);
    openSessionTab(thread.id);
  }, [currentProject, openSessionTab]);

  useEffect(() => {
    if (!currentProject) return;
    if (initializedProjectIdRef.current === currentProject.id) return;

    initializedProjectIdRef.current = currentProject.id;
    const store = useChatStore.getState();
    store.setActiveProject(currentProject.id);

    const activeThread = store.getActiveThread() ?? store.getProjectThreads(currentProject.id)[0] ?? null;
    if (activeThread) {
      openSessionTab(activeThread.id);
    }
  }, [currentProject, openSessionTab]);

  const handleCloseTab = useCallback((tabId: string) => {
    setOpenTabs((previousTabs) => {
      const nextTabs = previousTabs.filter((tab) => tab.id !== tabId);
      if (activeTabId === tabId) {
        setActiveTabId(nextTabs[nextTabs.length - 1]?.id ?? null);
      }
      return nextTabs;
    });
  }, [activeTabId]);

  const handleSelectTab = useCallback((tabId: string) => {
    setActiveTabId(tabId);
    const tab = openTabs.find((entry) => entry.id === tabId);
    if (tab?.kind === 'session' && 'threadId' in tab) {
      useChatStore.getState().switchThread(tab.threadId);
    }
  }, [openTabs]);

  const handleFileDeleted = useCallback((docId: string) => {
    handleCloseTab(`doc:${docId}`);
  }, [handleCloseTab]);

  const handleCitationClick = useCallback((filename: string, page: number, docId?: string) => {
    if (docId) {
      openDocumentTab(docId, filename);
    }
    setPdfPage(page);
  }, [openDocumentTab]);

  const handleUploadFiles = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = event.target.files;
    if (!fileList?.length || !projectId) return;

    try {
      for (let index = 0; index < fileList.length; index += 1) {
        await api.uploadFile(fileList[index], projectId);
      }
      setLibraryRefreshTrigger((value) => value + 1);
    } catch (error: any) {
      toastError(error?.message ?? 'Upload failed');
    } finally {
      event.target.value = '';
    }
  };

  const handleViewSwitch = useCallback((view: 'document' | 'graph' | 'chat' | 'canvas' | 'editor') => {
    if (view === 'chat') {
      handleNewSession();
    }
  }, [handleNewSession]);

  const handleWorkspaceModeChange = useCallback((mode: WorkspaceMode) => {
    setWorkspaceMode(mode);
  }, []);

  const handleExperimentSelect = useCallback((sessionId: string) => {
    setWorkspaceMode('experiment');
    setActiveExperimentSessionId(sessionId);
    useDiscoveryStore.getState().setActiveSession(sessionId);
  }, []);

  const handleExperimentCreated = useCallback((sessionId: string) => {
    setExperimentSetupOpen(false);
    setWorkspaceMode('experiment');
    if (sessionId) {
      setActiveExperimentSessionId(sessionId);
      useDiscoveryStore.getState().setActiveSession(sessionId);
    }
    void refreshDiscoverySessions();
  }, [refreshDiscoverySessions]);

  const refreshTasks = useCallback(async () => {
    if (!projectId) return;
    try {
      const list = await api.listTasks(projectId);
      setTasks(list);
      // If nothing selected and we have tasks, auto-pick the most recent running (or newest).
      if (!activeTaskId && list.length > 0) {
        const running = list.find(
          (t) => t.state !== 'completed' && t.state !== 'cancelled' && t.state !== 'failed'
        );
        setActiveTaskId((running || list[0]).id);
      }
    } catch (err) {
      // Non-fatal; user will see errors when creating.
    }
  }, [projectId, activeTaskId]);

  useEffect(() => {
    if (workspaceMode === 'experiment') {
      void refreshTasks();
      const poll = setInterval(() => void refreshTasks(), 4000);
      return () => clearInterval(poll);
    }
  }, [workspaceMode, refreshTasks]);

  const handleNewTask = useCallback(async () => {
    if (!projectId || creatingTask) return;
    setWorkspaceMode('experiment');
    setCreatingTask(true);
    try {
      const created = await api.createTask({ project_id: projectId });
      setTasks((prev) => [created, ...prev.filter((t) => t.id !== created.id)]);
      setActiveTaskId(created.id);
    } catch (err: any) {
      toastError(err?.message || 'Failed to create task');
    } finally {
      setCreatingTask(false);
    }
  }, [projectId, creatingTask]);

  const handleTaskSelect = useCallback((taskId: string) => {
    setWorkspaceMode('experiment');
    setActiveTaskId(taskId);
  }, []);

  const handleTaskCreated = useCallback((task: TaskInfo) => {
    setTasks((prev) => [task, ...prev.filter((t) => t.id !== task.id)]);
    setActiveTaskId(task.id);
  }, []);

  const activeTab = openTabs.find((entry) => entry.id === activeTabId);

  const renderChatContent = () => {
    if (!currentProject) return null;
    if (!activeTab) {
      return (
        <ChatLanding
          onNewChat={handleNewSession}
          onSwitchToExperiment={() => setWorkspaceMode('experiment')}
          onSwitchToPlugins={() => setWorkspaceMode('plugins')}
        />
      );
    }

    switch (activeTab.kind) {
      case 'session':
        return (
          <SessionTabContent
            projectId={currentProject.id}
            onCitationClick={handleCitationClick}
            pendingQuery={pendingOmniBarQuery}
            onQueryConsumed={() => setPendingOmniBarQuery(null)}
            onOpenRunHistory={() => {
              setAuditRunId(null);
              setRunHistoryOpen(true);
            }}
            onViewRunDetails={(runId) => {
              setAuditRunId(runId);
              setRunHistoryOpen(true);
            }}
            modelRegistry={modelRegistry}
            modelStatus={modelStatus}
            onLoadModel={handleLoadModel}
            isModelSwitching={isModelSwitching}
          />
        );

      case 'document': {
        const { docId, filename } = activeTab as Extract<WorkspaceTab, { kind: 'document' }>;
        if (/\.(txt|text|md|csv|log|json|xml)$/i.test(filename)) {
          return (
            <TextViewer
              fileUrl={api.getFileUrl(docId)}
              filename={filename}
              docId={docId}
              projectId={currentProject.id}
              onContextChange={() => {}}
            />
          );
        }

        return (
          <PDFViewer
            fileUrl={api.getFileUrl(docId)}
            filename={filename}
            docId={docId}
            projectId={currentProject.id}
            initialPage={pdfPage}
            onAskAboutPage={(query) => {
              useChatStore.getState().setPendingQuestion(query);
              handleNewSession();
            }}
            onRelatedPassageClick={handleCitationClick}
            onContextChange={() => {}}
          />
        );
      }

      case 'plugins':
        return <FrameworkPluginsTab />;

      default:
        return null;
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 bg-background px-8 text-center">
        <AlertCircle className="h-8 w-8 text-destructive/60" />
        <p className="max-w-md text-sm text-muted-foreground">{loadError}</p>
        <div className="flex gap-3">
          <button
            onClick={() => loadProjectRef.current?.()}
            className="inline-flex h-8 items-center gap-2 rounded bg-accent px-4 text-xs font-medium text-white transition-colors hover:bg-accent/90"
          >
            Retry
          </button>
          <button
            onClick={() => router.push('/')}
            className="inline-flex h-8 items-center gap-2 rounded border border-border px-4 text-xs text-muted-foreground transition-colors hover:bg-surface hover:text-foreground"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  if (!projectId || !currentProject) {
    return (
      <div className="flex h-screen items-center justify-center bg-background text-muted-foreground">
        Project not found.
      </div>
    );
  }

  return (
    <main className="flex h-screen w-screen flex-col overflow-hidden bg-background text-foreground">
      <input
        ref={uploadInputRef}
        type="file"
        className="hidden"
        accept=".pdf,.txt,.docx,.doc"
        multiple
        onChange={handleUploadFiles}
      />

      <div
        data-tauri-drag-region
        className="relative z-[100] flex h-9 shrink-0 items-center justify-between border-b border-border bg-card px-3"
      >
        <div className="no-drag flex items-center gap-2">
          <button
            onClick={() => router.push('/')}
            className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
            title="Back to projects"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="text-xs font-medium text-foreground/80">{currentProject.name}</span>
        </div>

        <div className="pointer-events-none absolute left-1/2 top-1/2 flex -translate-x-1/2 -translate-y-1/2 items-center gap-1.5 text-[11px] text-muted-foreground/50">
          <span className="text-accent/40">&#9670;</span>
          Atlas Framework
        </div>

        <div className="no-drag flex items-center gap-1">
          <button
            onClick={() => setSettingsOpen(true)}
            className="flex h-6 w-6 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
            title="Settings"
          >
            <Settings className="h-3.5 w-3.5" />
          </button>
          <WindowControls />
        </div>
      </div>

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <div className="flex w-[18.5rem] shrink-0 flex-col overflow-hidden border-r border-border bg-card">
          <ProjectSidebar
            projectId={currentProject.id}
            selectedDocId={
              activeTab?.kind === 'document'
                ? (activeTab as Extract<WorkspaceTab, { kind: 'document' }>).docId
                : null
            }
            onFileSelect={openDocumentTab}
            onFileDeleted={handleFileDeleted}
            onIngestionComplete={refreshGraph}
            onSessionSelect={openSessionTab}
            onNewSession={handleNewSession}
            activeSessionId={
              activeTab?.kind === 'session'
                ? (activeTab as Extract<WorkspaceTab, { kind: 'session' }>).threadId
                : null
            }
            refreshTrigger={libraryRefreshTrigger}
            onUploadClick={() => uploadInputRef.current?.click()}
            workspaceMode={workspaceMode}
            onWorkspaceModeChange={handleWorkspaceModeChange}
            discoverySessions={discoverySessions}
            activeExperimentSessionId={activeExperimentSessionId}
            onExperimentSelect={(sessionId) => void handleExperimentSelect(sessionId)}
            onNewExperiment={() => void handleNewTask()}
            tasks={tasks}
            activeTaskId={activeTaskId}
            onTaskSelect={handleTaskSelect}
            onNewTask={() => void handleNewTask()}
          />
        </div>

        <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-background">
          {workspaceMode === 'chat' && (
            <>
              <WorkspaceTabs
                tabs={openTabs}
                activeTabId={activeTabId}
                onSelectTab={handleSelectTab}
                onCloseTab={handleCloseTab}
                onReorder={(from, to) =>
                  setOpenTabs((previousTabs) => {
                    const nextTabs = [...previousTabs];
                    const [movedTab] = nextTabs.splice(from, 1);
                    nextTabs.splice(to, 0, movedTab);
                    return nextTabs;
                  })
                }
              />

              <div className="relative min-h-0 flex-1 overflow-hidden">
                {renderChatContent()}
              </div>
            </>
          )}

          {workspaceMode === 'experiment' && (
            <ExperimentWorkspace
              projectId={currentProject.id}
              activeTaskId={activeTaskId}
              onNewTask={() => void handleNewTask()}
              onTaskCreated={handleTaskCreated}
              creating={creatingTask}
            />
          )}

          {workspaceMode === 'plugins' && <FrameworkPluginsTab />}
        </div>
      </div>

      <StatusBar
        modelName={modelStatus?.active_model ?? modelRegistry?.active?.active_model ?? undefined}
        device={modelStatus?.device ?? undefined}
        nodeCount={nodeCount}
        connected={!!currentProject}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <OmniBar
        projectId={currentProject.id}
        onUpload={() => uploadInputRef.current?.click()}
        onExport={async () => {}}
        onSwitchView={handleViewSwitch}
        onSubmitQuery={(query: string, _mode: ChatMode) => {
          setWorkspaceMode('chat');
          const thread = useChatStore.getState().createThread(currentProject.id);
          openSessionTab(thread.id);
          setPendingOmniBarQuery(query);
        }}
        onOpenRunHistory={() => {
          setAuditRunId(null);
          setRunHistoryOpen(true);
        }}
      />

      <RunAuditPanel
        open={runHistoryOpen}
        onClose={() => setRunHistoryOpen(false)}
        projectId={currentProject.id}
        selectedRunId={auditRunId ?? undefined}
      />

      <SettingsModal
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        theme={theme}
        onThemeChange={(nextTheme) => setTheme(nextTheme)}
        onKeysUpdated={refreshModelRegistry}
      />

      {experimentSetupOpen && (
        <div className="fixed inset-0 z-[130] flex items-center justify-center bg-black/55 p-6 backdrop-blur-sm">
          <div className="relative h-[min(90vh,860px)] w-full max-w-5xl overflow-hidden rounded-3xl border border-border bg-card shadow-2xl">
            <MissionControl
              projectId={currentProject.id}
              onSuccess={(sessionId) => void handleExperimentCreated(sessionId)}
              onCancel={() => setExperimentSetupOpen(false)}
            />
          </div>
        </div>
      )}
    </main>
  );
}
