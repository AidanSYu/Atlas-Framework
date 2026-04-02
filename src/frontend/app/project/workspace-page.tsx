'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import { useParams, useRouter } from 'next/navigation';
import LibrarySidebar from '@/components/LibrarySidebar';
import PDFViewer from '@/components/PDFViewer';
import TextViewer from '@/components/TextViewer';
import KnowledgeGraph from '@/components/KnowledgeGraph';
import ChatShell from '@/components/chat/ChatShell';
import { ChatHistoryPanel } from '@/components/ChatHistoryPanel';
import { OmniBar } from '@/components/OmniBar';
import { AppMenuBar } from '@/components/AppMenuBar';
import { WorkspaceTabs, type WorkspaceTab } from '@/components/WorkspaceTabs';
import { RunAuditPanel } from '@/components/chat/RunAuditPanel';
import { DiscoveryWorkspaceTab } from '@/components/DiscoveryWorkspaceTab';
import { WelcomeTour } from '@/components/WelcomeTour';
import SettingsModal from '@/components/SettingsModal';
import { MissionControl } from '@/components/MissionControl';
import CandidateArtifactCard from '@/components/CandidateArtifact';
import { CapabilityGapArtifact } from '@/components/CapabilityGapArtifact';
import SpectroscopyArtifact from '@/components/SpectroscopyArtifact';
import type { ChatMode } from '@/hooks/useRunManager';

const WindowControls = dynamic(
  () => import('@/components/WindowControls').then((m) => ({ default: m.WindowControls })),
  { ssr: false }
);

import { api, type ModelStatusResponse, type ModelRegistryResponse, type ProjectInfo } from '@/lib/api';
import { STAGE_LABELS } from '@/lib/discovery-types';
import { useChatStore } from '@/stores/chatStore';
import { useGraphStore } from '@/stores/graphStore';
import { useDiscoveryStore } from '@/stores/discoveryStore';
import { useGoldenPathPipeline } from '@/hooks/useGoldenPathPipeline';
import { toastError, toast, toastSuccess } from '@/stores/toastStore';
import {
  ChevronLeft,
  MessageSquare,
  Cpu,
  Loader2,
  Trash2,
  Zap,
  Activity,
  FileText,
  Settings,
  X,
  FlaskConical,
  Plus,
  ArrowLeft,
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Stage empty / in-progress placeholder
// ---------------------------------------------------------------------------

function StageEmptyState({ stage }: { stage: number }) {
  const label = STAGE_LABELS[stage as keyof typeof STAGE_LABELS] ?? `Stage ${stage}`;
  return (
    <div className="flex h-full flex-col items-center justify-center text-muted-foreground select-none">
      <div className="rounded-2xl bg-primary/5 p-6 mb-5">
        <FlaskConical className="h-12 w-12 text-primary/30" />
      </div>
      <p className="font-serif text-xl text-foreground/50">
        Stage {stage}: {label}
      </p>
      <p className="mt-2 text-sm text-muted-foreground max-w-xs text-center">
        The pipeline is running in the background. Results will appear here when the stage completes.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main workspace
// ---------------------------------------------------------------------------

export default function ProjectWorkspacePage() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const projectId = decodeURIComponent(params.id || '');

  const { setActiveProject } = useChatStore();
  const activeThreadTitle = useChatStore((s) =>
    s.threads.find((t) => t.id === s.activeThreadId)?.title ?? 'Chat'
  );
  const { refreshGraph } = useGraphStore();

  const [currentProject, setCurrentProject] = useState<ProjectInfo | null>(null);
  const [loading, setLoading] = useState(true);

  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [selectedFilename, setSelectedFilename] = useState<string | null>(null);
  const [pdfPage, setPdfPage] = useState(1);
  /** VS Code-style workspace tabs: opening a file or session adds a tab; tabs can be reordered and closed */
  const [openTabs, setOpenTabs] = useState<WorkspaceTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);
  const [activeDiscoverySessionId, setActiveDiscoverySessionId] = useState<string | null>(null);

  const uploadInputRef = useRef<HTMLInputElement>(null);
  const importBibtexInputRef = useRef<HTMLInputElement>(null);
  const [libraryRefreshTrigger, setLibraryRefreshTrigger] = useState(0);

  const [modelRegistry, setModelRegistry] = useState<ModelRegistryResponse | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatusResponse | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [deletingProject, setDeletingProject] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [stageDimensions, setStageDimensions] = useState({ width: 800, height: 600 });

  // Chat sidebar — open/closed + which stage is visible ('list' = history, 'thread' = active chat)
  const [chatDrawerOpen, setChatDrawerOpen] = useState(false);
  const [chatView, setChatView] = useState<'list' | 'thread'>('list');

  // Mission Control
  const [missionControlOpen, setMissionControlOpen] = useState(false);

  // OmniBar query submission + Run Audit Panel
  const [pendingOmniBarQuery, setPendingOmniBarQuery] = useState<string | null>(null);
  const [runHistoryOpen, setRunHistoryOpen] = useState(false);
  const [auditRunId, setAuditRunId] = useState<string | null>(null);

  // Discovery store — read active epoch and session
  const activeEpochId = useDiscoveryStore((s) => s.activeEpochId);
  const epochs = useDiscoveryStore((s) => s.epochs);
  const sessionId = useDiscoveryStore((s) => s.sessionId);
  const activeEpoch = activeEpochId ? (epochs.get(activeEpochId) ?? null) : null;

  const { runHitGeneration, isPipelineRunning } = useGoldenPathPipeline();

  // Load project on mount
  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    let isMounted = true;

    const loadProject = async () => {
      if (!projectId) {
        if (isMounted) setLoading(false);
        return;
      }
      try {
        const projects = await api.listProjects();
        if (!isMounted) return;
        const project = projects.find((entry) => entry.id === projectId);
        if (!project) {
          router.push('/');
          return;
        }
        setCurrentProject(project);
        useChatStore.getState().setActiveProject(project.id);
        setLoading(false);
      } catch (error) {
        if (!isMounted) return;
        console.error('Failed to connect to backend. Retrying...', error);
        timeoutId = setTimeout(loadProject, 3000);
      }
    };

    setLoading(true);
    loadProject();

    return () => {
      isMounted = false;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clear active project on unmount (getState() avoids effect re-running when chat store updates)
  useEffect(() => {
    return () => {
      useChatStore.getState().setActiveProject(null);
    };
  }, []);

  // Track stage area dimensions for KnowledgeGraph
  useEffect(() => {
    const updateDimensions = () => {
      const el = document.getElementById('atlas-main-stage');
      if (!el) return;
      setStageDimensions({ width: el.clientWidth, height: el.clientHeight });
    };
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Ctrl+W / Cmd+W to close active tab
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'w') {
        e.preventDefault();
        if (activeTabId) {
          handleCloseTab(activeTabId);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeTabId]);

  // Load model registry
  const refreshModelRegistry = useCallback(async () => {
    try {
      const registry = await api.getModelRegistry();
      setModelRegistry(registry);
      setModelStatus(registry.active);
    } catch (err) {
      console.error('Failed to load model controls:', err);
    }
  }, []);

  useEffect(() => {
    refreshModelRegistry();

    const interval = window.setInterval(refreshModelRegistry, 8000);
    return () => window.clearInterval(interval);
  }, [refreshModelRegistry]);

  const handleModelChange = async (modelName: string) => {
    if (!modelName || modelStatus?.active_model === modelName) return;
    setModelLoading(true);
    try {
      const nextStatus = await api.loadModel(modelName);
      setModelStatus(nextStatus);
    } catch (err) {
      console.error('Failed to switch model:', err);
      toastError('Unable to switch model. Please check backend logs.');
    } finally {
      setModelLoading(false);
    }
  };

  const handleDeleteCurrentProject = async () => {
    const confirmed = window.confirm(`Delete project "${currentProject?.name}"? This cannot be undone.`);
    if (!confirmed || !currentProject) return;

    setDeletingProject(true);
    try {
      await api.deleteProject(currentProject.id);
      setActiveProject(null);
      router.replace('/');
      router.refresh();
    } catch (err) {
      console.error('Failed to delete current project:', err);
      toastError('Unable to delete project. Please check backend logs.');
    } finally {
      setDeletingProject(false);
    }
  };

  const handleIngestionComplete = useCallback(() => {
    refreshGraph();
  }, [refreshGraph]);

  const handleFileSelect = (docId: string, filename: string) => {
    const tabId = `document:${docId}:${filename}`;

    setOpenTabs((prev) => {
      // Check if this document is already open in a tab
      const existingIndex = prev.findIndex((t) => t.id === tabId);

      if (existingIndex !== -1) {
        // Document already has a tab - just switch to it
        setActiveTabId(tabId);
        return prev;
      }

      // Document not open yet
      if (activeTabId) {
        // Replace the currently active tab
        const activeIndex = prev.findIndex((t) => t.id === activeTabId);
        if (activeIndex !== -1) {
          const updated = [...prev];
          updated[activeIndex] = { kind: 'document', id: tabId, docId, filename };
          return updated;
        }
      }

      // No active tab or active tab not found - create new tab
      return [...prev, { kind: 'document', id: tabId, docId, filename }];
    });

    setActiveTabId(tabId);
    setSelectedDocId(docId);
    setSelectedFilename(filename);
    setActiveDiscoverySessionId(null);
    setPdfPage(1);
  };

  const handleSelectTab = (tabId: string) => {
    const tab = openTabs.find((t) => t.id === tabId);
    if (!tab) return;

    setActiveTabId(tabId);

    if (tab.kind === 'document') {
      setSelectedDocId(tab.docId);
      setSelectedFilename(tab.filename);
      setActiveDiscoverySessionId(null);
      setPdfPage(1);
    } else if (tab.kind === 'discovery') {
      setActiveDiscoverySessionId(tab.sessionId);
      setSelectedDocId(null);
      setSelectedFilename(null);
    }
  };

  const handleCloseTab = (tabId: string) => {
    setOpenTabs((prev) => {
      const next = prev.filter((t) => t.id !== tabId);
      const wasActive = activeTabId === tabId;
      if (wasActive && next.length > 0) {
        const newActive = next[0];
        setActiveTabId(newActive.id);
        if (newActive.kind === 'document') {
          setSelectedDocId(newActive.docId);
          setSelectedFilename(newActive.filename);
          setActiveDiscoverySessionId(null);
        } else if (newActive.kind === 'discovery') {
          setActiveDiscoverySessionId(newActive.sessionId);
          setSelectedDocId(null);
          setSelectedFilename(null);
        }
      } else if (wasActive && next.length === 0) {
        setActiveTabId(null);
        setSelectedDocId(null);
        setSelectedFilename(null);
        setActiveDiscoverySessionId(null);
      }
      return next;
    });
  };

  const handleReorderTabs = (fromIndex: number, toIndex: number) => {
    setOpenTabs((prev) => {
      const copy = [...prev];
      const [removed] = copy.splice(fromIndex, 1);
      copy.splice(toIndex, 0, removed);
      return copy;
    });
  };

  const handleFileDeleted = useCallback((docId: string) => {
    setOpenTabs((prev) => {
      const next = prev.filter((t) => !(t.kind === 'document' && t.docId === docId));
      const wasActive = activeTabId !== null && prev.some((t) => t.id === activeTabId && t.kind === 'document' && t.docId === docId);
      if (wasActive && next.length > 0) {
        const newActive = next[0];
        setActiveTabId(newActive.id);
        if (newActive.kind === 'document') {
          setSelectedDocId(newActive.docId);
          setSelectedFilename(newActive.filename);
          setActiveDiscoverySessionId(null);
        } else if (newActive.kind === 'discovery') {
          setActiveDiscoverySessionId(newActive.sessionId);
          setSelectedDocId(null);
          setSelectedFilename(null);
        }
      } else if (wasActive) {
        setActiveTabId(null);
        setSelectedDocId(null);
        setSelectedFilename(null);
        setActiveDiscoverySessionId(null);
      }
      return next;
    });
  }, [activeTabId]);

  const handleOpenDiscoveryTab = useCallback((sessionId: string, sessionName: string) => {
    const tabId = `discovery:${sessionId}`;
    setOpenTabs((prev) => {
      const exists = prev.find((t) => t.id === tabId);
      if (exists) {
        setActiveTabId(tabId);
        return prev;
      }
      return [...prev, { kind: 'discovery', id: tabId, sessionId, sessionName }];
    });
    setActiveTabId(tabId);
    setActiveDiscoverySessionId(sessionId);
    setSelectedDocId(null);
    setSelectedFilename(null);
    // Sync Zustand store so coordinator SSE calls use the correct session
    useDiscoveryStore.getState().setActiveSession(sessionId);
  }, []);

  const handleCitationClick = (filename: string, page: number, docId?: string) => {
    if (docId) {
      setSelectedDocId(docId);
      setSelectedFilename(filename);
    }
    setPdfPage(page);
  };

  const handleAskAboutPage = (question: string) => {
    useChatStore.getState().setPendingQuestion(question);
    setChatDrawerOpen(true);
    setChatView('thread');
  };

  const handleUploadClick = () => uploadInputRef.current?.click();
  const handleImportBibtexClick = () => importBibtexInputRef.current?.click();

  const handleUploadFiles = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList?.length || !projectId) return;
    try {
      for (let i = 0; i < fileList.length; i++) {
        await api.uploadFile(fileList[i], projectId);
      }
      setLibraryRefreshTrigger((t) => t + 1);
    } catch (err: any) {
      toastError(err?.message ?? 'Upload failed');
    } finally {
      e.target.value = '';
    }
  };

  const handleImportBibtexFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !projectId) return;
    try {
      toast('Importing bibliography...', 'info');
      const result = await api.importBibtex(file, projectId);
      if (result.total_imported > 0) {
        toastSuccess(`Imported ${result.total_imported} of ${result.total_entries} entries`);
        setLibraryRefreshTrigger((t) => t + 1);
      } else {
        toastError('No entries were imported');
      }
    } catch (err: any) {
      toastError(`Import failed: ${err?.message ?? 'Unknown error'}`);
    } finally {
      e.target.value = '';
    }
  };

  const handleExport = async (type: 'bibtex' | 'markdown' | 'chat') => {
    try {
      toast(`Exporting ${type}...`, 'info');

      if (type === 'bibtex') {
        const blob = await api.exportBibtexProject(projectId);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${currentProject?.name || 'atlas-project'}.bib`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toastSuccess('BibTeX export complete');
      } else if (type === 'markdown') {
        const editorContent = localStorage.getItem(`atlas-editor-${projectId}`) || '';
        if (!editorContent.trim()) {
          toastError('No editor content to export');
          return;
        }
        const result = await api.exportMarkdown({
          content: editorContent,
          citations: [],
          projectId: projectId,
          title: currentProject?.name || 'Research Synthesis',
          author: 'Atlas User',
          style: 'apa',
        });
        const blob = new Blob([result.markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = result.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toastSuccess('Markdown export complete');
      } else if (type === 'chat') {
        const store = useChatStore.getState();
        const activeThread = store.getActiveThread();
        const allMessages = activeThread?.messages ?? [];
        const exportMessages = allMessages.filter((m) => m.role === 'user' || m.role === 'assistant');
        if (exportMessages.length === 0) {
          toastError('No chat history to export');
          return;
        }

        const result = await api.exportChatHistory(exportMessages, currentProject?.name);
        const blob = new Blob([result.markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${currentProject?.name || 'chat'}-history.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toastSuccess('Chat history exported');
      }
    } catch (error: any) {
      toastError(`Export failed: ${error.message}`);
    }
  };

  const handleOmniBarQuery = useCallback((query: string, mode: ChatMode) => {
    useChatStore.getState().setChatMode(mode);
    setChatDrawerOpen(true);
    setChatView('thread');
    setPendingOmniBarQuery(query);
  }, []);

  const handleOpenRunHistory = useCallback(() => {
    setAuditRunId(null);
    setRunHistoryOpen(true);
  }, []);

  const handleViewRunDetails = useCallback((runId: string) => {
    setAuditRunId(runId);
    setRunHistoryOpen(true);
  }, []);

  // OmniBar view-switch: 'chat' opens drawer to the history list
  const handleViewSwitch = useCallback((view: string) => {
    if (view === 'chat') {
      setChatDrawerOpen(true);
      setChatView('list');
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Stage content renderer
  // ---------------------------------------------------------------------------

  function renderStageContent() {
    // Priority 1: Active discovery session tab
    if (activeDiscoverySessionId) {
      return (
        <DiscoveryWorkspaceTab
          sessionId={activeDiscoverySessionId}
          projectId={currentProject!.id}
        />
      );
    }

    // Priority 2: Existing epoch/document logic
    if (!activeEpoch && !selectedDocId) {
      return (
        <div className="flex h-full flex-col items-center justify-center text-muted-foreground">
          <div className="rounded-2xl bg-primary/5 p-6 mb-4">
            <FlaskConical className="h-10 w-10 text-primary/30" />
          </div>
          <p className="font-serif text-lg text-foreground/60">Workspace Ready</p>
          <p className="mt-1 text-sm text-muted-foreground max-w-sm text-center mb-6">
            Select a document from your library to review, or start a new Discovery Session from the sidebar to define your research objectives and constraints.
          </p>
        </div>
      );
    }

    // Unresolved capability gap takes priority
    const unresolvedGap = activeEpoch ? activeEpoch.capabilityGaps.find((g) => g.resolution === null) : null;
    if (unresolvedGap) {
      return (
        <div className="flex h-full items-center justify-center p-6">
          <div className="w-full max-w-2xl">
            <CapabilityGapArtifact gap={unresolvedGap} />
          </div>
        </div>
      );
    }

    const stageOrFallback = activeEpoch ? activeEpoch.currentStage : 1;

    switch (stageOrFallback) {
      case 1: {
        // Corpus viewer — show selected document or prompt to select one
        if (selectedDocId && selectedFilename) {
          return /\.(txt|text|md|csv|log|json|xml)$/i.test(selectedFilename) ? (
            <TextViewer
              fileUrl={api.getFileUrl(selectedDocId)}
              filename={selectedFilename}
              docId={selectedDocId}
              projectId={currentProject!.id}
              onContextChange={() => { }}
            />
          ) : (
            <PDFViewer
              fileUrl={api.getFileUrl(selectedDocId)}
              filename={selectedFilename}
              docId={selectedDocId}
              projectId={currentProject!.id}
              initialPage={pdfPage}
              onAskAboutPage={handleAskAboutPage}
              onRelatedPassageClick={handleCitationClick}
              onContextChange={() => { }}
            />
          );
        }
        return (
          <div className="flex h-full flex-col items-center justify-center text-muted-foreground">
            <div className="rounded-2xl bg-primary/5 p-6 mb-4">
              <FileText className="h-10 w-10 text-primary/30" />
            </div>
            <p className="font-serif text-lg text-foreground/60">No document selected</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Select a document from the library to begin corpus review
            </p>
          </div>
        );
      }

      case 4: {
        // Hit grid — Stage 4: SURFACE
        if (activeEpoch && activeEpoch.candidates.length > 0) {
          return (
            <div className="h-full overflow-y-auto p-6">
              <div className="mb-4 flex items-center gap-3">
                <span className="text-xs uppercase tracking-wider text-muted-foreground font-semibold">
                  {activeEpoch.candidates.length} candidate{activeEpoch.candidates.length !== 1 ? 's' : ''}
                </span>
                <span className="text-muted-foreground/40">·</span>
                <span className="text-xs text-muted-foreground">
                  {activeEpoch.candidates.filter((c) => c.status === 'approved').length} approved
                </span>
              </div>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                {activeEpoch.candidates.map((hit) => (
                  <CandidateArtifactCard key={hit.id} hit={hit} />
                ))}
              </div>
            </div>
          );
        }
        return <StageEmptyState stage={4} />;
      }

      case 6: {
        // Spectroscopy validation — Stage 6
        if (activeEpoch && activeEpoch.validations.length > 0) {
          return (
            <div className="h-full overflow-y-auto p-6">
              <SpectroscopyArtifact validation={activeEpoch.validations[0]} />
            </div>
          );
        }
        return <StageEmptyState stage={6} />;
      }

      case 7: {
        // Feedback loop — Stage 7: show knowledge graph
        return (
          <KnowledgeGraph
            height={stageDimensions.height}
            width={stageDimensions.width}
            projectId={currentProject!.id}
            documentId={selectedDocId || undefined}
          />
        );
      }

      default:
        return <StageEmptyState stage={stageOrFallback} />;
    }
  }

  // ---------------------------------------------------------------------------
  // Loading / error states
  // ---------------------------------------------------------------------------

  if (loading) {
    return (
      <div className="flex h-screen w-screen flex-col items-center justify-center bg-background overflow-hidden">
        {/* Ambient background glow */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] rounded-full bg-safety/4 blur-3xl" />
        </div>

        <div className="relative flex flex-col items-center gap-10">
          {/* Spinner with Atlas icon */}
          <div className="relative flex items-center justify-center">
            <div className="absolute h-24 w-24 rounded-full border border-safety/10 animate-ping" style={{ animationDuration: '2.5s' }} />
            <div className="absolute h-20 w-20 rounded-full border border-safety/20" />
            <div className="h-16 w-16 rounded-full border-2 border-safety/10 border-t-safety animate-spin" style={{ animationDuration: '0.9s' }} />
            <div className="absolute flex items-center justify-center">
              <Zap className="h-5 w-5 text-safety" />
            </div>
          </div>

          {/* Title */}
          <div className="flex flex-col items-center gap-2 text-center">
            <p className="text-sm font-medium text-text-primary tracking-wide">Initializing workspace</p>
            <p className="text-[11px] text-text-secondary/50">Connecting knowledge graph and AI agents…</p>
          </div>

          {/* Step indicators — staggered fade-in */}
          <div className="flex items-center gap-8">
            {([
              { icon: FileText, label: 'Library' },
              { icon: Activity, label: 'Graph' },
              { icon: Cpu, label: 'Agents' },
            ] as const).map(({ icon: Icon, label }, i) => (
              <div
                key={label}
                className="flex flex-col items-center gap-2 opacity-0 animate-in fade-in duration-400"
                style={{ animationDelay: `${200 + i * 150}ms`, animationFillMode: 'forwards' }}
              >
                <div className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-border/40 bg-card/60">
                  <Icon className="h-4 w-4 text-safety/60" />
                  {/* Pulsing dot */}
                  <span className="absolute -top-0.5 -right-0.5 h-2 w-2 rounded-full bg-safety/50 animate-pulse" style={{ animationDelay: `${i * 300}ms` }} />
                </div>
                <span className="text-[10px] text-text-secondary/40 tracking-widest uppercase">{label}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!projectId || !currentProject) {
    return (
      <div className="flex h-screen w-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <p className="text-sm text-muted-foreground">Project not found.</p>
          <button
            onClick={() => router.push('/')}
            className="rounded-lg border border-border bg-surface px-4 py-2 text-xs font-medium text-foreground transition-colors hover:bg-primary/10 hover:border-primary/30"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Main workspace
  // ---------------------------------------------------------------------------

  return (
    <main className="flex h-screen w-screen flex-col overflow-hidden bg-background text-foreground">

      {/* Hidden file inputs for File menu */}
      <input
        ref={uploadInputRef}
        type="file"
        className="hidden"
        accept=".pdf,.txt,.docx,.doc"
        multiple
        onChange={handleUploadFiles}
      />
      <input
        ref={importBibtexInputRef}
        type="file"
        className="hidden"
        accept=".bib,.ris"
        onChange={handleImportBibtexFile}
      />

      {/* ========== Header (Stacked layout: menu above, workspace toolbar below) ========== */}
      <div className="relative flex flex-col border-b border-border bg-card/80 backdrop-blur-sm z-[100]">
        {/* Top native-like Menu Bar */}
        <div data-tauri-drag-region className="relative flex h-8 w-full items-center justify-between px-2 shrink-0">

          <div className="flex items-center gap-4 h-full">
            {/* Atlas Logo */}
            <div className="flex items-center gap-1.5 no-drag select-none cursor-pointer" onClick={() => router.push('/')}>
              <div className="flex h-4 w-4 items-center justify-center rounded bg-safety/10 border border-safety/20">
                <Zap className="h-3 w-3 text-safety" />
              </div>
            </div>

            {/* Application Menu */}
            <div className="no-drag">
              <AppMenuBar
                projectId={currentProject.id}
                projectName={currentProject.name}
                onUploadDocuments={handleUploadClick}
                onImportBibtex={handleImportBibtexClick}
                onExport={handleExport}
                onOpenSettings={() => setSettingsOpen(true)}
                onSwitchView={(view) => { if (view === 'chat') { setChatDrawerOpen(true); setChatView('list'); } }}
                onStartDiscovery={() => setMissionControlOpen(true)}
                onRunHitGeneration={() => sessionId && activeEpoch && runHitGeneration({ projectId: currentProject.id, sessionId, epochId: activeEpoch.id, mock: false })}
                showDiscoveryActions={!!activeEpoch && !!sessionId}
                isPipelineRunning={isPipelineRunning}
              />
            </div>
          </div>

          {/* Centered Project Title */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none flex items-center justify-center">
            <span className="text-[12px] text-muted-foreground/80 font-medium tracking-wide">
              {currentProject.name} <span className="text-muted-foreground/40 font-normal mx-1">—</span> Atlas
            </span>
          </div>

          {/* Custom OS Window Controls (Minimize, Maximize, Close) */}
          <div className="h-full shrink-0 no-drag">
            <WindowControls />
          </div>
        </div>

        {/* Workspace Toolbar (Second row) */}
        <div className="flex h-11 w-full items-center justify-between border-t border-border/40 bg-surface/30 px-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => router.push('/')}
              className="inline-flex h-7 w-7 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition-colors hover:bg-primary/10 hover:border-primary/30 hover:text-primary"
              title="Back to Dashboard"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
          </div>

          <div className="flex items-center gap-3">
            {/* Status bar */}
            <div className="flex items-center gap-3 text-xs text-muted-foreground border-r border-border pr-3">
              <div className="flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" />
                <span>{useGraphStore.getState().nodes.length} nodes</span>
              </div>
              {modelStatus?.device && (
                <div className="flex items-center gap-1.5">
                  <Zap className="h-3.5 w-3.5 text-accent" />
                  <span>{modelStatus.device.toUpperCase()}</span>
                </div>
              )}
            </div>

            {/* Model selector: Local + Cloud API (always show both groups) */}
            <label className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Model</label>
            <select
              value={modelStatus?.active_model || ''}
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={modelLoading || ((modelRegistry?.local?.length ?? 0) + (modelRegistry?.api?.length ?? 0)) === 0}
              className="h-8 min-w-[200px] rounded-lg border border-border bg-surface px-2.5 text-xs text-foreground outline-none transition-colors focus:border-primary/50 focus:ring-1 focus:ring-primary/25 disabled:opacity-60"
            >
              {((modelRegistry?.local?.length ?? 0) + (modelRegistry?.api?.length ?? 0)) === 0 && (
                <option value="">No models found</option>
              )}
              {modelRegistry?.local && modelRegistry.local.length > 0 && (
                <optgroup label="Local">
                  {modelRegistry.local.map((m) => (
                    <option key={m.name} value={m.name}>{m.name}</option>
                  ))}
                </optgroup>
              )}
              <optgroup label="Cloud API">
                {modelRegistry?.api && modelRegistry.api.length > 0 ? (
                  modelRegistry.api.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.name}{m.has_key === false ? ' (no API key)' : ''}
                    </option>
                  ))
                ) : (
                  <option value="" disabled>
                    No cloud models — configure in Settings
                  </option>
                )}
              </optgroup>
            </select>

            <div className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-border bg-surface px-2.5 text-xs text-muted-foreground">
              {modelLoading ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
              ) : (
                <Cpu className="h-3.5 w-3.5" />
              )}
              <span>{(modelStatus?.device || 'unknown').toUpperCase()}</span>
            </div>

            {activeEpoch && sessionId && (
              <button
                onClick={() => {
                  runHitGeneration({
                    projectId: currentProject.id,
                    sessionId,
                    epochId: activeEpoch.id,
                    mock: false,
                  });
                }}
                disabled={isPipelineRunning}
                className="flex h-8 items-center gap-2 rounded-lg border border-orange-500/30 bg-orange-500/10 px-3 text-xs font-medium text-orange-500 transition-colors hover:bg-orange-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                title="Run LLM candidate generation then deterministic RDKit screen"
              >
                {isPipelineRunning ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <FlaskConical className="h-3.5 w-3.5" />
                )}
                Run Hit Generation
              </button>
            )}

            {/* Chat sidebar toggle */}
            <button
              onClick={() => {
                if (chatDrawerOpen) {
                  setChatDrawerOpen(false);
                } else {
                  setChatDrawerOpen(true);
                  setChatView('list');
                }
              }}
              className={`flex h-8 items-center gap-2 rounded-lg border px-3 text-xs font-medium transition-colors ${chatDrawerOpen
                ? 'border-primary/30 bg-primary/10 text-primary'
                : 'border-border bg-surface text-muted-foreground hover:bg-primary/10 hover:border-primary/30 hover:text-primary'
                }`}
              title="Toggle chat panel"
            >
              <MessageSquare className="h-3.5 w-3.5" />
              Chat
            </button>

            {/* Settings */}
            <button
              onClick={() => setSettingsOpen(true)}
              className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-surface text-muted-foreground transition-colors hover:bg-primary/10 hover:border-primary/30 hover:text-primary"
              title="Configure API Keys"
            >
              <Settings className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* ========== Body: Left | Stage | Pipeline ========== */}
      <div className="flex min-h-0 flex-1 overflow-hidden">

        {/* ---- Left sidebar ---- */}
        <div className="flex w-60 shrink-0 flex-col border-r border-border bg-card overflow-hidden">
          {/* Library */}
          <div className="min-h-0 flex-1 overflow-hidden">
            <LibrarySidebar
              onFileSelect={handleFileSelect}
              selectedDocId={selectedDocId}
              projectId={currentProject.id}
              onIngestionComplete={handleIngestionComplete}
              onFileDeleted={handleFileDeleted}
              onOpenDiscoverySession={handleOpenDiscoveryTab}
              onStartDiscovery={() => setMissionControlOpen(true)}
              refreshTrigger={libraryRefreshTrigger}
            />
          </div>

          {/* Project actions */}
          <div className="shrink-0 border-t border-border p-3">
            <button
              onClick={handleDeleteCurrentProject}
              disabled={deletingProject}
              className="inline-flex h-8 w-full items-center gap-2 rounded-lg border border-border bg-background px-2.5 text-xs text-muted-foreground transition-colors hover:bg-destructive/10 hover:border-destructive/30 hover:text-destructive disabled:opacity-50"
            >
              {deletingProject ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Trash2 className="h-3.5 w-3.5" />
              )}
              Delete Project
            </button>
          </div>
        </div>

        {/* ---- Center: Stage ---- */}
        <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-background">

          {/* Workspace tabs (VS Code-style): documents + discovery sessions */}
          {openTabs.length > 0 && (
            <WorkspaceTabs
              tabs={openTabs}
              activeTabId={activeTabId}
              onSelectTab={handleSelectTab}
              onCloseTab={handleCloseTab}
              onReorder={handleReorderTabs}
            />
          )}

          {/* Stage content */}
          <div id="atlas-main-stage" className="relative min-h-0 flex-1 overflow-hidden">
            {currentProject && renderStageContent()}
          </div>
        </div>

        {/* ---- Chat sidebar (push-aside, not an overlay) ---- */}
        {chatDrawerOpen && (
          <div
            className="flex w-[380px] shrink-0 flex-col border-l border-border bg-card overflow-hidden"
            style={{ animation: 'slideInRight 180ms ease-out' }}
          >
            {chatView === 'list' ? (
              /* ---- Stage 1: Chat history ---- */
              <ChatHistoryPanel
                projectId={currentProject.id}
                onSelectThread={(threadId) => {
                  useChatStore.getState().switchThread(threadId);
                  setChatView('thread');
                }}
                onNewChat={() => {
                  useChatStore.getState().createThread(currentProject.id);
                  setChatView('thread');
                }}
                onClose={() => setChatDrawerOpen(false)}
              />
            ) : (
              /* ---- Stage 2: Active thread ---- */
              <>
                {/* Thread header with back button */}
                <div className="flex h-11 shrink-0 items-center gap-1 border-b border-border px-2">
                  <button
                    onClick={() => setChatView('list')}
                    className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-surface hover:text-foreground"
                    title="Back to chat list"
                  >
                    <ArrowLeft className="h-4 w-4" />
                  </button>
                  <div className="flex min-w-0 flex-1 items-center gap-2 px-1">
                    <MessageSquare className="h-3.5 w-3.5 shrink-0 text-primary" />
                    <span className="truncate text-xs font-medium text-foreground/80">
                      {activeThreadTitle}
                    </span>
                  </div>
                  <button
                    onClick={() => setChatDrawerOpen(false)}
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-surface hover:text-foreground"
                    title="Close chat"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
                {/* Chat shell */}
                <div className="min-h-0 flex-1 overflow-hidden">
                  <ChatShell
                    onCitationClick={handleCitationClick}
                    projectId={currentProject.id}
                    autoSubmitQuery={pendingOmniBarQuery}
                    onAutoSubmitConsumed={() => setPendingOmniBarQuery(null)}
                    onOpenRunHistory={handleOpenRunHistory}
                    onViewRunDetails={handleViewRunDetails}
                  />
                </div>
              </>
            )}
          </div>
        )}

      </div>

      {/* ========== Global overlays ========== */}

      {/* OmniBar */}
      <OmniBar
        projectId={currentProject.id}
        onUpload={handleUploadClick}
        onExport={handleExport}
        onSwitchView={handleViewSwitch}
        onSubmitQuery={handleOmniBarQuery}
        onOpenRunHistory={handleOpenRunHistory}
      />

      {/* Run audit panel */}
      <RunAuditPanel
        open={runHistoryOpen}
        onClose={() => setRunHistoryOpen(false)}
        projectId={currentProject.id}
        selectedRunId={auditRunId}
      />

      {/* Welcome tour */}
      <WelcomeTour />

      {/* Settings modal */}
      <SettingsModal open={settingsOpen} onOpenChange={setSettingsOpen} onKeysUpdated={refreshModelRegistry} />

      {/* Mission Control Modal Overlay */}
      {missionControlOpen && currentProject && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm">
          <MissionControl
            projectId={currentProject.id}
            onSuccess={(sessionId, sessionName) => {
              setMissionControlOpen(false);
              // Immediately open the workspace tab for the new session
              handleOpenDiscoveryTab(sessionId, sessionName);
            }}
            onCancel={() => setMissionControlOpen(false)}
          />
        </div>
      )}

      {/* Slide-in animation keyframe */}
      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100%); opacity: 0; }
          to   { transform: translateX(0);    opacity: 1; }
        }
      `}</style>
    </main>
  );
}
