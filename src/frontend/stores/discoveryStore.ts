import { enableMapSet } from 'immer';
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

enableMapSet();

// Stable singleton references — prevents infinite re-render loops in Zustand
// selectors when no active session exists. `new Map()` / `[]` would create a
// fresh reference on every call, failing Object.is() equality.
const EMPTY_EPOCHS: Map<string, import('../lib/discovery-types').Epoch> = new Map();
const EMPTY_JOBS: import('../lib/discovery-types').BackgroundJob[] = [];

import type {
    Epoch,
    GoldenPathStage,
    CandidateArtifact,
    CapabilityGap,
    SpectroscopyValidation,
    BioassayResult,
    BackgroundJob,
    ProjectTargetParams,
    StageArtifact,
    CapabilityGapResolution,
} from '../lib/discovery-types';

// Multi-session state
export interface SessionState {
    sessionId: string;
    sessionName: string;
    projectId: string;
    createdAt: string;
    status: 'idle' | 'running' | 'complete';
    epochs: Map<string, Epoch>;
    activeEpochId: string | null;
    rootEpochId: string | null;
    backgroundJobs: BackgroundJob[];
    generatedFiles: string[];
}

export interface DiscoveryStore {
    // Multi-session dictionary
    sessions: Record<string, SessionState>;
    activeSessionId: string | null;

    // Backward-compat getters (legacy single-session consumers)
    get sessionId(): string | null;
    get epochs(): Map<string, Epoch>;
    get activeEpochId(): string | null;
    get rootEpochId(): string | null;
    get backgroundJobs(): BackgroundJob[];

    // Derived (computed via getters, not stored)
    get activeEpoch(): Epoch | null;
    get activeStageArtifact(): StageArtifact | null;

    // Session actions
    setActiveSession: (sessionId: string) => void;

    // Epoch actions
    initializeSession: (params: ProjectTargetParams, sessionId?: string) => string;
    forkEpoch: (
        parentEpochId: string,
        reason: string,
        paramOverrides?: Partial<ProjectTargetParams>,
        startStage?: GoldenPathStage,
    ) => string;
    switchToEpoch: (epochId: string) => void;

    // Stage actions (operate on active epoch)
    approveHit: (hitId: string) => void;
    rejectHit: (hitId: string) => void;
    resolveCapabilityGap: (gapId: string, resolution: CapabilityGapResolution) => void;
    advanceToStage: (stage: GoldenPathStage) => void;
    submitRawDataFile: (hitId: string, reportFile: File) => void;
    submitExperimentalResult: (hitId: string, result: BioassayResult) => void;

    // Jobs (Golden Path pipeline)
    addJob: (job: BackgroundJob) => void;
    updateJob: (jobId: string, partial: Partial<BackgroundJob>) => void;

    // Stage 4: set candidates from pipeline or Discovery chat
    setCandidatesForEpoch: (epochId: string, candidates: CandidateArtifact[]) => void;

    // Sync a session that exists in the backend but not in local store (e.g. after restart)
    upsertBackendSession: (sessionId: string, sessionName: string, createdAt: string, projectId?: string) => void;
}

export const useDiscoveryStore = create<DiscoveryStore>()(
    persist(
        immer((set, get) => ({
            sessions: {},
            activeSessionId: null,

            // Backward-compat getters
            get sessionId() {
                return get().activeSessionId;
            },
            get epochs() {
                const state = get();
                return state.sessions[state.activeSessionId ?? '']?.epochs ?? EMPTY_EPOCHS;
            },
            get activeEpochId() {
                const state = get();
                return state.sessions[state.activeSessionId ?? '']?.activeEpochId ?? null;
            },
            get rootEpochId() {
                const state = get();
                return state.sessions[state.activeSessionId ?? '']?.rootEpochId ?? null;
            },
            get backgroundJobs() {
                const state = get();
                return state.sessions[state.activeSessionId ?? '']?.backgroundJobs ?? EMPTY_JOBS;
            },

            get activeEpoch() {
                const state = get();
                const activeEpochId = state.activeEpochId;
                if (!activeEpochId) return null;
                const epochs = state.epochs;
                return epochs.get(activeEpochId) || null;
            },

            get activeStageArtifact(): StageArtifact | null {
                const epoch = get().activeEpoch;
                if (!epoch) return null;

                // Determine artifact based on current stage and epoch state.
                // This is a basic mapping; can be refined as components are built.
                switch (epoch.currentStage) {
                    case 1:
                        // Assuming corpus viewer is active if we have corpus docs
                        if (epoch.targetParams.corpusDocumentIds.length > 0) {
                            return { type: 'corpus_viewer', documentId: epoch.targetParams.corpusDocumentIds[0] };
                        }
                        return null;
                    case 4:
                        return { type: 'hit_grid', hits: epoch.candidates };
                    case 5:
                        // Placeholder for synthesis plan tree
                        const approvedHitForSynthesis = epoch.candidates.find(c => c.status === 'approved');
                        if (approvedHitForSynthesis && approvedHitForSynthesis.synthesisPlanRunId) {
                            return { type: 'synthesis_plan_tree', hitId: approvedHitForSynthesis.id, runId: approvedHitForSynthesis.synthesisPlanRunId };
                        }
                        return null;
                    case 6:
                        // Placeholder: showing the first validation if available
                        if (epoch.validations.length > 0) {
                            return { type: 'spectroscopy_validation', hitId: epoch.validations[0].hitId, validationId: epoch.validations[0].id };
                        }
                        return null;
                    case 7:
                        // Placeholder, assumes project id is part of the session, though not directly in target params. Using a placeholder for now.
                        return { type: 'knowledge_graph', projectId: 'current-project' };
                    default:
                        return null;
                }
            },

            setActiveSession: (sessionId) => {
                set((draft) => {
                    if (draft.sessions[sessionId]) {
                        draft.activeSessionId = sessionId;
                    } else {
                        console.warn(`Attempted to activate non-existent session: ${sessionId}`);
                    }
                });
            },

            initializeSession: (params, sessionId) => {
                const sid = sessionId ?? crypto.randomUUID();
                const rootEpochId = crypto.randomUUID();
                const rootEpoch: Epoch = {
                    id: rootEpochId,
                    parentEpochId: null,
                    forkReason: 'Initial Session',
                    targetParams: params,
                    currentStage: 1,
                    createdAt: Date.now(),
                    candidates: [],
                    capabilityGaps: [],
                    validations: [],
                    feedbackResults: [],
                    stageRuns: {},
                };

                const sessionName = params.objective || `Session ${new Date().toLocaleDateString()}`;
                const newSession: SessionState = {
                    sessionId: sid,
                    sessionName,
                    projectId: params.projectId ?? '',
                    createdAt: new Date().toISOString(),
                    status: 'idle',
                    epochs: new Map([[rootEpochId, rootEpoch]]),
                    activeEpochId: rootEpochId,
                    rootEpochId,
                    backgroundJobs: [],
                    generatedFiles: [],
                };

                set((state) => {
                    state.sessions[sid] = newSession;
                    // Don't auto-set activeSessionId - let the tab selection handle it
                });

                return rootEpochId;
            },

            forkEpoch: (parentEpochId, reason, paramOverrides, startStage = 2) => {
                const state = get();
                const activeSession = state.sessions[state.activeSessionId ?? ''];
                if (!activeSession) throw new Error('No active session');

                const parentEpoch = activeSession.epochs.get(parentEpochId);
                if (!parentEpoch) {
                    throw new Error(`Cannot fork: Parent epoch ${parentEpochId} not found.`);
                }

                const newEpochId = crypto.randomUUID();
                const targetParamsClone: ProjectTargetParams = JSON.parse(JSON.stringify(parentEpoch.targetParams));

                const newEpoch: Epoch = {
                    id: newEpochId,
                    parentEpochId,
                    forkReason: reason,
                    targetParams: { ...targetParamsClone, ...paramOverrides },
                    currentStage: startStage,
                    createdAt: Date.now(),
                    candidates: [],
                    capabilityGaps: [],
                    validations: [],
                    feedbackResults: [],
                    stageRuns: {},
                };

                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (session) {
                        session.epochs.set(newEpochId, newEpoch);
                    }
                });

                return newEpochId;
            },

            switchToEpoch: (epochId) => {
                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (session && session.epochs.has(epochId)) {
                        session.activeEpochId = epochId;
                    } else {
                        console.warn(`Attempted to switch to non-existent epoch: ${epochId}`);
                    }
                });
            },

            approveHit: (hitId) => {
                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (!session) throw new Error("No active session");
                    const epochId = session.activeEpochId;
                    if (!epochId) throw new Error("No active epoch");
                    const epoch = session.epochs.get(epochId);
                    if (!epoch) return;

                    const hit = epoch.candidates.find(c => c.id === hitId);
                    if (hit) {
                        hit.status = 'approved';
                    }
                });
            },

            rejectHit: (hitId) => {
                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (!session) throw new Error("No active session");
                    const epochId = session.activeEpochId;
                    if (!epochId) throw new Error("No active epoch");
                    const epoch = session.epochs.get(epochId);
                    if (!epoch) return;

                    const hit = epoch.candidates.find(c => c.id === hitId);
                    if (hit) {
                        hit.status = 'rejected';
                    }
                });
            },

            resolveCapabilityGap: (gapId, resolution) => {
                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (!session) throw new Error("No active session");
                    const epochId = session.activeEpochId;
                    if (!epochId) throw new Error("No active epoch");
                    const epoch = session.epochs.get(epochId);
                    if (!epoch) return;

                    const gap = epoch.capabilityGaps.find(g => g.id === gapId);
                    if (gap) {
                        gap.resolution = resolution;
                    }
                });
            },

            advanceToStage: (stage) => {
                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (!session) throw new Error("No active session");
                    const epochId = session.activeEpochId;
                    if (!epochId) throw new Error("No active epoch");
                    const epoch = session.epochs.get(epochId);
                    if (epoch) {
                        epoch.currentStage = stage;
                    }
                });
            },

            submitRawDataFile: (hitId, reportFile) => {
                // Placeholder for actual file processing logic which would likely involve backend uploads
                console.log(`Submitting raw data file ${reportFile.name} for hit ${hitId}`);
                // In a real implementation, this might trigger a background job to process the file
            },

            submitExperimentalResult: (hitId, result) => {
                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (!session) throw new Error("No active session");
                    const epochId = session.activeEpochId;
                    if (!epochId) throw new Error("No active epoch");
                    const epoch = session.epochs.get(epochId);
                    if (!epoch) return;

                    epoch.feedbackResults.push(result);
                });
            },

            addJob: (job) => {
                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (session) {
                        session.backgroundJobs.push(job);
                    }
                });
            },

            updateJob: (jobId, partial) => {
                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (session) {
                        const job = session.backgroundJobs.find((j) => j.id === jobId);
                        if (job) Object.assign(job, partial);
                    }
                });
            },

            setCandidatesForEpoch: (epochId, candidates) => {
                set((draft) => {
                    const session = draft.sessions[draft.activeSessionId ?? ''];
                    if (!session) return;
                    const epoch = session.epochs.get(epochId);
                    if (epoch) {
                        epoch.candidates = candidates;
                        epoch.currentStage = 4;
                    }
                });
            },

            upsertBackendSession: (sessionId, sessionName, createdAt, projectId) => {
                set((draft) => {
                    // Only add if not already present — never overwrite richer local state
                    if (!draft.sessions[sessionId]) {
                        draft.sessions[sessionId] = {
                            sessionId,
                            sessionName,
                            projectId: projectId ?? '',
                            createdAt,
                            status: 'idle',
                            epochs: new Map(),
                            activeEpochId: null,
                            rootEpochId: null,
                            backgroundJobs: [],
                            generatedFiles: [],
                        };
                    }
                });
            },

        })),
        {
            name: 'atlas-discovery-store',
            version: 2, // Bumped for multi-session refactor
            storage: createJSONStorage(() => localStorage),
            partialize: (state) => ({
                sessions: Object.fromEntries(
                    Object.entries(state.sessions).map(([sid, session]) => [
                        sid,
                        {
                            ...session,
                            epochs: Array.from(session.epochs.entries()),
                        },
                    ])
                ),
                activeSessionId: state.activeSessionId,
            }),
            merge: (persistedState: any, currentState) => {
                if (!persistedState) return currentState;
                const sessions: Record<string, SessionState> = {};
                if (persistedState.sessions) {
                    Object.entries(persistedState.sessions).forEach(([sid, session]: [string, any]) => {
                        sessions[sid] = {
                            ...session,
                            epochs: new Map(session.epochs || []),
                        };
                    });
                }
                return {
                    ...currentState,
                    sessions,
                    activeSessionId: persistedState.activeSessionId ?? null,
                };
            },
        }
    )
);
