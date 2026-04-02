/**
 * Discovery OS — Golden Path Shared Types
 *
 * Barrel file for every interface and type used across the Discovery OS
 * frontend. No logic — pure declarations only.
 *
 * Source: docs/DiscoveryOS_GoldenPath_Plan.md v3.1
 */

// ---------------------------------------------------------------------------
// Domain
// ---------------------------------------------------------------------------

export interface DomainSchema {
  domain: string;
  target_schema: string[];
}

export interface ProjectTargetParams {
  domain: string;
  objective: string;
  propertyConstraints: PropertyConstraint[];
  domainSpecificConstraints: Record<string, any>;
  corpusDocumentIds: string[];
  projectId?: string;
}

export interface PropertyConstraint {
  property: 'MW' | 'LogP' | 'TPSA' | 'HBD' | 'HBA' | 'RotBonds' | (string & {});
  operator: '<' | '>' | '<=' | '>=' | 'between';
  value: number | [number, number];
}

// ---------------------------------------------------------------------------
// Golden Path Stages
// ---------------------------------------------------------------------------

export type GoldenPathStage = 1 | 2 | 3 | 4 | 5 | 6 | 7;

export const STAGE_LABELS: Record<GoldenPathStage, string> = {
  1: 'PRIME',
  2: 'GENERATE',
  3: 'SCREEN',
  4: 'SURFACE',
  5: 'SYNTHESIS_PLAN',
  6: 'SPECTROSCOPY_VALIDATION',
  7: 'FEEDBACK',
} as const;

// ---------------------------------------------------------------------------
// Epochs (Branching State Model)
// ---------------------------------------------------------------------------

export interface Epoch {
  id: string;
  parentEpochId: string | null;
  forkReason: string;
  targetParams: ProjectTargetParams;
  currentStage: GoldenPathStage;
  createdAt: number;

  candidates: CandidateArtifact[];
  capabilityGaps: CapabilityGap[];
  validations: SpectroscopyValidation[];
  feedbackResults: BioassayResult[];

  stageRuns: Partial<Record<GoldenPathStage, string>>;
}

// ---------------------------------------------------------------------------
// Candidate Artifacts (Stage 4)
// ---------------------------------------------------------------------------

export type CandidateRenderType =
  | 'molecule_2d'
  | 'crystal_3d'
  | 'polymer_chain'
  | 'data_table';

export interface CandidateArtifact {
  id: string;
  rank: number;
  score: number;
  renderType: CandidateRenderType;
  renderData: any;
  properties: PredictedProperty[];
  sourceReasoning: string;
  sourceDocumentIds: string[];
  status: 'pending' | 'approved' | 'rejected' | 'flagged';
  synthesisPlanRunId?: string;
}

export interface PredictedProperty {
  name: string;
  value: number | string | boolean;
  unit?: string;
  passesConstraint: boolean | null;
  model: string;
}

// ---------------------------------------------------------------------------
// Stage Artifacts (what the Stage renders)
// ---------------------------------------------------------------------------

export type StageArtifact =
  | { type: 'corpus_viewer'; documentId: string }
  | { type: 'hit_grid'; hits: CandidateArtifact[] }
  | { type: 'synthesis_plan_tree'; hitId: string; runId: string }
  | { type: 'spectroscopy_validation'; hitId: string; validationId: string }
  | { type: 'knowledge_graph'; projectId: string }
  | { type: 'capability_gap'; gap: CapabilityGap };

// ---------------------------------------------------------------------------
// Capability Gaps (Missing Tool Protocol)
// ---------------------------------------------------------------------------

export interface CapabilityGap {
  id: string;
  runId: string;
  stage: GoldenPathStage;
  requiredFunction: string;
  inputSchema: Record<string, string>;
  outputSchema: Record<string, string>;
  standardReference?: string;
  resolution: CapabilityGapResolution | null;
}

export interface CapabilityGapResolution {
  method: 'local_script' | 'api_endpoint' | 'plugin' | 'skip';
  config: Record<string, any>;
}

// ---------------------------------------------------------------------------
// Spectroscopy Validation (Stage 6)
// ---------------------------------------------------------------------------

export interface SpectroscopyPeak {
  position: number;
  intensity: number;
  assignment?: string;
}

export interface PeakMatch {
  predicted: SpectroscopyPeak;
  observed: SpectroscopyPeak | null;
  matched: boolean;
  deviation?: number;
}

export type SpectroscopyVerdict =
  | 'full_match'
  | 'partial_match'
  | 'no_match'
  | 'no_prediction_available';

export interface SpectroscopyValidation {
  id: string;
  hitId: string;
  runId: string;
  verdict: SpectroscopyVerdict;
  verdictText: string;
  observedPeaks: SpectroscopyPeak[];
  predictedPeaks: SpectroscopyPeak[];
  matches: PeakMatch[];
  missing: SpectroscopyPeak[];
}

// ---------------------------------------------------------------------------
// Bioassay Feedback (Stage 7)
// ---------------------------------------------------------------------------

export interface BioassayResult {
  id: string;
  hitId: string;
  epochId: string;
  resultName: string;
  resultValue: number;
  unit: string;
  passed: boolean;
  notes: string;
  submittedAt: number;
}

// ---------------------------------------------------------------------------
// Background Jobs
// ---------------------------------------------------------------------------

export type BackgroundJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface BackgroundJob {
  id: string;
  epochId: string;
  runId: string;
  label: string;
  status: BackgroundJobStatus;
  startedAt: number;
  completedAt: number | null;
  resultSummary: string | null;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Discovery Store
// ---------------------------------------------------------------------------

export interface DiscoveryStore {
  // Epoch tree
  epochs: Map<string, Epoch>;
  activeEpochId: string | null;
  rootEpochId: string | null;

  // Derived (computed via getters, not stored)
  activeEpoch: Epoch | null;
  activeStageArtifact: StageArtifact | null;

  // Jobs
  backgroundJobs: BackgroundJob[];

  // Epoch actions
  initializeSession: (params: ProjectTargetParams) => string;
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
}

// ---------------------------------------------------------------------------
// Cross-Store Bridge: Chat ↔ Discovery Context
// ---------------------------------------------------------------------------

export interface StageContextBundle {
  activeEpochId: string | null;
  activeStage: GoldenPathStage | null;
  targetParams: ProjectTargetParams | null;
  activeArtifact: StageArtifactSummary | null;
  focusedCandidateId: string | null;
  focusedCandidate: CandidateArtifact | null;
  recentToolInvocations: TruncatedToolInvocation[];
}

export interface StageArtifactSummary {
  type: StageArtifact['type'];
  label: string;
  candidateCount?: number;
  validationVerdict?: string;
}

export interface TruncatedToolInvocation {
  tool: string;
  inputPreview: string;
  outputPreview: string;
  status: 'completed' | 'failed';
}

// ---------------------------------------------------------------------------
// Follow-Up Taxonomy
// ---------------------------------------------------------------------------

export interface FollowUpSuggestion {
  label: string;
  query: string;
}

export interface FollowUpSuggestions {
  depth: FollowUpSuggestion;
  breadth: FollowUpSuggestion;
  opposition: FollowUpSuggestion;
}
