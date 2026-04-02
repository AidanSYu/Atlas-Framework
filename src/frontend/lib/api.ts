import { streamSSE, NormalizedEvent, FailureCategory } from './stream-adapter';
export type { NormalizedEvent, FailureCategory };

// Detect if running in Tauri desktop app
declare global {
  interface Window {
    __TAURI__?: {
      tauri?: any;
      window?: any;
    };
  }
}

/**
 * Get the API base URL dynamically.
 */
function getApiBaseUrl(): string {
  if (typeof window !== 'undefined' && window.__TAURI__) {
    return 'http://127.0.0.1:8000';
  }
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
}

const API_BASE_URL = getApiBaseUrl();

export function getApiBase(): string {
  return API_BASE_URL;
}

const API_TIMEOUT = 30000;
const CHAT_TIMEOUT = 180000;
const SWARM_TIMEOUT = 300000; // 5 minutes for swarm (multi-step)

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    const errorMessage = errorData.detail || response.statusText;
    throw new Error(`API Error (${response.status}): ${errorMessage}`);
  }
  return response.json();
}

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeout: number = API_TIMEOUT
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  // Compose timeout signal with any externally-provided signal
  const signals: AbortSignal[] = [controller.signal];
  if (options.signal) signals.push(options.signal);
  const composedSignal = signals.length > 1 ? AbortSignal.any(signals) : controller.signal;
  try {
    return await fetch(url, { ...options, signal: composedSignal });
  } finally {
    clearTimeout(timeoutId);
  }
}

// ============================================================
// TYPES
// ============================================================

export interface ProjectInfo {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
}

export interface FileInfo {
  filename: string;
  doc_id: string;
  status: string;
  size_bytes?: number;
  uploaded_at?: string;
  processed_at?: string;
  total_chunks?: number;
  processed_chunks?: number;
  progress?: number;
  project_id?: string;
  page_count?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

export interface Citation {
  source: string;
  page: number;
  doc_id?: string;
  relevance?: number;
  text?: string;
}

export interface FollowUpSuggestion {
  label: string;
  query: string;
}

export interface FollowUpSuggestions {
  depth: FollowUpSuggestion;
  breadth: FollowUpSuggestion;
  opposition: FollowUpSuggestion;
}

export interface ChatResponse {
  answer: string;
  reasoning: string;
  citations: Citation[];
  relationships?: Array<{ source: string; type: string; target: string; context?: string }>;
  context_sources?: any;
  // Follow-up taxonomy (D4)
  follow_ups?: FollowUpSuggestions;
}

export interface EntityInfo {
  id: string;
  name: string;
  type: string;
  description?: string;
  document_id: string;
}

export interface RelationshipInfo {
  id: string;
  source_id: string;
  source_name: string;
  target_id: string;
  target_name: string;
  type: string;
  context?: string;
}

export interface ModelInfo {
  name: string;
  path: string;
}

export interface ModelsResponse {
  models_dir: string;
  llm: ModelInfo[];
  embeddings: ModelInfo[];
  ner: ModelInfo[];
  other: ModelInfo[];
  message?: string;
}

export interface ModelStatusResponse {
  active_model: string | null;
  model_type: string;
  device: string;
  gpu_layers: number;
  fallback: boolean;
  model_source: 'local' | 'api';          // Atlas 3.0
  api_models_available: boolean;           // Atlas 3.0
}

// Atlas 3.0: Model Registry types
export interface RegistryModel {
  name: string;
  source: 'local' | 'api';
  provider: string;
  has_key?: boolean;
}

export interface ModelRegistryResponse {
  local: RegistryModel[];
  api: RegistryModel[];
  active: ModelStatusResponse;
}

export interface GroundingEvent {
  claim: string;
  status: 'GROUNDED' | 'SUPPORTED' | 'UNVERIFIED' | 'INFERRED';
  confidence: number;
  source?: string;
  page?: number;
}

export interface SwarmEvidence {
  source: string;
  page: number;
  excerpt: string;
  relevance: number;
  grounding_status?: 'GROUNDED' | 'SUPPORTED' | 'UNVERIFIED' | 'INFERRED';
}

export interface SwarmResponse {
  brain_used: 'navigator' | 'cortex' | string;
  hypothesis: string;
  evidence: SwarmEvidence[];
  reasoning_trace: string[];
  status: string;
  confidence_score?: number;
  iterations?: number;
  final_answer?: string;
  contradictions?: Array<{
    claim_a: string;
    claim_b: string;
    severity: 'HIGH' | 'LOW';
    resolution?: string;
  }>;
}

// Discovery OS types
export interface DiscoveryCandidate {
  smiles: string;
  properties?: Record<string, any>;
  toxicity?: Record<string, any>;
}

export interface DiscoveryResponse extends SwarmResponse {
  candidates: DiscoveryCandidate[];
}

export interface ToolCallEvent {
  tool: string;
  input: Record<string, any>;
}

export interface ToolResultEvent {
  tool: string;
  output: Record<string, any>;
}

export interface SpectrumUploadResponse {
  file_id: string;
  filename: string;
  file_path: string;
}

// Phase 4: Workspace Drafts
export interface WorkspaceDraft {
  id: string;
  filename: string;
  size?: number;
  content?: Record<string, any>;
  updated_at: number;
}

// Phase 4: Context Engine types
export interface PaperStructure {
  title: string;
  authors: string[];
  year: number | null;
  abstract: string;
  methodology: string;
  key_findings: string[];
  limitations: string[];
  paper_type: string;
  page_count: number;
  total_chars: number;
}

export interface DocumentStructureResponse {
  doc_id: string;
  filename: string;
  status: string;
  uploaded_at: string | null;
  structure: PaperStructure;
}

export interface RelatedPassage {
  text: string;
  source: string;
  page: number;
  doc_id: string;
  score: number;
  chunk_id: string;
}

export interface DocumentChunk {
  chunk_id: string;
  text: string;
  chunk_index: number;
  page_number: number | null;
  start_char: number | null;
  end_char: number | null;
  metadata: Record<string, any>;
}

export interface ContextSuggestions {
  related_passages: RelatedPassage[];
  connected_concepts: Array<{
    id: string;
    name: string;
    type: string;
    document_id: string;
    confidence: number;
  }>;
  suggestions: string[];
}

// Phase 5: Import/Export types
export interface ImportResult {
  status: string;
  imported: Array<{
    doc_id: string;
    bibtex_key: string;
    title: string;
    authors: string[];
    year: number | null;
  }>;
  skipped: Array<{
    bibtex_key: string;
    reason: string;
  }>;
  total_entries: number;
  total_imported: number;
  error?: string;
}

export interface MarkdownExportResult {
  markdown: string;
  bibtex: string;
  filename: string;
}

export interface FormattedCitation {
  doc_id: string;
  filename: string;
  citation: string;
  bibtex_key: string;
}

export interface CitationFormatResult {
  citations: FormattedCitation[];
  style: string;
}

// Phase 6: API Key Configuration
export interface ConfigKeys {
  has_openai: boolean;
  has_anthropic: boolean;
  has_deepseek: boolean;
  has_minimax: boolean;
}

export interface ConfigKeysUpdate {
  OPENAI_API_KEY?: string;
  ANTHROPIC_API_KEY?: string;
  DEEPSEEK_API_KEY?: string;
  MINIMAX_API_KEY?: string;
}

export interface ConfigKeysVerifyResponse {
  openai: boolean;
  anthropic: boolean;
  deepseek: boolean;
  minimax: boolean;
}

// ============================================================
// API CLIENT
// ============================================================

export const api = {
  // ---- Config / API Keys ----

  async getApiKeysStatus(): Promise<ConfigKeys> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/config/keys`);
    return handleResponse(response);
  },

  async updateApiKeys(keys: ConfigKeysUpdate): Promise<{ status: string }> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/config/keys`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(keys),
    });
    return handleResponse(response);
  },

  async verifyApiKeys(): Promise<ConfigKeysVerifyResponse> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/config/keys/verify`, {
      method: 'POST',
    });
    return handleResponse(response);
  },

  // ---- Projects ----

  async createProject(name: string, description?: string): Promise<ProjectInfo> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description }),
    });
    return handleResponse(response);
  },

  async listProjects(): Promise<ProjectInfo[]> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/projects`);
    return handleResponse(response);
  },

  async deleteProject(projectId: string): Promise<any> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/projects/${encodeURIComponent(projectId)}`, {
      method: 'DELETE',
    });
    return handleResponse(response);
  },

  // ---- File Management ----

  async uploadFile(file: File, projectId?: string): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const url = projectId
      ? `${API_BASE_URL}/ingest?project_id=${encodeURIComponent(projectId)}`
      : `${API_BASE_URL}/ingest`;
    const response = await fetchWithTimeout(url, { method: 'POST', body: formData });
    return handleResponse(response);
  },

  async listFiles(projectId?: string): Promise<FileInfo[]> {
    const params = new URLSearchParams();
    if (projectId) params.append('project_id', projectId);
    const url = params.toString()
      ? `${API_BASE_URL}/files?${params.toString()}`
      : `${API_BASE_URL}/files`;
    const response = await fetchWithTimeout(url);
    return handleResponse(response);
  },

  async deleteFile(docId: string): Promise<any> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/files/${encodeURIComponent(docId)}`,
      { method: 'DELETE' }
    );
    return handleResponse(response);
  },

  getFileUrl(docId: string): string {
    return `${API_BASE_URL}/files/${encodeURIComponent(docId)}`;
  },

  // ---- Chat (Librarian RAG) ----

  async chat(query: string, projectId?: string, signal?: AbortSignal, stageContext?: Record<string, any> | null): Promise<ChatResponse> {
    const payload: Record<string, any> = { query, project_id: projectId };
    if (stageContext) payload.stage_context = stageContext;
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/chat`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal,
      },
      CHAT_TIMEOUT
    );
    return handleResponse(response);
  },

  // ---- Swarm (Two-Brain Agentic RAG) / MoE (Atlas 3.0) ----

  async runSwarm(query: string, projectId: string): Promise<SwarmResponse> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/swarm/run`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, project_id: projectId }),
      },
      SWARM_TIMEOUT
    );
    return handleResponse(response);
  },

  // ---- Atlas 3.0: MoE (Mixture of Experts) ----

  async runMoE(query: string, projectId: string): Promise<SwarmResponse> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/moe/run`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, project_id: projectId }),
      },
      SWARM_TIMEOUT
    );
    return handleResponse(response);
  },

  async streamMoE(
    query: string,
    projectId: string,
    onEvent: (type: string, data: any) => void,
    sessionId?: string,
    signal?: AbortSignal
  ): Promise<void> {
    await streamSSE(
      `${API_BASE_URL}/api/moe/stream`,
      { query, project_id: projectId, session_id: sessionId },
      (event) => {
        if (event.type === 'error') {
          onEvent('error', { message: event.message });
        } else if (event.type === 'cancelled') {
          onEvent('cancelled', {});
        } else if (event.type === 'complete') {
          onEvent('complete', event.result);
        } else if (event.type === 'routing') {
          onEvent('routing', { brain: event.mode, intent: event.intent });
        } else if (event.type === 'evidence') {
          onEvent('evidence', { count: event.count });
        } else if (event.type === 'grounding') {
          onEvent('grounding', { claim: event.claim, status: event.status, confidence: event.confidence });
        } else if (event.type === 'chunk') {
          onEvent('chunk', { content: event.content });
        } else {
          onEvent(event.type, event);
        }
      },
      { signal, timeout: SWARM_TIMEOUT },
    );
  },

  async streamMoEHypotheses(
    query: string,
    projectId: string,
    onEvent: (type: string, data: any) => void,
    sessionId?: string,
    signal?: AbortSignal
  ): Promise<void> {
    await streamSSE(
      `${API_BASE_URL}/api/moe/hypotheses`,
      { query, project_id: projectId, session_id: sessionId },
      (event) => {
        if (event.type === 'error') {
          onEvent('error', { message: event.message });
        } else if (event.type === 'cancelled') {
          onEvent('cancelled', {});
        } else if (event.type === 'hypotheses') {
          onEvent('hypotheses', { items: event.items });
        } else if (event.type === 'routing') {
          onEvent('routing', { brain: event.mode, intent: event.intent });
        } else {
          onEvent(event.type, event);
        }
      },
      { signal, timeout: SWARM_TIMEOUT },
    );
  },

  // ---- Entities & Graph ----

  async listEntities(params?: {
    entity_type?: string;
    document_id?: string;
    project_id?: string;
    limit?: number;
  }): Promise<EntityInfo[]> {
    const q = new URLSearchParams();
    if (params?.entity_type) q.append('entity_type', params.entity_type);
    if (params?.document_id) q.append('document_id', params.document_id);
    if (params?.project_id) q.append('project_id', params.project_id);
    q.append('limit', String(params?.limit ?? 50));
    const response = await fetchWithTimeout(`${API_BASE_URL}/entities?${q.toString()}`);
    return handleResponse(response);
  },

  async getEntityRelationships(
    entityId: string,
    direction: 'outgoing' | 'incoming' | 'both' = 'both'
  ): Promise<RelationshipInfo[]> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/entities/${encodeURIComponent(entityId)}/relationships?direction=${direction}`
    );
    return handleResponse(response);
  },

  async getEntityTypes(projectId?: string): Promise<Array<{ type: string; count: number }>> {
    const params = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';
    const response = await fetchWithTimeout(`${API_BASE_URL}/graph/types${params}`);
    const data = await handleResponse<{ entity_types: Array<{ type: string; count: number }> }>(response);
    return data.entity_types;
  },

  async getFullGraph(
    documentId?: string,
    projectId?: string
  ): Promise<{ nodes: EntityInfo[]; edges: RelationshipInfo[] }> {
    const params = new URLSearchParams();
    if (documentId) params.append('document_id', documentId);
    if (projectId) params.append('project_id', projectId);
    const url = params.toString()
      ? `${API_BASE_URL}/graph/full?${params.toString()}`
      : `${API_BASE_URL}/graph/full`;
    const response = await fetchWithTimeout(url);
    return handleResponse(response);
  },

  async listModels(): Promise<ModelsResponse> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/models`);
    return handleResponse(response);
  },

  async getModelStatus(): Promise<ModelStatusResponse> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/models/status`);
    return handleResponse(response);
  },

  async loadModel(modelName: string): Promise<ModelStatusResponse> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/models/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_name: modelName }),
    });
    return handleResponse(response);
  },

  // Atlas 3.0: Model Registry (local + cloud API models)
  async getModelRegistry(): Promise<ModelRegistryResponse> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/models/registry`);
    return handleResponse(response);
  },

  // ---- Workspace (Phase 4) ----

  async listWorkspaceDrafts(projectId: string): Promise<WorkspaceDraft[]> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/api/workspace/${encodeURIComponent(projectId)}/drafts`);
    return handleResponse(response);
  },

  async getWorkspaceDraft(projectId: string, draftId: string): Promise<WorkspaceDraft> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/api/workspace/${encodeURIComponent(projectId)}/drafts/${encodeURIComponent(draftId)}`);
    return handleResponse(response);
  },

  async saveWorkspaceDraft(projectId: string, draftId: string, content: Record<string, any>): Promise<any> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/workspace/${encodeURIComponent(projectId)}/drafts/${encodeURIComponent(draftId)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      }
    );
    return handleResponse(response);
  },

  async deleteWorkspaceDraft(projectId: string, draftId: string): Promise<any> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/workspace/${encodeURIComponent(projectId)}/drafts/${encodeURIComponent(draftId)}`,
      {
        method: 'DELETE',
      }
    );
    return handleResponse(response);
  },

  // ---- Context Engine (Phase 4) ----

  async getDocumentStructure(docId: string): Promise<DocumentStructureResponse> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/files/${encodeURIComponent(docId)}/structure`
    );
    return handleResponse(response);
  },

  async getRelatedPassages(
    docId: string,
    text: string,
    projectId?: string,
    limit: number = 8
  ): Promise<RelatedPassage[]> {
    const params = new URLSearchParams({ text, limit: String(limit) });
    if (projectId) params.append('project_id', projectId);
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/files/${encodeURIComponent(docId)}/related?${params.toString()}`
    );
    return handleResponse(response);
  },

  async getDocumentChunks(
    docId: string,
    page?: number,
    limit: number = 50
  ): Promise<DocumentChunk[]> {
    const params = new URLSearchParams({ limit: String(limit) });
    if (page !== undefined) params.append('page', String(page));
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/files/${encodeURIComponent(docId)}/chunks?${params.toString()}`
    );
    return handleResponse(response);
  },

  async getContextSuggestions(
    projectId: string,
    selectedText?: string,
    currentDocId?: string,
    currentPage?: number
  ): Promise<ContextSuggestions> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/context`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: projectId,
          selected_text: selectedText,
          current_doc_id: currentDocId,
          current_page: currentPage,
        }),
      }
    );
    return handleResponse(response);
  },

  // ---- Import / Export (Phase 5) ----

  async importBibtex(file: File, projectId: string): Promise<ImportResult> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/import/bibtex?project_id=${encodeURIComponent(projectId)}`,
      { method: 'POST', body: formData }
    );
    return handleResponse(response);
  },

  async exportBibtexProject(projectId: string): Promise<Blob> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/export/bibtex/${encodeURIComponent(projectId)}`
    );
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Export failed: ${err.detail || response.statusText}`);
    }
    return response.blob();
  },

  async exportBibtexSelection(docIds: string[]): Promise<Blob> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/export/bibtex`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_ids: docIds, style: 'apa' }),
      }
    );
    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(`Export failed: ${err.detail || response.statusText}`);
    }
    return response.blob();
  },

  async exportMarkdown(params: {
    content: string;
    citations: Array<{ source: string; page: number; doc_id?: string }>;
    projectId: string;
    title?: string;
    author?: string;
    style?: string;
  }): Promise<MarkdownExportResult> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/export/markdown`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: params.content,
          citations: params.citations,
          project_id: params.projectId,
          title: params.title || 'Research Synthesis',
          author: params.author || '',
          style: params.style || 'apa',
        }),
      }
    );
    return handleResponse(response);
  },

  async exportChatHistory(
    messages: Array<{ role: string; content: string; citations?: any[] }>,
    projectName?: string
  ): Promise<{ markdown: string }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/export/chat`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages,
          project_name: projectName || 'Atlas Research',
        }),
      }
    );
    return handleResponse(response);
  },

  async formatCitations(
    docIds: string[],
    style: string = 'apa'
  ): Promise<CitationFormatResult> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/export/citations/format`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_ids: docIds, style }),
      }
    );
    return handleResponse(response);
  },

  // ---- Swarm (Two-Brain Agentic RAG) ----

  async streamSwarm(
    query: string,
    projectId: string,
    onEvent: (type: string, data: any) => void,
    sessionId?: string,
    signal?: AbortSignal
  ): Promise<void> {
    await streamSSE(
      `${API_BASE_URL}/api/swarm/stream`,
      { query, project_id: projectId, session_id: sessionId },
      (event) => {
        if (event.type === 'error') {
          onEvent('error', { message: event.message });
        } else if (event.type === 'cancelled') {
          onEvent('cancelled', {});
        } else if (event.type === 'complete') {
          onEvent('complete', event.result);
        } else if (event.type === 'routing') {
          onEvent('routing', { brain: event.mode, intent: event.intent });
        } else if (event.type === 'graph_analysis') {
          onEvent('graph_analysis', event.data);
        } else if (event.type === 'evidence') {
          onEvent('evidence', { count: event.count });
        } else if (event.type === 'grounding') {
          onEvent('grounding', { claim: event.claim, status: event.status, confidence: event.confidence });
        } else if (event.type === 'chunk') {
          onEvent('chunk', { content: event.content });
        } else {
          onEvent(event.type, event);
        }
      },
      { signal, timeout: SWARM_TIMEOUT },
    );
  },

  // ============================================================
  // DISCOVERY OS (Phase 1: Deterministic Tool-Calling)
  // ============================================================

  async uploadSpectrum(file: File, projectId: string): Promise<SpectrumUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const url = `${API_BASE_URL}/api/discovery/upload-spectrum?project_id=${encodeURIComponent(projectId)}`;
    const response = await fetchWithTimeout(url, { method: 'POST', body: formData });
    return handleResponse(response);
  },

  async runDiscovery(query: string, projectId: string, spectrumFilePath?: string): Promise<DiscoveryResponse> {
    const body: Record<string, any> = { query, project_id: projectId };
    if (spectrumFilePath) body.spectrum_file_path = spectrumFilePath;
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/discovery/run`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      },
      SWARM_TIMEOUT
    );
    return handleResponse(response);
  },

  async streamDiscovery(
    query: string,
    projectId: string,
    onEvent: (type: string, data: any) => void,
    sessionId?: string,
    signal?: AbortSignal,
    spectrumFilePath?: string
  ): Promise<void> {
    const body: Record<string, any> = {
      query,
      project_id: projectId,
      session_id: sessionId,
    };
    if (spectrumFilePath) body.spectrum_file_path = spectrumFilePath;

    await streamSSE(
      `${API_BASE_URL}/api/discovery/stream`,
      body,
      (event) => {
        if (event.type === 'error') {
          onEvent('error', { message: event.message });
        } else if (event.type === 'cancelled') {
          onEvent('cancelled', {});
        } else if (event.type === 'complete') {
          onEvent('complete', event.result);
        } else if (event.type === 'routing') {
          onEvent('routing', { brain: event.mode, intent: event.intent });
        } else if (event.type === 'tool_call') {
          onEvent('tool_call', { tool: event.tool, input: event.input });
        } else if (event.type === 'tool_result') {
          onEvent('tool_result', { tool: event.tool, output: event.output });
        } else if (event.type === 'evidence') {
          onEvent('evidence', { count: event.count });
        } else if (event.type === 'chunk') {
          onEvent('chunk', { content: event.content });
        } else {
          onEvent(event.type, event);
        }
      },
      { signal, timeout: SWARM_TIMEOUT },
    );
  },

  // ---- Intent routing (lightweight, no execution) ----

  async routeIntent(query: string, projectId: string): Promise<{ intent: string }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/route`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, project_id: projectId }),
      },
      API_TIMEOUT,
    );
    return handleResponse(response);
  },

  async parseBrainstorm(text: string, domain: string): Promise<{
    objective: string;
    propertyConstraints: any[];
    domainSpecificConstraints: Record<string, string>;
  }> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/api/discovery/parse-brainstorm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, domain }),
    });
    return handleResponse(response);
  },

  // ---- Discovery Sessions (Golden Path) ----

  async listDiscoverySessions(projectId?: string): Promise<Array<{
    session_id: string;
    session_name: string;
    created_at: string | null;
    status: string;
    folder_exists: boolean;
  }>> {
    const url = projectId
      ? `${API_BASE_URL}/api/discovery/sessions?project_id=${encodeURIComponent(projectId)}`
      : `${API_BASE_URL}/api/discovery/sessions`;
    const response = await fetchWithTimeout(url);
    return handleResponse(response);
  },

  async getSessionFiles(sessionId: string): Promise<Array<{
    filename: string;
    path: string;
    size_bytes: number;
  }>> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/discovery/${encodeURIComponent(sessionId)}/files`
    );
    return handleResponse(response);
  },

  async readSessionFile(sessionId: string, filePath: string): Promise<{
    filename: string;
    path: string;
    content: string;
    size_bytes: number;
  }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/discovery/${encodeURIComponent(sessionId)}/files/${filePath}`
    );
    return handleResponse(response);
  },

  async getSessionMemory(sessionId: string): Promise<{
    session_id: string;
    initialized_at: string;
    domain: string | null;
    corpus_context: {
      entities: string[];
      document_ids: string[];
      summary: string;
    } | null;
    research_goals: string[];
    constraints: Record<string, any>;
    agents_completed: string[];
    current_stage: string;
    metadata: Record<string, any>;
  }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/discovery/${encodeURIComponent(sessionId)}/memory`
    );
    return handleResponse(response);
  },

  // ---- Executor Agent (Phase 5) ----

  async startExecutor(sessionId: string, autoApprove: boolean = false): Promise<{ status: string }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/discovery/${encodeURIComponent(sessionId)}/executor/start`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auto_approve: autoApprove }),
      }
    );
    return handleResponse(response);
  },

  async approveScript(
    sessionId: string,
    decision: 'approve' | 'reject' | 'edit',
    editedCode?: string
  ): Promise<{ status: string; decision: string }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/discovery/${encodeURIComponent(sessionId)}/executor/approve`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          decision,
          edited_code: editedCode || null,
        }),
      }
    );
    return handleResponse(response);
  },

  // ---- Plugin Visibility (Phase 5 - Part 2) ----

  async getPlugins(): Promise<{
    plugins: Array<{
      name: string;
      description: string;
      loaded: boolean;
      available: boolean;
      unavailable_reason?: string | null;
      type: 'deterministic' | 'semantic';
      input_schema: any;
      output_schema: any;
    }>;
    orchestrator_provider: string;
    orchestrator_model: string;
    tool_provider: string;
    tool_model: string;
  }> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/api/discovery/plugins`);
    return handleResponse(response);
  },

  async unloadPlugin(pluginName: string): Promise<{ status: string; plugin: string }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/discovery/plugins/${encodeURIComponent(pluginName)}/unload`,
      { method: 'POST' }
    );
    return handleResponse(response);
  },

  async getDiscoverySchema(): Promise<{ domain: string; target_schema: string[] }> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/api/discovery/schema`);
    return handleResponse(response);
  },

  async initializeDiscoverySession(params: {
    domain: string;
    objective: string;
    propertyConstraints: any[];
    domainSpecificConstraints: Record<string, any>;
    corpusDocumentIds: string[];
  }): Promise<{ session_id: string; session_name: string; epoch_id: string; folder_path: string; status: string }> {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/discovery/initialize`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      },
    );
    return handleResponse(response);
  },

  streamDiscoveryChat(
    sessionId: string,
    message: string | null,
    projectId: string,
    onEvent: (event: NormalizedEvent) => void,
    options?: {
      action?: string;
      smilesList?: string[];
      signal?: AbortSignal;
      timeout?: number;
    },
  ): Promise<void> {
    const body: Record<string, any> = {
      message,
      project_id: projectId,
    };
    if (options?.action) body.action = options.action;
    if (options?.smilesList) body.smiles_list = options.smilesList;

    return streamSSE(
      `${API_BASE_URL}/api/discovery/${sessionId}/chat`,
      body,
      onEvent,
      { signal: options?.signal, timeout: options?.timeout ?? 600_000 },
    );
  },
};

