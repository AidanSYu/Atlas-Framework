/**
 * Unified SSE Stream Adapter
 *
 * Replaces four copy-pasted SSE parsing loops in api.ts with one
 * canonical implementation. All streaming endpoints are normalized
 * into a single NormalizedEvent discriminated union.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type FailureCategory =
  | 'connectivity'
  | 'timeout'
  | 'stream_parse'
  | 'backend_validation'
  | 'backend_runtime'
  | 'user_cancelled';

export type NormalizedEvent =
  | { type: 'routing'; mode: string; intent: string }
  | { type: 'progress'; node: string; message: string }
  | { type: 'thinking'; content: string }
  | { type: 'tool_call'; tool: string; input: Record<string, any> }
  | { type: 'tool_result'; tool: string; output: Record<string, any> }
  | { type: 'evidence'; count: number }
  | { type: 'grounding'; claim: string; status: string; confidence: number }
  | { type: 'hypotheses'; items: any[] }
  | { type: 'graph_analysis'; data: any }
  | { type: 'chunk'; content: string }
  | { type: 'complete'; result: any }
  | { type: 'error'; message: string; category: FailureCategory }
  | { type: 'cancelled' }
  // Phase 4: Coordinator HITL events
  | { type: 'coordinator_thinking'; content: string }
  | { type: 'coordinator_question'; question: string; options: string[]; context?: string; turn: number; goalsSoFar: string[] }
  | { type: 'coordinator_complete'; extractedGoals: string[]; summary: string; corpusEntities?: string[]; corpusSummary?: string }
  // Phase 5: Executor script sandbox events
  | { type: 'executor_script_generated'; filename: string; code: string; description: string; requiredPackages: string[] }
  | { type: 'executor_awaiting_approval'; filename: string; preview: string; description: string }
  | { type: 'executor_executing'; filename: string; iteration: number }
  | { type: 'executor_artifact'; filename: string; artifactType: string }
  | { type: 'executor_complete'; artifacts: string[]; summary: string }
  | { type: 'executor_candidates'; candidates: Array<{
      smiles: string;
      name: string;
      properties: Record<string, number>;
      toxicity: { clean: boolean; alert_count: number } | null;
    }> }
  // Pipeline execution events (Chemistry Engine Plan)
  | { type: 'pipeline_planning'; stages: string[]; moleculeCount: number }
  | { type: 'pipeline_stage_start'; stage: string; index: number }
  | { type: 'pipeline_stage_complete'; stage: string; index: number; resultCount: number }
  | { type: 'pipeline_complete'; summary: string; artifacts: string[] }
  // Unified Discovery Chat events
  | { type: 'session_update'; stage: string; sessionId: string }
  | { type: 'message'; content: string }
  | { type: 'plan_proposed'; planId: string; summary: string; reasoning: string;
      moleculeNotes: string; moleculeCount: number; iteration: number;
      estimatedTotalSeconds: number; warnings: string[]; isDemoData: boolean;
      stages: Array<{ stageId: number; plugin: string; description: string; estimatedSeconds: number }> }
  | { type: 'tool_start'; stageId: number; plugin: string; description: string;
      thinking?: string; totalStages: number }
  | { type: 'tool_complete'; stageId: number; plugin: string; summary: string;
      totalStages: number; candidatesSoFar?: number; error?: boolean;
      stats?: Record<string, any> }
  | { type: 'analysis'; keyFindings: string[];
      topCandidates: Array<{ smiles: string; reasoning: string; composite_score?: number }>;
      concerns: string[]; recommendations: Array<{ action: string; description: string; priority: string }>;
      missingCapabilities: string[] }
  | { type: 'recommendation'; recommendations?: Array<{ action: string; description: string; priority: string }>;
      missingCapabilities?: string[] };

export interface StreamSSEOptions {
  signal?: AbortSignal;
  timeout?: number;
}

// ---------------------------------------------------------------------------
// Core SSE parser
// ---------------------------------------------------------------------------

/**
 * Opens an SSE connection to `url`, POSTing `body` as JSON, and emits
 * raw (eventType, parsedData) pairs to `onRawEvent`.
 *
 * Handles: buffering across chunks, JSON parse failures, abort signals,
 * timeouts, and connection errors — all surfaced as NormalizedEvents via
 * `onEvent`.
 */
export async function streamSSE(
  url: string,
  body: Record<string, any>,
  onEvent: (event: NormalizedEvent) => void,
  options?: StreamSSEOptions,
): Promise<void> {
  const { signal, timeout = 300_000 } = options ?? {};

  const timeoutController = new AbortController();
  const timeoutId = setTimeout(() => timeoutController.abort(), timeout);

  const signals: AbortSignal[] = [timeoutController.signal];
  if (signal) signals.push(signal);
  const composedSignal = AbortSignal.any(signals);

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: composedSignal,
    });
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err?.name === 'AbortError') {
      if (signal?.aborted) {
        onEvent({ type: 'cancelled' });
      } else {
        onEvent({ type: 'error', message: 'The request took too long. The backend may be overloaded.', category: 'timeout' });
      }
      return;
    }
    onEvent({ type: 'error', message: 'Cannot reach the backend. Is the server running?', category: 'connectivity' });
    return;
  }

  if (!response.ok) {
    clearTimeout(timeoutId);
    let detail = response.statusText;
    try {
      const errBody = await response.json();
      detail = errBody.detail || detail;
    } catch { /* use statusText */ }

    const category: FailureCategory = response.status >= 400 && response.status < 500
      ? 'backend_validation'
      : 'backend_runtime';
    onEvent({ type: 'error', message: `The backend ${category === 'backend_validation' ? 'rejected the request' : 'encountered an error'}: ${detail}`, category });
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    clearTimeout(timeoutId);
    onEvent({ type: 'error', message: 'Received an unexpected response format.', category: 'stream_parse' });
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';

      for (const block of blocks) {
        if (!block.trim()) continue;

        let eventType = '';
        let data: any = null;

        for (const line of block.split('\n')) {
          if (line.startsWith('event: ')) {
            eventType = line.substring(7).trim();
          } else if (line.startsWith('data: ')) {
            try {
              data = JSON.parse(line.substring(6).trim());
            } catch {
              onEvent({ type: 'error', message: 'Received an unexpected response format.', category: 'stream_parse' });
            }
          }
        }

        if (eventType && data) {
          onEvent(mapRawEvent(eventType, data));
        }
      }
    }
  } catch (err: any) {
    if (err?.name === 'AbortError') {
      if (signal?.aborted) {
        onEvent({ type: 'cancelled' });
      } else {
        onEvent({ type: 'error', message: 'The request took too long. The backend may be overloaded.', category: 'timeout' });
      }
    } else {
      onEvent({ type: 'error', message: 'Cannot reach the backend. Is the server running?', category: 'connectivity' });
    }
  } finally {
    clearTimeout(timeoutId);
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// Raw SSE event -> NormalizedEvent mapping
// ---------------------------------------------------------------------------

/**
 * Maps a raw SSE (eventType, data) pair into our NormalizedEvent union.
 * Backend SSE event names are already close to our types; this function
 * handles the remaining normalization.
 */
function mapRawEvent(eventType: string, data: any): NormalizedEvent {
  switch (eventType) {
    case 'routing':
      return { type: 'routing', mode: data.brain || '', intent: data.intent || '' };

    case 'progress':
      return { type: 'progress', node: data.node || '', message: data.message || '' };

    case 'thinking':
      return { type: 'thinking', content: data.content || '' };

    case 'tool_call':
      return { type: 'tool_call', tool: data.tool || '', input: data.input || {} };

    case 'tool_result':
      return { type: 'tool_result', tool: data.tool || '', output: data.output || {} };

    case 'evidence':
      return { type: 'evidence', count: data.count || 1 };

    case 'grounding':
      return { type: 'grounding', claim: data.claim || '', status: data.status || '', confidence: data.confidence || 0 };

    case 'hypotheses':
      return { type: 'hypotheses', items: data.items || [] };

    case 'graph_analysis':
      return { type: 'graph_analysis', data };

    case 'chunk':
      return { type: 'chunk', content: data.content || '' };

    case 'complete':
      return { type: 'complete', result: data };

    case 'cancelled':
      return { type: 'cancelled' };

    case 'error':
      return { type: 'error', message: data.message || 'Unknown error', category: 'backend_runtime' };

    // Phase 4: Coordinator events
    case 'coordinator_thinking':
      return { type: 'coordinator_thinking', content: data.content || '' };

    case 'coordinator_question':
      return {
        type: 'coordinator_question',
        question: data.question || '',
        options: data.options || [],
        context: data.context,
        turn: data.turn || 0,
        goalsSoFar: data.goals_so_far || [],
      };

    case 'coordinator_complete':
      return {
        type: 'coordinator_complete',
        extractedGoals: data.extracted_goals || [],
        summary: data.summary || '',
        corpusEntities: data.corpus_entities,
        corpusSummary: data.corpus_summary,
      };

    // Phase 5: Executor events
    case 'executor_thinking':
      return { type: 'thinking', content: data.content || '' };

    case 'executor_script_generated':
      return {
        type: 'executor_script_generated',
        filename: data.filename || '',
        code: data.code || '',
        description: data.description || '',
        requiredPackages: data.required_packages || [],
      };

    case 'executor_awaiting_approval':
      return {
        type: 'executor_awaiting_approval',
        filename: data.filename || '',
        preview: data.preview || '',
        description: data.description || '',
      };

    case 'executor_executing':
      return {
        type: 'executor_executing',
        filename: data.filename || '',
        iteration: data.iteration || 0,
      };

    case 'executor_artifact':
      return {
        type: 'executor_artifact',
        filename: data.filename || '',
        artifactType: data.type || 'txt',
      };

    case 'executor_complete':
      return {
        type: 'executor_complete',
        artifacts: data.artifacts || [],
        summary: data.summary || '',
      };

    case 'executor_candidates':
      return {
        type: 'executor_candidates',
        candidates: data.candidates || [],
      };

    // Pipeline execution events (Chemistry Engine Plan)
    case 'pipeline_planning':
      return { type: 'pipeline_planning', stages: data.stages || [], moleculeCount: data.molecule_count || 0 };
    case 'pipeline_stage_start':
      return { type: 'pipeline_stage_start', stage: data.stage || '', index: data.index || 0 };
    case 'pipeline_stage_complete':
      return { type: 'pipeline_stage_complete', stage: data.stage || '', index: data.index || 0, resultCount: data.result_count || 0 };
    case 'pipeline_complete':
      return { type: 'pipeline_complete', summary: data.summary || '', artifacts: data.artifacts || [] };

    // Unified Discovery Chat events
    case 'session_update':
      return { type: 'session_update', stage: data.stage || '', sessionId: data.session_id || '' };

    case 'message':
      return { type: 'message', content: data.content || '' };

    case 'plan_proposed':
      // #region agent log
      fetch('http://127.0.0.1:7272/ingest/f7210b22-43a9-4e37-b68b-7a5242f3a0fe',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'93fd50'},body:JSON.stringify({sessionId:'93fd50',location:'stream-adapter.ts:337',message:'plan_proposed raw data.warnings',data:{rawWarnings:data.warnings,rawType:typeof data.warnings,isArray:Array.isArray(data.warnings)},timestamp:Date.now(),hypothesisId:'A'})}).catch(()=>{});
      // #endregion
      return {
        type: 'plan_proposed',
        planId: data.plan_id || '',
        summary: data.summary || '',
        reasoning: data.reasoning || '',
        moleculeNotes: data.molecule_notes || '',
        moleculeCount: data.molecule_count || 0,
        iteration: data.iteration || 1,
        estimatedTotalSeconds: data.estimated_total_seconds || 0,
        warnings: data.warnings || [],
        isDemoData: data.is_demo_data || false,
        stages: (data.stages || []).map((s: any) => ({
          stageId: s.stage_id || 0,
          plugin: s.plugin || '',
          description: s.description || '',
          estimatedSeconds: s.estimated_seconds || 0,
        })),
      };

    case 'tool_start':
      return {
        type: 'tool_start',
        stageId: data.stage_id || 0,
        plugin: data.plugin || '',
        description: data.description || '',
        totalStages: data.total_stages || 0,
      };

    case 'tool_complete':
      return {
        type: 'tool_complete',
        stageId: data.stage_id || 0,
        plugin: data.plugin || '',
        summary: data.summary || '',
        totalStages: data.total_stages || 0,
        candidatesSoFar: data.candidates_so_far,
        error: data.error,
      };

    case 'analysis':
      return {
        type: 'analysis',
        keyFindings: data.key_findings || [],
        topCandidates: (data.top_candidates || []).map((c: any) => ({
          smiles: c.smiles || '', reasoning: c.reasoning || '',
        })),
        concerns: data.concerns || [],
        recommendations: (data.recommendations || []).map((r: any) => ({
          action: r.action || '', description: r.description || '', priority: r.priority || 'medium',
        })),
        missingCapabilities: data.missing_capabilities || [],
      };

    case 'recommendation':
      return {
        type: 'recommendation',
        recommendations: data.recommendations?.map((r: any) => ({
          action: r.action || '', description: r.description || '', priority: r.priority || 'medium',
        })),
        missingCapabilities: data.missing_capabilities,
      };

    default:
      console.warn(`[stream-adapter] Unknown SSE event type: ${eventType}`);
      return { type: 'progress', node: eventType, message: JSON.stringify(data) };
  }
}
