'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  FileText,
  Network,
  Lightbulb,
  BookOpen,
  Brain,
  Sparkles,
  TrendingUp,
  Loader2,
  MessageSquare,
  ExternalLink,
  Tag,
} from 'lucide-react';
import { useChatStore, ChatMessage } from '@/stores/chatStore';
import { useGraphStore } from '@/stores/graphStore';
import CitationCard from './generative/CitationCard';
import { api, RelatedPassage, PaperStructure, DocumentStructureResponse } from '@/lib/api';

interface ContextEngineProps {
  projectId: string;
  selectedDocId: string | null;
  selectedFilename: string | null;
  onCitationClick: (filename: string, page: number, docId?: string) => void;
  suggestions?: any;
  isProcessing?: boolean;
}

export default function ContextEngine({
  projectId,
  selectedDocId,
  selectedFilename,
  onCitationClick,
  suggestions,
  isProcessing,
}: ContextEngineProps) {
  const threads = useChatStore((s) => s.threads);
  const activeThreadId = useChatStore((s) => s.activeThreadId);
  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId) ?? null,
    [activeThreadId, threads]
  );
  const librarianMessages = activeThread?.messages ?? [];
  const cortexMessages = activeThread?.messages ?? [];
  const nodes = useGraphStore((s) => s.nodes);
  const links = useGraphStore((s) => s.links);
  const focusedNodeId = useGraphStore((s) => s.focusedNodeId);

  // Phase 4: Proactive context state
  const [docStructure, setDocStructure] = useState<DocumentStructureResponse | null>(null);
  const [structureLoading, setStructureLoading] = useState(false);
  const lastFetchedDocId = useRef<string | null>(null);

  const relatedPassages = suggestions?.related_passages || [];
  const connectedConcepts = suggestions?.connected_concepts || [];
  const contextLoading = isProcessing || false;

  // Fetch document structure when a new document is selected
  useEffect(() => {
    if (!selectedDocId || selectedDocId === lastFetchedDocId.current) return;
    lastFetchedDocId.current = selectedDocId;

    let cancelled = false;
    setStructureLoading(true);
    setDocStructure(null);

    api.getDocumentStructure(selectedDocId, projectId)
      .then((data) => {
        if (!cancelled) setDocStructure(data);
      })
      .catch((err) => {
        console.error('Failed to load document structure:', err);
      })
      .finally(() => {
        if (!cancelled) setStructureLoading(false);
      });

    return () => { cancelled = true; };
  }, [selectedDocId, projectId]);

  const allCitations = useMemo(() => {
    const citations: Array<{
      source: string;
      page: number;
      excerpt?: string;
      relevance?: number;
      docId?: string;
    }> = [];
    const seen = new Set<string>();

    [...librarianMessages, ...cortexMessages].forEach((msg) => {
      if (msg.citations) {
        msg.citations.forEach((c) => {
          const key = `${c.source}:${c.page}`;
          if (!seen.has(key)) {
            seen.add(key);
            citations.push({
              source: c.source,
              page: c.page,
              excerpt: c.text,
              relevance: c.relevance,
              docId: c.doc_id,
            });
          }
        });
      }
    });

    return citations.reverse().slice(0, 20);
  }, [librarianMessages, cortexMessages]);

  const focusedNode = useMemo(() => {
    if (!focusedNodeId) return null;
    return nodes.find((n) => n.id === focusedNodeId) || null;
  }, [focusedNodeId, nodes]);

  const focusedNeighbors = useMemo(() => {
    if (!focusedNodeId) return [];
    const neighborIds = new Set<string>();

    links.forEach((link) => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      if (sourceId === focusedNodeId) neighborIds.add(targetId);
      if (targetId === focusedNodeId) neighborIds.add(sourceId);
    });

    return nodes.filter((n) => neighborIds.has(n.id));
  }, [focusedNodeId, nodes, links]);

  const focusedEdges = useMemo(() => {
    if (!focusedNodeId) return [];
    return links.filter((link) => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      return sourceId === focusedNodeId || targetId === focusedNodeId;
    });
  }, [focusedNodeId, links]);

  const graphStats = useMemo(() => {
    if (nodes.length === 0) return null;
    const typeMap: Record<string, number> = {};
    nodes.forEach((n) => {
      typeMap[n.type] = (typeMap[n.type] || 0) + 1;
    });
    const topTypes = Object.entries(typeMap)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
    return { totalNodes: nodes.length, totalEdges: links.length, topTypes };
  }, [nodes, links]);

  const lastBrainActivity = useMemo(() => {
    const allMsgs = [...cortexMessages].reverse();
    return allMsgs.find((m: ChatMessage) => m.brainActivity) || null;
  }, [cortexMessages]);

  const s = docStructure?.structure;
  const hasAnyContent = selectedFilename || focusedNode || allCitations.length > 0 || graphStats || lastBrainActivity;

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto custom-scrollbar">
        {/* Active Document + Structure (Phase 4) */}
        {selectedFilename && (
          <div className="border-b border-border p-3">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              <BookOpen className="h-3 w-3" />
              Active Document
            </div>
            <div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/15">
                  <FileText className="h-4 w-4 text-primary" />
                </div>
                <div className="min-w-0">
                  <p className="truncate text-xs font-medium text-foreground">
                    {s?.title || selectedFilename}
                  </p>
                  {s?.authors && s.authors.length > 0 ? (
                    <p className="truncate text-[10px] text-muted-foreground mt-0.5">
                      {s.authors.slice(0, 3).join(', ')}
                      {s.authors.length > 3 ? ' et al.' : ''}
                    </p>
                  ) : (
                    <p className="text-[10px] text-muted-foreground mt-0.5">Currently viewing</p>
                  )}
                </div>
              </div>
              {/* Paper metadata badges */}
              {s && (s.year || s.paper_type) && (
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  {s.year && (
                    <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                      {s.year}
                    </span>
                  )}
                  {s.paper_type && s.paper_type !== 'other' && (
                    <span className="rounded-full bg-accent/10 px-2 py-0.5 text-[10px] font-medium text-accent capitalize">
                      {s.paper_type}
                    </span>
                  )}
                  {s.page_count > 0 && (
                    <span className="text-[10px] text-muted-foreground">
                      {s.page_count} pages
                    </span>
                  )}
                </div>
              )}
              {/* Key findings preview */}
              {s?.key_findings && s.key_findings.length > 0 && (
                <div className="mt-2.5 space-y-1">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Key Findings
                  </p>
                  {s.key_findings.slice(0, 2).map((finding, i) => (
                    <p key={i} className="flex items-start gap-1.5 text-[10px] text-foreground/70 leading-relaxed">
                      <Lightbulb className="mt-0.5 h-2.5 w-2.5 shrink-0 text-primary/50" />
                      <span className="line-clamp-2">{finding}</span>
                    </p>
                  ))}
                  {s.key_findings.length > 2 && (
                    <p className="text-[10px] text-muted-foreground pl-4">
                      +{s.key_findings.length - 2} more
                    </p>
                  )}
                </div>
              )}
              {structureLoading && (
                <div className="mt-2 flex items-center gap-1.5">
                  <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                  <span className="text-[10px] text-muted-foreground">Analyzing structure...</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Connected Concepts from Context Engine (Phase 4) */}
        {connectedConcepts.length > 0 && (
          <div className="border-b border-border p-3">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              <Tag className="h-3 w-3" />
              Connected Concepts
            </div>
            <div className="flex flex-wrap gap-1.5">
              {connectedConcepts.slice(0, 12).map((concept: any) => (
                <span
                  key={concept.id}
                  className="inline-flex items-center gap-1 rounded-full border border-border bg-surface px-2 py-0.5 text-[10px] text-foreground/70"
                  title={`${concept.type} (${Math.round(concept.confidence * 100)}% confidence)`}
                >
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{
                      backgroundColor:
                        concept.type === 'concept' ? 'var(--primary)' :
                          concept.type === 'person' ? 'var(--accent)' :
                            concept.type === 'method' ? '#10b981' :
                              'var(--muted-foreground)',
                    }}
                  />
                  {concept.name}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Related Passages from Other Documents (Phase 4) - always show when a doc is selected */}
        {selectedDocId && (
          <div className="border-b border-border p-3">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              <ExternalLink className="h-3 w-3" />
              Related in Other Docs
            </div>
            {contextLoading ? (
              <div className="flex items-center gap-1.5 py-2">
                <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                <span className="text-[10px] text-muted-foreground">Finding connections...</span>
              </div>
            ) : relatedPassages.length > 0 ? (
              <div className="space-y-1.5">
                {relatedPassages.slice(0, 4).map((passage: any, i: number) => (
                  <button
                    key={`${passage.chunk_id}-${i}`}
                    onClick={() => onCitationClick(passage.source, passage.page, passage.doc_id)}
                    className="w-full rounded-lg border border-border bg-background p-2 text-left transition-all hover:border-primary/30 hover:bg-primary/5"
                  >
                    <p className="text-[10px] text-foreground/70 leading-relaxed line-clamp-2">
                      {passage.text}
                    </p>
                    <div className="mt-1 flex items-center gap-2">
                      <span className="truncate text-[10px] font-medium text-primary">
                        {passage.source}
                      </span>
                      <span className="text-[9px] text-muted-foreground">p.{passage.page}</span>
                      <span className="ml-auto text-[9px] text-muted-foreground">
                        {Math.round(passage.score * 100)}%
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-[10px] text-muted-foreground py-2">
                No related passages found in other documents.
              </p>
            )}
          </div>
        )}

        {/* Graph Stats */}
        {graphStats && !focusedNode && (
          <div className="border-b border-border p-3">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              <Network className="h-3 w-3" />
              Knowledge Graph
            </div>
            <div className="rounded-lg border border-border bg-card/50 p-3">
              <div className="flex items-center gap-4 mb-3">
                <div className="text-center">
                  <p className="text-lg font-bold text-primary">{graphStats.totalNodes}</p>
                  <p className="text-[10px] text-muted-foreground">Entities</p>
                </div>
                <div className="h-8 w-px bg-border" />
                <div className="text-center">
                  <p className="text-lg font-bold text-accent">{graphStats.totalEdges}</p>
                  <p className="text-[10px] text-muted-foreground">Relations</p>
                </div>
              </div>
              <div className="space-y-1">
                {graphStats.topTypes.map(([type, count]) => (
                  <div key={type} className="flex items-center justify-between text-[11px]">
                    <span className="text-muted-foreground">{type}</span>
                    <span className="rounded bg-surface px-1.5 py-0.5 text-[10px] font-medium text-foreground/70">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Focused Graph Node */}
        {focusedNode && (
          <div className="border-b border-border p-3">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              <Network className="h-3 w-3" />
              Graph Focus
            </div>
            <div className="rounded-lg border border-border bg-card p-3">
              <div className="flex items-center gap-2.5 mb-2">
                <div
                  className="h-3.5 w-3.5 rounded-full ring-2 ring-offset-1 ring-offset-card"
                  style={{ backgroundColor: focusedNode.color, ['--tw-ring-color' as string]: focusedNode.color }}
                />
                <p className="text-sm font-medium text-foreground">{focusedNode.name}</p>
                <span className="ml-auto rounded-full bg-surface px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                  {focusedNode.type}
                </span>
              </div>

              {focusedNode.properties?.description && (
                <p className="text-[11px] text-muted-foreground leading-relaxed mb-3 pl-6">
                  {focusedNode.properties.description}
                </p>
              )}

              {focusedNeighbors.length > 0 && (
                <div className="mt-2 pl-6">
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
                    Connections ({focusedNeighbors.length})
                  </p>
                  <div className="space-y-1">
                    {focusedNeighbors.slice(0, 8).map((neighbor) => {
                      const edge = focusedEdges.find((e) => {
                        const sId = typeof e.source === 'string' ? e.source : e.source.id;
                        const tId = typeof e.target === 'string' ? e.target : e.target.id;
                        return (sId === neighbor.id || tId === neighbor.id);
                      });

                      return (
                        <div
                          key={neighbor.id}
                          className="flex items-center gap-2 rounded-md bg-surface/50 px-2 py-1.5 text-[11px]"
                        >
                          <div
                            className="h-2 w-2 rounded-full shrink-0"
                            style={{ backgroundColor: neighbor.color }}
                          />
                          <span className="truncate text-foreground/80">{neighbor.name}</span>
                          {edge && (
                            <span className="ml-auto shrink-0 text-[10px] text-accent italic">
                              {edge.label}
                            </span>
                          )}
                        </div>
                      );
                    })}
                    {focusedNeighbors.length > 8 && (
                      <p className="text-[10px] text-muted-foreground px-2">
                        +{focusedNeighbors.length - 8} more
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Last Brain Activity */}
        {lastBrainActivity?.brainActivity && (
          <div className="border-b border-border p-3">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              <Brain className="h-3 w-3" />
              Last Analysis
            </div>
            <div className="rounded-lg border border-accent/20 bg-accent/5 p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-accent/15">
                  <Brain className="h-3 w-3 text-accent" />
                </div>
                <span className="text-xs font-medium text-foreground capitalize">
                  {lastBrainActivity.brainActivity.brain}
                </span>
                <span className="ml-auto text-[10px] text-muted-foreground">
                  {lastBrainActivity.brainActivity.trace.length} steps
                </span>
              </div>
              <div className="rounded-md bg-background/50 p-2 max-h-24 overflow-y-auto custom-scrollbar">
                {lastBrainActivity.brainActivity.trace.slice(-3).map((step, i) => (
                  <p key={i} className="text-[10px] text-muted-foreground leading-relaxed">
                    <span className="text-accent/50 mr-1">&gt;</span>
                    {step}
                  </p>
                ))}
              </div>
              {lastBrainActivity.brainActivity.evidence.length > 0 && (
                <p className="mt-2 text-[10px] text-muted-foreground flex items-center gap-1">
                  <TrendingUp className="h-3 w-3 text-primary" />
                  {lastBrainActivity.brainActivity.evidence.length} sources analyzed
                </p>
              )}
            </div>
          </div>
        )}

        {/* Citations from Chat */}
        {allCitations.length > 0 && (
          <div className="border-b border-border p-3">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              <FileText className="h-3 w-3" />
              Recent Citations
            </div>
            <div className="space-y-1.5">
              {allCitations.slice(0, 10).map((citation, idx) => (
                <CitationCard
                  key={`${citation.source}-${citation.page}-${idx}`}
                  source={citation.source}
                  page={citation.page}
                  excerpt={citation.excerpt}
                  relevance={citation.relevance}
                  onClick={() => onCitationClick(citation.source, citation.page, citation.docId)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Suggestions */}
        <div className="p-3">
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
            <Lightbulb className="h-3 w-3" />
            Suggestions
          </div>
          {!hasAnyContent ? (
            <div className="rounded-xl border border-dashed border-border bg-card/30 p-6 text-center">
              <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-gradient-to-br from-primary/10 to-accent/10">
                <Sparkles className="h-5 w-5 text-primary/50" />
              </div>
              <p className="text-xs font-medium text-foreground/60 mb-1">Context Engine</p>
              <p className="text-[11px] text-muted-foreground leading-relaxed max-w-[200px] mx-auto">
                Insights will surface as you explore documents, ask questions, and navigate the knowledge graph.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {selectedFilename && (
                <button className="w-full flex items-start gap-2 rounded-lg border border-accent/20 bg-accent/5 p-2.5 text-left transition-all hover:bg-accent/10 hover:border-accent/30">
                  <MessageSquare className="mt-0.5 h-3 w-3 shrink-0 text-accent" />
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    Ask: &ldquo;What are the key findings in {selectedFilename}?&rdquo;
                  </p>
                </button>
              )}
              {focusedNode && (
                <button className="w-full flex items-start gap-2 rounded-lg border border-primary/20 bg-primary/5 p-2.5 text-left transition-all hover:bg-primary/10 hover:border-primary/30">
                  <Network className="mt-0.5 h-3 w-3 shrink-0 text-primary" />
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    Explore: &ldquo;How does {focusedNode.name} relate to other concepts?&rdquo;
                  </p>
                </button>
              )}
              {graphStats && graphStats.totalNodes > 10 && !focusedNode && (
                <button className="w-full flex items-start gap-2 rounded-lg border border-primary/20 bg-primary/5 p-2.5 text-left transition-all hover:bg-primary/10 hover:border-primary/30">
                  <Brain className="mt-0.5 h-3 w-3 shrink-0 text-primary" />
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    Analyze: &ldquo;What are the main themes across my documents?&rdquo;
                  </p>
                </button>
              )}
              {allCitations.length === 0 && !focusedNode && selectedFilename && (
                <button className="w-full flex items-start gap-2 rounded-lg border border-accent/20 bg-accent/5 p-2.5 text-left transition-all hover:bg-accent/10 hover:border-accent/30">
                  <Brain className="mt-0.5 h-3 w-3 shrink-0 text-accent" />
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    Deep dive: &ldquo;Find connections between this paper and others&rdquo;
                  </p>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
