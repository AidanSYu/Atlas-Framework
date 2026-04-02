'use client';

import React, { useRef, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import type { ChatMessage, FollowUpSuggestions } from '@/stores/chatStore';
import type { Citation, GroundingEvent } from '@/lib/api';
import {
  Bot,
  User,
  Brain,
  BookOpen,
  Sparkles,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  RotateCw,
  AlertTriangle,
  ArrowRight,
  Network,
  Zap,
  FileText,
  Search,
  History,
} from 'lucide-react';
import CitationCard from '@/components/generative/CitationCard';
import { ComparisonTable } from '@/components/generative/ComparisonTable';
import { MetricCard } from '@/components/generative/MetricCard';
import { spring, animations } from '@/lib/design-system/motion';
import { RunErrorDisplay } from './RunErrorDisplay';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ConversationViewProps {
  messages: ChatMessage[];
  onCitationClick: (filename: string, page: number, docId?: string) => void;
  groundingMap?: Map<string, GroundingEvent>;
  chatMode?: string;
  isLoading?: boolean;
  onRetry?: (query: string) => void;
  onQuickQuery?: (query: string) => void;
  onViewRunDetails?: (runId: string) => void;
  // Follow-up pills (D4)
  onFollowUpClick?: (query: string) => void;
}

// ---------------------------------------------------------------------------
// Quick-start queries per mode
// ---------------------------------------------------------------------------

const QUICK_QUERIES: Record<string, { queries: string[]; title: string; desc: string }> = {
  librarian: {
    title: 'Document Librarian',
    desc: 'Answers backed by your documents with full citations.',
    queries: [
      'What are the key findings in this paper?',
      'Compare the methods across my documents',
      'Summarize the conclusions about...',
      'Find all mentions of...',
    ],
  },
  cortex: {
    title: 'Research Cortex',
    desc: 'Deep multi-agent analysis with reflection loops.',
    queries: [
      'What connections exist between these concepts?',
      'Generate a hypothesis about...',
      'What contradictions exist in the literature?',
      'Trace the evidence chain for...',
    ],
  },
  moe: {
    title: 'Mixture of Experts',
    desc: 'Expert team: hypothesis, retrieval, writing, critique.',
    queries: [
      'Synthesize the current understanding of...',
      'What is the strongest evidence for...?',
      'Write a critical analysis of...',
      'Evaluate the methodology of...',
    ],
  },
};

// ---------------------------------------------------------------------------
// Markdown / Generative UI renderers
// ---------------------------------------------------------------------------

function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
        em: ({ children }) => <em className="italic text-foreground/90">{children}</em>,
        h1: ({ children }) => <h1 className="text-base font-bold text-foreground mt-3 mb-1.5 font-serif">{children}</h1>,
        h2: ({ children }) => <h2 className="text-sm font-bold text-foreground mt-2.5 mb-1 font-serif">{children}</h2>,
        h3: ({ children }) => <h3 className="text-sm font-semibold text-foreground mt-2 mb-1">{children}</h3>,
        ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>,
        li: ({ children }) => <li className="text-foreground/90">{children}</li>,
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-primary/40 pl-3 my-2 text-muted-foreground italic">{children}</blockquote>
        ),
        code: ({ className, children }) => {
          const isBlock = className?.includes('language-');
          if (isBlock) {
            return (
              <pre className="rounded-lg bg-zinc-950/50 p-3 my-2 overflow-x-auto">
                <code className="text-xs font-mono text-foreground/90">{children}</code>
              </pre>
            );
          }
          return <code className="rounded bg-surface px-1.5 py-0.5 text-xs font-mono text-accent">{children}</code>;
        },
        table: ({ children }) => (
          <div className="my-3 overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-xs">{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className="bg-surface/50 text-muted-foreground">{children}</thead>,
        th: ({ children }) => <th className="px-3 py-2 text-left font-semibold border-b border-border">{children}</th>,
        td: ({ children }) => <td className="px-3 py-2 border-b border-border/50">{children}</td>,
        a: ({ href, children }) => (
          <a href={href} className="text-accent underline underline-offset-2 hover:text-accent/80" target="_blank" rel="noopener noreferrer">{children}</a>
        ),
        hr: () => <hr className="my-3 border-border/50" />,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function GenerativeRenderer({ content }: { content: string }) {
  const tableRegex = /\|(.+)\|\n\|[-:\s|]+\|\n((?:\|.+\|\n)+)/;
  const metricRegex = /\*\*([^:]+):\*\*\s*([0-9.,]+)(%|\s*[a-zA-Z\s]+)?/g;

  const tableMatch = content.match(tableRegex);
  const metricMatches: RegExpExecArray[] = [];
  let match;
  while ((match = metricRegex.exec(content)) !== null) {
    metricMatches.push(match);
  }

  let tableConfig = null;
  if (tableMatch) {
    try {
      const headers = tableMatch[1].split('|').map(h => h.trim()).filter(Boolean);
      const rowsRaw = tableMatch[2].trim().split('\n').filter(Boolean);
      const rows = rowsRaw.map(r => {
        const cells = r.split('|').map(c => c.trim()).filter(Boolean);
        return {
          feature: cells[0] || '',
          values: cells.slice(1).map(c => {
            if (c.toLowerCase() === 'yes' || c === '\u2713') return true;
            if (c.toLowerCase() === 'no' || c === '\u2717' || c === 'x') return false;
            if (c === '-' || c === '') return null;
            return c;
          }),
        };
      });
      tableConfig = { headers: headers.slice(1), rows };
    } catch (e) {
      console.error('Generative UI table parser error:', e);
    }
  }

  const metricsConfig = metricMatches.map(m => ({
    label: m[1].trim(),
    value: m[2].trim(),
    unit: m[3] ? m[3].trim() : undefined,
  }));

  if (!tableConfig && metricsConfig.length === 0) {
    return <MarkdownContent content={content} />;
  }

  let remainingContent = content;
  if (tableMatch) {
    remainingContent = remainingContent.replace(tableMatch[0], '');
  }
  if (metricsConfig.length > 0) {
    remainingContent = remainingContent.replace(metricRegex, '');
  }

  return (
    <div className="space-y-4">
      {remainingContent.trim() && <MarkdownContent content={remainingContent} />}

      {metricsConfig.length > 0 && (
        <div className="flex flex-wrap gap-3 mt-4">
          {metricsConfig.map((m, i) => (
            <MetricCard
              key={i}
              label={m.label}
              value={m.value}
              unit={m.unit}
              color={i % 3 === 0 ? 'primary' : i % 3 === 1 ? 'accent' : 'success'}
            />
          ))}
        </div>
      )}

      {tableConfig && (
        <div className="mt-4">
          <ComparisonTable headers={tableConfig.headers} rows={tableConfig.rows} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mode icon helper
// ---------------------------------------------------------------------------

function getModeIcon(mode: string) {
  switch (mode) {
    case 'librarian': return BookOpen;
    case 'cortex': return Brain;
    case 'moe': return Network;
    default: return BookOpen;
  }
}

function getModeColor(mode: string) {
  switch (mode) {
    case 'librarian': return 'from-primary/20 to-primary/5';
    case 'cortex': return 'from-accent/20 to-accent/5';
    case 'moe': return 'from-blue-500/20 to-blue-500/5';
    default: return 'from-primary/20 to-primary/5';
  }
}

function getModeIconColor(mode: string) {
  switch (mode) {
    case 'librarian': return 'text-primary';
    case 'cortex': return 'text-accent';
    case 'moe': return 'text-blue-500';
    default: return 'text-primary';
  }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

/**
 * Follow-up taxonomy pills component (D4)
 */
function FollowUpPills({
  followUps,
  onClick,
  isLastMessage,
}: {
  followUps: FollowUpSuggestions;
  onClick: (query: string) => void;
  isLastMessage: boolean;
}) {
  // Only show on the last assistant message
  if (!isLastMessage) return null;

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {/* Depth pill - muted blue */}
      <button
        onClick={() => onClick(followUps.depth.query)}
        className="inline-flex items-center gap-1.5 rounded-full bg-blue-500/10 px-3 py-1.5 text-xs font-medium text-blue-600 hover:bg-blue-500/20 transition-colors"
        title={followUps.depth.query}
      >
        <span className="text-blue-500">↓</span>
        {followUps.depth.label}
      </button>

      {/* Breadth pill - muted teal */}
      <button
        onClick={() => onClick(followUps.breadth.query)}
        className="inline-flex items-center gap-1.5 rounded-full bg-teal-500/10 px-3 py-1.5 text-xs font-medium text-teal-600 hover:bg-teal-500/20 transition-colors"
        title={followUps.breadth.query}
      >
        <span className="text-teal-500">↔</span>
        {followUps.breadth.label}
      </button>

      {/* Opposition pill - muted amber */}
      <button
        onClick={() => onClick(followUps.opposition.query)}
        className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/10 px-3 py-1.5 text-xs font-medium text-amber-600 hover:bg-amber-500/20 transition-colors"
        title={followUps.opposition.query}
      >
        <span className="text-amber-500">✗</span>
        {followUps.opposition.label}
      </button>
    </div>
  );
}

export function ConversationView({
  messages,
  onCitationClick,
  groundingMap = new Map(),
  chatMode = 'librarian',
  isLoading = false,
  onRetry,
  onQuickQuery,
  onViewRunDetails,
  onFollowUpClick,
}: ConversationViewProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [expandedTraces, setExpandedTraces] = useState<Set<string>>(new Set());

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleTrace = (msgId: string) => {
    setExpandedTraces((prev) => {
      const next = new Set(prev);
      if (next.has(msgId)) next.delete(msgId);
      else next.add(msgId);
      return next;
    });
  };

  const renderCitations = (citations: Citation[], brainEvidence?: any[]) => {
    const evidenceItems = brainEvidence || [];
    return (
      <div className="mt-3 space-y-1.5">
        {citations.map((citation, idx) => {
          const matchingEvidence = evidenceItems.find(
            (e: any) => e.source === citation.source && e.page === citation.page
          );
          const groundingKey = `${citation.source}:${citation.page}`;
          const groundingStatus = groundingMap.get(groundingKey)?.status;
          return (
            <CitationCard
              key={`${citation.source}-${citation.page}-${idx}`}
              source={citation.source}
              page={citation.page}
              excerpt={citation.text || matchingEvidence?.excerpt}
              relevance={citation.relevance || matchingEvidence?.relevance}
              groundingStatus={groundingStatus}
              onClick={() => onCitationClick(citation.source, citation.page, citation.doc_id)}
            />
          );
        })}
      </div>
    );
  };

  // ---- Empty state: quick-start grid ----
  if (messages.length === 0 && !isLoading) {
    const config = QUICK_QUERIES[chatMode] || QUICK_QUERIES.librarian;
    const ModeIcon = getModeIcon(chatMode);

    return (
      <div className="flex flex-col items-center justify-center h-full px-8 py-12">
        {/* Hero */}
        <div className={`rounded-2xl bg-gradient-to-br ${getModeColor(chatMode)} p-5 mb-5`}>
          <ModeIcon className={`h-10 w-10 ${getModeIconColor(chatMode)}`} />
        </div>
        <h3 className="font-display text-lg font-semibold text-foreground mb-1">
          {config.title}
        </h3>
        <p className="text-sm text-muted-foreground/70 max-w-sm text-center leading-relaxed mb-8">
          {config.desc}
        </p>

        {/* Quick-start query grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full max-w-lg">
          {config.queries.map((query, i) => (
            <button
              key={i}
              onClick={() => onQuickQuery?.(query)}
              className="group flex items-start gap-3 rounded-xl border border-border/30 bg-surface/20 px-4 py-3.5 text-left transition-all hover:border-border/60 hover:bg-surface/40"
            >
              <Zap className={`h-3.5 w-3.5 mt-0.5 shrink-0 ${getModeIconColor(chatMode)} opacity-40 group-hover:opacity-80 transition-opacity`} />
              <span className="text-xs text-muted-foreground group-hover:text-foreground transition-colors leading-relaxed">
                {query}
              </span>
            </button>
          ))}
        </div>

        {/* Hint */}
        <p className="mt-8 text-[11px] text-muted-foreground/40">
          Press <kbd className="rounded bg-muted/50 px-1.5 py-0.5 font-mono text-[10px]">Ctrl+K</kbd> to ask from any view
        </p>
      </div>
    );
  }

  // ---- Messages ----
  return (
    <>
      <AnimatePresence mode="popLayout">
        {messages.map((message, idx) => {
          // Error messages
          if (message.errorInfo) {
            return (
              <motion.div
                key={message.id}
                layout
                layoutId={message.id}
                {...animations.slideUp}
                transition={spring}
              >
                <RunErrorDisplay
                  category={message.errorInfo.category}
                  message={message.errorInfo.message}
                  onRetry={message.errorInfo.retryable && onRetry
                    ? () => {
                        const userMsg = messages.findLast((m) => m.role === 'user');
                        if (userMsg) onRetry(userMsg.content);
                      }
                    : undefined}
                />
              </motion.div>
            );
          }

          return (
          <motion.div
            key={message.id}
            layout
            layoutId={message.id}
            {...animations.slideUp}
            transition={spring}
            className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {message.role === 'assistant' && (
              <div className={`mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl ${
                message.brainActivity
                ? message.brainActivity.brain.includes('MoE')
                  ? 'bg-gradient-to-br from-blue-500/20 to-blue-500/10 border border-blue-500/20'
                  : 'bg-gradient-to-br from-accent/20 to-accent/10 border border-accent/20'
                : 'bg-gradient-to-br from-primary/20 to-primary/10 border border-primary/20'
                }`}>
                {message.brainActivity ? (
                  message.brainActivity.brain.includes('MoE') ? (
                    <Network className="h-4 w-4 text-blue-500" />
                  ) : (
                    <Brain className="h-4 w-4 text-accent" />
                  )
                ) : (
                  <Bot className="h-4 w-4 text-primary" />
                )}
              </div>
            )}

            <div className="max-w-[80%]">
              <div
                className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${message.role === 'user'
                  ? 'bg-gradient-to-br from-primary to-primary/90 text-primary-foreground rounded-br-sm shadow-sm'
                  : (message as any).isThinking
                    ? 'rounded-bl-sm border border-emerald-500/20 bg-emerald-500/5 text-emerald-300/90 font-mono text-xs'
                    : 'rounded-bl-sm border border-border bg-card text-foreground'
                  }`}
              >
                {message.role === 'assistant' ? (
                  <div className="prose-sm">
                    <GenerativeRenderer content={message.content} />
                  </div>
                ) : (
                  <div className="whitespace-pre-wrap break-words">{message.content}</div>
                )}
              </div>

              {/* Brain Activity Trace */}
              {message.brainActivity && message.brainActivity.trace.length > 0 && (
                <div className="mt-2">
                  <button
                    onClick={() => toggleTrace(message.id)}
                    className="flex items-center gap-1.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors group"
                  >
                    {expandedTraces.has(message.id) ? (
                      <ChevronUp className="h-3 w-3" />
                    ) : (
                      <ChevronDown className="h-3 w-3" />
                    )}
                    <Sparkles className="h-3 w-3 text-accent group-hover:text-accent" />
                    <span className="capitalize">{message.brainActivity.brain}</span> reasoning
                    <span className="rounded-full bg-surface px-1.5 py-0.5 text-[9px]">
                      {message.brainActivity.trace.length} steps
                    </span>
                  </button>

                  {(message.brainActivity.confidenceScore !== undefined || message.brainActivity.iterations !== undefined) && (
                    <div className="mt-2 flex gap-3 text-xs text-muted-foreground">
                      {message.brainActivity.confidenceScore !== undefined && (
                        <div className="flex items-center gap-1.5">
                          <ShieldCheck className="h-3.5 w-3.5 text-success" />
                          <span>Confidence: {(message.brainActivity.confidenceScore * 100).toFixed(0)}%</span>
                        </div>
                      )}
                      {message.brainActivity.iterations !== undefined && (
                        <div className="flex items-center gap-1.5">
                          <RotateCw className="h-3.5 w-3.5 text-accent" />
                          <span>{message.brainActivity.iterations} reflection{message.brainActivity.iterations !== 1 ? 's' : ''}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {expandedTraces.has(message.id) && (
                    <div className="mt-1.5 rounded-xl border border-border/50 bg-zinc-950/30 px-3 py-2.5 space-y-1.5">
                      {message.brainActivity.trace.map((step, i) => (
                        <div key={i} className="flex items-start gap-2 text-[11px]">
                          <span className="shrink-0 flex h-4 w-4 items-center justify-center rounded-full bg-accent/10 text-[9px] font-mono text-accent/60">
                            {i + 1}
                          </span>
                          <span className="leading-relaxed text-muted-foreground">{step}</span>
                        </div>
                      ))}

                      {message.brainActivity.contradictions && message.brainActivity.contradictions.length > 0 && (
                        <div className="mt-3 border-t border-border pt-3">
                          <div className="mb-2 text-xs font-medium text-warning">
                            Contradictions Identified ({message.brainActivity.contradictions.length})
                          </div>
                          {message.brainActivity.contradictions.map((contradiction, i) => (
                            <div key={i} className="mb-2 rounded-lg bg-warning/5 border border-warning/20 p-3 text-xs">
                              <div className="mb-1 flex items-center gap-1.5 font-medium text-warning">
                                <AlertTriangle className="h-3 w-3" />
                                {contradiction.severity}
                              </div>
                              <div className="space-y-1.5 text-muted-foreground">
                                <div><span className="text-foreground">A:</span> {contradiction.claim_a}</div>
                                <div><span className="text-foreground">B:</span> {contradiction.claim_b}</div>
                                {contradiction.resolution && (
                                  <div className="mt-2 border-t border-warning/10 pt-1.5 text-xs text-foreground">
                                    <span className="text-success">Resolution:</span> {contradiction.resolution}
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Citations */}
              {message.citations && message.citations.length > 0 && (
                renderCitations(message.citations, message.brainActivity?.evidence)
              )}

              {/* Librarian Metadata */}
              {message.librarianMetadata?.reasoning && (
                <div className="mt-3 rounded-lg bg-muted/30 p-3 text-xs">
                  <div className="mb-1.5 font-medium text-foreground">Reasoning</div>
                  <div className="text-muted-foreground">{message.librarianMetadata.reasoning}</div>
                </div>
              )}

              {message.librarianMetadata?.relationships && message.librarianMetadata.relationships.length > 0 && (
                <div className="mt-2 space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">Graph Relationships</div>
                  {message.librarianMetadata.relationships.map((rel, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="text-foreground">{rel.source}</span>
                      <ArrowRight className="h-3 w-3 text-accent" />
                      <span className="rounded bg-accent/10 px-1.5 py-0.5 text-accent">{rel.type}</span>
                      <ArrowRight className="h-3 w-3 text-accent" />
                      <span className="text-foreground">{rel.target}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* View run details link */}
              {message.role === 'assistant' && message.runId && onViewRunDetails && (
                <button
                  onClick={() => onViewRunDetails(message.runId!)}
                  className="mt-2 flex items-center gap-1.5 text-[10px] text-muted-foreground/50 hover:text-muted-foreground transition-colors"
                >
                  <History className="h-3 w-3" />
                  View run details
                </button>
              )}

              {/* Follow-up taxonomy pills (D4) - only on last assistant message */}
              {message.role === 'assistant' && message.followUps && onFollowUpClick && (
                <FollowUpPills
                  followUps={message.followUps}
                  onClick={onFollowUpClick}
                  isLastMessage={
                    // Find the last assistant message index
                    messages.map((m, i) => ({ ...m, originalIndex: i }))
                      .filter(m => m.role === 'assistant')
                      .pop()?.originalIndex === idx
                  }
                />
              )}
            </div>

            {message.role === 'user' && (
              <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-primary/20 border border-primary/30">
                <User className="h-4 w-4 text-primary" />
              </div>
            )}
          </motion.div>
          );
        })}
      </AnimatePresence>

      <div ref={messagesEndRef} />
    </>
  );
}
