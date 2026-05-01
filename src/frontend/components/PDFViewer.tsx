'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  RotateCw,
  Loader2,
  PanelRightOpen,
  PanelRightClose,
  Users,
  BookOpen,
  Beaker,
  Lightbulb,
  AlertTriangle,
  FileText,
  Search,
  ExternalLink,
} from 'lucide-react';
import { api, PaperStructure, DocumentStructureResponse, RelatedPassage } from '@/lib/api';

// Configure PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

interface PDFViewerProps {
  fileUrl: string;
  filename: string;
  docId?: string;
  projectId?: string;
  initialPage?: number;
  onAskAboutPage?: (question: string) => void;
  onRelatedPassageClick?: (filename: string, page: number, docId?: string) => void;
  onContextChange?: (selectedText?: string, docId?: string, pageNumber?: number) => void;
}

export default function PDFViewer({
  fileUrl,
  filename,
  docId,
  projectId,
  initialPage = 1,
  onAskAboutPage,
  onRelatedPassageClick,
  onContextChange,
}: PDFViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const [pageNumber, setPageNumber] = useState<number>(initialPage);
  const [scale, setScale] = useState<number>(1.0);
  const [rotation, setRotation] = useState<number>(0);
  const [fitMode, setFitMode] = useState<'fit' | 'actual'>('fit');
  const [containerWidth, setContainerWidth] = useState<number>(1000);

  // Smart sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [structure, setStructure] = useState<DocumentStructureResponse | null>(null);
  const [structureLoading, setStructureLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [relatedPassages, setRelatedPassages] = useState<RelatedPassage[]>([]);
  const [relatedLoading, setRelatedLoading] = useState(false);

  useEffect(() => {
    setPageNumber(initialPage);
  }, [initialPage, fileUrl]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const observer = new ResizeObserver(() => {
      setContainerWidth(el.clientWidth);
    });

    setContainerWidth(el.clientWidth);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Fetch paper structure when sidebar is opened
  useEffect(() => {
    if (!sidebarOpen || !docId || !projectId || structure?.doc_id === docId) return;

    let cancelled = false;
    setStructureLoading(true);

    api.getDocumentStructure(docId, projectId)
      .then((data) => {
        if (!cancelled) setStructure(data);
      })
      .catch((err) => {
        console.error('Failed to load document structure:', err);
      })
      .finally(() => {
        if (!cancelled) setStructureLoading(false);
      });

    return () => { cancelled = true; };
  }, [sidebarOpen, docId, projectId, structure?.doc_id]);

  // Listen for text selection in the PDF viewer
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();

      // Ensure selection is within the PDF viewer to prevent Context Engine from reloading on external clicks
      if (selection && selection.anchorNode && containerRef.current && !containerRef.current.contains(selection.anchorNode)) {
        return;
      }

      const text = selection?.toString().trim() || '';
      if (text.length > 20) {
        setSelectedText(text);
      } else {
        setSelectedText('');
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  // Report context changes
  useEffect(() => {
    if (onContextChange) {
      onContextChange(selectedText, docId, pageNumber);
    }
  }, [selectedText, docId, pageNumber, onContextChange]);

  // Find related passages when text is selected
  const handleFindRelated = useCallback(async () => {
    if (!selectedText || !docId || !projectId) return;
    setRelatedLoading(true);
    try {
      const passages = await api.getRelatedPassages(docId, selectedText, projectId, 5);
      setRelatedPassages(passages);
      if (!sidebarOpen) setSidebarOpen(true);
    } catch (err) {
      console.error('Failed to find related passages:', err);
    } finally {
      setRelatedLoading(false);
    }
  }, [selectedText, docId, projectId, sidebarOpen]);

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    setPageNumber((prev) => Math.min(Math.max(prev, 1), numPages));
  };

  const s = structure?.structure;

  return (
    <div className="flex h-full bg-background">
      {/* Main PDF Area */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Toolbar */}
        <div className="flex h-11 items-center justify-between border-b border-border bg-card px-3">
          <p className="max-w-[35%] truncate text-xs text-muted-foreground">{filename}</p>

          <div className="flex items-center gap-1">
            <button
              onClick={() => setPageNumber((p) => Math.max(p - 1, 1))}
              disabled={pageNumber <= 1}
              className="inline-flex h-7 w-7 items-center justify-center border border-border bg-surface text-muted-foreground transition-colors hover:bg-accent/15 hover:text-foreground disabled:opacity-40"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>

            <span className="min-w-[4.5rem] text-center text-xs text-foreground">
              {pageNumber} / {numPages || '-'}
            </span>

            <button
              onClick={() => setPageNumber((p) => Math.min(p + 1, numPages))}
              disabled={pageNumber >= numPages}
              className="inline-flex h-7 w-7 items-center justify-center border border-border bg-surface text-muted-foreground transition-colors hover:bg-accent/15 hover:text-foreground disabled:opacity-40"
            >
              <ChevronRight className="h-4 w-4" />
            </button>

            <div className="mx-1 h-5 w-px bg-border" />

            <button
              onClick={() => {
                setFitMode('actual');
                setScale((s) => Math.max(s - 0.1, 0.5));
              }}
              className="inline-flex h-7 w-7 items-center justify-center border border-border bg-surface text-muted-foreground transition-colors hover:bg-accent/15 hover:text-foreground"
            >
              <ZoomOut className="h-3 w-3" />
            </button>

            <span className="w-10 text-center text-[11px] text-muted-foreground">
              {Math.round(scale * 100)}%
            </span>

            <button
              onClick={() => {
                setFitMode('actual');
                setScale((s) => Math.min(s + 0.1, 3.0));
              }}
              className="inline-flex h-7 w-7 items-center justify-center border border-border bg-surface text-muted-foreground transition-colors hover:bg-accent/15 hover:text-foreground"
            >
              <ZoomIn className="h-3 w-3" />
            </button>

            <button
              onClick={() => setRotation((r) => (r + 90) % 360)}
              className="inline-flex h-7 items-center justify-center border border-border bg-surface px-2 text-xs text-muted-foreground transition-colors hover:bg-accent/15 hover:text-foreground"
            >
              <RotateCw className="mr-1 h-3 w-3" />
              Rotate
            </button>

            <button
              onClick={() => {
                setFitMode('fit');
                setScale(1);
              }}
              className={`inline-flex h-7 items-center justify-center border border-border px-2 text-xs transition-colors ${fitMode === 'fit'
                ? 'bg-background text-foreground'
                : 'bg-surface text-muted-foreground hover:bg-accent/15 hover:text-foreground'
                }`}
            >
              Fit
            </button>

            <div className="mx-1 h-5 w-px bg-border" />

            {/* Find Related button (visible when text is selected) */}
            {selectedText && docId && (
              <button
                onClick={handleFindRelated}
                disabled={relatedLoading}
                className="inline-flex h-7 items-center gap-1.5 border border-primary/30 bg-primary/10 px-2.5 text-xs text-primary transition-colors hover:bg-primary/20"
              >
                {relatedLoading ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Search className="h-3 w-3" />
                )}
                Find Related
              </button>
            )}

            {/* Smart Sidebar Toggle */}
            {docId && (
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className={`inline-flex h-7 w-7 items-center justify-center border border-border transition-colors ${sidebarOpen
                  ? 'bg-primary/10 text-primary border-primary/30'
                  : 'bg-surface text-muted-foreground hover:bg-accent/15 hover:text-foreground'
                  }`}
                title={sidebarOpen ? 'Hide paper info' : 'Show paper info'}
              >
                {sidebarOpen ? (
                  <PanelRightClose className="h-3.5 w-3.5" />
                ) : (
                  <PanelRightOpen className="h-3.5 w-3.5" />
                )}
              </button>
            )}
          </div>
        </div>

        {/* PDF Content */}
        <div ref={containerRef} className="flex-1 overflow-auto bg-background p-6">
          <div className="mx-auto flex min-h-full w-full justify-center">
            <Document
              file={fileUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              loading={
                <div className="flex h-64 w-full items-center justify-center">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              }
              error={
                <div className="flex h-64 w-full items-center justify-center text-sm text-destructive">
                  Failed to load PDF.
                </div>
              }
              className="border border-border bg-surface"
            >
              <Page
                pageNumber={pageNumber}
                scale={scale}
                rotate={rotation}
                renderTextLayer={true}
                renderAnnotationLayer={true}
                className="bg-white"
                width={
                  fitMode === 'fit'
                    ? Math.max(320, Math.min(1100, Math.floor((sidebarOpen ? containerWidth : containerWidth) - 80)))
                    : undefined
                }
              />
            </Document>
          </div>
        </div>
      </div>

      {/* Smart Sidebar */}
      {sidebarOpen && docId && (
        <div className="flex w-72 shrink-0 flex-col border-l border-border bg-card overflow-hidden">
          <div className="min-h-0 flex-1 overflow-y-auto custom-scrollbar">
            {structureLoading ? (
              <div className="flex h-32 items-center justify-center">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : s ? (
              <>
                {/* Title & Authors */}
                <div className="border-b border-border p-3">
                  <h3 className="text-sm font-semibold text-foreground leading-snug">
                    {s.title || filename}
                  </h3>
                  {s.authors.length > 0 && (
                    <div className="mt-2 flex items-start gap-1.5">
                      <Users className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
                      <p className="text-[11px] text-muted-foreground leading-relaxed">
                        {s.authors.join(', ')}
                      </p>
                    </div>
                  )}
                  <div className="mt-2 flex items-center gap-2">
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
                </div>

                {/* Abstract */}
                {s.abstract && (
                  <div className="border-b border-border p-3">
                    <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                      <BookOpen className="h-3 w-3" />
                      Abstract
                    </div>
                    <p className="text-[11px] text-foreground/80 leading-relaxed line-clamp-6">
                      {s.abstract}
                    </p>
                  </div>
                )}

                {/* Methodology */}
                {s.methodology && (
                  <div className="border-b border-border p-3">
                    <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                      <Beaker className="h-3 w-3" />
                      Methodology
                    </div>
                    <p className="text-[11px] text-foreground/80 leading-relaxed">
                      {s.methodology}
                    </p>
                  </div>
                )}

                {/* Key Findings */}
                {s.key_findings.length > 0 && (
                  <div className="border-b border-border p-3">
                    <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                      <Lightbulb className="h-3 w-3" />
                      Key Findings
                    </div>
                    <ul className="space-y-1.5">
                      {s.key_findings.map((finding, i) => (
                        <li key={i} className="flex items-start gap-1.5 text-[11px] text-foreground/80 leading-relaxed">
                          <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary/50" />
                          {finding}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Limitations */}
                {s.limitations.length > 0 && (
                  <div className="border-b border-border p-3">
                    <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                      <AlertTriangle className="h-3 w-3" />
                      Limitations
                    </div>
                    <ul className="space-y-1">
                      {s.limitations.map((limitation, i) => (
                        <li key={i} className="text-[11px] text-foreground/70 leading-relaxed pl-3">
                          {limitation}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <div className="flex h-32 flex-col items-center justify-center text-muted-foreground">
                <FileText className="h-6 w-6 mb-2 opacity-50" />
                <p className="text-xs">No structure data available</p>
              </div>
            )}

            {/* Related Passages (when text is selected and searched) */}
            {relatedPassages.length > 0 && (
              <div className="border-b border-border p-3">
                <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                  <ExternalLink className="h-3 w-3" />
                  Related in Other Documents
                </div>
                <div className="space-y-2">
                  {relatedPassages.map((passage, i) => (
                    <button
                      key={`${passage.chunk_id}-${i}`}
                      onClick={() => onRelatedPassageClick?.(passage.source, passage.page, passage.doc_id)}
                      className="w-full rounded-lg border border-border bg-background p-2.5 text-left transition-all hover:border-primary/30 hover:bg-primary/5"
                    >
                      <p className="text-[11px] text-foreground/80 leading-relaxed line-clamp-3">
                        {passage.text}
                      </p>
                      <div className="mt-1.5 flex items-center gap-2">
                        <span className="truncate text-[10px] font-medium text-primary">
                          {passage.source}
                        </span>
                        <span className="text-[10px] text-muted-foreground">
                          p. {passage.page}
                        </span>
                        <span className="ml-auto text-[10px] text-muted-foreground">
                          {Math.round(passage.score * 100)}%
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Ask About This Page */}
            {onAskAboutPage && (
              <div className="p-3">
                <button
                  onClick={() => onAskAboutPage(`What are the key points on page ${pageNumber} of "${filename}"?`)}
                  className="w-full rounded-lg border border-accent/20 bg-accent/5 p-2.5 text-left transition-all hover:bg-accent/10 hover:border-accent/30"
                >
                  <p className="text-[11px] font-medium text-accent">
                    Ask about page {pageNumber}
                  </p>
                  <p className="mt-0.5 text-[10px] text-muted-foreground">
                    Get AI analysis of this page&apos;s content
                  </p>
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
