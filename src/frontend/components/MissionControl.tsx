'use client';

import { useCallback, useEffect, useState } from 'react';
import { getApiBase } from '../lib/api';
import { api } from '../lib/api';
import type {
  DomainSchema,
  ProjectTargetParams,
  PropertyConstraint,
} from '../lib/discovery-types';
import { useDiscoveryStore } from '../stores/discoveryStore';

const OPERATORS: Array<'<' | '>' | '<=' | '>=' | 'between'> = ['<', '>', '<=', '>=', 'between'];

const DEFAULT_DOMAIN = 'organic_chemistry';

export interface CorpusFile {
  file: File;
  docId: string | null;
  status: 'uploading' | 'ingested' | 'error';
  error?: string;
}

export interface MissionControlProps {
  projectId: string;
  onSuccess?: (sessionId: string, sessionName: string) => void;
  onCancel?: () => void;
}

export function MissionControl({ projectId, onSuccess, onCancel }: MissionControlProps) {
  const initializeSession = useDiscoveryStore((s) => s.initializeSession);

  const [schema, setSchema] = useState<DomainSchema | null>(null);
  const [schemaLoading, setSchemaLoading] = useState(true);
  const [schemaError, setSchemaError] = useState<string | null>(null);

  const [domain, setDomain] = useState(DEFAULT_DOMAIN);
  const [objective, setObjective] = useState('');
  const [propertyConstraints, setPropertyConstraints] = useState<PropertyConstraint[]>([]);
  const [domainSpecificText, setDomainSpecificText] = useState('');
  const [showAddConstraint, setShowAddConstraint] = useState(false);
  const [addProperty, setAddProperty] = useState('');
  const [addOperator, setAddOperator] = useState<'<' | '>' | '<=' | '>=' | 'between'>('<');
  const [addValue, setAddValue] = useState('');
  const [addValueHigh, setAddValueHigh] = useState('');

  const [corpusFiles, setCorpusFiles] = useState<CorpusFile[]>([]);
  const [initError, setInitError] = useState<string | null>(null);
  const [initLoading, setInitLoading] = useState(false);

  // Brainstorm state
  const [brainstormText, setBrainstormText] = useState('');
  const [isParsing, setIsParsing] = useState(false);
  const [mode, setMode] = useState<'brainstorm' | 'review'>('brainstorm');

  // Fetch DomainSchema on mount to populate field labels
  useEffect(() => {
    let cancelled = false;
    setSchemaLoading(true);
    setSchemaError(null);
    fetch(`${getApiBase()}/api/discovery/schema`)
      .then((res) => {
        if (!res.ok) throw new Error(`Schema: ${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((data: DomainSchema) => {
        if (!cancelled) {
          setSchema(data);
          if (data.domain) setDomain(data.domain);
        }
      })
      .catch((err) => {
        if (!cancelled) setSchemaError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setSchemaLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const corpusDocumentIds = corpusFiles
    .filter((c) => c.docId != null)
    .map((c) => c.docId as string);

  // We no longer require the user to manually select an objective or corpus
  // By default, if they don't upload a corpus, the backend defaults to searching the existing DB.
  const canInitialize = true;

  const handleAddConstraint = useCallback(() => {
    const prop = addProperty.trim() || 'MW';
    const op = addOperator;
    if (op === 'between') {
      const lo = parseFloat(addValue);
      const hi = parseFloat(addValueHigh);
      if (!Number.isNaN(lo) && !Number.isNaN(hi)) {
        setPropertyConstraints((prev) => [...prev, { property: prop, operator: 'between', value: [lo, hi] }]);
      }
    } else {
      const val = parseFloat(addValue);
      if (!Number.isNaN(val)) {
        setPropertyConstraints((prev) => [...prev, { property: prop, operator: op, value: val }]);
      }
    }
    setShowAddConstraint(false);
    setAddProperty('');
    setAddValue('');
    setAddValueHigh('');
    setAddOperator('<');
  }, [addProperty, addOperator, addValue, addValueHigh]);

  const removeConstraint = useCallback((index: number) => {
    setPropertyConstraints((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const parseDomainSpecificConstraints = useCallback((text: string): Record<string, unknown> => {
    const out: Record<string, unknown> = {};
    text.split('\n').forEach((line) => {
      const idx = line.indexOf(':');
      if (idx > 0) {
        const key = line.slice(0, idx).trim();
        const value = line.slice(idx + 1).trim();
        if (key) out[key] = value;
      }
    });
    return out;
  }, []);

  const handleFileDrop = useCallback(
    (files: FileList | null) => {
      if (!files?.length) return;
      Array.from(files).forEach((file) => {
        const entry: CorpusFile = { file, docId: null, status: 'uploading' };
        setCorpusFiles((prev) => [...prev, entry]);
        api
          .uploadFile(file, projectId)
          .then((res: any) => {
            const docId = res?.doc_id ?? res?.id ?? null;
            setCorpusFiles((prev) =>
              prev.map((c) =>
                c.file === file ? { ...c, docId, status: docId ? 'ingested' : 'error', error: docId ? undefined : 'No doc_id in response' } : c
              )
            );
          })
          .catch((err) => {
            setCorpusFiles((prev) =>
              prev.map((c) =>
                c.file === file ? { ...c, status: 'error' as const, error: err instanceof Error ? err.message : String(err) } : c
              )
            );
          });
      });
    },
    [projectId]
  );

  const handleParseBrainstorm = useCallback(async () => {
    if (!brainstormText.trim()) return;
    setIsParsing(true);
    setInitError(null);
    try {
      const parsed = await api.parseBrainstorm(brainstormText, domain);
      if (parsed) {
        setObjective(parsed.objective || '');
        if (parsed.propertyConstraints) {
          setPropertyConstraints(parsed.propertyConstraints);
        }
        if (parsed.domainSpecificConstraints) {
          const lines = Object.entries(parsed.domainSpecificConstraints)
            .map(([k, v]) => `${k}: ${v}`)
            .join('\n');
          setDomainSpecificText(lines);
        }
        setMode('review');
      }
    } catch (err) {
      setInitError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsParsing(false);
    }
  }, [brainstormText, domain]);

  const handleInitialize = useCallback(async () => {
    if (!canInitialize) return;
    setInitError(null);
    setInitLoading(true);
    const params: ProjectTargetParams = {
      domain,
      objective: objective.trim(),
      propertyConstraints,
      domainSpecificConstraints: parseDomainSpecificConstraints(domainSpecificText),
      corpusDocumentIds,
      projectId,
    };
    try {
      const res = await fetch(`${getApiBase()}/api/discovery/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errData.detail ?? res.statusText);
      }
      const data = (await res.json()) as { session_id?: string; epoch_id?: string; status?: string };
      const newSessionId = data.session_id ?? undefined;
      const newSessionName = params.objective.trim() || `Session ${new Date().toLocaleDateString()}`;
      initializeSession(params, newSessionId);
      onSuccess?.(newSessionId ?? '', newSessionName);
    } catch (err) {
      setInitError(err instanceof Error ? err.message : String(err));
    } finally {
      setInitLoading(false);
    }
  }, [
    canInitialize,
    domain,
    objective,
    propertyConstraints,
    domainSpecificText,
    corpusDocumentIds,
    parseDomainSpecificConstraints,
    initializeSession,
    onSuccess,
  ]);

  const schemaLabel = schema?.target_schema?.length
    ? schema.target_schema.map((s) => s.replace(/([A-Z])/g, ' $1').trim()).join(', ')
    : 'Key Property Constraints';

  return (
    <div
      className="flex h-full w-full items-center justify-center bg-background p-6"
      role="region"
      aria-labelledby="mission-control-title"
    >
      <div className="flex max-h-[90vh] w-full max-w-2xl flex-col rounded-lg border border-neutral-700 bg-neutral-900 shadow-xl">
        <header className="flex items-center justify-between border-b border-neutral-700 px-6 py-4">
          <h2 id="mission-control-title" className="text-lg font-semibold text-neutral-100">
            Exploration Setup
          </h2>
          <button
            onClick={onCancel}
            className="text-neutral-400 hover:text-neutral-200"
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <div className="flex-1 overflow-y-auto px-6 py-5">
          {schemaLoading && (
            <p className="text-sm text-neutral-400">Loading domain schema…</p>
          )}
          {schemaError && (
            <p className="text-sm text-red-400" role="alert">
              {schemaError}
            </p>
          )}

          {schema && (
            <div className="space-y-5">
              <div>
                <label htmlFor="mission-domain" className="mb-1 block text-sm font-medium text-neutral-300">
                  Active Domain
                </label>
                <select
                  id="mission-domain"
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                  className="w-full rounded border border-neutral-600 bg-neutral-800 px-3 py-2 text-neutral-100 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                >
                  <option value="organic_chemistry">Organic Chemistry</option>
                </select>
              </div>

              {mode === 'brainstorm' ? (
                <div className="space-y-4 rounded-lg bg-neutral-800/50 p-4 border border-primary-900/40">
                  <h3 className="text-sm font-semibold text-neutral-100">Mind Splatter</h3>
                  <p className="text-sm text-neutral-400">Describe what you are trying to find. Don't worry about formatting, our agents will parse the objective and constraints for you.</p>
                  <textarea
                    value={brainstormText}
                    onChange={(e) => setBrainstormText(e.target.value)}
                    rows={4}
                    placeholder="e.g. I want to find an ATP-competitive inhibitor for EGFR with Molecular Weight around 500 and logP < 3. Avoid reactive warheads."
                    className="w-full rounded border border-neutral-600 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                  />
                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={handleParseBrainstorm}
                      disabled={isParsing || !brainstormText.trim()}
                      className="rounded bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-500 disabled:opacity-50"
                    >
                      {isParsing ? 'Parsing...' : 'Parse Ideas'}
                    </button>
                  </div>
                  <div className="text-center mt-2">
                    <button
                      type="button"
                      onClick={() => setMode('review')}
                      className="text-xs text-neutral-500 hover:text-neutral-300 underline"
                    >
                      skip and enter manually
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-5 rounded-lg border border-neutral-700 bg-neutral-800/30 p-4">
                  <div className="flex justify-between items-center mb-1">
                    <p className="text-sm text-neutral-400">Define your target based on domain schema.</p>
                    <button
                      type="button"
                      onClick={() => setMode('brainstorm')}
                      className="text-xs text-primary-400 hover:text-primary-300"
                    >
                      Back to Brainstorm
                    </button>
                  </div>

                  <div>
                    <label htmlFor="mission-objective" className="mb-1 block text-sm font-medium text-neutral-300">
                      Target Objective <span className="text-neutral-500 font-normal">(Optional)</span>
                    </label>
                    <input
                      id="mission-objective"
                      type="text"
                      value={objective}
                      onChange={(e) => setObjective(e.target.value)}
                      placeholder="e.g. EGFR kinase inhibition — ATP-competitive"
                      className="w-full rounded border border-neutral-600 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>

                  <div>
                    <span className="mb-1 block text-sm font-medium text-neutral-300">
                      {schemaLabel}
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {propertyConstraints.map((c, i) => (
                        <span
                          key={i}
                          className="inline-flex items-center gap-1 rounded-full bg-neutral-700 px-3 py-1 text-sm text-neutral-200"
                        >
                          {c.property} {c.operator}{' '}
                          {Array.isArray(c.value) ? `${c.value[0]}–${c.value[1]}` : c.value}
                          <button
                            type="button"
                            onClick={() => removeConstraint(i)}
                            className="ml-1 rounded text-neutral-400 hover:text-neutral-200"
                            aria-label="Remove constraint"
                          >
                            ×
                          </button>
                        </span>
                      ))}
                      {!showAddConstraint ? (
                        <button
                          type="button"
                          onClick={() => setShowAddConstraint(true)}
                          className="rounded-full border border-dashed border-neutral-500 px-3 py-1 text-sm text-neutral-400 hover:border-neutral-400 hover:text-neutral-300"
                        >
                          + Add constraint
                        </button>
                      ) : (
                        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-neutral-600 bg-neutral-800 p-2">
                          <input
                            type="text"
                            value={addProperty}
                            onChange={(e) => setAddProperty(e.target.value)}
                            placeholder="Property"
                            className="w-24 rounded border border-neutral-600 bg-neutral-700 px-2 py-1 text-sm text-neutral-100"
                          />
                          <select
                            value={addOperator}
                            onChange={(e) => setAddOperator(e.target.value as PropertyConstraint['operator'])}
                            className="rounded border border-neutral-600 bg-neutral-700 px-2 py-1 text-sm text-neutral-100"
                          >
                            {OPERATORS.map((op) => (
                              <option key={op} value={op}>
                                {op}
                              </option>
                            ))}
                          </select>
                          {addOperator === 'between' ? (
                            <>
                              <input
                                type="number"
                                value={addValue}
                                onChange={(e) => setAddValue(e.target.value)}
                                placeholder="Low"
                                className="w-20 rounded border border-neutral-600 bg-neutral-700 px-2 py-1 text-sm text-neutral-100"
                              />
                              <input
                                type="number"
                                value={addValueHigh}
                                onChange={(e) => setAddValueHigh(e.target.value)}
                                placeholder="High"
                                className="w-20 rounded border border-neutral-600 bg-neutral-700 px-2 py-1 text-sm text-neutral-100"
                              />
                            </>
                          ) : (
                            <input
                              type="number"
                              value={addValue}
                              onChange={(e) => setAddValue(e.target.value)}
                              placeholder="Value"
                              className="w-20 rounded border border-neutral-600 bg-neutral-700 px-2 py-1 text-sm text-neutral-100"
                            />
                          )}
                          <button
                            type="button"
                            onClick={handleAddConstraint}
                            className="rounded bg-primary-600 px-2 py-1 text-sm text-white hover:bg-primary-500"
                          >
                            Add
                          </button>
                          <button
                            type="button"
                            onClick={() => setShowAddConstraint(false)}
                            className="rounded px-2 py-1 text-sm text-neutral-400 hover:text-neutral-200"
                          >
                            Cancel
                          </button>
                        </div>
                      )}
                    </div>
                  </div>

                  <div>
                    <label htmlFor="mission-domain-specific" className="mb-1 block text-sm font-medium text-neutral-300">
                      Domain-Specific Constraints (e.g. Forbidden Substructures)
                    </label>
                    <textarea
                      id="mission-domain-specific"
                      value={domainSpecificText}
                      onChange={(e) => setDomainSpecificText(e.target.value)}
                      rows={3}
                      placeholder="key: value (one per line)"
                      className="w-full rounded border border-neutral-600 bg-neutral-800 px-3 py-2 text-neutral-100 placeholder-neutral-500 focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
                    />
                  </div>
                </div>
              )}

              <div className="mt-5">
                <span className="mb-1 block text-sm font-medium text-neutral-300">
                  Corpus Context <span className="text-neutral-500 font-normal">(Optional)</span>
                </span>
                <div
                  role="button"
                  tabIndex={0}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    handleFileDrop(e.dataTransfer.files);
                  }}
                  onClick={() => document.getElementById('mission-corpus-input')?.click()}
                  className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-neutral-600 bg-neutral-800/50 py-8 text-neutral-400 transition-colors hover:border-neutral-500 hover:text-neutral-300"
                >
                  <span className="text-sm">Drag PDFs here or click to upload</span>
                </div>
                <input
                  id="mission-corpus-input"
                  type="file"
                  multiple
                  accept=".pdf,application/pdf"
                  className="sr-only"
                  onChange={(e) => handleFileDrop(e.target.files)}
                />
                {corpusFiles.length > 0 && (
                  <ul className="mt-3 space-y-2">
                    {corpusFiles.map((cf, i) => (
                      <li
                        key={i}
                        className="flex items-center gap-2 rounded border border-neutral-700 bg-neutral-800 px-3 py-2 text-sm text-neutral-200"
                      >
                        <span className="flex-1 truncate">{cf.file.name}</span>
                        {cf.status === 'uploading' && (
                          <span className="text-neutral-500">uploading…</span>
                        )}
                        {cf.status === 'ingested' && (
                          <span className="text-emerald-400">✓ ingested</span>
                        )}
                        {cf.status === 'error' && (
                          <span className="text-red-400">{cf.error ?? 'Error'}</span>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {initError && (
                <p className="rounded bg-red-900/30 px-3 py-2 text-sm text-red-300" role="alert">
                  {initError}
                </p>
              )}
            </div>
          )}
        </div>

        <footer className="border-t border-neutral-700 px-6 py-4">
          <button
            type="button"
            onClick={handleInitialize}
            disabled={!canInitialize || initLoading}
            className="w-full rounded-lg bg-primary-600 py-2.5 font-medium text-white transition-colors hover:bg-primary-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {initLoading ? 'Initializing…' : 'Initialize Discovery Session →'}
          </button>
        </footer>
      </div>
    </div>
  );
}
