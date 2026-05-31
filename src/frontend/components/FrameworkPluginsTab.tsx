'use client';

import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Braces,
  CheckCircle2,
  Cpu,
  Info,
  Loader2,
  Play,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Wrench,
  Zap,
  Package,
  Activity,
  CircleDot,
} from 'lucide-react';

import {
  api,
  type FrameworkCatalogResponse,
  type FrameworkMachineProfile,
  type FrameworkPluginInvokeResponse,
  type FrameworkPluginProofResponse,
  type FrameworkPluginRuntimeInfo,
  type FrameworkRuntimeResponse,
  type FrameworkToolInfo,
} from '@/lib/api';

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function formatMb(value: number): string {
  if (!value) return 'n/a';
  if (value >= 1024) return `${(value / 1024).toFixed(1)} GB`;
  return `${value} MB`;
}

/* ================================================================
   Status styling — uses accent (red-orange) instead of pastels
   ================================================================ */

const STATUS_META: Record<string, { classes: string; tooltip: string }> = {
  passed: {
    classes: 'border-accent/30 bg-accent/10 text-accent',
    tooltip: 'Self-test passed — plugin works on this machine',
  },
  success: {
    classes: 'border-accent/30 bg-accent/10 text-accent',
    tooltip: 'Plugin executed successfully',
  },
  ok: {
    classes: 'border-accent/30 bg-accent/10 text-accent',
    tooltip: 'Preflight checks passed — all dependencies met',
  },
  attention: {
    classes: 'border-warning/30 bg-warning/10 text-warning',
    tooltip: 'Some checks need review — plugin may have limited functionality',
  },
  unverified: {
    classes: 'border-muted-foreground/30 bg-muted/40 text-muted-foreground',
    tooltip: 'Not yet verified — run a proof to check if this plugin works',
  },
  advisory: {
    classes: 'border-warning/30 bg-warning/10 text-warning',
    tooltip: 'Advisory notes present — review before use',
  },
  unsupported: {
    classes: 'border-destructive/30 bg-destructive/10 text-destructive',
    tooltip: 'Not supported on this machine — check dependencies',
  },
  blocked: {
    classes: 'border-destructive/30 bg-destructive/10 text-destructive',
    tooltip: 'Blocked — missing critical dependencies or incompatible environment',
  },
  failed: {
    classes: 'border-destructive/30 bg-destructive/10 text-destructive',
    tooltip: 'Last run failed — check the result for details',
  },
};

const DEFAULT_STATUS_META = {
  classes: 'border-border bg-background/60 text-muted-foreground',
  tooltip: 'Unknown status',
};

function getStatusMeta(status: string) {
  return STATUS_META[status] ?? DEFAULT_STATUS_META;
}

function StatusBadge({ status, className = '' }: { status: string; className?: string }) {
  const meta = getStatusMeta(status);
  return (
    <span
      title={meta.tooltip}
      className={`rounded border px-2.5 py-1 text-[10px] font-medium uppercase tracking-[0.14em] cursor-default ${meta.classes} ${className}`}
    >
      {status}
    </span>
  );
}

function machineSummary(machine: FrameworkMachineProfile | null): string {
  if (!machine) return 'Machine profile unavailable.';
  const gpu = machine.gpu_devices[0];
  const gpuText = gpu ? `${gpu.name} (${formatMb(gpu.total_vram_mb)})` : 'No CUDA GPU detected';
  return `${machine.platform} · Python ${machine.python_version} · ${formatMb(machine.total_ram_mb)} RAM · ${gpuText}`;
}

function defaultArgumentsFor(
  plugin: FrameworkToolInfo | null,
  runtimePlugin: FrameworkPluginRuntimeInfo | null,
): string {
  if (runtimePlugin?.supports_self_test && Object.keys(runtimePlugin.default_proof_arguments).length > 0) {
    return prettyJson(runtimePlugin.default_proof_arguments);
  }
  if (!plugin) return '{}';
  return plugin.self_test ? '{\n  "mode": "self_test"\n}' : '{}';
}

export function FrameworkPluginsTab() {
  const [catalog, setCatalog] = useState<FrameworkCatalogResponse | null>(null);
  const [runtime, setRuntime] = useState<FrameworkRuntimeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPluginName, setSelectedPluginName] = useState<string | null>(null);
  const [argumentsText, setArgumentsText] = useState('{}');
  const [runResult, setRunResult] = useState<FrameworkPluginInvokeResponse | null>(null);
  const [proofResult, setProofResult] = useState<FrameworkPluginProofResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [proofRunning, setProofRunning] = useState(false);

  async function loadCatalog() {
    setLoading(true);
    setError(null);
    try {
      const [catalogResponse, runtimeResponse] = await Promise.all([
        api.getFrameworkTools(),
        api.getFrameworkRuntime(),
      ]);
      setCatalog(catalogResponse);
      setRuntime(runtimeResponse);

      const nextSelected = selectedPluginName ?? catalogResponse.plugins[0]?.name ?? null;
      if (nextSelected) {
        const plugin = catalogResponse.plugins.find((item) => item.name === nextSelected) ?? null;
        const runtimePlugin = runtimeResponse.plugins.find((item) => item.name === nextSelected) ?? null;
        setSelectedPluginName(nextSelected);
        setArgumentsText(defaultArgumentsFor(plugin, runtimePlugin));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load framework plugins.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadCatalog();
  }, []);

  const selectedPlugin = catalog?.plugins.find((plugin) => plugin.name === selectedPluginName) ?? null;
  const selectedRuntime = runtime?.plugins.find((plugin) => plugin.name === selectedPluginName) ?? null;
  const selectedProof = proofResult?.plugin_name === selectedPluginName ? proofResult : null;

  function selectPlugin(plugin: FrameworkToolInfo) {
    const runtimePlugin = runtime?.plugins.find((item) => item.name === plugin.name) ?? null;
    setSelectedPluginName(plugin.name);
    setArgumentsText(defaultArgumentsFor(plugin, runtimePlugin));
    setRunResult(null);
    setProofResult(null);
    setError(null);
  }

  function loadProofPayload() {
    if (!selectedPlugin || !selectedRuntime) return;
    setArgumentsText(defaultArgumentsFor(selectedPlugin, selectedRuntime));
  }

  async function runSelectedPlugin() {
    if (!selectedPlugin) return;

    let parsedArguments: Record<string, any>;
    try {
      parsedArguments = JSON.parse(argumentsText);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Arguments must be valid JSON.');
      return;
    }

    setRunning(true);
    setError(null);
    try {
      const response = await api.invokeFrameworkPlugin(
        selectedPlugin.name,
        parsedArguments,
        { manual_clickthrough: true },
      );
      setRunResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Plugin invocation failed.');
      setRunResult(null);
    } finally {
      setRunning(false);
    }
  }

  async function runSelectedProof() {
    if (!selectedPlugin) return;
    setProofRunning(true);
    setError(null);
    try {
      const response = await api.proveFrameworkPlugin(selectedPlugin.name);
      setProofResult(response);
      if (response.arguments && Object.keys(response.arguments).length > 0) {
        setArgumentsText(prettyJson(response.arguments));
      }
      const nextRuntime = await api.getFrameworkRuntime();
      setRuntime(nextRuntime);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Plugin proof failed.');
      setProofResult(null);
    } finally {
      setProofRunning(false);
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!catalog || !runtime) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <div className="rounded-xl border border-border bg-card p-6 text-center">
          <p className="text-sm text-foreground">{error ?? 'Framework runtime is unavailable.'}</p>
          <button
            type="button"
            onClick={() => void loadCatalog()}
            className="mt-4 inline-flex items-center gap-2 rounded border border-border px-4 py-2 text-sm text-foreground transition-colors hover:bg-surface"
          >
            <RefreshCw className="h-4 w-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  const pluginCount = catalog.plugins.length;
  const okCount = runtime.plugins.filter((p) => p.preflight_status === 'ok' || p.preflight_status === 'passed').length;
  const blockedCount = runtime.plugins.filter((p) => p.preflight_status === 'blocked').length;

  return (
    <div className="flex h-full min-h-0 overflow-hidden">
      {/* ---- LEFT SIDEBAR: Plugin catalog ---- */}
      <aside className="flex w-[340px] shrink-0 flex-col border-r border-border bg-card">
        {/* Header */}
        <div className="border-b border-border px-4 py-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Package className="h-4 w-4 text-accent" />
              <p className="text-sm font-semibold text-foreground">Plugins</p>
              <span className="rounded bg-accent/10 px-1.5 py-0.5 text-[10px] font-medium text-accent">
                {pluginCount}
              </span>
            </div>
            <button
              type="button"
              onClick={() => void loadCatalog()}
              className="inline-flex h-7 w-7 items-center justify-center rounded border border-border text-muted-foreground transition-colors hover:bg-surface hover:text-foreground"
              title="Refresh plugin catalog"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>

          {/* Quick summary stats */}
          <div className="flex items-center gap-3 mt-3 text-[11px] text-muted-foreground">
            <span className="flex items-center gap-1" title="Plugins that passed preflight">
              <CircleDot className="h-3 w-3 text-accent" />
              {okCount} ready
            </span>
            {blockedCount > 0 && (
              <span className="flex items-center gap-1" title="Plugins blocked by missing dependencies">
                <AlertTriangle className="h-3 w-3 text-destructive" />
                {blockedCount} blocked
              </span>
            )}
            <span className="flex items-center gap-1" title="Total registered plugins">
              {pluginCount - okCount - blockedCount} other
            </span>
          </div>
        </div>

        {/* Machine preflight */}
        <div className="border-b border-border px-4 py-3">
          <div className="rounded-lg border border-border bg-background/50 p-3">
            <div className="flex items-center gap-2 text-xs font-medium text-foreground">
              <Cpu className="h-3.5 w-3.5 text-accent/70" />
              Machine
            </div>
            <p className="mt-1.5 text-[11px] leading-5 text-muted-foreground">{machineSummary(runtime.machine)}</p>
          </div>
        </div>

        {/* Plugin list */}
        <div className="min-h-0 flex-1 overflow-y-auto p-2">
          <div className="space-y-1">
            {catalog.plugins.map((plugin) => {
              const runtimePlugin = runtime.plugins.find((item) => item.name === plugin.name) ?? null;
              const isActive = plugin.name === selectedPluginName;
              const statusMeta = getStatusMeta(runtimePlugin?.preflight_status ?? 'unverified');
              return (
                <button
                  key={plugin.name}
                  type="button"
                  onClick={() => selectPlugin(plugin)}
                  className={`w-full rounded-lg border p-3 text-left transition-all ${
                    isActive
                      ? 'border-accent/40 bg-accent/8 ring-1 ring-accent/15'
                      : 'border-transparent hover:border-border hover:bg-surface/50'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className={`text-sm font-medium ${isActive ? 'text-accent' : 'text-foreground'}`}>
                          {plugin.name}
                        </span>
                      </div>
                      <div className="mt-0.5 text-[11px] text-muted-foreground/70">{plugin.source_type}</div>
                    </div>
                    <StatusBadge status={runtimePlugin?.preflight_status ?? 'unverified'} />
                  </div>
                  <p className="mt-2 text-[12px] leading-5 text-muted-foreground line-clamp-2">{plugin.description}</p>
                  {runtimePlugin?.supports_self_test && (
                    <div className="mt-2 flex items-center gap-1 text-[10px] text-accent/70">
                      <ShieldCheck className="h-3 w-3" />
                      Self-test available
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </aside>

      {/* ---- MAIN CONTENT: Plugin details ---- */}
      <section className="min-w-0 flex-1 overflow-y-auto bg-background">
        {selectedPlugin && selectedRuntime ? (
          <div className="mx-auto flex max-w-5xl flex-col gap-5 px-6 py-5">
            {/* Plugin header */}
            <div className="rounded-xl border border-border bg-card p-5">
              <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                <div className="max-w-3xl">
                  <div className="inline-flex items-center gap-2 rounded border border-border px-2.5 py-1 text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                    <Wrench className="h-3 w-3" />
                    Plugin Workbench
                  </div>
                  <h2 className="mt-3 text-xl font-semibold tracking-tight text-foreground">{selectedPlugin.name}</h2>
                  <p className="mt-1.5 text-sm leading-6 text-muted-foreground">{selectedPlugin.description}</p>
                </div>

                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void runSelectedProof()}
                    disabled={proofRunning || !selectedRuntime.supports_self_test}
                    title={!selectedRuntime.supports_self_test ? 'This plugin does not expose a self-test' : 'Run the plugin self-test to verify it works'}
                    className="inline-flex items-center justify-center gap-2 rounded border border-border px-3 py-1.5 text-[13px] text-foreground transition-colors hover:bg-surface disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {proofRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ShieldCheck className="h-3.5 w-3.5" />}
                    Run proof
                  </button>
                  <button
                    type="button"
                    onClick={loadProofPayload}
                    disabled={!selectedRuntime.supports_self_test}
                    title="Load the default self-test payload into the arguments editor"
                    className="inline-flex items-center justify-center gap-2 rounded border border-border px-3 py-1.5 text-[13px] text-foreground transition-colors hover:bg-surface disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <Braces className="h-3.5 w-3.5" />
                    Use proof payload
                  </button>
                  <button
                    type="button"
                    onClick={() => void runSelectedPlugin()}
                    disabled={running}
                    title="Execute the plugin with the current arguments"
                    className="inline-flex items-center justify-center gap-2 rounded border border-accent/40 bg-accent/10 px-3 py-1.5 text-[13px] text-accent transition-colors hover:bg-accent/15 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                    Run plugin
                  </button>
                </div>
              </div>

              {/* Status badges */}
              <div className="mt-4 flex flex-wrap gap-2">
                <StatusBadge status={selectedRuntime.preflight_status} />
                <span
                  title={selectedRuntime.loaded ? 'Plugin module is loaded in memory' : 'Plugin will be loaded on first use'}
                  className={`rounded border px-2.5 py-1 text-[10px] font-medium uppercase tracking-[0.14em] cursor-default ${
                    selectedRuntime.loaded
                      ? 'border-accent/20 bg-accent/8 text-accent'
                      : 'border-muted-foreground/20 bg-muted/40 text-muted-foreground'
                  }`}
                >
                  {selectedRuntime.loaded ? 'loaded' : 'lazy-loaded'}
                </span>
                <span
                  title="Software license for this plugin"
                  className="rounded border border-border/70 px-2.5 py-1 text-[10px] font-medium text-muted-foreground cursor-default"
                >
                  license: {selectedRuntime.license || 'unspecified'}
                </span>
                {selectedPlugin.tags.map((tag) => (
                  <span key={tag} className="rounded border border-border/70 px-2.5 py-1 text-[10px] font-medium text-muted-foreground cursor-default">
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            {/* Two-column grid */}
            <div className="grid gap-5 xl:grid-cols-[1.05fr_0.95fr]">
              {/* Left column */}
              <div className="space-y-5">
                {/* Runtime Preflight */}
                <div className="rounded-xl border border-border bg-card p-5">
                  <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                    <ShieldAlert className="h-4 w-4 text-accent/70" />
                    Runtime Preflight
                  </div>
                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    <div className="rounded-lg border border-border bg-background/70 p-3">
                      <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Machine fit</div>
                      <p className="mt-1.5 text-sm font-medium text-foreground">
                        {selectedRuntime.resource_assessment.status}
                      </p>
                      <p className="mt-1 text-[11px] leading-5 text-muted-foreground">
                        {machineSummary(runtime.machine)}
                      </p>
                    </div>
                    <div className="rounded-lg border border-border bg-background/70 p-3">
                      <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Self-test</div>
                      <p className="mt-1.5 text-sm font-medium text-foreground">
                        {selectedRuntime.supports_self_test ? 'Available' : 'Not exposed'}
                      </p>
                      <p className="mt-1 text-[11px] leading-5 text-muted-foreground">
                        Default payload: {Object.keys(selectedRuntime.default_proof_arguments).length > 0
                          ? JSON.stringify(selectedRuntime.default_proof_arguments)
                          : 'none'}
                      </p>
                    </div>
                  </div>

                  {selectedRuntime.blocking_issues.length > 0 && (
                    <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-[13px] text-destructive">
                      {selectedRuntime.blocking_issues.map((issue) => (
                        <div key={issue} className="flex items-start gap-2">
                          <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                          <span>{issue}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {selectedRuntime.advisory_notes.length > 0 && (
                    <div className="mt-3 rounded-lg border border-warning/30 bg-warning/5 px-4 py-3 text-[13px] text-warning">
                      {selectedRuntime.advisory_notes.map((note) => (
                        <div key={note} className="flex items-start gap-2">
                          <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                          <span>{note}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {selectedRuntime.dependency_statuses.length > 0 && (
                    <div className="mt-3">
                      <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">Dependencies</div>
                      <div className="flex flex-wrap gap-1.5">
                        {selectedRuntime.dependency_statuses.map((dependency) => (
                          <span
                            key={dependency.package}
                            title={dependency.available ? `${dependency.package} is installed` : `${dependency.package} is missing — install to unblock`}
                            className={`rounded border px-2 py-0.5 text-[10px] font-medium cursor-default ${
                              dependency.available
                                ? 'border-accent/20 bg-accent/5 text-accent/80'
                                : 'border-destructive/20 bg-destructive/5 text-destructive/80'
                            }`}
                          >
                            {dependency.package}: {dependency.available ? 'ok' : 'missing'}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Invocation Arguments */}
                <div className="rounded-xl border border-border bg-card p-5">
                  <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                    <Braces className="h-4 w-4 text-accent/70" />
                    Invocation Arguments
                  </div>
                  <p className="mt-1.5 text-[11px] text-muted-foreground">
                    Load the proof payload for a guided self-test, or supply any valid JSON arguments directly.
                  </p>
                  <textarea
                    value={argumentsText}
                    onChange={(event) => setArgumentsText(event.target.value)}
                    spellCheck={false}
                    className="mt-3 min-h-[240px] w-full rounded-lg border border-border bg-background p-3 font-mono text-xs text-foreground outline-none transition-colors focus:border-accent/40 focus:ring-1 focus:ring-accent/20"
                  />
                  {error && (
                    <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-[13px] text-destructive">
                      {error}
                    </div>
                  )}
                </div>

                {/* Input Schema */}
                <div className="rounded-xl border border-border bg-card p-5">
                  <div className="text-sm font-medium text-foreground">Input Schema</div>
                  <pre className="mt-3 overflow-x-auto rounded-lg border border-border bg-background p-3 text-xs text-muted-foreground">
                    {prettyJson(selectedPlugin.input_schema)}
                  </pre>
                </div>
              </div>

              {/* Right column */}
              <div className="space-y-5">
                {/* Latest Proof */}
                <div className="rounded-xl border border-border bg-card p-5">
                  <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                    <ShieldCheck className="h-4 w-4 text-accent/70" />
                    Latest Proof
                  </div>
                  {selectedProof ? (
                    <>
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        <StatusBadge status={selectedProof.proof_status} />
                        <span className="text-xs text-muted-foreground">{selectedProof.summary}</span>
                        <span className="text-[11px] text-muted-foreground/60">{selectedProof.duration_ms} ms</span>
                      </div>
                      <pre className="mt-3 overflow-x-auto rounded-lg border border-border bg-background p-3 text-xs text-muted-foreground">
                        {prettyJson(selectedProof.result)}
                      </pre>
                    </>
                  ) : (
                    <div className="mt-3 rounded-lg border border-dashed border-border bg-background/50 p-4 text-center">
                      <ShieldCheck className="h-5 w-5 mx-auto text-muted-foreground/30 mb-2" />
                      <p className="text-[12px] text-muted-foreground">
                        No proof run yet. Use the proof button to validate this plugin.
                      </p>
                    </div>
                  )}
                </div>

                {/* Latest Manual Result */}
                <div className="rounded-xl border border-border bg-card p-5">
                  <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                    <Activity className="h-4 w-4 text-accent/70" />
                    Latest Manual Result
                  </div>
                  {runResult ? (
                    <>
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        <StatusBadge status={runResult.status} />
                        <span className="text-xs text-muted-foreground">{runResult.summary}</span>
                      </div>
                      <pre className="mt-3 overflow-x-auto rounded-lg border border-border bg-background p-3 text-xs text-muted-foreground">
                        {prettyJson(runResult.result)}
                      </pre>
                    </>
                  ) : (
                    <div className="mt-3 rounded-lg border border-dashed border-border bg-background/50 p-4 text-center">
                      <Play className="h-5 w-5 mx-auto text-muted-foreground/30 mb-2" />
                      <p className="text-[12px] text-muted-foreground">
                        No invocation yet. Run the plugin to see raw outputs.
                      </p>
                    </div>
                  )}
                </div>

                {/* Output Schema */}
                <div className="rounded-xl border border-border bg-card p-5">
                  <div className="text-sm font-medium text-foreground">Output Schema</div>
                  <pre className="mt-3 overflow-x-auto rounded-lg border border-border bg-background p-3 text-xs text-muted-foreground">
                    {prettyJson(selectedPlugin.output_schema)}
                  </pre>
                </div>

                {/* Execution Posture */}
                <div className="rounded-xl border border-border bg-card p-5">
                  <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                    <CheckCircle2 className="h-4 w-4 text-accent/70" />
                    Execution Posture
                  </div>
                  <p className="mt-2 text-[12px] leading-5 text-muted-foreground">
                    Preflight checks machine fit and library availability. Proof runs the plugin's own self-test and shows the raw result. These are independent — a passing preflight does not guarantee the plugin works.
                  </p>
                  {selectedRuntime.load_error && (
                    <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-[13px] text-destructive">
                      Load error: {selectedRuntime.load_error}
                    </div>
                  )}
                  {!selectedRuntime.supports_self_test && (
                    <div className="mt-3 rounded-lg border border-warning/30 bg-warning/5 px-3 py-2 text-[12px] text-warning">
                      This plugin does not expose a self-test route. Manual invocation is the verification path.
                    </div>
                  )}
                  {selectedRuntime.preflight_status === 'blocked' && selectedRuntime.blocking_issues.length === 0 && (
                    <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-[13px] text-destructive">
                      <div className="flex items-center gap-2">
                        <AlertTriangle className="h-3.5 w-3.5" />
                        Runtime preflight is blocked on this machine.
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center text-center px-6">
            <Package className="h-8 w-8 text-muted-foreground/25 mb-3" />
            <p className="text-sm text-muted-foreground">Select a plugin from the sidebar</p>
            <p className="mt-1 text-[11px] text-muted-foreground/60">
              {pluginCount} plugin{pluginCount !== 1 ? 's' : ''} registered
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
