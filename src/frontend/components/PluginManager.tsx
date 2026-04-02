'use client';

import React, { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import {
  Cpu,
  Database,
  Trash2,
  RefreshCw,
  Zap,
  ChevronDown,
  ChevronRight,
  Search,
  Maximize2,
  X,
  AlertTriangle,
} from 'lucide-react';

interface Plugin {
  name: string;
  description: string;
  loaded: boolean;
  available: boolean;
  unavailable_reason?: string | null;
  type: 'deterministic' | 'semantic';
  input_schema: any;
  output_schema: any;
}

interface PluginManagerProps {
  projectId?: string;
}

export function PluginManager({ projectId: _projectId }: PluginManagerProps) {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [orchestratorProvider, setOrchestratorProvider] = useState<string>('');
  const [orchestratorModel, setOrchestratorModel] = useState<string>('');
  const [toolProvider, setToolProvider] = useState<string>('');
  const [toolModel, setToolModel] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);
  const [expandedPlugin, setExpandedPlugin] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [showAllModal, setShowAllModal] = useState(false);

  const loadPlugins = async () => {
    setIsLoading(true);
    try {
      const data = await api.getPlugins();
      setPlugins(data.plugins);
      setOrchestratorProvider(data.orchestrator_provider);
      setOrchestratorModel(data.orchestrator_model);
      setToolProvider(data.tool_provider);
      setToolModel(data.tool_model);
    } catch (err) {
      console.error('Failed to load plugins:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadPlugins();
  }, []);

  useEffect(() => {
    if (!showAllModal) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setShowAllModal(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [showAllModal]);

  const handleUnload = async (pluginName: string) => {
    try {
      await api.unloadPlugin(pluginName);
      await loadPlugins();
    } catch (err) {
      console.error('Failed to unload plugin:', err);
    }
  };

  const normalizedSearch = search.trim().toLowerCase();
  const visiblePlugins = plugins
    .filter((plugin) => {
      if (!normalizedSearch) return true;
      const searchable = [
        plugin.name,
        plugin.description,
        plugin.type,
        plugin.available ? 'available' : 'unavailable',
      ].join(' ').toLowerCase();
      return searchable.includes(normalizedSearch);
    })
    .sort((a, b) => {
      if (a.available !== b.available) return a.available ? -1 : 1;
      if (a.loaded !== b.loaded) return a.loaded ? -1 : 1;
      return a.name.localeCompare(b.name);
    });

  const loadedCount = plugins.filter((p) => p.loaded).length;
  const availableCount = plugins.filter((p) => p.available).length;
  const unavailableCount = plugins.length - availableCount;

  const renderPluginCard = (plugin: Plugin) => (
    <div
      key={plugin.name}
      className="flex flex-col gap-2 p-3 bg-background border border-border rounded-lg hover:border-primary/30 transition-colors"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={`h-2 w-2 rounded-full ${
              plugin.loaded ? 'bg-green-500' : plugin.available ? 'bg-amber-400' : 'bg-destructive'
            }`}
            title={plugin.loaded ? 'Loaded' : plugin.available ? 'Available (not loaded)' : 'Unavailable'}
          />
          <span className="text-xs font-medium text-foreground">{plugin.name}</span>
          <span className="text-[10px] text-muted-foreground px-1.5 py-0.5 bg-accent/10 rounded">
            {plugin.type}
          </span>
          <span
            className={`text-[10px] px-1.5 py-0.5 rounded border ${
              plugin.available
                ? 'text-emerald-500 border-emerald-500/30 bg-emerald-500/10'
                : 'text-destructive border-destructive/30 bg-destructive/10'
            }`}
          >
            {plugin.available ? 'available' : 'unavailable'}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() =>
              setExpandedPlugin(expandedPlugin === plugin.name ? null : plugin.name)
            }
            className="flex items-center gap-1 text-xs text-primary hover:underline"
          >
            {expandedPlugin === plugin.name ? (
              <>
                <ChevronDown className="h-3 w-3" />
                Hide
              </>
            ) : (
              <>
                <ChevronRight className="h-3 w-3" />
                Schema
              </>
            )}
          </button>
          {plugin.loaded && (
            <button
              onClick={() => handleUnload(plugin.name)}
              className="ml-2 p-1 rounded hover:bg-destructive/10 text-destructive transition-colors"
              title="Unload plugin to free memory"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      <p className="text-xs text-muted-foreground">{plugin.description}</p>

      {!plugin.available && (
        <div className="flex items-start gap-2 rounded border border-destructive/30 bg-destructive/10 p-2 text-[11px] text-destructive">
          <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
          <span className="break-all">{plugin.unavailable_reason || 'Missing dependency or import failure.'}</span>
        </div>
      )}

      {expandedPlugin === plugin.name && (
        <div className="flex flex-col gap-2 mt-1 p-2 bg-accent/5 rounded text-[10px]">
          <div>
            <span className="font-semibold text-foreground">Input:</span>
            <pre className="mt-1 p-2 bg-background rounded overflow-x-auto text-muted-foreground">
              {JSON.stringify(plugin.input_schema, null, 2)}
            </pre>
          </div>
          <div>
            <span className="font-semibold text-foreground">Output:</span>
            <pre className="mt-1 p-2 bg-background rounded overflow-x-auto text-muted-foreground">
              {JSON.stringify(plugin.output_schema, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="flex flex-col gap-4 p-4 bg-card rounded-lg border border-border">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Plugin Registry</h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowAllModal(true)}
            className="flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-xs hover:bg-accent/10 transition-colors"
          >
            <Maximize2 className="h-3.5 w-3.5" />
            Show all
          </button>
          <button
            onClick={loadPlugins}
            disabled={isLoading}
            className="flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-xs hover:bg-accent/10 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Plugin inventory at a glance */}
      <div className="grid grid-cols-3 gap-2 text-[10px]">
        <div className="rounded border border-border/50 bg-background/50 px-2 py-1.5">
          <p className="text-muted-foreground uppercase tracking-wide">Total</p>
          <p className="font-semibold text-foreground">{plugins.length}</p>
        </div>
        <div className="rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-1.5">
          <p className="text-emerald-500 uppercase tracking-wide">Available</p>
          <p className="font-semibold text-emerald-500">{availableCount} ({loadedCount} loaded)</p>
        </div>
        <div className="rounded border border-destructive/30 bg-destructive/10 px-2 py-1.5">
          <p className="text-destructive uppercase tracking-wide">Unavailable</p>
          <p className="font-semibold text-destructive">{unavailableCount}</p>
        </div>
      </div>

      {/* LLM Configuration */}
      <div className="grid grid-cols-2 gap-3 p-3 bg-accent/5 rounded border border-border/50">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Orchestrator</span>
          <div className="flex items-center gap-2">
            <Zap className="h-3.5 w-3.5 text-orange-500" />
            <span className="text-xs font-medium text-foreground">
              {orchestratorProvider}/{orchestratorModel || 'local'}
            </span>
          </div>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Tool Caller</span>
          <div className="flex items-center gap-2">
            <Database className="h-3.5 w-3.5 text-blue-500" />
            <span className="text-xs font-medium text-foreground">
              {toolProvider}/{toolModel || 'local'}
            </span>
          </div>
        </div>
      </div>

      <div className="relative">
        <Search className="pointer-events-none absolute left-2 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search plugins..."
          className="w-full rounded border border-border bg-background py-2 pl-7 pr-2 text-xs text-foreground outline-none focus:ring-1 focus:ring-primary/40"
        />
      </div>

      {/* Plugin List */}
      <div className="flex flex-col gap-2">
        {visiblePlugins.length === 0 && !isLoading && (
          <div className="text-xs text-muted-foreground text-center py-4">
            No plugins match your filter
          </div>
        )}
        {visiblePlugins.map(renderPluginCard)}
      </div>

      {showAllModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-4">
          <div className="flex max-h-[85vh] w-full max-w-4xl flex-col rounded-xl border border-border bg-card shadow-2xl">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <div>
                <h3 className="text-sm font-semibold text-foreground">All Discovery Plugins</h3>
                <p className="text-xs text-muted-foreground">
                  Showing all registered and unavailable plugins with dependency status.
                </p>
              </div>
              <button
                onClick={() => setShowAllModal(false)}
                className="rounded-md p-1.5 text-muted-foreground hover:bg-accent/20 hover:text-foreground"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="overflow-y-auto p-4">
              <div className="grid gap-2">
                {visiblePlugins.map(renderPluginCard)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
