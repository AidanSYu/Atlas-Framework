/**
 * Knowledge Graph Store
 * 
 * Manages graph state for document-specific rendering and focus mode.
 */
import { create } from 'zustand';
import { api, EntityInfo, RelationshipInfo } from '@/lib/api';

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  color: string;
  val: number;
  properties?: Record<string, any>;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number;
  fy?: number;
  // For focus mode
  isFocused?: boolean;
  isNeighbor?: boolean;
  isDimmed?: boolean;
  isRelated?: boolean;
}

export interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
  label: string;
  color?: string;
  isDimmed?: boolean;
}

interface GraphState {
  // Data
  nodes: GraphNode[];
  links: GraphLink[];
  allNodes: GraphNode[]; // Cache of all project nodes for related suggestions
  allLinks: GraphLink[];
  
  // UI State
  loading: boolean;
  error: string | null;
  selectedNodeId: string | null;
  focusedNodeId: string | null;
  documentId: string | null;
  projectId: string | null;
  
  // Performance tracking
  lastRenderTime: number;
  
  // Actions
  loadGraph: (projectId: string, documentId?: string) => Promise<void>;
  setSelectedNode: (nodeId: string | null) => void;
  setFocusedNode: (nodeId: string | null) => void;
  expandNode: (nodeId: string) => Promise<void>;
  clearFocus: () => void;
  refreshGraph: () => Promise<void>;
}

// Professional color palette - dark slate with neon accents
const NODE_COLORS: Record<string, string> = {
  // Core entity types
  Person: '#10b981',      // emerald-500
  Organization: '#06b6d4', // cyan-500
  Location: '#8b5cf6',    // violet-500
  Date: '#f59e0b',        // amber-500
  Event: '#ef4444',       // red-500
  Concept: '#ec4899',     // pink-500
  Document: '#3b82f6',    // blue-500
  // Scientific types
  Chemical: '#14b8a6',    // teal-500
  Experiment: '#f97316',  // orange-500
  Result: '#84cc16',      // lime-500
  Method: '#6366f1',      // indigo-500
  // Default
  Entity: '#64748b',      // slate-500
};

const getNodeColor = (type: string): string => {
  return NODE_COLORS[type] || NODE_COLORS['Entity'];
};

const getNodeSize = (type: string): number => {
  const sizes: Record<string, number> = {
    Document: 12,
    Person: 10,
    Organization: 10,
    Event: 9,
    Concept: 8,
  };
  return sizes[type] || 8;
};

export const useGraphStore = create<GraphState>((set, get) => ({
  nodes: [],
  links: [],
  allNodes: [],
  allLinks: [],
  loading: false,
  error: null,
  selectedNodeId: null,
  focusedNodeId: null,
  documentId: null,
  projectId: null,
  lastRenderTime: 0,

  loadGraph: async (projectId, documentId) => {
    const startTime = performance.now();
    set({ loading: true, error: null, projectId, documentId });
    
    try {
      // Load document-specific graph if documentId provided, else project-wide
      const graphData = await api.getFullGraph(projectId, documentId || undefined);
      
      const nodes: GraphNode[] = graphData.nodes.map((e: EntityInfo) => ({
        id: e.id,
        name: e.name,
        type: e.type,
        color: getNodeColor(e.type),
        val: getNodeSize(e.type),
        properties: { 
          description: e.description, 
          document_id: e.document_id 
        },
      }));
      
      const links: GraphLink[] = graphData.edges.map((e: RelationshipInfo) => ({
        source: e.source_id,
        target: e.target_id,
        label: e.type,
      }));

      // If we have a document filter, also load global data for "related" suggestions
      let allNodes: GraphNode[] = [];
      let allLinks: GraphLink[] = [];
      
      if (documentId && projectId) {
        try {
          const globalData = await api.getFullGraph(projectId);
          allNodes = globalData.nodes.map((e: EntityInfo) => ({
            id: e.id,
            name: e.name,
            type: e.type,
            color: getNodeColor(e.type),
            val: getNodeSize(e.type) * 0.7, // Slightly smaller for related nodes
            properties: { description: e.description, document_id: e.document_id },
          }));
          allLinks = globalData.edges.map((e: RelationshipInfo) => ({
            source: e.source_id,
            target: e.target_id,
            label: e.type,
          }));
        } catch (e) {
          console.warn('Failed to load global graph for related suggestions:', e);
        }
      }

      const renderTime = performance.now() - startTime;
      
      set({
        nodes,
        links,
        allNodes,
        allLinks,
        loading: false,
        lastRenderTime: renderTime,
      });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to load graph',
        loading: false,
      });
    }
  },

  setSelectedNode: (nodeId) => set({ selectedNodeId: nodeId }),

  setFocusedNode: (nodeId) => {
    const { nodes, links } = get();
    
    if (!nodeId) {
      // Clear focus - reset all nodes
      set({
        focusedNodeId: null,
        nodes: nodes.map(n => ({ ...n, isFocused: false, isNeighbor: false, isDimmed: false })),
        links: links.map(l => ({ ...l, isDimmed: false })),
      });
      return;
    }

    // Find neighbors
    const neighborIds = new Set<string>();
    neighborIds.add(nodeId);
    
    links.forEach(link => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      
      if (sourceId === nodeId) neighborIds.add(targetId);
      if (targetId === nodeId) neighborIds.add(sourceId);
    });

    // Update node states
    const updatedNodes = nodes.map(node => ({
      ...node,
      isFocused: node.id === nodeId,
      isNeighbor: node.id !== nodeId && neighborIds.has(node.id),
      isDimmed: node.id !== nodeId && !neighborIds.has(node.id),
    }));

    // Update link states
    const updatedLinks = links.map(link => {
      const sourceId = typeof link.source === 'string' ? link.source : link.source.id;
      const targetId = typeof link.target === 'string' ? link.target : link.target.id;
      const isConnected = (sourceId === nodeId || targetId === nodeId);
      
      return {
        ...link,
        isDimmed: !isConnected,
        color: isConnected ? '#06b6d4' : undefined, // cyan for focused connections
      };
    });

    set({
      focusedNodeId: nodeId,
      nodes: updatedNodes,
      links: updatedLinks,
    });
  },

  expandNode: async (nodeId) => {
    const { projectId } = get();
    if (!projectId) return;
    try {
      const rels = await api.getEntityRelationships(nodeId, projectId, 'both');
      const { nodes, links } = get();
      
      const existingNodeIds = new Set(nodes.map(n => n.id));
      const existingLinkKeys = new Set(
        links.map(l => {
          const sourceId = typeof l.source === 'string' ? l.source : l.source.id;
          const targetId = typeof l.target === 'string' ? l.target : l.target.id;
          return `${sourceId}-${targetId}-${l.label}`;
        })
      );

      const newNodes: GraphNode[] = [];
      const newLinks: GraphLink[] = [];

      rels.forEach(rel => {
        // Add source node if new
        if (!existingNodeIds.has(rel.source_id)) {
          newNodes.push({
            id: rel.source_id,
            name: rel.source_name || rel.source_id,
            type: 'Entity',
            color: getNodeColor('Entity'),
            val: getNodeSize('Entity'),
            properties: {},
          });
          existingNodeIds.add(rel.source_id);
        }

        // Add target node if new
        if (!existingNodeIds.has(rel.target_id)) {
          newNodes.push({
            id: rel.target_id,
            name: rel.target_name || rel.target_id,
            type: 'Entity',
            color: getNodeColor('Entity'),
            val: getNodeSize('Entity'),
            properties: {},
          });
          existingNodeIds.add(rel.target_id);
        }

        // Add link if new
        const linkKey = `${rel.source_id}-${rel.target_id}-${rel.type}`;
        if (!existingLinkKeys.has(linkKey)) {
          newLinks.push({
            source: rel.source_id,
            target: rel.target_id,
            label: rel.type,
          });
          existingLinkKeys.add(linkKey);
        }
      });

      set({
        nodes: [...nodes, ...newNodes],
        links: [...links, ...newLinks],
      });
    } catch (error) {
      console.error('Failed to expand node:', error);
    }
  },

  clearFocus: () => {
    const { nodes, links } = get();
    set({
      focusedNodeId: null,
      nodes: nodes.map(n => ({ ...n, isFocused: false, isNeighbor: false, isDimmed: false })),
      links: links.map(l => ({ ...l, isDimmed: false, color: undefined })),
    });
  },

  refreshGraph: async () => {
    const { projectId, documentId } = get();
    if (!projectId) return;
    await get().loadGraph(projectId, documentId || undefined);
  },
}));
