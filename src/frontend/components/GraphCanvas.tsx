'use client';

import React, { useEffect, useRef, useState, useCallback, useMemo, Suspense, lazy } from 'react';
import dynamic from 'next/dynamic';
import { api, EntityInfo, RelationshipInfo } from '@/lib/api';
import { X } from 'lucide-react';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

interface GraphCanvasProps {
  height: number;
  width: number;
  projectId: string;
}

type NodeData = { id: string; name: string; type: string; color: string; properties?: Record<string, any> };
type LinkData = { source: string; target: string; label: string };

export default function GraphCanvas({ height, width, projectId }: GraphCanvasProps) {
  const [nodes, setNodes] = useState<NodeData[]>([]);
  const [links, setLinks] = useState<LinkData[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null);
  const fgRef = useRef<any>();
  const nodePositions = useRef<Map<string, { x: number; y: number }>>(new Map());

  const loadGraph = useCallback(async () => {
    try {
      setLoading(true);
      // Use /graph/full endpoint to get both nodes and edges at once
      const graphData = await api.getFullGraph(projectId);
      
      const newNodes: NodeData[] = graphData.nodes.map(e => ({
        id: e.id,
        name: e.name,
        type: e.type,
        color: getNodeColor(e.type),
        properties: { description: e.description, document_id: e.document_id },
      }));
      
      const newLinks: LinkData[] = graphData.edges.map(e => ({
        source: e.source_id,
        target: e.target_id,
        label: e.type,
      }));
      
      // Clear positions on initial load
      nodePositions.current.clear();
      setNodes(newNodes);
      setLinks(newLinks);
    } catch (error) {
      console.error('Failed to load graph:', error);
      // Fallback to just loading nodes if full graph fails
      try {
        const ents: EntityInfo[] = await api.listEntities({ project_id: projectId, limit: 100 });
        const newNodes: NodeData[] = ents.map(e => ({
          id: e.id,
          name: e.name,
          type: e.type,
          color: getNodeColor(e.type),
          properties: { description: e.description, document_id: e.document_id },
        }));
        setNodes(newNodes);
        setLinks([]);
      } catch (fallbackError) {
        console.error('Fallback graph load also failed:', fallbackError);
      }
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  // Transform data for force graph - memoize to prevent unnecessary recalculations
  const forceGraphData = useMemo(() => {
    return {
      nodes: nodes.map((n) => ({ id: n.id, name: n.name, val: 10, color: n.color, properties: n.properties, type: n.type })),
      links: links.map((l) => ({ source: l.source, target: l.target, label: l.label })),
    };
  }, [nodes, links]);

  function getNodeColor(label: string): string {
    const colors: Record<string, string> = {
      Document: '#3b82f6',
      Entity: '#10b981',
      Experiment: '#f59e0b',
      Chemical: '#8b5cf6',
      Result: '#ef4444',
    };
    return colors[label] || '#6b7280';
  }

  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node as NodeData);
    // On node click, fetch relationships and expand neighboring nodes
    // Use functional updates to avoid dependency on nodes/links state
    (async () => {
      try {
        const rels: RelationshipInfo[] = await api.getEntityRelationships(node.id, projectId, 'both');
        
        setNodes(prevNodes => {
          const existingIds = new Set(prevNodes.map(n => n.id));
          const newNodesToAdd: NodeData[] = [];
          
          rels.forEach(r => {
            // Ensure source node exists
            if (!existingIds.has(r.source_id)) {
              newNodesToAdd.push({ 
                id: r.source_id, 
                name: r.source_name || r.source_id, 
                type: 'Entity', 
                color: getNodeColor('Entity'),
                properties: {}
              });
              existingIds.add(r.source_id);
            }
            // Ensure target node exists
            if (!existingIds.has(r.target_id)) {
              newNodesToAdd.push({ 
                id: r.target_id, 
                name: r.target_name || r.target_id, 
                type: 'Entity', 
                color: getNodeColor('Entity'),
                properties: {}
              });
              existingIds.add(r.target_id);
            }
          });
          
          // Only add new nodes, don't reset the graph
          return newNodesToAdd.length > 0 ? [...prevNodes, ...newNodesToAdd] : prevNodes;
        });
        
        setLinks(prevLinks => {
          const existingLinkKeys = new Set(
            prevLinks.map(l => `${l.source}-${l.target}-${l.label}`)
          );
          const newLinksToAdd: LinkData[] = [];
          
          rels.forEach(r => {
            const linkKey = `${r.source_id}-${r.target_id}-${r.type}`;
            if (!existingLinkKeys.has(linkKey)) {
              newLinksToAdd.push({ 
                source: r.source_id, 
                target: r.target_id, 
                label: r.type 
              });
              existingLinkKeys.add(linkKey);
            }
          });
          
          // Only add new links, don't reset the graph
          if (newLinksToAdd.length > 0) {
            // When adding new links, unlock positions temporarily to allow simulation to adjust
            if (fgRef.current) {
              fgRef.current.getGraphData().nodes.forEach((node: any) => {
                // Only unlock if it's not manually positioned
                if (node.fx !== undefined && node.fy !== undefined) {
                  const pos = nodePositions.current.get(node.id);
                  if (pos) {
                    // Keep locked but allow slight adjustment
                    node.fx = pos.x;
                    node.fy = pos.y;
                  }
                }
              });
            }
            return [...prevLinks, ...newLinksToAdd];
          }
          return prevLinks;
        });
      } catch (e) {
        console.error('Failed to expand relationships', e);
      }
    })();
  }, [projectId]); // Remove nodes dependency to prevent graph reset

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-gray-500">Loading graph...</div>
      </div>
    );
  }

  if (forceGraphData.nodes.length === 0) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-gray-500 mb-2">No graph data available yet.</p>
          <p className="text-sm text-gray-400">
            Upload and index documents to populate the knowledge graph.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full bg-gray-900">
      <ForceGraph2D
        ref={fgRef}
        graphData={forceGraphData}
        height={height}
        width={width}
        nodeLabel={(node: any) => `${node.name} (${node.type || 'Entity'})`}
        nodeColor={(node: any) => node.color}
        linkLabel={(link: any) => link.label}
        linkDirectionalArrowLength={6}
        linkDirectionalArrowRelPos={1}
        linkColor={() => '#64748b'}
        linkWidth={2}
        onNodeClick={handleNodeClick}
        onNodeDrag={(node: any) => {
          // Lock position during drag
          node.fx = node.x;
          node.fy = node.y;
        }}
        onNodeDragEnd={(node: any) => {
          // Keep position locked after drag
          node.fx = node.x;
          node.fy = node.y;
        }}
        onNodeRightClick={(node: any) => {
          // Unlock position on right click to allow simulation to reposition
          node.fx = undefined;
          node.fy = undefined;
        }}
        onEngineTick={() => {
          // Store positions of all nodes on each tick to preserve them when new nodes are added
          if (fgRef.current) {
            fgRef.current.getGraphData().nodes.forEach((node: any) => {
              if (node.x !== undefined && node.y !== undefined && !node.fx && !node.fy) {
                // Only store positions for nodes that aren't locked
                nodePositions.current.set(node.id, { x: node.x, y: node.y });
              }
            });
          }
        }}
        onEngineStop={() => {
          // When simulation stops, lock all node positions to prevent reset
          if (fgRef.current) {
            fgRef.current.getGraphData().nodes.forEach((node: any) => {
              if (node.x !== undefined && node.y !== undefined && !node.fx && !node.fy) {
                // Lock positions to prevent reset when new nodes are added
                node.fx = node.x;
                node.fy = node.y;
                nodePositions.current.set(node.id, { x: node.x, y: node.y });
              }
            });
          }
        }}
        cooldownTicks={100}
        enablePanInteraction={true}
        enableZoomInteraction={true}
        nodeCanvasObject={(node: any, ctx, globalScale) => {
          const label = node.name;
          const fontSize = 12 / globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          
          // Draw node circle
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
          ctx.fillStyle = node.color;
          ctx.fill();
          ctx.strokeStyle = '#fff';
          ctx.lineWidth = 2 / globalScale;
          ctx.stroke();

          // Draw label
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = '#fff';
          ctx.fillText(label, node.x, node.y + node.val + fontSize);
        }}
      />

      {/* Node Details Panel */}
      {selectedNode && (
        <div className="absolute top-4 right-4 w-80 bg-white rounded-lg shadow-lg border border-gray-200 max-h-[80vh] overflow-y-auto">
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">Node Details</h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="p-1 hover:bg-gray-100 rounded"
            >
              <X className="h-4 w-4 text-gray-500" />
            </button>
          </div>

          <div className="p-4 space-y-4">
            {/* Labels */}
            <div>
              <label className="text-xs font-semibold text-gray-500 uppercase">
                Type
              </label>
              <div className="flex flex-wrap gap-1 mt-1">
                <span className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded">
                  {selectedNode.type || 'Entity'}
                </span>
              </div>
            </div>

            {/* Properties */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-semibold text-gray-500 uppercase">
                  Properties
                </label>
              </div>

              <div className="space-y-2">
                {Object.entries(selectedNode.properties || {}).map(([key, value]) => (
                  <div key={key}>
                    <label className="text-xs text-gray-600 block mb-1">{key}</label>
                    <p className="text-sm text-gray-900 bg-gray-50 px-2 py-1 rounded">{String(value)}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-lg border border-gray-200 p-3">
        <h4 className="text-xs font-semibold text-gray-700 mb-2">Node Types</h4>
        <div className="space-y-1">
          {[
            { label: 'Document', color: '#3b82f6' },
            { label: 'Entity', color: '#10b981' },
            { label: 'Experiment', color: '#f59e0b' },
            { label: 'Chemical', color: '#8b5cf6' },
            { label: 'Result', color: '#ef4444' },
          ].map((type) => (
            <div key={type.label} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: type.color }}
              />
              <span className="text-xs text-gray-600">{type.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
