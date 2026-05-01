"""Always-on Atlas Framework tools backed by the local knowledge substrate."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.core.database import Document, get_project_session
from app.services.graph import GraphService
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class CoreToolManifest(BaseModel):
    """Validated manifest for a built-in Atlas Framework tool."""

    schema_version: str = "1.0"
    name: str
    version: str = "1.0.0"
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 100
    tags: List[str] = Field(default_factory=list)


@dataclass
class RegisteredCoreTool:
    """Runtime record for an always-on Atlas core tool."""

    manifest: CoreToolManifest
    handler: Any


class SearchLiteratureTool:
    """Hybrid RAG query tool over Atlas' local knowledge substrate."""

    def __init__(self) -> None:
        self._service: Optional[RetrievalService] = None

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(arguments or {})
        runtime = dict(context or {})

        query = str(payload.get("query") or runtime.get("user_prompt") or "").strip()
        project_id = payload.get("project_id") or runtime.get("project_id")

        if not query:
            return {
                "status": "error",
                "summary": "search_literature requires a non-empty query.",
                "error": "missing_query",
            }

        if self._service is None:
            self._service = RetrievalService()

        result = await self._service.query_atlas(query, project_id=project_id)
        context_block = result.get("context") or {}
        vector_chunks = context_block.get("vector_chunks", [])
        graph_nodes = context_block.get("graph_nodes", [])
        graph_edges = context_block.get("graph_edges", [])

        evidence: List[Dict[str, Any]] = []
        for chunk in vector_chunks[:5]:
            metadata = chunk.get("metadata", {})
            evidence.append(
                {
                    "source": metadata.get("filename", "Unknown"),
                    "page": metadata.get("page"),
                    "excerpt": (chunk.get("text") or "")[:400],
                }
            )

        summary = result.get("answer") or (
            f"Retrieved {len(vector_chunks)} text chunks, "
            f"{len(graph_nodes)} graph nodes, and {len(graph_edges)} graph edges."
        )

        return {
            "status": result.get("status", "unknown"),
            "summary": summary,
            "answer": result.get("answer", ""),
            "evidence": evidence,
            "context_summary": {
                "vector_chunks": len(vector_chunks),
                "graph_nodes": len(graph_nodes),
                "graph_edges": len(graph_edges),
            },
        }


class QueryVectorDBTool:
    """Direct semantic retrieval over the Qdrant vector store."""

    def __init__(self) -> None:
        self._service: Optional[RetrievalService] = None

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(arguments or {})
        runtime = dict(context or {})

        query = str(payload.get("query") or runtime.get("user_prompt") or "").strip()
        project_id = payload.get("project_id") or runtime.get("project_id")
        limit = int(payload.get("limit") or 5)
        limit = max(1, min(limit, 20))

        if not query:
            return {
                "status": "error",
                "summary": "query_vector_db requires a non-empty query.",
                "error": "missing_query",
            }

        if self._service is None:
            self._service = RetrievalService()

        active_doc_ids = self._get_active_document_ids(project_id=project_id)
        if not active_doc_ids:
            return {
                "status": "no_documents",
                "summary": "No completed documents are available for semantic retrieval.",
                "matches": [],
            }

        query_embedding = await self._service._embed_text(query)
        loop = asyncio.get_running_loop()

        def _search() -> Any:
            return self._service.qdrant_client.query_points(
                collection_name=self._service.collection_name,
                query=query_embedding,
                limit=max(limit * 3, 10),
            ).points

        raw_results = await loop.run_in_executor(None, _search)
        matches: List[Dict[str, Any]] = []
        for item in raw_results:
            payload_block = item.payload or {}
            if payload_block.get("doc_id") not in active_doc_ids:
                continue
            matches.append(
                {
                    "chunk_id": str(item.id),
                    "doc_id": payload_block.get("doc_id"),
                    "score": float(item.score),
                    "text": payload_block.get("text", ""),
                    "metadata": payload_block.get("metadata", {}),
                }
            )
            if len(matches) >= limit:
                break

        if not matches:
            return {
                "status": "no_results",
                "summary": f"No semantic matches were found for '{query}'.",
                "matches": [],
            }

        top_sources = []
        for match in matches[:3]:
            metadata = match.get("metadata", {})
            source = metadata.get("filename")
            if source:
                top_sources.append(source)

        summary = (
            f"Found {len(matches)} semantic match(es) for '{query}'. "
            f"Top sources: {', '.join(top_sources) if top_sources else 'local corpus'}."
        )
        return {
            "status": "success",
            "summary": summary,
            "matches": matches,
        }

    @staticmethod
    def _get_active_document_ids(project_id: str) -> set[str]:
        session = get_project_session(project_id)
        try:
            query = session.query(Document).filter(Document.status == "completed")
            return {str(document.id) for document in query.all()}
        finally:
            session.close()


class WalkKnowledgeGraphTool:
    """Traverse the local Rustworkx knowledge graph from a seed query or node."""

    def __init__(self) -> None:
        self._service = GraphService()

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(arguments or {})
        runtime = dict(context or {})

        node_id = str(payload.get("node_id") or "").strip() or None
        query = str(payload.get("query") or runtime.get("user_prompt") or "").strip()
        project_id = payload.get("project_id") or runtime.get("project_id")
        depth = int(payload.get("depth") or 2)
        limit = int(payload.get("limit") or 25)
        graph_limit = int(payload.get("graph_limit") or 500)

        depth = max(1, min(depth, 4))
        limit = max(1, min(limit, 100))
        graph_limit = max(limit, min(graph_limit, 2000))

        graph, id_to_idx = await self._service.get_rustworkx_subgraph(
            project_id=project_id,
            limit=graph_limit,
        )
        if len(id_to_idx) == 0:
            return {
                "status": "empty_graph",
                "summary": "The knowledge graph does not contain any active nodes yet.",
                "nodes": [],
                "edges": [],
            }

        weighted_edges = self._weighted_edge_list(graph)
        seed_indices = self._resolve_seed_indices(
            graph=graph,
            id_to_idx=id_to_idx,
            node_id=node_id,
            query=query,
        )
        if not seed_indices:
            label = node_id or query or "the requested seed"
            return {
                "status": "no_seed_match",
                "summary": f"I could not find a graph seed matching '{label}'.",
                "nodes": [],
                "edges": [],
            }

        selected_indices, selected_edges = self._walk_graph(
            seed_indices=seed_indices,
            weighted_edges=weighted_edges,
            depth=depth,
            limit=limit,
        )

        nodes = [graph.get_node_data(index) for index in selected_indices]
        edges = [
            {
                "source": graph.get_node_data(source),
                "target": graph.get_node_data(target),
                "relationship": weight,
            }
            for source, target, weight in selected_edges
        ]

        anchor_names = [graph.get_node_data(index).get("name", "unknown") for index in seed_indices[:3]]
        summary = (
            f"Walked {len(nodes)} node(s) and {len(edges)} edge(s) from seed "
            f"{', '.join(anchor_names)} with depth={depth}."
        )
        return {
            "status": "success",
            "summary": summary,
            "seeds": anchor_names,
            "nodes": nodes,
            "edges": edges,
        }

    @staticmethod
    def _weighted_edge_list(graph: Any) -> List[Tuple[int, int, Any]]:
        if hasattr(graph, "weighted_edge_list"):
            return list(graph.weighted_edge_list())
        if hasattr(graph, "edge_list"):
            return [(source, target, {}) for source, target in graph.edge_list()]
        return []

    @staticmethod
    def _resolve_seed_indices(
        graph: Any,
        id_to_idx: Dict[str, int],
        node_id: Optional[str],
        query: str,
    ) -> List[int]:
        if node_id and node_id in id_to_idx:
            return [id_to_idx[node_id]]

        if not query:
            return []

        lowered = query.lower()
        matches: List[int] = []
        for node_identifier, index in id_to_idx.items():
            node = graph.get_node_data(index)
            haystacks = [
                str(node_identifier),
                str(node.get("id", "")),
                str(node.get("name", "")),
                str(node.get("type", "")),
                str(node.get("description", "")),
            ]
            if any(lowered in value.lower() for value in haystacks if value):
                matches.append(index)
        return matches[:5]

    @staticmethod
    def _walk_graph(
        seed_indices: List[int],
        weighted_edges: List[Tuple[int, int, Any]],
        depth: int,
        limit: int,
    ) -> Tuple[List[int], List[Tuple[int, int, Any]]]:
        adjacency: Dict[int, List[Tuple[int, int, Any]]] = {}
        for source, target, weight in weighted_edges:
            adjacency.setdefault(source, []).append((source, target, weight))
            adjacency.setdefault(target, []).append((source, target, weight))

        visited: List[int] = []
        seen = set(seed_indices)
        edge_keys = set()
        selected_edges: List[Tuple[int, int, Any]] = []
        queue: deque[Tuple[int, int]] = deque((seed, 0) for seed in seed_indices)

        while queue and len(visited) < limit:
            current, level = queue.popleft()
            if current not in visited:
                visited.append(current)

            if level >= depth:
                continue

            for source, target, weight in adjacency.get(current, []):
                neighbor = target if source == current else source
                edge_key = (source, target, json.dumps(weight, sort_keys=True, default=str))
                if edge_key not in edge_keys:
                    edge_keys.add(edge_key)
                    selected_edges.append((source, target, weight))
                if neighbor not in seen and len(seen) < limit:
                    seen.add(neighbor)
                    queue.append((neighbor, level + 1))

        allowed = set(visited)
        filtered_edges = [
            edge
            for edge in selected_edges
            if edge[0] in allowed and edge[1] in allowed
        ]
        return visited, filtered_edges


class GetTraceabilitySubgraphTool:
    """Extract a bidirectional neighborhood from a root node id for PROV auditing.

    Output is shaped for direct consumption by the `traceability_compliance`
    plugin's `graph_data` argument. A typical tool chain is:

        get_traceability_subgraph(root_node_id="board-8842")
          -> traceability_compliance(mode="report", graph_data=<result>)
    """

    def __init__(self) -> None:
        self._service = GraphService()

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(arguments or {})
        runtime = dict(context or {})

        root_node_id = str(payload.get("root_node_id") or "").strip()
        if not root_node_id:
            return {
                "status": "error",
                "summary": "root_node_id is required.",
                "root_node_id": "",
                "nodes": [],
                "edges": [],
            }

        project_id = payload.get("project_id") or runtime.get("project_id")
        max_depth = max(1, min(int(payload.get("max_depth") or 6), 12))
        graph_limit = max(50, min(int(payload.get("graph_limit") or 500), 5000))

        graph, id_to_idx = await self._service.get_rustworkx_subgraph(
            project_id=project_id, limit=graph_limit,
        )
        if len(id_to_idx) == 0:
            return {
                "status": "empty_graph",
                "summary": "The knowledge graph does not contain any active nodes yet.",
                "root_node_id": root_node_id,
                "nodes": [],
                "edges": [],
            }
        if root_node_id not in id_to_idx:
            return {
                "status": "no_root",
                "summary": f"Root node '{root_node_id}' was not found in the graph.",
                "root_node_id": root_node_id,
                "nodes": [],
                "edges": [],
            }

        # Bidirectional BFS — traceability must walk upstream and downstream.
        root_idx = id_to_idx[root_node_id]
        depth: Dict[int, int] = {root_idx: 0}
        order: List[int] = [root_idx]
        visited_edges: List[Tuple[int, int, Dict[str, Any]]] = []
        edge_keys: set = set()
        cursor = 0
        while cursor < len(order):
            cur = order[cursor]
            cursor += 1
            if depth[cur] >= max_depth:
                continue
            neighbors: List[Tuple[int, int, Dict[str, Any]]] = [
                (s, t, d) for s, t, d in graph.out_edges(cur)
            ] + [
                (s, t, d) for s, t, d in graph.in_edges(cur)
            ]
            for src_idx, tgt_idx, data in neighbors:
                other = tgt_idx if src_idx == cur else src_idx
                key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx), id(data))
                if key in edge_keys:
                    continue
                edge_keys.add(key)
                visited_edges.append((src_idx, tgt_idx, data))
                if other not in depth:
                    depth[other] = depth[cur] + 1
                    order.append(other)

        # Reshape into the dict form `traceability_compliance` expects.
        shaped_nodes: List[Dict[str, Any]] = []
        for idx in order:
            node_data = graph.get_node_data(idx)
            metadata = {k: v for k, v in node_data.items() if k not in ("id", "type")}
            shaped_nodes.append({
                "id": node_data["id"],
                "type": node_data.get("type", "Entity"),
                "metadata": metadata,
            })
        shaped_edges: List[Dict[str, Any]] = []
        for src_idx, tgt_idx, data in visited_edges:
            src = graph.get_node_data(src_idx)
            tgt = graph.get_node_data(tgt_idx)
            metadata = {k: v for k, v in data.items() if k != "type"}
            shaped_edges.append({
                "source": src["id"],
                "target": tgt["id"],
                "relation": data.get("type", "wasDerivedFrom"),
                "metadata": metadata,
            })

        return {
            "status": "success",
            "summary": (
                f"Traced {len(shaped_nodes)} node(s) and {len(shaped_edges)} edge(s) "
                f"from '{root_node_id}' at depth<={max_depth}."
            ),
            "root_node_id": root_node_id,
            "nodes": shaped_nodes,
            "edges": shaped_edges,
        }


class CoreToolRegistry:
    """Registry for Atlas' always-on knowledge substrate tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredCoreTool] = {}
        self._register_defaults()

    def refresh(self) -> None:
        """Core tools are static; refresh is a no-op kept for API symmetry."""

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": record.manifest.name,
                "description": record.manifest.description,
                "priority": record.manifest.priority,
                "input_schema": record.manifest.input_schema,
                "output_schema": record.manifest.output_schema,
                "tags": record.manifest.tags,
                "source": "atlas-core",
                "source_type": "core",
                "loaded": True,
                "load_error": None,
            }
            for record in self._ordered_tools()
        ]

    def tool_names(self) -> List[str]:
        return [record.manifest.name for record in self._ordered_tools()]

    def build_toolkit_prompt(self) -> str:
        tool_blocks: List[str] = []
        for record in self._ordered_tools():
            manifest = record.manifest
            required = manifest.input_schema.get("required", []) if manifest.input_schema else []
            tool_blocks.append(
                "\n".join(
                    [
                        f"TOOL: {manifest.name}",
                        "KIND: core_tool",
                        f"DESCRIPTION: {manifest.description}",
                        f"REQUIRED PARAMETERS: {json.dumps(required, ensure_ascii=True)}",
                        f"INPUT SCHEMA: {json.dumps(manifest.input_schema, ensure_ascii=True)}",
                    ]
                )
            )
        return "\n\n".join(tool_blocks)

    async def invoke(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = self._tools.get(tool_name)
        if record is None:
            raise ValueError(f"Unknown Atlas core tool: {tool_name}")

        try:
            result = await record.handler.invoke(arguments or {}, context or {})
        except Exception as exc:
            logger.error("Atlas core tool '%s' failed: %s", tool_name, exc, exc_info=True)
            return {
                "summary": f"{tool_name} failed: {exc}",
                "error": str(exc),
                "tool": tool_name,
            }

        if isinstance(result, dict):
            if "summary" not in result:
                result["summary"] = self._summarize_result(tool_name, result)
            return result

        return {
            "summary": f"{tool_name} returned a non-dict result",
            "raw_result": result,
        }

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def _register_defaults(self) -> None:
        self._register(
            CoreToolManifest(
                name="search_literature",
                description=(
                    "Query Atlas hybrid RAG across SQLite, Qdrant, BM25, and the knowledge graph. "
                    "Use this first whenever you need grounded evidence from the local corpus."
                ),
                priority=200,
                tags=["retrieval", "hybrid-rag", "grounding", "core"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Grounded literature or project-memory query.",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project scope for retrieval.",
                        },
                    },
                    "required": ["query"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                        "answer": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "object"}},
                        "context_summary": {"type": "object"},
                    },
                },
            ),
            SearchLiteratureTool(),
        )
        self._register(
            CoreToolManifest(
                name="query_vector_db",
                description=(
                    "Run direct semantic retrieval against the local Qdrant vector store. "
                    "Use this when you want the raw nearest-neighbor passages without the full hybrid synthesis."
                ),
                priority=180,
                tags=["retrieval", "semantic-search", "qdrant", "core"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Semantic search query for the vector store.",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project scope for retrieval.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of vector matches to return.",
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                        "matches": {"type": "array", "items": {"type": "object"}},
                    },
                },
            ),
            QueryVectorDBTool(),
        )
        self._register(
            CoreToolManifest(
                name="walk_knowledge_graph",
                description=(
                    "Traverse the always-on Rustworkx knowledge graph from a seed query or node id. "
                    "Use this to inspect connected entities, relationships, and neighborhood structure."
                ),
                priority=170,
                tags=["knowledge-graph", "rustworkx", "core"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Seed text used to find matching graph nodes.",
                        },
                        "node_id": {
                            "type": "string",
                            "description": "Optional exact node id to anchor the walk.",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project scope for traversal.",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Maximum neighborhood depth to traverse.",
                            "minimum": 1,
                            "maximum": 4,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of nodes to include in the traversal result.",
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                        "seeds": {"type": "array", "items": {"type": "string"}},
                        "nodes": {"type": "array", "items": {"type": "object"}},
                        "edges": {"type": "array", "items": {"type": "object"}},
                    },
                },
            ),
            WalkKnowledgeGraphTool(),
        )
        self._register(
            CoreToolManifest(
                name="get_traceability_subgraph",
                description=(
                    "Extract a bidirectional neighborhood from the Rustworkx graph rooted at "
                    "an exact node id, shaped as a provenance bundle input for the "
                    "traceability_compliance plugin. Use this when a user asks to audit, "
                    "trace, or certify a specific lot / board / batch by id."
                ),
                priority=165,
                tags=["knowledge-graph", "traceability", "provenance", "core"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "root_node_id": {
                            "type": "string",
                            "description": "Exact graph node id to anchor the walk (e.g. 'board-8842').",
                        },
                        "project_id": {
                            "type": "string",
                            "description": "Optional project scope for traversal.",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum bidirectional traversal depth. Default 6.",
                            "minimum": 1,
                            "maximum": 12,
                        },
                        "graph_limit": {
                            "type": "integer",
                            "description": "Maximum nodes to fetch from the substrate before walking.",
                            "minimum": 50,
                            "maximum": 5000,
                        },
                    },
                    "required": ["root_node_id"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                        "root_node_id": {"type": "string"},
                        "nodes": {"type": "array", "items": {"type": "object"}},
                        "edges": {"type": "array", "items": {"type": "object"}},
                    },
                },
            ),
            GetTraceabilitySubgraphTool(),
        )

    def _register(self, manifest: CoreToolManifest, handler: Any) -> None:
        self._tools[manifest.name] = RegisteredCoreTool(manifest=manifest, handler=handler)

    def _ordered_tools(self) -> List[RegisteredCoreTool]:
        return sorted(
            self._tools.values(),
            key=lambda record: (-record.manifest.priority, record.manifest.name),
        )

    @staticmethod
    def _summarize_result(tool_name: str, payload: Dict[str, Any]) -> str:
        keys = ", ".join(sorted(payload.keys())[:6])
        return f"{tool_name} completed. Keys: {keys or 'none'}."


_core_tool_registry: Optional[CoreToolRegistry] = None


def get_core_tool_registry() -> CoreToolRegistry:
    """Return the Atlas Framework core tool registry singleton."""
    global _core_tool_registry
    if _core_tool_registry is None:
        _core_tool_registry = CoreToolRegistry()
    return _core_tool_registry
