"""Always-on Atlas Framework tools backed by the local knowledge substrate."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.core.database import Document, TaskEvent, get_project_session
from app.services.graph import GraphService
from app.services.memory import MemoryService
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class CoreToolManifest(BaseModel):
    """Validated manifest for a built-in Atlas Framework tool."""

    model_config = ConfigDict(protected_namespaces=())

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

        if not project_id:
            return {
                "status": "error",
                "summary": "query_vector_db requires a project_id (per-project corpus).",
                "error": "missing_project_id",
                "matches": [],
            }

        loop = asyncio.get_running_loop()
        active_doc_ids = await loop.run_in_executor(
            None, self._get_active_document_ids, project_id
        )
        if not active_doc_ids:
            return {
                "status": "no_documents",
                "summary": "No completed documents are available for semantic retrieval.",
                "matches": [],
            }

        query_embedding = await self._service._embed_text(query)

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


class RecallMemoryTool:
    """Recall lessons from prior completed campaigns (cross-campaign memory)."""

    def __init__(self) -> None:
        self._service = MemoryService()

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(arguments or {})
        runtime = dict(context or {})
        query = str(payload.get("query") or runtime.get("user_prompt") or "").strip()
        project_id = payload.get("project_id") or runtime.get("project_id")
        limit = max(1, min(int(payload.get("limit") or 5), 20))

        if not project_id:
            return {
                "status": "error",
                "summary": "recall_memory requires a project_id.",
                "error": "missing_project_id",
                "memories": [],
            }

        loop = asyncio.get_running_loop()
        memories = await loop.run_in_executor(
            None, lambda: self._service.recall(project_id=project_id, query=query, limit=limit)
        )
        if not memories:
            return {
                "status": "no_memories",
                "summary": "No prior-campaign memories match this query yet.",
                "memories": [],
            }
        top = memories[0]
        summary = (
            f"Recalled {len(memories)} prior-campaign memory(ies). Most relevant: "
            f"\"{top['goal'][:80]}\" ({top.get('created_at') or '?'}). Treat as hypotheses."
        )
        return {"status": "success", "summary": summary, "memories": memories}


class RecallPriorOutcomesTool:
    """Recall prior plans + the dead-ends to skip for a related target."""

    def __init__(self) -> None:
        self._service = MemoryService()

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(arguments or {})
        runtime = dict(context or {})
        target = str(
            payload.get("target") or payload.get("query") or runtime.get("user_prompt") or ""
        ).strip()
        project_id = payload.get("project_id") or runtime.get("project_id")
        limit = max(1, min(int(payload.get("limit") or 5), 20))

        if not project_id:
            return {
                "status": "error",
                "summary": "recall_prior_outcomes requires a project_id.",
                "error": "missing_project_id",
            }
        if not target:
            return {
                "status": "error",
                "summary": "recall_prior_outcomes requires a target.",
                "error": "missing_target",
            }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._service.recall_prior_outcomes(
                project_id=project_id, target=target, limit=limit
            ),
        )


class ReadArtifactTool:
    """Re-read the full payload of an earlier tool result that was projected
    down for the model's context (restorable offload). The full result lives in
    the task event log keyed by the tool call_id surfaced in the projection."""

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(arguments or {})
        runtime = dict(context or {})
        project_id = payload.get("project_id") or runtime.get("project_id")
        task_id = payload.get("task_id") or runtime.get("task_id")
        ref = str(
            payload.get("artifact_id") or payload.get("call_id") or payload.get("ref") or ""
        ).strip()
        if ref.startswith("artifact:"):
            ref = ref[len("artifact:"):]

        if not project_id or not ref:
            return {
                "status": "error",
                "summary": "read_artifact requires project_id and an artifact_id.",
                "error": "missing_args",
            }

        loop = asyncio.get_running_loop()
        full = await loop.run_in_executor(None, self._fetch, project_id, task_id, ref)
        if full is None:
            return {"status": "not_found", "summary": f"No artifact found for id '{ref}'."}
        return {"status": "success", "summary": f"Full payload for artifact '{ref}'.", "artifact": full}

    @staticmethod
    def _fetch(project_id: str, task_id: Optional[str], call_id: str) -> Optional[Any]:
        session = get_project_session(project_id)
        try:
            query = session.query(TaskEvent).filter(
                TaskEvent.event_type == "TOOL_EXECUTION_RESULT"
            )
            if task_id:
                query = query.filter(TaskEvent.task_id == task_id)
            for evt in query.order_by(TaskEvent.sequence.desc()).limit(300):
                if (evt.payload or {}).get("call_id") == call_id:
                    return (evt.payload or {}).get("output")
            return None
        finally:
            session.close()


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
                "status": "error",
                "summary": f"{tool_name} failed: {exc}",
                "error": str(exc),
                "tool": tool_name,
            }

        if isinstance(result, dict):
            if "summary" not in result:
                result["summary"] = self._summarize_result(tool_name, result)
            return result

        return {
            "status": "error",
            "summary": f"{tool_name} returned a non-dict result",
            "error": "non_dict_result",
            "tool": tool_name,
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
                name="recall_prior_outcomes",
                description=(
                    "Recall what PRIOR campaigns on a related target already learned — reusable "
                    "plan skeletons and the dead-ends to SKIP — before you start exploring. Call "
                    "this early on any campaign so you don't re-derive facts or re-run chains a "
                    "previous run already ruled out. Results are unverified hypotheses; cite them."
                ),
                priority=175,
                tags=["memory", "compounding", "campaign", "core"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "The current target / goal to find related prior campaigns for.",
                        },
                        "project_id": {"type": "string", "description": "Optional project scope."},
                        "limit": {
                            "type": "integer",
                            "description": "Max prior campaigns to recall.",
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["target"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                        "prior_campaigns": {"type": "array", "items": {"type": "object"}},
                        "dead_ends_to_skip": {"type": "array", "items": {"type": "object"}},
                    },
                },
            ),
            RecallPriorOutcomesTool(),
        )
        self._register(
            CoreToolManifest(
                name="recall_memory",
                description=(
                    "Recall lessons (key facts, outcomes) from prior completed campaigns by "
                    "semantic relevance to a query. Use to avoid re-discovering what the lab "
                    "already knows. Results are unverified hypotheses with provenance; cite them."
                ),
                priority=165,
                tags=["memory", "recall", "core"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to recall from prior-campaign memory.",
                        },
                        "project_id": {"type": "string", "description": "Optional project scope."},
                        "limit": {
                            "type": "integer",
                            "description": "Max memories to recall.",
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
                        "memories": {"type": "array", "items": {"type": "object"}},
                    },
                },
            ),
            RecallMemoryTool(),
        )
        self._register(
            CoreToolManifest(
                name="read_artifact",
                description=(
                    "Re-read the FULL result of an earlier tool call that was summarized for "
                    "your context. Pass the artifact_id (the call ref shown in a truncated tool "
                    "result) to retrieve every field you couldn't see inline."
                ),
                priority=90,
                tags=["context", "offload", "core"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "artifact_id": {
                            "type": "string",
                            "description": "The call ref shown in a truncated/offloaded tool result.",
                        },
                        "project_id": {"type": "string", "description": "Optional project scope."},
                    },
                    "required": ["artifact_id"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                        "artifact": {"type": "object"},
                    },
                },
            ),
            ReadArtifactTool(),
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
