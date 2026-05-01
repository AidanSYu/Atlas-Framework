"""Graph service for querying nodes and edges (per-project SQLite backend)."""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import joinedload
from sqlalchemy import func

from app.core.database import (
    Document,
    Edge,
    Node,
    get_project_session,
)

import asyncio
from async_lru import alru_cache


class GraphService:
    """Manages knowledge graph queries — every call is project-scoped."""

    def __init__(self):
        pass

    def list_nodes(
        self,
        project_id: str,
        label: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        session = get_project_session(project_id)
        try:
            query = (
                session.query(Node)
                .join(Document, Node.document_id == Document.id)
                .filter(Document.status == "completed")
            )

            if label:
                query = query.filter(Node.label == label)
            if document_id:
                query = query.filter(Node.document_id == document_id)

            nodes = query.limit(limit).all()
            return [self._node_to_dict(n) for n in nodes]
        finally:
            session.close()

    def get_node_relationships(
        self, project_id: str, node_id: str, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        session = get_project_session(project_id)
        try:
            relationships = []

            if direction in ["outgoing", "both"]:
                query = (
                    session.query(Edge)
                    .options(joinedload(Edge.source_node), joinedload(Edge.target_node))
                    .filter(Edge.source_id == node_id)
                )
                relationships.extend(query.all())

            if direction in ["incoming", "both"]:
                query = (
                    session.query(Edge)
                    .options(joinedload(Edge.source_node), joinedload(Edge.target_node))
                    .filter(Edge.target_id == node_id)
                )
                relationships.extend(query.all())

            return [self._edge_to_dict(r) for r in relationships]
        finally:
            session.close()

    def get_node_types(self, project_id: str) -> List[Dict[str, Any]]:
        session = get_project_session(project_id)
        try:
            results = (
                session.query(Node.label, func.count(Node.id).label("count"))
                .group_by(Node.label)
                .all()
            )
            return [{"type": r[0], "count": r[1]} for r in results]
        finally:
            session.close()

    def get_full_graph(
        self,
        project_id: str,
        document_id: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        session = get_project_session(project_id)
        try:
            node_query = (
                session.query(Node)
                .join(Document, Node.document_id == Document.id)
                .filter(Document.status == "completed")
            )

            if document_id:
                node_query = node_query.filter(Node.document_id == document_id)

            nodes = node_query.limit(limit).all()
            node_dicts = [self._node_to_dict(n) for n in nodes]
            node_ids = {n.id for n in nodes}

            edge_dicts = []
            if node_ids:
                node_id_list = list(node_ids)
                edges = (
                    session.query(Edge)
                    .options(joinedload(Edge.source_node), joinedload(Edge.target_node))
                    .filter(Edge.source_id.in_(node_id_list), Edge.target_id.in_(node_id_list))
                    .all()
                )
                edge_dicts = [self._edge_to_dict(e) for e in edges]

            return {"nodes": node_dicts, "edges": edge_dicts}
        finally:
            session.close()

    @alru_cache(maxsize=32, ttl=300)
    async def get_full_graph_cached(
        self,
        project_id: str,
        document_id: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_full_graph(
                project_id=project_id,
                document_id=document_id,
                limit=limit,
            ),
        )

    @alru_cache(maxsize=32, ttl=300)
    async def get_rustworkx_subgraph(
        self,
        project_id: str,
        document_id: Optional[str] = None,
        limit: int = 500,
    ) -> Any:
        import rustworkx as rx

        loop = asyncio.get_running_loop()

        def _fetch_data():
            return self.get_full_graph(
                project_id=project_id, document_id=document_id, limit=limit
            )

        graph_data = await loop.run_in_executor(None, _fetch_data)

        G = rx.PyDiGraph()
        id_to_idx = {}

        for node in graph_data["nodes"]:
            idx = G.add_node({
                "id": node["id"],
                "name": node["name"],
                "type": node["type"],
                "description": node.get("description", ""),
                "document_id": node.get("document_id", ""),
            })
            id_to_idx[node["id"]] = idx

        for edge in graph_data["edges"]:
            src_id = edge["source_id"]
            tgt_id = edge["target_id"]

            if src_id in id_to_idx and tgt_id in id_to_idx:
                G.add_edge(
                    id_to_idx[src_id],
                    id_to_idx[tgt_id],
                    {
                        "type": edge["type"],
                        "source_name": edge.get("source_name", ""),
                        "target_name": edge.get("target_name", ""),
                    },
                )

        return G, id_to_idx

    def invalidate_cache(self):
        try:
            self.get_full_graph_cached.cache_clear()
        except Exception:
            pass
        try:
            self.get_rustworkx_subgraph.cache_clear()
        except Exception:
            pass

    def create_or_update_feedback_node(
        self,
        project_id: str,
        hit_id: str,
        epoch_id: str,
        result_name: str,
        result_value: float,
        unit: str,
        passed: bool,
        notes: str,
        smiles: Optional[str] = None,
    ) -> List[str]:
        """Create or update a knowledge graph node with bioassay feedback."""
        import uuid
        from datetime import datetime

        session = get_project_session(project_id)
        updated_nodes: List[str] = []

        try:
            existing_node = None

            nodes_by_hit = (
                session.query(Node)
                .filter(Node.properties.contains({"hit_id": hit_id}))
                .all()
            )
            if nodes_by_hit:
                existing_node = nodes_by_hit[0]
            elif smiles:
                nodes_by_smiles = (
                    session.query(Node)
                    .filter(Node.properties.contains({"smiles": smiles}))
                    .all()
                )
                if nodes_by_smiles:
                    existing_node = nodes_by_smiles[0]

            if existing_node:
                props = existing_node.properties or {}
                feedback_entry = {
                    "result_name": result_name,
                    "result_value": result_value,
                    "unit": unit,
                    "passed": passed,
                    "notes": notes,
                    "epoch_id": epoch_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                if "feedback_history" not in props:
                    props["feedback_history"] = []

                props["feedback_history"].append(feedback_entry)
                props["latest_feedback"] = feedback_entry

                existing_node.properties = props
                session.commit()
                updated_nodes.append(str(existing_node.id))
            else:
                node_id = str(uuid.uuid4())

                placeholder_doc_id = f"discovery_{project_id}"
                doc = session.query(Document).filter(Document.id == placeholder_doc_id).first()
                if not doc:
                    doc = session.query(Document).first()

                document_id = doc.id if doc else placeholder_doc_id

                feedback_entry = {
                    "result_name": result_name,
                    "result_value": result_value,
                    "unit": unit,
                    "passed": passed,
                    "notes": notes,
                    "epoch_id": epoch_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                new_node = Node(
                    id=node_id,
                    label=hit_id,
                    document_id=document_id,
                    project_id=project_id,
                    properties={
                        "hit_id": hit_id,
                        "epoch_id": epoch_id,
                        "smiles": smiles,
                        "name": f"Candidate {hit_id[:8]}",
                        "result_name": result_name,
                        "result_value": result_value,
                        "unit": unit,
                        "passed": passed,
                        "notes": notes,
                        "feedback_history": [feedback_entry],
                        "latest_feedback": feedback_entry,
                    },
                )
                session.add(new_node)
                session.commit()
                updated_nodes.append(node_id)

            return updated_nodes

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _node_to_dict(self, node: Node) -> Dict[str, Any]:
        props = node.properties or {}
        return {
            "id": str(node.id),
            "name": props.get("name", "Unknown"),
            "type": node.label,
            "description": props.get("description"),
            "document_id": str(node.document_id) if node.document_id else "",
        }

    def _edge_to_dict(self, edge: Edge) -> Dict[str, Any]:
        source = edge.source_node
        target = edge.target_node

        source_props = source.properties if source else {}
        target_props = target.properties if target else {}

        return {
            "id": str(edge.id),
            "source_id": str(edge.source_id),
            "source_name": source_props.get("name", "Unknown") if source else "Unknown",
            "target_id": str(edge.target_id),
            "target_name": target_props.get("name", "Unknown") if target else "Unknown",
            "type": edge.type,
            "context": edge.properties.get("context") if edge.properties else None,
        }
