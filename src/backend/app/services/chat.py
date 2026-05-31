"""
Chat Service - Grounded chat interface over the Atlas retrieval substrate.

Single-orchestrator: uses RetrievalService for document + graph-backed Q&A.
The legacy "librarian" / "cortex" personas were removed.
"""
from typing import Dict, Any, Optional

from app.services.retrieval import RetrievalService


class ChatService:
    """Handles chat interactions with the local LLM."""

    def __init__(self):
        self.retrieval_service = RetrievalService()

    async def chat(
        self,
        user_question: str,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a grounded chat query using hybrid RAG retrieval.

        Returns:
            Dict with answer, reasoning, citations, relationships, context_sources
        """
        result = await self.retrieval_service.query_atlas(
            user_question,
            project_id=project_id,
        )
        context = result.get("context") or {
            "vector_chunks": [],
            "graph_nodes": [],
            "graph_edges": [],
        }
        vector_chunks = context.get("vector_chunks", [])
        graph_edges = context.get("graph_edges", [])
        graph_nodes = context.get("graph_nodes", [])

        # Format citations
        citations = []
        for chunk in vector_chunks[:5]:
            metadata = chunk.get("metadata", {})
            page = metadata.get("page")
            citations.append(
                {
                    "source": metadata.get("filename", "Unknown"),
                    "doc_id": chunk.get("doc_id"),
                    "page": page if page is not None else 1,
                    "text": (chunk.get("text") or "")[:200]
                    + ("..." if len(chunk.get("text", "")) > 200 else ""),
                }
            )

        # Format relationships
        relationships = []
        for edge in graph_edges[:10]:
            relationships.append(
                {
                    "source": edge.get("source_id", ""),
                    "target": edge.get("target_id", ""),
                    "type": edge.get("type", ""),
                    "properties": edge.get("properties", {}),
                }
            )

        answer = result.get("answer") or ""
        status = result.get("status")
        if not answer and status in (
            "no_results",
            "no_documents",
            "extraction_failed",
            "generation_failed",
        ):
            answer = (
                "I couldn't find an answer from your documents. "
                "Try rephrasing your question or adding more relevant documents."
            )

        return {
            "answer": answer,
            "reasoning": (
                f"Reviewed {len(vector_chunks)} text chunks, "
                f"{len(graph_nodes)} graph nodes, and {len(graph_edges)} graph edges."
            ),
            "citations": citations,
            "relationships": relationships,
            "context_sources": {
                "vector_chunks": len(vector_chunks),
                "graph_nodes": len(graph_nodes),
                "graph_edges": len(graph_edges),
            },
        }
