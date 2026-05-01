"""Context Engine — proactive context-aware retrieval (per-project)."""
from typing import Any, Dict, List, Optional
import logging

from sqlalchemy import func

from app.core.database import (
    Document,
    DocumentChunk,
    Node,
    get_project_session,
)
from app.core.qdrant_store import PROJECT_QDRANT_COLLECTION, get_qdrant_client
from app.services.llm import LLMService
from app.services.rerank import get_rerank_service
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams

logger = logging.getLogger(__name__)

_REPR_PATTERNS = (
    "RefItem(",
    "ContentLayer.",
    "parent=RefItem",
    "self_ref=",
    "cref=",
)


def _sanitize_passage_text(text: str) -> str:
    """Replace Docling/Python repr-like chunk text with a short fallback for UI."""
    if not text or not text.strip():
        return ""
    t = text.strip()
    for pattern in _REPR_PATTERNS:
        if pattern in t:
            return "Excerpt from document."
    return t[:500]


class ContextEngineService:
    """Proactive context-aware retrieval, scoped per project."""

    def __init__(self):
        self.llm_service = LLMService.get_instance()
        self.collection_name = PROJECT_QDRANT_COLLECTION
        self.reranker = get_rerank_service()
        # Cache: keyed by (project_id, doc_id, page_number)
        self._related_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._concepts_cache: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("ContextEngineService initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_document_structure(
        self, project_id: str, doc_id: str
    ) -> Optional[Dict[str, Any]]:
        session = get_project_session(project_id)
        try:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if not document:
                return None

            metadata = document.doc_metadata or {}
            return {
                "doc_id": doc_id,
                "filename": document.filename,
                "status": document.status,
                "uploaded_at": document.uploaded_at.isoformat() if document.uploaded_at else None,
                "structure": {
                    "title": metadata.get("title", document.filename),
                    "authors": metadata.get("authors", []),
                    "year": metadata.get("year"),
                    "abstract": metadata.get("abstract", ""),
                    "methodology": metadata.get("methodology", ""),
                    "key_findings": metadata.get("key_findings", []),
                    "limitations": metadata.get("limitations", []),
                    "paper_type": metadata.get("paper_type", "other"),
                    "page_count": metadata.get("page_count", 0),
                    "total_chars": metadata.get("total_chars", 0),
                },
            }
        finally:
            session.close()

    def _document_exists(self, project_id: str, doc_id: str) -> bool:
        session = get_project_session(project_id)
        try:
            return session.query(Document).filter(Document.id == doc_id).first() is not None
        finally:
            session.close()

    def _collection_exists(self, project_id: str) -> bool:
        try:
            qdrant_client = get_qdrant_client(project_id)
            names = [c.name for c in qdrant_client.get_collections().collections]
            return self.collection_name in names
        except Exception:
            return False

    async def get_related_passages(
        self,
        project_id: str,
        doc_id: str,
        text: str,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Find passages in OTHER documents related to the given text."""
        if not doc_id or not self._document_exists(project_id, doc_id):
            return []

        if not self._collection_exists(project_id):
            return []

        if getattr(self.llm_service, "is_generating", False):
            logger.info("Skipping background embedding request due to active LLM generation loop.")
            return []

        embedding = await self.llm_service.embed(text)

        must_not = [FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        search_filter = Filter(must_not=must_not)

        try:
            qdrant_client = get_qdrant_client(project_id)
            results = qdrant_client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                query_filter=search_filter,
                limit=limit * 2,
                search_params=SearchParams(exact=True),
            ).points
        except Exception as e:
            logger.warning(f"Qdrant query failed (collection may be empty): {e}")
            return []

        passages = []
        for r in results:
            payload = r.payload or {}
            meta = payload.get("metadata", {})
            r_doc_id = payload.get("doc_id", "")
            if r_doc_id == doc_id:
                continue
            raw_text = payload.get("text", "")[:500]
            passages.append({
                "text": _sanitize_passage_text(raw_text),
                "source": meta.get("filename", "Unknown"),
                "page": meta.get("page", 1),
                "doc_id": r_doc_id,
                "score": r.score,
                "chunk_id": payload.get("chunk_id", ""),
            })

        if self.reranker and passages:
            try:
                passages = await self.reranker.rerank(
                    query=text,
                    documents=passages,
                    top_n=limit,
                )
            except Exception as e:
                logger.warning(f"Reranking failed in context engine: {e}")
                passages = passages[:limit]
        else:
            passages = passages[:limit]

        filtered_passages = []
        for p in passages:
            score = p.get("score", 0.0)
            is_reranked = "rerank_score" in p

            if is_reranked:
                if score < 0.01:
                    continue
            else:
                if score < 0.25:
                    continue
                score = max(0.0, min(1.0, (score - 0.25) * 2.0))
                p["score"] = score

            filtered_passages.append(p)

        return filtered_passages

    async def get_context_suggestions(
        self,
        project_id: str,
        selected_text: Optional[str] = None,
        current_doc_id: Optional[str] = None,
        current_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "related_passages": [],
            "connected_concepts": [],
            "suggestions": [],
        }

        if selected_text and len(selected_text.strip()) > 10:
            if current_doc_id and self._document_exists(project_id, current_doc_id):
                results["related_passages"] = await self.get_related_passages(
                    project_id=project_id,
                    doc_id=current_doc_id,
                    text=selected_text,
                    limit=5,
                )
            results["connected_concepts"] = await self._find_connected_entities(
                project_id, selected_text
            )

        elif current_doc_id and current_page:
            cache_key = f"{project_id}:{current_doc_id}:{current_page}"
            if cache_key in self._related_cache:
                results["related_passages"] = self._related_cache[cache_key]
                results["connected_concepts"] = self._concepts_cache.get(cache_key, [])
                return results

            if not self._document_exists(project_id, current_doc_id):
                return results
            page_text = await self._get_page_text(project_id, current_doc_id, current_page)
            if page_text:
                related = await self.get_related_passages(
                    project_id=project_id,
                    doc_id=current_doc_id,
                    text=page_text[:500],
                    limit=3,
                )
                concepts = await self._find_connected_entities(
                    project_id, page_text[:500]
                )
                results["related_passages"] = related
                results["connected_concepts"] = concepts
                self._related_cache[cache_key] = related
                self._concepts_cache[cache_key] = concepts

        return results

    async def get_document_chunks(
        self,
        project_id: str,
        doc_id: str,
        page_number: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        session = get_project_session(project_id)
        try:
            query = session.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc_id
            )
            if page_number is not None:
                query = query.filter(DocumentChunk.page_number == page_number)

            query = query.order_by(DocumentChunk.chunk_index).limit(limit)
            chunks = query.all()

            return [
                {
                    "chunk_id": c.id,
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "page_number": c.page_number,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "metadata": c.chunk_metadata or {},
                }
                for c in chunks
            ]
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _find_connected_entities(
        self, project_id: str, text: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        session = get_project_session(project_id)
        try:
            words = sorted(set(
                w.lower()
                for w in text.split()
                if len(w) > 3 and w.isalpha()
            ))

            if not words:
                return []

            node_query = session.query(Node)

            matching_nodes = []
            for word in list(words)[:15]:
                nodes = node_query.filter(
                    func.lower(func.json_extract(Node.properties, "$.name")).contains(word)
                ).limit(5).all()
                matching_nodes.extend(nodes)

            seen_ids = set()
            unique_nodes = []
            for node in matching_nodes:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    props = node.properties or {}
                    unique_nodes.append({
                        "id": node.id,
                        "name": props.get("name", ""),
                        "type": node.label,
                        "document_id": node.document_id,
                        "confidence": props.get("confidence", 0.0),
                    })

            return unique_nodes[:limit]
        finally:
            session.close()

    async def _get_page_text(
        self, project_id: str, doc_id: str, page_number: int
    ) -> Optional[str]:
        session = get_project_session(project_id)
        try:
            chunks = (
                session.query(DocumentChunk)
                .filter(
                    DocumentChunk.document_id == doc_id,
                    DocumentChunk.page_number == page_number,
                )
                .order_by(DocumentChunk.chunk_index)
                .all()
            )
            if not chunks:
                return None
            return " ".join(c.text for c in chunks)
        finally:
            session.close()
