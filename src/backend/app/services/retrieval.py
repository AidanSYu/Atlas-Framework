"""
Retrieval Service - Hybrid RAG retrieval logic.

Atlas 3.0: Implements hybrid search combining:
1. Vector search (semantic similarity via embedded Qdrant)
2. BM25 sparse search (keyword matching via bm25s)
3. Reciprocal Rank Fusion (RRF) to merge vector + BM25 results
4. Entity-based matching (knowledge graph nodes)
5. Exact text matching (dates, numbers, specific phrases)
6. Graph expansion (1-hop neighborhood)
7. FlashRank cross-encoder reranking
8. Document filtering (only active/completed documents)

Production Desktop Sidecar: SQLite + embedded Qdrant + BM25 + bundled LLMs.
"""
from typing import List, Dict, Any, Optional, Set
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, func
import json
import logging
import asyncio

from app.core.database import (
    Document,
    DocumentChunk,
    Edge,
    Node,
    get_project_session,
)
from app.core.config import settings
from app.services.llm import LLMService
from app.core.qdrant_store import PROJECT_QDRANT_COLLECTION, get_qdrant_client
from app.services.rerank import get_rerank_service
from app.services.bm25_index import get_bm25_service, rrf_fuse
logger = logging.getLogger(__name__)


class RetrievalService:
    """Handles hybrid retrieval from vector store, BM25, and knowledge graph."""

    def __init__(self):
        self.llm_service = LLMService.get_instance()
        # Per-project Qdrant client + collection are resolved at method-call time.
        self.collection_name = PROJECT_QDRANT_COLLECTION
        self.reranker = get_rerank_service()
        # Atlas 3.0: BM25 sparse retrieval
        self.bm25_service = get_bm25_service()
        logger.info("RetrievalService initialized")

    async def _embed_text(self, text: str) -> List[float]:
        """Generate embedding using bundled LLM service."""
        return await self.llm_service.embed(text)

    async def _extract_query_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities, dates, and key terms from query using LLM."""
        prompt = f"""Extract key information from this query. Focus on:
- Entities: people, places, organizations, concepts
- Dates and time periods
- Specific numbers, measurements, or values
- Key phrases that should be matched exactly

Query: {query}

Return ONLY a JSON object with:
{{
  "entities": ["entity1", "entity2"],
  "dates": ["1920", "1930"],
  "date_ranges": [{{"start": "1920", "end": "1930"}}],
  "key_phrases": ["exact phrase to match"]
}}

If no dates/entities found, return empty arrays."""

        try:
            response_text = await self.llm_service.generate(
                prompt=prompt, temperature=0.1, max_tokens=512
            )
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except Exception as e:
            logger.debug(f"Entity extraction failed: {e}")

        return {"entities": [], "dates": [], "date_ranges": [], "key_phrases": []}

    async def _get_active_document_ids(self, session: Session) -> Set[str]:
        """Get set of active (completed) document IDs."""
        active_docs = session.query(Document).filter(Document.status == "completed").all()
        return {str(doc.id) for doc in active_docs}

    @staticmethod
    def _build_answer_prompt(
        user_question: str,
        context_str: str,
    ) -> tuple[str, str]:
        system_msg = (
            "You are Atlas, a grounded research assistant. Answer the user's question "
            "based only on the provided project context (documents and graph relationships). "
            "You may synthesize across sources, but you must clearly separate direct "
            "evidence from higher-level inference. If the answer is not in the context, "
            'say "I cannot find this information in the available documents."'
        )
        instructions = [
            "Base your answer only on the provided context.",
            "Cite the source document name and page number for every factual claim.",
            "Use this exact citation format: [Source: filename.pdf, Page: X]",
            "When you connect ideas across sources or graph relationships, label that as synthesis or inference.",
            "Do not claim that you can run plugins, experiments, or tools from this chat.",
        ]

        user_msg = (
            f"QUESTION: {user_question}\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            + "\n".join(f"{index}. {instruction}" for index, instruction in enumerate(instructions, start=1))
        )
        return system_msg, user_msg

    async def query_atlas(
        self,
        user_question: str,
        project_id: str,
    ) -> Dict[str, Any]:
        """
        Main retrieval function - implements hybrid RAG workflow.

        Args:
            user_question: The query string
            project_id: Project scope (required)

        Returns:
            Dict with answer, context (vector_chunks, graph_nodes, graph_edges)
        """
        if not project_id:
            raise ValueError("project_id is required for retrieval")
        qdrant_client = get_qdrant_client(project_id)
        session = get_project_session(project_id)
        try:
            # Step 0: Extract entities/dates from query
            query_info = await self._extract_query_entities(user_question)
            # Use .get() with defaults to handle malformed LLM output
            entities = query_info.get('entities', [])
            dates = query_info.get('dates', [])
            logger.info(
                f"Extracted from query: entities={entities}, dates={dates}"
            )

            # Step 0.5: Get active document IDs (optionally scoped to project)
            doc_query = session.query(Document).filter(Document.status == "completed")
            if project_id:
                doc_query = doc_query.filter(Document.project_id == project_id)
            active_docs = doc_query.all()
            active_doc_ids = {str(doc.id) for doc in active_docs}

            if not active_doc_ids:
                return {
                    "status": "no_documents",
                    "answer": "",
                    "context": {"vector_chunks": [], "graph_nodes": [], "graph_edges": []},
                }

            loop = asyncio.get_running_loop()
            vector_results = []
            try:
                query_embedding = await self._embed_text(user_question)

                def _qdrant_search():
                    return qdrant_client.query_points(
                        collection_name=self.collection_name,
                        query=query_embedding,
                        limit=20,
                    ).points

                vector_results = await loop.run_in_executor(None, _qdrant_search)
                vector_results = [
                    r for r in vector_results if (r.payload or {}).get("doc_id") in active_doc_ids
                ]
            except Exception as e:
                logger.warning(f"Vector search failed (non-fatal): {e}")

            # Step 2: Entity-based matching
            entity_matched_chunks = []
            if entities:
                for entity_name in entities:
                    # SQLite JSON: use json_extract for property queries
                    node_query = session.query(Node).join(
                        Document, Node.document_id == Document.id
                    ).filter(Document.status == "completed")
                    if project_id:
                        node_query = node_query.filter(Node.project_id == project_id)

                    # Task 0.2: Fix Entity Matching Performance
                    # Use SQLite json_extract for SQL-level filtering (no Python loop)
                    # This replaces the old logic that loaded 100+ nodes into memory
                    matching_nodes = node_query.filter(
                        func.lower(
                            func.json_extract(Node.properties, '$.name')
                        ).contains(entity_name.lower())
                    ).limit(20).all()

                    seen_chunk_ids = set()
                    for node in matching_nodes:
                        chunk_id = (node.properties or {}).get("chunk_id")
                        if chunk_id and chunk_id not in seen_chunk_ids:
                            seen_chunk_ids.add(chunk_id)
                            try:
                                chunk_result = await loop.run_in_executor(
                                    None,
                                    lambda cid=chunk_id: qdrant_client.retrieve(
                                        collection_name=self.collection_name, ids=[cid]
                                    ),
                                )
                                if chunk_result and chunk_result[0].payload.get("doc_id") in active_doc_ids:
                                    payload = chunk_result[0].payload
                                    entity_matched_chunks.append(
                                        {
                                            "text": payload.get("text", ""),
                                            "doc_id": payload.get("doc_id"),
                                            "metadata": {**payload.get("metadata", {}), "chunk_id": chunk_id},
                                            "relevance_score": 0.95,
                                            "match_type": "entity",
                                        }
                                    )
                            except Exception as e:
                                logger.debug(f"Error retrieving chunk {chunk_id}: {e}")

            # Step 3: Exact text matching for dates/key phrases
            exact_matched_chunks = []
            key_phrases = query_info.get('key_phrases', [])
            search_terms = [str(term).strip() for term in dates + key_phrases if str(term).strip()]
            if search_terms:
                seen_exact_chunks: set[str] = set()
                exact_query = (
                    session.query(DocumentChunk, Document)
                    .join(Document, DocumentChunk.document_id == Document.id)
                    .filter(Document.status == "completed")
                )
                if project_id:
                    exact_query = exact_query.filter(Document.project_id == project_id)

                for term in search_terms[:10]:
                    try:
                        term_matches = (
                            exact_query
                            .filter(func.lower(DocumentChunk.text).contains(term.lower()))
                            .limit(20)
                            .all()
                        )

                        for chunk, document in term_matches:
                            chunk_id = str(chunk.id)
                            if chunk_id in seen_exact_chunks:
                                continue
                            seen_exact_chunks.add(chunk_id)
                            exact_matched_chunks.append(
                                {
                                    "text": chunk.text,
                                    "doc_id": str(document.id),
                                    "metadata": {
                                        **(chunk.chunk_metadata or {}),
                                        "chunk_id": chunk_id,
                                        "filename": document.filename,
                                        "page": chunk.page_number,
                                    },
                                    "relevance_score": 0.98,
                                    "match_type": "exact",
                                    "matched_term": term,
                                }
                            )
                    except Exception as e:
                        logger.debug(f"Error searching exact matches for term '{term}': {e}")

            # Step 3.5 (Atlas 3.0): BM25 sparse keyword search
            bm25_chunks = []
            if self.bm25_service.is_available() and self.bm25_service.corpus_size > 0:
                try:
                    bm25_results = self.bm25_service.search(
                        query=user_question,
                        top_k=20,
                        doc_ids=active_doc_ids,
                    )
                    bm25_chunks = bm25_results
                    logger.info(f"BM25 returned {len(bm25_chunks)} results")
                except Exception as e:
                    logger.warning(f"BM25 search failed (non-fatal): {e}")

            # Step 4: Combine results using Reciprocal Rank Fusion (RRF)
            # Atlas 3.0: RRF fuses vector, BM25, entity, and exact match results

            # Convert vector results to standard format
            vector_chunk_list = []
            for result in vector_results[:15]:
                payload = result.payload
                chunk_id = str(result.id)
                vector_chunk_list.append({
                    "text": payload.get("text", ""),
                    "doc_id": payload.get("doc_id"),
                    "metadata": {**payload.get("metadata", {}), "chunk_id": chunk_id},
                    "relevance_score": float(result.score),
                    "match_type": "vector",
                })

            # Use RRF to fuse all result lists
            if bm25_chunks:
                candidate_chunks = rrf_fuse(
                    vector_chunk_list,
                    bm25_chunks,
                    entity_matched_chunks,
                    exact_matched_chunks,
                    k=60,
                )
                logger.info(f"RRF fused {len(vector_chunk_list)} vector + {len(bm25_chunks)} BM25 + "
                           f"{len(entity_matched_chunks)} entity + {len(exact_matched_chunks)} exact "
                           f"-> {len(candidate_chunks)} candidates")
            else:
                # Fallback: manual merge when BM25 is not available
                all_chunks: Dict[str, Dict] = {}
                for chunk in vector_chunk_list:
                    chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
                    if chunk_id and chunk_id not in all_chunks:
                        all_chunks[chunk_id] = chunk

                for chunk in entity_matched_chunks:
                    chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
                    if chunk_id and chunk_id not in all_chunks:
                        all_chunks[chunk_id] = chunk
                    elif chunk_id in all_chunks:
                        all_chunks[chunk_id]["relevance_score"] = max(
                            all_chunks[chunk_id]["relevance_score"], 0.95
                        )

                for chunk in exact_matched_chunks:
                    chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
                    if chunk_id and chunk_id not in all_chunks:
                        all_chunks[chunk_id] = chunk
                    elif chunk_id in all_chunks:
                        all_chunks[chunk_id]["relevance_score"] = max(
                            all_chunks[chunk_id]["relevance_score"], 0.98
                        )

                candidate_chunks = sorted(all_chunks.values(), key=lambda x: x["relevance_score"], reverse=True)

            candidate_chunks = candidate_chunks[:20]
            if not candidate_chunks:
                return {
                    "status": "no_results",
                    "answer": "",
                    "context": {"vector_chunks": [], "graph_nodes": [], "graph_edges": []},
                }

            # Step 4.5: Reranking (Phase B1) - Cross-encoder precision scoring
            if settings.ENABLE_RERANKING and len(candidate_chunks) > 0:
                try:
                    vector_chunks = await self.reranker.rerank(
                        query=user_question,
                        documents=candidate_chunks,
                        top_n=settings.RERANK_TOP_N,
                    )
                    logger.info(f"Reranked {len(candidate_chunks)} chunks -> top {len(vector_chunks)}")
                except Exception as e:
                    logger.warning(f"Reranking failed, using original ordering: {e}")
                    vector_chunks = candidate_chunks[:10]
            else:
                vector_chunks = candidate_chunks[:10]

            # Step 5: Graph Expansion - 1-hop neighborhood
            node_ids_from_chunks: set = set()
            for chunk in vector_chunks:
                metadata = chunk.get("metadata", {})
                if "node_ids" in metadata:
                    node_ids = metadata["node_ids"]
                    if isinstance(node_ids, str):
                        node_ids = [nid.strip() for nid in node_ids.split(",") if nid.strip()]
                    if isinstance(node_ids, list):
                        node_ids_from_chunks.update(node_ids)
                chunk_id = metadata.get("chunk_id")
                if chunk_id:
                    try:
                        nodes = session.query(Node).filter(
                            Node.properties.like(f'%"chunk_id": "{chunk_id}"%')
                        ).all()
                        # Fallback: filter in Python for JSON
                        if not nodes:
                            all_nodes = session.query(Node).limit(500).all()
                            nodes = [
                                n for n in all_nodes
                                if (n.properties or {}).get("chunk_id") == chunk_id
                            ]
                        for node in nodes:
                            node_ids_from_chunks.add(str(node.id))
                    except Exception:
                        pass

            graph_nodes = []
            graph_edges = []

            if node_ids_from_chunks:
                node_id_list = list(node_ids_from_chunks)
                nodes = (
                    session.query(Node)
                    .options(
                        joinedload(Node.outgoing_edges).joinedload(Edge.target_node),
                        joinedload(Node.incoming_edges).joinedload(Edge.source_node),
                    )
                    .filter(Node.id.in_(node_id_list))
                    .all()
                )

                for node in nodes:
                    graph_nodes.append(
                        {"id": str(node.id), "label": node.label, "properties": node.properties}
                    )

                edges = (
                    session.query(Edge)
                    .options(joinedload(Edge.source_node), joinedload(Edge.target_node))
                    .filter(or_(Edge.source_id.in_(node_id_list), Edge.target_id.in_(node_id_list)))
                    .all()
                )

                connected_node_ids = set()
                for edge in edges:
                    graph_edges.append(
                        {
                            "id": str(edge.id),
                            "source_id": str(edge.source_id),
                            "target_id": str(edge.target_id),
                            "type": edge.type,
                            "properties": edge.properties,
                        }
                    )
                    connected_node_ids.add(edge.source_id)
                    connected_node_ids.add(edge.target_id)

                if connected_node_ids:
                    connected_nodes = (
                        session.query(Node).filter(Node.id.in_(list(connected_node_ids))).all()
                    )
                    for node in connected_nodes:
                        if str(node.id) not in [n["id"] for n in graph_nodes]:
                            graph_nodes.append(
                                {"id": str(node.id), "label": node.label, "properties": node.properties}
                            )

            # Step 6: Build context string for LLM
            context_parts = []
            context_parts.append("=" * 70)
            context_parts.append("RELEVANT TEXT CHUNKS FROM DOCUMENTS")
            context_parts.append("=" * 70)
            for i, chunk in enumerate(vector_chunks, 1):
                match_types = chunk.get("match_type", "vector")
                if isinstance(match_types, list):
                    match_types = ", ".join(match_types)
                context_parts.append(
                    f"\n[Chunk {i}] (Match: {match_types}, Score: {chunk['relevance_score']:.3f})"
                )
                context_parts.append(f"Source: {chunk['metadata'].get('filename', 'Unknown')}")
                if chunk["metadata"].get("page"):
                    context_parts.append(f"Page: {chunk['metadata']['page']}")
                context_parts.append(f"Full Text: {chunk['text']}")

            if graph_nodes:
                context_parts.append("\n" + "=" * 70)
                context_parts.append("RELATED CONCEPTS FROM KNOWLEDGE GRAPH")
                context_parts.append("=" * 70)
                for node in graph_nodes[:10]:
                    props = node["properties"]
                    name = props.get("name", "Unknown")
                    description = props.get("description", "")
                    context_parts.append(f"\n* {name} ({node['label']})")
                    if description:
                        context_parts.append(f"  Description: {description[:200]}")

            if graph_edges:
                context_parts.append("\n" + "=" * 70)
                context_parts.append("KEY RELATIONSHIPS")
                context_parts.append("=" * 70)
                for edge in graph_edges[:10]:
                    source_node = next((n for n in graph_nodes if n["id"] == edge["source_id"]), None)
                    target_node = next((n for n in graph_nodes if n["id"] == edge["target_id"]), None)
                    source_name = source_node["properties"].get("name", "Unknown") if source_node else "Unknown"
                    target_name = target_node["properties"].get("name", "Unknown") if target_node else "Unknown"
                    context_parts.append(f"-> {source_name} --[{edge['type']}]--> {target_name}")

            context_str = "\n".join(context_parts)

            # Step 7: Generate answer using the selected grounded chat mode
            system_msg, user_msg = self._build_answer_prompt(
                user_question=user_question,
                context_str=context_str,
            )

            try:
                answer = await self.llm_service.generate_chat(
                    system_message=system_msg,
                    user_message=user_msg,
                    temperature=0.1,
                    max_tokens=2048,
                )
                return {
                    "status": "success",
                    "answer": answer,
                    "context": {
                        "vector_chunks": vector_chunks,
                        "graph_nodes": graph_nodes,
                        "graph_edges": graph_edges,
                    },
                }
            except Exception as e:
                logger.error(f"LLM generation failed: {e}", exc_info=True)
                return {
                    "status": "generation_failed",
                    "answer": "",
                    "context": {
                        "vector_chunks": vector_chunks,
                        "graph_nodes": graph_nodes,
                        "graph_edges": graph_edges,
                    },
                }
        finally:
            session.close()
