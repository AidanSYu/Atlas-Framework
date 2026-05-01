"""Document service for file management (per-project SQLite + per-project Qdrant)."""
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi.responses import FileResponse
import logging

from app.core.database import (
    Document,
    DocumentChunk,
    Edge,
    Node,
    get_project_session,
)
from app.core.qdrant_store import PROJECT_QDRANT_COLLECTION, get_qdrant_client
from app.services.bm25_index import get_bm25_service

logger = logging.getLogger(__name__)


class DocumentService:
    """Manages document storage and retrieval, scoped per project."""

    def __init__(self):
        pass

    def list_documents(
        self,
        project_id: str,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List documents in a project. Auto-deletes records whose backing file is gone."""
        session = get_project_session(project_id)
        try:
            query = session.query(Document)
            if status:
                query = query.filter(Document.status == status)

            documents = query.order_by(Document.uploaded_at.desc()).limit(limit).all()

            result = []
            orphaned_docs = []

            for doc in documents:
                try:
                    file_path = Path(doc.file_path)
                    if not file_path.exists():
                        orphaned_docs.append(doc)
                        continue
                except Exception:
                    orphaned_docs.append(doc)
                    continue

                result.append(self._document_to_dict(doc))

            if orphaned_docs:
                logger.info(f"Auto-deleting {len(orphaned_docs)} orphaned document records")
                for orphaned_doc in orphaned_docs:
                    try:
                        session.query(DocumentChunk).filter(
                            DocumentChunk.document_id == orphaned_doc.id
                        ).delete(synchronize_session=False)
                        session.delete(orphaned_doc)
                    except Exception as e:
                        logger.error(f"Error deleting orphaned document {orphaned_doc.id}: {e}")
                try:
                    session.commit()
                except Exception as e:
                    logger.error(f"Error committing orphaned document deletions: {e}")
                    session.rollback()

            return result
        finally:
            session.close()

    def get_document(self, project_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        session = get_project_session(project_id)
        try:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if not document:
                return None
            return self._document_to_dict(document)
        finally:
            session.close()

    def get_document_file(self, project_id: str, doc_id: str) -> Optional[FileResponse]:
        session = get_project_session(project_id)
        try:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if not document:
                return None

            file_path = Path(document.file_path)
            if not file_path.exists():
                logger.warning(f"File not found for document {doc_id}: {file_path}")
                return None

            mime_map = {
                ".pdf": "application/pdf",
                ".txt": "text/plain",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".doc": "application/msword",
            }
            ext = file_path.suffix.lower()
            media_type = mime_map.get(ext, "application/octet-stream")

            return FileResponse(
                path=file_path, media_type=media_type, filename=document.filename
            )
        finally:
            session.close()

    def delete_document(self, project_id: str, doc_id: str) -> bool:
        """Delete a document from every store (vectors, chunks, graph, file, row)."""
        session = get_project_session(project_id)
        try:
            document = session.query(Document).filter(Document.id == doc_id).first()
            if not document:
                return False

            doc_id_str = str(doc_id)

            # 1. Delete from this project's Qdrant
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                qdrant_client = get_qdrant_client(project_id)

                filter_condition = Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id_str))]
                )

                point_ids = []
                offset = None
                while True:
                    scroll_result = qdrant_client.scroll(
                        collection_name=PROJECT_QDRANT_COLLECTION,
                        scroll_filter=filter_condition,
                        limit=100,
                        offset=offset,
                    )
                    points, next_offset = scroll_result
                    if not points:
                        break
                    point_ids.extend([point.id for point in points])
                    if next_offset is None:
                        break
                    offset = next_offset

                if point_ids:
                    qdrant_client.delete(
                        collection_name=PROJECT_QDRANT_COLLECTION, points_selector=point_ids
                    )
                    logger.info(f"Deleted {len(point_ids)} vectors from Qdrant for doc {doc_id_str}")
            except Exception as e:
                logger.warning(f"Error deleting from Qdrant for doc {doc_id_str}: {e}")

            # 2. BM25 index removal (in-memory or per-project file — see bm25 service)
            try:
                get_bm25_service().remove_document(doc_id_str)
                logger.info(f"Removed doc {doc_id_str} from BM25 index")
            except Exception as e:
                logger.warning(f"Error removing from BM25 index for doc {doc_id_str}: {e}")

            # 3. Delete chunks
            try:
                session.query(DocumentChunk).filter(
                    DocumentChunk.document_id == doc_id
                ).delete(synchronize_session=False)
            except Exception as e:
                logger.warning(f"Error deleting chunks for doc {doc_id_str}: {e}")

            # 4. Delete related nodes and edges
            try:
                nodes = session.query(Node).filter(Node.document_id == doc_id).all()
                node_ids = [node.id for node in nodes]

                if node_ids:
                    session.query(Edge).filter(
                        (Edge.source_id.in_(node_ids)) | (Edge.target_id.in_(node_ids))
                    ).delete(synchronize_session=False)

                    session.query(Node).filter(
                        Node.document_id == doc_id
                    ).delete(synchronize_session=False)
            except Exception as e:
                logger.warning(f"Error deleting nodes/edges for doc {doc_id_str}: {e}")

            # 5. Delete the on-disk file
            try:
                file_path = Path(document.file_path)
                if file_path.exists():
                    file_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Error deleting file {document.file_path}: {e}")

            # 6. Delete the row
            session.delete(document)
            session.commit()

            logger.info(f"Successfully deleted document {doc_id_str}")
            return True
        except Exception as e:
            logger.error(f"Error during document deletion: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def _document_to_dict(self, document: Document) -> Dict[str, Any]:
        status = document.status
        if status == "completed":
            status = "indexed"

        progress = 0.0
        if document.total_chunks and document.total_chunks > 0:
            progress = min(100.0, (document.processed_chunks / document.total_chunks) * 100.0)

        return {
            "filename": document.filename,
            "doc_id": str(document.id),
            "status": status,
            "size_bytes": document.file_size,
            "uploaded_at": document.uploaded_at.isoformat() if document.uploaded_at else None,
            "processed_at": document.processed_at.isoformat() if document.processed_at else None,
            "total_chunks": document.total_chunks,
            "processed_chunks": document.processed_chunks,
            "progress": round(progress, 1),
            "project_id": document.project_id,
        }
