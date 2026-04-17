"""Managed Workspace lifecycle.

Each workspace is an isolated, AppData-rooted folder:

    {ATLAS_WORKSPACES_DIR}/{workspace_id}/
        workspace.json   — manifest (schema_version, name, created_at, ...)
        files/           — uploaded documents
        drafts/          — editor drafts

Portable archives (`.atlas`) bundle the folder together with the relational + vector
slices of the shared stores that belong to this workspace, so a workspace can be
re-hydrated on a different machine without dragging global state along.
"""
from __future__ import annotations

import io
import json
import logging
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import (
    Document,
    DocumentChunk,
    Edge,
    Node,
    Project,
    get_session,
)
from app.core.qdrant_store import get_qdrant_client

logger = logging.getLogger(__name__)

ARCHIVE_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "workspace.json"
DB_DUMP_FILENAME = "db.json"
QDRANT_DUMP_FILENAME = "qdrant_points.jsonl"
FILES_SUBDIR = "files"
DRAFTS_SUBDIR = "drafts"


class WorkspaceManager:
    """Owns on-disk workspace folders and archive I/O."""

    def __init__(self) -> None:
        self.root = Path(settings.ATLAS_WORKSPACES_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Folder lifecycle
    # ------------------------------------------------------------------

    def workspace_path(self, workspace_id: str) -> Path:
        return self.root / workspace_id

    def files_path(self, workspace_id: str) -> Path:
        path = self.workspace_path(workspace_id) / FILES_SUBDIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def drafts_path(self, workspace_id: str) -> Path:
        path = self.workspace_path(workspace_id) / DRAFTS_SUBDIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_attachments_path(self, workspace_id: str, task_id: str) -> Path:
        """Dir where task-scoped attachments (non-ingested) live.

        Attachments are user-uploaded files passed to the tool loop (NMR
        spectra, inspection images, CSVs) that should NOT be embedded into
        the RAG corpus. Keep them separate from `files/` which feeds ingest.
        """
        path = self.workspace_path(workspace_id) / "task_attachments" / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_folder(self, workspace_id: str, name: str, description: Optional[str] = None) -> Path:
        ws_path = self.workspace_path(workspace_id)
        (ws_path / FILES_SUBDIR).mkdir(parents=True, exist_ok=True)
        (ws_path / DRAFTS_SUBDIR).mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "workspace_id": workspace_id,
            "name": name,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
        }
        (ws_path / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("Created workspace folder at %s", ws_path)
        return ws_path

    def delete_folder(self, workspace_id: str) -> bool:
        ws_path = self.workspace_path(workspace_id)
        if not ws_path.exists():
            return False
        shutil.rmtree(ws_path, ignore_errors=True)
        logger.info("Removed workspace folder %s", ws_path)
        return True

    def read_manifest(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        manifest_path = self.workspace_path(workspace_id) / MANIFEST_FILENAME
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read manifest for %s: %s", workspace_id, exc)
            return None

    # ------------------------------------------------------------------
    # Archive export
    # ------------------------------------------------------------------

    def export_archive(self, workspace_id: str) -> bytes:
        """Build a `.atlas` archive in memory for the given workspace."""
        session: Session = get_session()
        try:
            project = session.query(Project).filter(Project.id == workspace_id).first()
            if project is None:
                raise ValueError(f"Workspace {workspace_id} not found")

            documents = session.query(Document).filter(Document.project_id == workspace_id).all()
            doc_ids = [d.id for d in documents]
            chunks = (
                session.query(DocumentChunk)
                .filter(DocumentChunk.document_id.in_(doc_ids))
                .all()
                if doc_ids
                else []
            )
            nodes = session.query(Node).filter(Node.project_id == workspace_id).all()
            edges = session.query(Edge).filter(Edge.project_id == workspace_id).all()

            db_dump = {
                "project": _project_to_dict(project),
                "documents": [_document_to_dict(d) for d in documents],
                "chunks": [_chunk_to_dict(c) for c in chunks],
                "nodes": [_node_to_dict(n) for n in nodes],
                "edges": [_edge_to_dict(e) for e in edges],
            }
        finally:
            session.close()

        qdrant_lines = _dump_qdrant_points([c["id"] for c in db_dump["chunks"]])

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = self.read_manifest(workspace_id) or {
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "workspace_id": workspace_id,
                "name": db_dump["project"]["name"],
                "description": db_dump["project"].get("description"),
                "created_at": db_dump["project"].get("created_at"),
            }
            manifest = dict(manifest)
            manifest["exported_at"] = datetime.utcnow().isoformat()
            manifest["schema_version"] = ARCHIVE_SCHEMA_VERSION
            zf.writestr(MANIFEST_FILENAME, json.dumps(manifest, indent=2))
            zf.writestr(DB_DUMP_FILENAME, json.dumps(db_dump, indent=2))
            zf.writestr(QDRANT_DUMP_FILENAME, "\n".join(qdrant_lines))

            ws_path = self.workspace_path(workspace_id)
            for sub in (FILES_SUBDIR, DRAFTS_SUBDIR):
                sub_root = ws_path / sub
                if not sub_root.exists():
                    continue
                for file_path in sub_root.rglob("*"):
                    if file_path.is_file():
                        arcname = f"{sub}/{file_path.relative_to(sub_root).as_posix()}"
                        zf.write(file_path, arcname)
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Archive import
    # ------------------------------------------------------------------

    def import_archive(self, archive_bytes: bytes) -> Project:
        """Restore a `.atlas` archive as a brand-new workspace with fresh IDs."""
        try:
            zf = zipfile.ZipFile(io.BytesIO(archive_bytes))
        except zipfile.BadZipFile as exc:
            raise ValueError("Not a valid .atlas archive") from exc

        with zf:
            names = set(zf.namelist())
            if MANIFEST_FILENAME not in names or DB_DUMP_FILENAME not in names:
                raise ValueError("Archive is missing manifest or db dump")

            manifest = json.loads(zf.read(MANIFEST_FILENAME).decode("utf-8"))
            db_dump = json.loads(zf.read(DB_DUMP_FILENAME).decode("utf-8"))
            qdrant_raw = (
                zf.read(QDRANT_DUMP_FILENAME).decode("utf-8")
                if QDRANT_DUMP_FILENAME in names
                else ""
            )

            new_workspace_id = str(uuid.uuid4())
            ws_path = self.workspace_path(new_workspace_id)
            files_dir = ws_path / FILES_SUBDIR
            drafts_dir = ws_path / DRAFTS_SUBDIR
            files_dir.mkdir(parents=True, exist_ok=True)
            drafts_dir.mkdir(parents=True, exist_ok=True)

            for member in zf.namelist():
                if member.startswith(f"{FILES_SUBDIR}/") and not member.endswith("/"):
                    _safe_extract(zf, member, files_dir, strip_prefix=f"{FILES_SUBDIR}/")
                elif member.startswith(f"{DRAFTS_SUBDIR}/") and not member.endswith("/"):
                    _safe_extract(zf, member, drafts_dir, strip_prefix=f"{DRAFTS_SUBDIR}/")

        desired_name = _unique_project_name(manifest.get("name") or "Imported Workspace")

        doc_id_map: Dict[str, str] = {d["id"]: str(uuid.uuid4()) for d in db_dump["documents"]}
        chunk_id_map: Dict[str, str] = {c["id"]: str(uuid.uuid4()) for c in db_dump["chunks"]}
        node_id_map: Dict[str, str] = {n["id"]: str(uuid.uuid4()) for n in db_dump["nodes"]}

        session: Session = get_session()
        try:
            project = Project(
                id=new_workspace_id,
                name=desired_name,
                description=db_dump["project"].get("description"),
                created_at=datetime.utcnow(),
            )
            session.add(project)
            session.flush()

            for d in db_dump["documents"]:
                old_path = Path(d.get("file_path") or "")
                new_path = files_dir / old_path.name if old_path.name else files_dir / d["filename"]
                session.add(
                    Document(
                        id=doc_id_map[d["id"]],
                        filename=d["filename"],
                        file_hash=d["file_hash"],
                        file_path=str(new_path),
                        file_size=d.get("file_size"),
                        mime_type=d.get("mime_type"),
                        project_id=new_workspace_id,
                        uploaded_at=_parse_dt(d.get("uploaded_at")) or datetime.utcnow(),
                        processed_at=_parse_dt(d.get("processed_at")),
                        status=d.get("status", "completed"),
                        total_chunks=d.get("total_chunks", 0) or 0,
                        processed_chunks=d.get("processed_chunks", 0) or 0,
                        doc_metadata=d.get("doc_metadata") or {},
                    )
                )

            for c in db_dump["chunks"]:
                session.add(
                    DocumentChunk(
                        id=chunk_id_map[c["id"]],
                        document_id=doc_id_map[c["document_id"]],
                        text=c["text"],
                        chunk_index=c.get("chunk_index"),
                        page_number=c.get("page_number"),
                        start_char=c.get("start_char"),
                        end_char=c.get("end_char"),
                        chunk_metadata=c.get("chunk_metadata") or {},
                    )
                )

            for n in db_dump["nodes"]:
                session.add(
                    Node(
                        id=node_id_map[n["id"]],
                        label=n["label"],
                        properties=n.get("properties") or {},
                        document_id=doc_id_map.get(n["document_id"]) if n.get("document_id") else None,
                        project_id=new_workspace_id,
                    )
                )

            for e in db_dump["edges"]:
                source = node_id_map.get(e["source_id"])
                target = node_id_map.get(e["target_id"])
                if not source or not target:
                    continue
                session.add(
                    Edge(
                        id=str(uuid.uuid4()),
                        source_id=source,
                        target_id=target,
                        type=e["type"],
                        properties=e.get("properties") or {},
                        document_id=doc_id_map.get(e["document_id"]) if e.get("document_id") else None,
                        project_id=new_workspace_id,
                    )
                )

            session.commit()
            project_view = _project_to_dict(
                session.query(Project).filter(Project.id == new_workspace_id).first()
            )
        except Exception:
            session.rollback()
            shutil.rmtree(ws_path, ignore_errors=True)
            raise
        finally:
            session.close()

        manifest_out = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "workspace_id": new_workspace_id,
            "name": desired_name,
            "description": project_view.get("description"),
            "created_at": project_view.get("created_at"),
            "imported_from": manifest.get("workspace_id"),
            "imported_at": datetime.utcnow().isoformat(),
        }
        (ws_path / MANIFEST_FILENAME).write_text(
            json.dumps(manifest_out, indent=2), encoding="utf-8"
        )

        _restore_qdrant_points(qdrant_raw, chunk_id_map, doc_id_map)

        session = get_session()
        try:
            return session.query(Project).filter(Project.id == new_workspace_id).first()
        finally:
            session.close()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _project_to_dict(p: Project) -> Dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "created_at": _isoformat(p.created_at),
    }


def _document_to_dict(d: Document) -> Dict[str, Any]:
    return {
        "id": d.id,
        "filename": d.filename,
        "file_hash": d.file_hash,
        "file_path": d.file_path,
        "file_size": d.file_size,
        "mime_type": d.mime_type,
        "uploaded_at": _isoformat(d.uploaded_at),
        "processed_at": _isoformat(d.processed_at),
        "status": d.status,
        "total_chunks": d.total_chunks,
        "processed_chunks": d.processed_chunks,
        "doc_metadata": d.doc_metadata or {},
    }


def _chunk_to_dict(c: DocumentChunk) -> Dict[str, Any]:
    return {
        "id": c.id,
        "document_id": c.document_id,
        "text": c.text,
        "chunk_index": c.chunk_index,
        "page_number": c.page_number,
        "start_char": c.start_char,
        "end_char": c.end_char,
        "chunk_metadata": c.chunk_metadata or {},
    }


def _node_to_dict(n: Node) -> Dict[str, Any]:
    return {
        "id": n.id,
        "label": n.label,
        "properties": n.properties or {},
        "document_id": n.document_id,
    }


def _edge_to_dict(e: Edge) -> Dict[str, Any]:
    return {
        "id": e.id,
        "source_id": e.source_id,
        "target_id": e.target_id,
        "type": e.type,
        "properties": e.properties or {},
        "document_id": e.document_id,
    }


def _dump_qdrant_points(point_ids: List[str]) -> List[str]:
    """Retrieve Qdrant points (with vectors) for the given chunk IDs."""
    if not point_ids:
        return []
    try:
        client = get_qdrant_client()
    except Exception as exc:
        logger.warning("Qdrant unavailable during export: %s", exc)
        return []

    lines: List[str] = []
    # Retrieve in chunks to avoid overlong requests.
    batch_size = 256
    for i in range(0, len(point_ids), batch_size):
        batch = point_ids[i : i + batch_size]
        try:
            records = client.retrieve(
                collection_name=settings.QDRANT_COLLECTION,
                ids=batch,
                with_vectors=True,
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Qdrant retrieve failed for batch: %s", exc)
            continue
        for r in records:
            lines.append(
                json.dumps(
                    {
                        "id": str(r.id),
                        "vector": list(r.vector) if r.vector is not None else None,
                        "payload": r.payload or {},
                    }
                )
            )
    return lines


def _restore_qdrant_points(
    raw_jsonl: str,
    chunk_id_map: Dict[str, str],
    doc_id_map: Dict[str, str],
) -> None:
    if not raw_jsonl.strip():
        return
    try:
        from qdrant_client.models import PointStruct  # lazy import

        client = get_qdrant_client()
    except Exception as exc:
        logger.warning("Qdrant unavailable during import: %s", exc)
        return

    points: List[Any] = []
    for line in raw_jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        old_id = str(row.get("id", ""))
        vector = row.get("vector")
        if not old_id or vector is None:
            continue
        new_id = chunk_id_map.get(old_id, old_id)
        payload = dict(row.get("payload") or {})
        if "chunk_id" in payload:
            payload["chunk_id"] = chunk_id_map.get(payload["chunk_id"], payload["chunk_id"])
        if "doc_id" in payload:
            payload["doc_id"] = doc_id_map.get(payload["doc_id"], payload["doc_id"])
        points.append(PointStruct(id=new_id, vector=vector, payload=payload))

    if not points:
        return
    try:
        client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    except Exception as exc:
        logger.warning("Qdrant upsert failed during import: %s", exc)


def _unique_project_name(desired: str) -> str:
    """Return `desired`, suffixing ' (imported)' / ' (imported N)' on collision."""
    session: Session = get_session()
    try:
        candidate = desired
        if not session.query(Project).filter(Project.name == candidate).first():
            return candidate
        candidate = f"{desired} (imported)"
        if not session.query(Project).filter(Project.name == candidate).first():
            return candidate
        i = 2
        while True:
            candidate = f"{desired} (imported {i})"
            if not session.query(Project).filter(Project.name == candidate).first():
                return candidate
            i += 1
    finally:
        session.close()


def _safe_extract(zf: zipfile.ZipFile, member: str, dest_root: Path, strip_prefix: str) -> None:
    """Extract a single archive member under dest_root, defending against zip-slip."""
    rel = member[len(strip_prefix):] if member.startswith(strip_prefix) else member
    if not rel or rel.endswith("/"):
        return
    target = (dest_root / rel).resolve()
    if dest_root.resolve() not in target.parents and target != dest_root.resolve():
        logger.warning("Skipping suspicious archive path: %s", member)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as source, open(target, "wb") as dest:
        shutil.copyfileobj(source, dest)


# Singleton accessor
_workspace_manager: Optional[WorkspaceManager] = None


def get_workspace_manager() -> WorkspaceManager:
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = WorkspaceManager()
    return _workspace_manager
