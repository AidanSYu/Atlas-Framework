"""Managed workspace lifecycle.

Each workspace is an isolated, portable folder owned by a single project.
Layout (see app.core.project_paths):

    {project_root}/
        workspace.json        — manifest
        project.db            — per-project SQLite
        qdrant/               — per-project Qdrant storage
        files/                — uploaded documents
        drafts/               — editor drafts
        task_attachments/     — task-scoped uploads
        traces/               — failure / orchestration traces
        .atlas_cache/         — plugin asset cache

`.atlas` archives are now just zipped folders. Export = zip the folder,
import = unzip into a fresh folder + add to the registry.
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
from typing import Any, Dict, Optional

from app.core import project_paths, registry
from app.core.config import settings
from app.core.database import close_project_engine, init_project_db
from app.core.qdrant_store import close_qdrant_client

logger = logging.getLogger(__name__)

ARCHIVE_SCHEMA_VERSION = 2  # bumped: archive is now the project folder itself
MANIFEST_FILENAME = "workspace.json"


class WorkspaceManager:
    """Owns on-disk workspace folders and archive I/O."""

    def __init__(self) -> None:
        self.root = Path(settings.ATLAS_WORKSPACES_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Folder lifecycle
    # ------------------------------------------------------------------

    def workspace_path(self, workspace_id: str) -> Path:
        entry = registry.get_project(workspace_id)
        if entry is not None:
            return entry.folder()
        return self.root / workspace_id

    def files_path(self, workspace_id: str) -> Path:
        path = project_paths.project_files_path(self.workspace_path(workspace_id))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def drafts_path(self, workspace_id: str) -> Path:
        path = project_paths.project_drafts_path(self.workspace_path(workspace_id))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_attachments_path(self, workspace_id: str, task_id: str) -> Path:
        path = project_paths.project_task_attachments_path(
            self.workspace_path(workspace_id), task_id
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_folder(
        self, workspace_id: str, name: str, description: Optional[str] = None
    ) -> Path:
        ws_path = self.workspace_path(workspace_id)
        project_paths.ensure_project_folder(ws_path)
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
        # Caller (delete_project route) already closes engine + qdrant client;
        # do it again here in case this is called from elsewhere.
        close_project_engine(workspace_id)
        close_qdrant_client(workspace_id)

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
    # Archive export — zip the entire folder
    # ------------------------------------------------------------------

    def export_archive(self, workspace_id: str) -> bytes:
        """Build a `.atlas` archive in memory by zipping the project folder."""
        entry = registry.get_project(workspace_id)
        if entry is None:
            raise ValueError(f"Workspace {workspace_id} not found")

        # Caller should have closed open handles already, but flush again to be safe.
        close_project_engine(workspace_id)
        close_qdrant_client(workspace_id)

        ws_path = entry.folder()
        if not ws_path.exists():
            raise ValueError(f"Workspace folder missing: {ws_path}")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in ws_path.rglob("*"):
                if path.is_file():
                    arcname = path.relative_to(ws_path).as_posix()
                    zf.write(path, arcname)
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Archive import — unzip into a fresh folder + register
    # ------------------------------------------------------------------

    def import_archive(self, archive_bytes: bytes) -> registry.ProjectEntry:
        """Unzip a `.atlas` archive into a fresh project folder and register it."""
        try:
            zf = zipfile.ZipFile(io.BytesIO(archive_bytes))
        except zipfile.BadZipFile as exc:
            raise ValueError("Not a valid .atlas archive") from exc

        with zf:
            names = zf.namelist()
            if MANIFEST_FILENAME not in names:
                raise ValueError("Archive is missing workspace.json")

            manifest = json.loads(zf.read(MANIFEST_FILENAME).decode("utf-8"))
            new_workspace_id = str(uuid.uuid4())
            ws_path = self.root / new_workspace_id
            ws_path.mkdir(parents=True, exist_ok=False)

            for member in names:
                if member.endswith("/"):
                    continue
                _safe_extract(zf, member, ws_path)

        # Rewrite the manifest with the fresh id (the original id may collide
        # with an existing project on this machine).
        desired_name = _unique_project_name(manifest.get("name") or "Imported Workspace")
        new_manifest = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "workspace_id": new_workspace_id,
            "name": desired_name,
            "description": manifest.get("description"),
            "created_at": datetime.utcnow().isoformat(),
            "imported_from": manifest.get("workspace_id"),
            "imported_at": datetime.utcnow().isoformat(),
        }
        (ws_path / MANIFEST_FILENAME).write_text(
            json.dumps(new_manifest, indent=2), encoding="utf-8"
        )

        entry = registry.add_project(
            project_id=new_workspace_id,
            name=desired_name,
            description=manifest.get("description"),
            path=ws_path,
        )
        # Touch the engine once so any post-import migrations or table additions
        # are applied to the imported DB.
        try:
            init_project_db(new_workspace_id)
        except Exception as exc:
            logger.warning("init_project_db after import failed for %s: %s", new_workspace_id, exc)
        return entry

    # ------------------------------------------------------------------
    # Register an existing on-disk folder (VSCode "open folder")
    # ------------------------------------------------------------------

    def register_existing(self, folder: Path) -> registry.ProjectEntry:
        folder = Path(folder).resolve()
        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")
        manifest_path = folder / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise ValueError(f"No workspace.json found at {folder}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        workspace_id = manifest.get("workspace_id") or str(uuid.uuid4())
        name = manifest.get("name") or folder.name

        if registry.get_project(workspace_id) is not None:
            raise ValueError(f"Project {workspace_id} is already registered")
        if registry.find_by_name(name):
            name = _unique_project_name(name)

        entry = registry.add_project(
            project_id=workspace_id,
            name=name,
            description=manifest.get("description"),
            path=folder,
        )
        try:
            init_project_db(workspace_id)
        except Exception as exc:
            logger.warning("init_project_db on register failed for %s: %s", workspace_id, exc)
        return entry


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _unique_project_name(desired: str) -> str:
    """Return `desired`, suffixing ' (imported)' / ' (imported N)' on collision."""
    if registry.find_by_name(desired) is None:
        return desired
    candidate = f"{desired} (imported)"
    if registry.find_by_name(candidate) is None:
        return candidate
    i = 2
    while True:
        candidate = f"{desired} (imported {i})"
        if registry.find_by_name(candidate) is None:
            return candidate
        i += 1


def _safe_extract(zf: zipfile.ZipFile, member: str, dest_root: Path) -> None:
    """Extract a single archive member under dest_root, defending against zip-slip."""
    target = (dest_root / member).resolve()
    if dest_root.resolve() != target and dest_root.resolve() not in target.parents:
        logger.warning("Skipping suspicious archive path: %s", member)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as source, open(target, "wb") as dest:
        shutil.copyfileobj(source, dest)


# ----------------------------------------------------------------------
# Singleton accessor
# ----------------------------------------------------------------------

_workspace_manager: Optional[WorkspaceManager] = None


def get_workspace_manager() -> WorkspaceManager:
    global _workspace_manager
    if _workspace_manager is None:
        _workspace_manager = WorkspaceManager()
    return _workspace_manager
