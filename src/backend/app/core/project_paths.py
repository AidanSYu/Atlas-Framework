"""Filesystem layout for a single project folder.

Each project is fully self-contained at:

    {project_root}/
        workspace.json    — manifest
        project.db        — per-project SQLite
        qdrant/           — per-project Qdrant storage
        files/            — original uploaded documents (full copies)
        drafts/
        task_attachments/{task_id}/
        traces/
        .atlas_cache/

`project_root` defaults to `{ATLAS_WORKSPACES_DIR}/{project_id}/` but the
registry can point at any absolute path on disk, so a folder can be moved
or imported without copying.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.config import settings


WORKSPACE_MANIFEST = "workspace.json"
PROJECT_DB_FILENAME = "project.db"
QDRANT_SUBDIR = "qdrant"
FILES_SUBDIR = "files"
DRAFTS_SUBDIR = "drafts"
TASK_ATTACHMENTS_SUBDIR = "task_attachments"
TRACES_SUBDIR = "traces"
PLUGIN_CACHE_SUBDIR = ".atlas_cache"

PROJECT_QDRANT_COLLECTION = "docs"


def default_project_root(project_id: str) -> Path:
    """Default folder for a freshly created project."""
    return Path(settings.ATLAS_WORKSPACES_DIR) / project_id


def project_db_path(project_root: Path) -> Path:
    return project_root / PROJECT_DB_FILENAME


def project_qdrant_path(project_root: Path) -> Path:
    return project_root / QDRANT_SUBDIR


def project_files_path(project_root: Path) -> Path:
    return project_root / FILES_SUBDIR


def project_drafts_path(project_root: Path) -> Path:
    return project_root / DRAFTS_SUBDIR


def project_task_attachments_path(project_root: Path, task_id: Optional[str] = None) -> Path:
    base = project_root / TASK_ATTACHMENTS_SUBDIR
    return base / task_id if task_id else base


def project_traces_path(project_root: Path) -> Path:
    return project_root / TRACES_SUBDIR


def project_plugin_cache_path(project_root: Path) -> Path:
    return project_root / PLUGIN_CACHE_SUBDIR


def ensure_project_folder(project_root: Path) -> None:
    """Create the empty subfolders for a project. Idempotent."""
    project_root.mkdir(parents=True, exist_ok=True)
    for sub in (
        FILES_SUBDIR,
        DRAFTS_SUBDIR,
        TASK_ATTACHMENTS_SUBDIR,
        TRACES_SUBDIR,
        PLUGIN_CACHE_SUBDIR,
        QDRANT_SUBDIR,
    ):
        (project_root / sub).mkdir(parents=True, exist_ok=True)
