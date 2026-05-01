"""App-level project registry.

The registry is a single JSON file at `{ATLAS_HOME}/projects.json`. It owns
nothing about a project's contents — only the pointer to the project folder
on disk, plus a few convenience fields the UI can show without having to
open every project's database.

This is intentionally NOT a SQLite database. Keeping it as a flat JSON file
lets the user (or an installer) edit the registry directly, and removes any
shared schema between Atlas and individual project folders. A project folder
remains fully portable: zip it, move it, drop it on another machine, and
register it via `register_project(path=...)`.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA_VERSION = 1


def _atlas_home() -> Path:
    """App-level data root (where the registry lives)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Atlas"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Atlas"
    return Path.home() / ".atlas"


def registry_path() -> Path:
    return _atlas_home() / "projects.json"


@dataclass
class ProjectEntry:
    id: str
    name: str
    path: str
    description: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_opened: Optional[str] = None

    def folder(self) -> Path:
        return Path(self.path)


_lock = threading.RLock()


def _empty_registry() -> dict:
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "projects": []}


def _load() -> dict:
    path = registry_path()
    if not path.exists():
        return _empty_registry()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "projects" not in data:
            logger.warning("Registry malformed; resetting to empty")
            return _empty_registry()
        return data
    except Exception as exc:
        logger.warning("Failed to read registry %s: %s — resetting", path, exc)
        return _empty_registry()


def _save(data: dict) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def init_registry() -> None:
    """Ensure the registry file exists. Idempotent."""
    with _lock:
        if not registry_path().exists():
            _save(_empty_registry())


def list_projects() -> List[ProjectEntry]:
    with _lock:
        data = _load()
        return [ProjectEntry(**p) for p in data.get("projects", [])]


def get_project(project_id: str) -> Optional[ProjectEntry]:
    with _lock:
        data = _load()
        for p in data.get("projects", []):
            if p.get("id") == project_id:
                return ProjectEntry(**p)
        return None


def find_by_name(name: str) -> Optional[ProjectEntry]:
    with _lock:
        data = _load()
        for p in data.get("projects", []):
            if p.get("name") == name:
                return ProjectEntry(**p)
        return None


def add_project(
    *,
    name: str,
    path: Path,
    description: Optional[str] = None,
    project_id: Optional[str] = None,
) -> ProjectEntry:
    """Add a project to the registry. Caller is responsible for the folder existing."""
    entry = ProjectEntry(
        id=project_id or str(uuid.uuid4()),
        name=name,
        description=description,
        path=str(Path(path).resolve()),
    )
    with _lock:
        data = _load()
        if any(p.get("id") == entry.id for p in data["projects"]):
            raise ValueError(f"Project id {entry.id} already registered")
        if any(p.get("name") == entry.name for p in data["projects"]):
            raise ValueError(f"Project name '{entry.name}' already registered")
        data["projects"].append(asdict(entry))
        _save(data)
    return entry


def update_project(
    project_id: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    last_opened: Optional[str] = None,
) -> Optional[ProjectEntry]:
    with _lock:
        data = _load()
        for p in data.get("projects", []):
            if p.get("id") == project_id:
                if name is not None:
                    p["name"] = name
                if description is not None:
                    p["description"] = description
                if last_opened is not None:
                    p["last_opened"] = last_opened
                _save(data)
                return ProjectEntry(**p)
        return None


def remove_project(project_id: str) -> bool:
    with _lock:
        data = _load()
        before = len(data["projects"])
        data["projects"] = [p for p in data["projects"] if p.get("id") != project_id]
        if len(data["projects"]) == before:
            return False
        _save(data)
        return True


def touch_last_opened(project_id: str) -> None:
    update_project(project_id, last_opened=datetime.utcnow().isoformat())
