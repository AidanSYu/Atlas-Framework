"""Per-project embedded Qdrant clients.

Each project owns its own Qdrant storage at `{project_root}/qdrant/`. The
embedded Qdrant binary holds an exclusive file lock on its storage path,
so only one client can be open per project at a time. We cap the total
number of open clients with an LRU and dispose the rest; reopening is
cheap (just re-acquires the lock + memory-maps the segments).

Within a single project's Qdrant, the collection name is fixed:
`PROJECT_QDRANT_COLLECTION` (= "docs"). There is no need to namespace
collections by project_id because the storage is already isolated.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Optional

from qdrant_client import QdrantClient

from app.core import project_paths, registry

logger = logging.getLogger(__name__)

PROJECT_QDRANT_COLLECTION = project_paths.PROJECT_QDRANT_COLLECTION
_QDRANT_CLIENT_CACHE_MAX = 8

_client_cache: "OrderedDict[str, QdrantClient]" = OrderedDict()
_lock = RLock()


def _clear_stale_lock(storage_path: Path) -> None:
    """Remove a `.lock` file left by a previous unclean shutdown."""
    lock_file = storage_path / ".lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
            logger.info("Removed stale Qdrant lock file: %s", lock_file)
        except OSError as exc:
            logger.warning("Could not remove lock file %s: %s", lock_file, exc)


def _resolve_storage_path(project_id: str) -> Path:
    entry = registry.get_project(project_id)
    if entry is None:
        raise KeyError(f"Project {project_id} not registered")
    return project_paths.project_qdrant_path(entry.folder())


def _open_client(storage_path: Path) -> QdrantClient:
    storage_path.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(storage_path))
    except RuntimeError as exc:
        if "already accessed" in str(exc):
            logger.warning("Qdrant storage locked at %s — clearing stale lock", storage_path)
            _clear_stale_lock(storage_path)
            return QdrantClient(path=str(storage_path))
        raise


def _evict_one_locked() -> Optional[str]:
    """Close the LRU client. Caller holds _lock. Returns the evicted id."""
    if not _client_cache:
        return None
    oldest_id, oldest_client = next(iter(_client_cache.items()))
    try:
        oldest_client.close()
    except Exception as exc:
        logger.warning("Qdrant close failed for %s: %s", oldest_id, exc)
    _client_cache.pop(oldest_id, None)
    return oldest_id


def get_qdrant_client(project_id: str) -> QdrantClient:
    """Return the embedded Qdrant client for a project, opening one if needed."""
    with _lock:
        client = _client_cache.get(project_id)
        if client is not None:
            _client_cache.move_to_end(project_id)
            return client

        if len(_client_cache) >= _QDRANT_CLIENT_CACHE_MAX:
            _evict_one_locked()

        storage_path = _resolve_storage_path(project_id)
        client = _open_client(storage_path)
        _client_cache[project_id] = client
        logger.info("Opened Qdrant client for project %s at %s", project_id, storage_path)
        return client


def close_qdrant_client(project_id: str) -> None:
    """Close the client for `project_id` so its storage can be deleted/moved."""
    with _lock:
        client = _client_cache.pop(project_id, None)
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                logger.warning("Qdrant close failed for %s: %s", project_id, exc)


def close_all_qdrant_clients() -> None:
    """Close every cached client. Used on shutdown."""
    with _lock:
        for project_id, client in list(_client_cache.items()):
            try:
                client.close()
            except Exception as exc:
                logger.warning("Qdrant close failed for %s: %s", project_id, exc)
        _client_cache.clear()
