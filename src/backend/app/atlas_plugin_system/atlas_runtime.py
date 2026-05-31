"""
Runtime support for .atlas packages — asset extraction, caching, and
helpers for non-Python payloads (GGUF models, native libs, ONNX, etc.).

When a .atlas package contains an asset bundle (model weights, shared
libraries, data files), this module extracts them to a persistent cache
directory so the thin Python wrapper can load them by path.

Cache layout::

    {ATLAS_ASSET_CACHE}/
        {plugin_name}-{content_hash}/
            model.gguf
            libscoring.dll
            ...

The content hash is derived from the raw asset bytes so re-extraction is
skipped when the cache is warm.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "atlas_asset_cache"


def get_asset_cache_dir() -> Path:
    """Return the asset cache root, respecting ATLAS_ASSET_CACHE env var."""
    return Path(os.environ.get("ATLAS_ASSET_CACHE", str(_DEFAULT_CACHE_DIR)))


def extract_assets(
    plugin_name: str,
    assets_bytes: bytes,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Extract the asset zip bundle to a cache directory.

    Returns the path to the extracted directory.  If the cache already
    contains an identical extraction (matched by content hash), the
    existing directory is returned without re-extracting.

    Parameters
    ----------
    plugin_name : str
        Plugin name (used as a human-readable prefix in the cache dir).
    assets_bytes : bytes
        Raw zip archive bytes from the .atlas asset section.
    cache_dir : Path or None
        Override the cache root (default: ATLAS_ASSET_CACHE or system temp).

    Returns
    -------
    Path
        Directory containing the extracted assets.
    """
    if not assets_bytes:
        # No assets — return an empty temp directory
        base = (cache_dir or get_asset_cache_dir()) / f"{plugin_name}-empty"
        base.mkdir(parents=True, exist_ok=True)
        return base

    content_hash = hashlib.sha256(assets_bytes).hexdigest()[:16]
    base = (cache_dir or get_asset_cache_dir()) / f"{plugin_name}-{content_hash}"

    # Check if already extracted (marker file)
    marker = base / ".atlas_extracted"
    if marker.exists():
        logger.debug("Asset cache hit for %s at %s", plugin_name, base)
        return base

    # Extract — guard against zip-slip (members with `..` or absolute paths).
    base.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting assets for %s to %s", plugin_name, base)

    base_resolved = base.resolve()
    with zipfile.ZipFile(io.BytesIO(assets_bytes)) as zf:
        for member in zf.infolist():
            target = (base / member.filename).resolve()
            if base_resolved != target and base_resolved not in target.parents:
                raise RuntimeError(
                    f"Refusing to extract '{member.filename}': "
                    f"escapes asset cache for plugin '{plugin_name}'"
                )
        zf.extractall(base)

    marker.write_text("ok", encoding="utf-8")
    return base


def find_asset(assets_dir: Path, pattern: str) -> Optional[Path]:
    """Find a single asset file matching a glob pattern.

    Useful in wrappers::

        model_path = find_asset(self.__atlas_assets__, "*.gguf")
    """
    matches = list(assets_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning("Multiple matches for pattern '%s' in %s: %s", pattern, assets_dir, matches)
    return matches[0]
