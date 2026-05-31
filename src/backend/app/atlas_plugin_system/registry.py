"""Filesystem-backed plugin registry for the Atlas Framework."""

from __future__ import annotations

import logging
import os
import time
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings

logger = logging.getLogger(__name__)


class ResourceRequirements(BaseModel):
    """Hardware requirements for a plugin."""

    min_vram_mb: int = 0
    min_ram_mb: int = 0
    gpu_required: bool = False
    recommended_vram_mb: int = 0
    # When true, the orchestrator evicts itself from GPU before dispatching
    # this tool so the plugin has exclusive VRAM. Reserve for heavy workloads
    # (training, large VLM inference) — reloading Nemotron costs ~10-15s.
    exclusive_gpu: bool = False


class PluginManifest(BaseModel):
    """Validated plugin manifest loaded from manifest.json."""

    model_config = ConfigDict(protected_namespaces=())

    schema_version: str = "1.0"
    name: str
    version: str = "0.1.0"
    description: str
    entry_point: str = "wrapper.py"
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 50
    tags: List[str] = Field(default_factory=list)
    runtime: str = "python"  # python | gguf | onnx | native | generic
    license: str = ""  # e.g. "Apache-2.0", "GPL-3.0"
    optional_dependencies: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)  # embedded asset filenames
    resource_requirements: ResourceRequirements = Field(default_factory=ResourceRequirements)
    self_test: str = ""  # shell command to validate the plugin is working
    fallback_used: str = ""  # describes what capability is lost without optional deps
    # Controls what the orchestrator (model) sees of this tool's result so a
    # large payload doesn't blow the 8K context. Shape:
    #   {"salient_fields": ["canonical_smiles", "inchi_key"], "max_chars": 200}
    # The FULL payload always streams to the UI event log; this only shrinks the
    # model-facing view. Empty {} = conservative structural default.
    to_model_projection: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class RegisteredPlugin:
    """Runtime plugin record."""

    manifest: PluginManifest
    source_path: Path
    wrapper_reference: str
    wrapper_instance: Any = None
    load_error: Optional[str] = None
    source_type: str = "directory"


class _FunctionWrapper:
    """Normalize module-level invoke() functions into the wrapper protocol."""

    def __init__(self, fn: Any):
        self._fn = fn

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = self._fn(arguments or {}, context or {})
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, dict):
            return result
        return {"summary": str(result), "raw_result": result}


class PluginRegistry:
    """Scan the plugins directory and lazily load wrapper runtimes."""

    def __init__(self, plugin_dir: Optional[Path] = None):
        self.plugin_dir = Path(plugin_dir or settings.ATLAS_PLUGIN_DIR)
        self._plugins: Dict[str, RegisteredPlugin] = {}
        self._last_scan_signature: Optional[Tuple[float, int]] = None
        self._last_scan_monotonic: float = 0.0
        self.refresh()

    def _directory_signature(self) -> Optional[Tuple[float, int]]:
        """Return (latest_mtime, child_count) across the plugin tree.

        Catches additions, removals, and edits at one or two levels deep
        — same depth ``_iter_candidates`` walks. Returns None when the
        directory cannot be statted.
        """
        try:
            latest = self.plugin_dir.stat().st_mtime
            count = 0
            for item in self.plugin_dir.iterdir():
                if item.name.startswith("."):
                    continue
                try:
                    latest = max(latest, item.stat().st_mtime)
                except OSError:
                    continue
                count += 1
                if item.is_dir():
                    try:
                        for sub in item.iterdir():
                            if sub.name.startswith("."):
                                continue
                            try:
                                latest = max(latest, sub.stat().st_mtime)
                            except OSError:
                                continue
                            count += 1
                    except OSError:
                        continue
            return (latest, count)
        except OSError:
            return None

    def refresh_if_stale(self, ttl_seconds: Optional[float] = None) -> None:
        """Refresh only when the plugin tree changed or the cache is too old.

        The full ``refresh`` rescans every plugin candidate and re-parses
        manifests — wasteful on the hot path if nothing changed. This
        compares a (latest_mtime, child_count) signature against the
        previous scan and skips work when both match within ``ttl_seconds``.
        """
        ttl = settings.ATLAS_PLUGIN_CATALOG_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        signature = self._directory_signature()
        now = time.monotonic()
        if (
            signature is not None
            and signature == self._last_scan_signature
            and (now - self._last_scan_monotonic) < ttl
        ):
            return
        self.refresh()
        self._last_scan_signature = signature
        self._last_scan_monotonic = now

    def _iter_candidates(self, root: Path) -> List[Path]:
        """Yield plugin candidates from root, supporting one level of grouping.

        Supports both flat layout (plugins/my_plugin/) and grouped layout
        (plugins/<group>/<plugin>/). A directory is a group folder if it
        contains no manifest.json but contains subdirectories that do.

        Any path that resolves outside `root` (e.g. via symlink) is rejected.
        """
        root_resolved = root.resolve()

        def _is_inside_root(p: Path) -> bool:
            try:
                resolved = p.resolve()
            except OSError:
                return False
            return resolved == root_resolved or root_resolved in resolved.parents

        candidates: List[Path] = []
        root_dir_names = {
            item.name
            for item in root.iterdir()
            if item.is_dir() and not item.name.startswith(".") and _is_inside_root(item)
        }
        for item in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if item.name.startswith("."):
                continue
            if not _is_inside_root(item):
                logger.warning(
                    "Refusing plugin candidate %s: resolves outside plugin dir %s",
                    item, root_resolved,
                )
                continue
            if item.is_dir():
                manifest_path = item / "manifest.json"
                if manifest_path.exists():
                    # Direct plugin directory
                    candidates.append(item)
                else:
                    # Potential group folder — recurse one level
                    child_dir_names = {
                        sub.name
                        for sub in item.iterdir()
                        if sub.is_dir() and not sub.name.startswith(".") and _is_inside_root(sub)
                    }
                    for sub in sorted(item.iterdir(), key=lambda p: p.name.lower()):
                        if sub.name.startswith("."):
                            continue
                        if not _is_inside_root(sub):
                            logger.warning(
                                "Refusing plugin candidate %s: resolves outside plugin dir %s",
                                sub, root_resolved,
                            )
                            continue
                        if (
                            sub.is_file()
                            and sub.suffix.lower() in {".atlas", ".zip"}
                            and sub.stem in child_dir_names
                        ):
                            continue
                        candidates.append(sub)
            else:
                # .atlas or .zip file at root level
                if item.suffix.lower() in {".atlas", ".zip"} and item.stem in root_dir_names:
                    continue
                candidates.append(item)
        return candidates

    def refresh(self) -> None:
        """Rescan the plugins directory for folders and zip archives."""
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        previous = self._plugins
        discovered: Dict[str, RegisteredPlugin] = {}

        for candidate in self._iter_candidates(self.plugin_dir):
            if candidate.name.startswith("."):
                continue
            record = self._build_record(candidate)
            if record is None:
                continue
            existing = previous.get(record.manifest.name)
            if (
                existing is not None
                and existing.source_path == record.source_path
                and existing.wrapper_reference == record.wrapper_reference
            ):
                record.wrapper_instance = existing.wrapper_instance
                record.load_error = existing.load_error
            if record.manifest.name in discovered:
                previous_record = discovered[record.manifest.name]
                if self._source_priority(previous_record.source_type) > self._source_priority(record.source_type):
                    logger.warning(
                        "Duplicate Atlas plugin name '%s' from %s; keeping %s over lower-priority %s source",
                        record.manifest.name,
                        candidate,
                        previous_record.source_path,
                        record.source_type,
                    )
                    continue
                logger.warning(
                    "Duplicate Atlas plugin name '%s' from %s; overriding previous entry",
                    record.manifest.name,
                    candidate,
                )
            discovered[record.manifest.name] = record

        self._plugins = discovered
        self._last_scan_signature = self._directory_signature()
        self._last_scan_monotonic = time.monotonic()
        logger.info("Atlas PluginRegistry loaded %d plugin(s)", len(self._plugins))

    def _build_record(self, candidate: Path) -> Optional[RegisteredPlugin]:
        if candidate.is_dir():
            manifest_path = candidate / "manifest.json"
            if not manifest_path.exists():
                return None
            try:
                manifest = PluginManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                logger.warning("Skipping invalid plugin manifest at %s: %s", manifest_path, exc)
                return None
            return RegisteredPlugin(
                manifest=manifest,
                source_path=candidate,
                wrapper_reference=manifest.entry_point,
                source_type="directory",
            )

        if candidate.is_file() and candidate.suffix.lower() == ".atlas":
            return self._build_atlas_record(candidate)

        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(candidate) as archive:
                    manifest_member = self._resolve_archive_member(
                        archive.namelist(),
                        "manifest.json",
                    )
                    if manifest_member is None:
                        return None
                    manifest = PluginManifest.model_validate_json(
                        archive.read(manifest_member).decode("utf-8")
                    )
                    wrapper_member = self._resolve_archive_member(
                        archive.namelist(),
                        manifest.entry_point,
                    )
                    if wrapper_member is None:
                        logger.warning(
                            "Skipping zip plugin %s because %s was not found",
                            candidate,
                            manifest.entry_point,
                        )
                        return None
            except Exception as exc:
                logger.warning("Skipping unreadable zip plugin %s: %s", candidate, exc)
                return None

            return RegisteredPlugin(
                manifest=manifest,
                source_path=candidate,
                wrapper_reference=wrapper_member,
                source_type="zip",
            )

        return None

    def _build_atlas_record(self, candidate: Path) -> Optional[RegisteredPlugin]:
        """Build a RegisteredPlugin from a .atlas binary package."""
        from app.atlas_plugin_system.atlas_format import inspect_atlas

        try:
            info = inspect_atlas(candidate)
            manifest = PluginManifest.model_validate(info["manifest"])
        except Exception as exc:
            logger.warning("Skipping invalid .atlas package %s: %s", candidate, exc)
            return None

        return RegisteredPlugin(
            manifest=manifest,
            source_path=candidate,
            wrapper_reference="<atlas>",
            source_type="atlas",
        )

    @staticmethod
    def _source_priority(source_type: str) -> int:
        return {
            "directory": 3,
            "atlas": 2,
            "zip": 1,
        }.get(source_type, 0)

    @staticmethod
    def _resolve_archive_member(names: List[str], target_name: str) -> Optional[str]:
        direct_matches = [name for name in names if name.rstrip("/") == target_name]
        if direct_matches:
            return direct_matches[0]

        suffix = "/" + target_name
        nested_matches = [name for name in names if name.endswith(suffix)]
        if nested_matches:
            nested_matches.sort(key=len)
            return nested_matches[0]

        return None

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return plugin metadata for API responses and prompt construction."""
        return [
            {
                "name": record.manifest.name,
                "description": record.manifest.description,
                "priority": record.manifest.priority,
                "input_schema": record.manifest.input_schema,
                "output_schema": record.manifest.output_schema,
                "tags": record.manifest.tags,
                "license": record.manifest.license,
                "optional_dependencies": record.manifest.optional_dependencies,
                "artifacts": record.manifest.artifacts,
                "resource_requirements": record.manifest.resource_requirements.model_dump(),
                "self_test": record.manifest.self_test,
                "fallback_used": record.manifest.fallback_used,
                "to_model_projection": record.manifest.to_model_projection,
                "source": str(record.source_path),
                "source_type": record.source_type,
                "loaded": record.wrapper_instance is not None,
                "load_error": record.load_error,
            }
            for record in self._ordered_plugins()
        ]

    def is_exclusive_gpu(self, plugin_name: str) -> bool:
        """Return True if the plugin's manifest requests exclusive GPU access."""
        record = self._plugins.get(plugin_name)
        if record is None:
            return False
        return bool(record.manifest.resource_requirements.exclusive_gpu)

    async def invoke(
        self,
        plugin_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke a plugin wrapper by name."""
        record = self._plugins.get(plugin_name)
        if record is None:
            raise ValueError(f"Unknown Atlas Framework plugin: {plugin_name}")

        wrapper = self._load_wrapper(record)
        try:
            result = await wrapper.invoke(arguments or {}, context or {})
        except Exception as exc:
            logger.error("Atlas plugin '%s' failed: %s", plugin_name, exc, exc_info=True)
            return {
                "status": "error",
                "summary": f"{plugin_name} failed: {exc}",
                "error": str(exc),
                "plugin": plugin_name,
            }

        if isinstance(result, dict):
            if "summary" not in result:
                result["summary"] = self._summarize_result(plugin_name, result)
            return result

        return {
            "summary": f"{plugin_name} returned a non-dict result",
            "raw_result": result,
        }

    def _load_wrapper(self, record: RegisteredPlugin) -> Any:
        if record.wrapper_instance is not None:
            return record.wrapper_instance

        try:
            if record.source_type == "atlas":
                module = self._load_atlas_module(record)
            else:
                module_name = f"atlas_framework_plugin_{record.manifest.name}"
                module = types.ModuleType(module_name)
                module.__file__ = f"{record.source_path}:{record.wrapper_reference}"

                if record.source_type == "directory":
                    wrapper_path = (record.source_path / record.wrapper_reference).resolve()
                    plugin_dir_resolved = record.source_path.resolve()
                    if plugin_dir_resolved not in wrapper_path.parents:
                        raise RuntimeError(
                            f"Refusing to load wrapper '{record.wrapper_reference}': "
                            f"escapes plugin directory {plugin_dir_resolved}"
                        )
                    source = wrapper_path.read_text(encoding="utf-8")
                    origin = str(wrapper_path)
                else:
                    with zipfile.ZipFile(record.source_path) as archive:
                        source = archive.read(record.wrapper_reference).decode("utf-8")
                    origin = f"{record.source_path}!{record.wrapper_reference}"

                exec(compile(source, origin, "exec"), module.__dict__)

            wrapper = module.__dict__.get("PLUGIN")
            if wrapper is None and "create_plugin" in module.__dict__:
                wrapper = module.__dict__["create_plugin"]()
            if wrapper is None and "invoke" in module.__dict__:
                wrapper = _FunctionWrapper(module.__dict__["invoke"])

            if wrapper is None or not hasattr(wrapper, "invoke"):
                raise RuntimeError(
                    f"Wrapper {record.wrapper_reference} did not expose PLUGIN, create_plugin(), or invoke()."
                )

            record.wrapper_instance = wrapper
            record.load_error = None
            return wrapper
        except Exception as exc:
            record.load_error = str(exc)
            raise

    @staticmethod
    def _load_atlas_module(record: RegisteredPlugin) -> types.ModuleType:
        """Load a .atlas binary package into a module via bytecode unmarshalling."""
        from app.atlas_plugin_system.atlas_format import load_atlas_module, read_atlas

        passphrase = os.environ.get("ATLAS_PLUGIN_KEY")
        package = read_atlas(record.source_path, passphrase=passphrase)
        return load_atlas_module(package)

    @staticmethod
    def _summarize_result(plugin_name: str, payload: Dict[str, Any]) -> str:
        keys = ", ".join(sorted(payload.keys())[:6])
        return f"{plugin_name} completed. Keys: {keys or 'none'}."

    def _ordered_plugins(self) -> List[RegisteredPlugin]:
        return sorted(
            self._plugins.values(),
            key=lambda record: (-record.manifest.priority, record.manifest.name),
        )


_plugin_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Return the Atlas Framework plugin registry singleton."""
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry()
    return _plugin_registry
