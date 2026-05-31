"""Merged Atlas Framework tool catalog for core tools and optional plugins."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.atlas_plugin_system.core_tools import CoreToolRegistry, get_core_tool_registry
from app.atlas_plugin_system.registry import PluginRegistry, get_plugin_registry

logger = logging.getLogger(__name__)


class ToolCatalog:
    """Expose Atlas core tools and optional plugins through one interface."""

    def __init__(
        self,
        core_tools: Optional[CoreToolRegistry] = None,
        plugins: Optional[PluginRegistry] = None,
    ) -> None:
        self.core_tools = core_tools or get_core_tool_registry()
        self.plugins = plugins or get_plugin_registry()

    def refresh(self) -> None:
        self.core_tools.refresh()
        self.plugins.refresh()

    def refresh_if_stale(self) -> None:
        """Refresh only if the plugin directory has changed or scan is old.

        Avoids a per-request filesystem rescan when nothing has changed —
        the orchestrator's hot path calls this on every prompt.
        """
        self.core_tools.refresh()
        if hasattr(self.plugins, "refresh_if_stale"):
            self.plugins.refresh_if_stale()
        else:
            self.plugins.refresh()

    def list_core_tools(self) -> List[Dict[str, Any]]:
        return self.core_tools.list_tools()

    def list_plugins(self) -> List[Dict[str, Any]]:
        return self.plugins.list_plugins()

    def list_tools(self) -> List[Dict[str, Any]]:
        tools = list(self.list_core_tools())
        core_names = {tool["name"] for tool in tools}
        for plugin in self.list_plugins():
            if plugin["name"] in core_names:
                logger.warning(
                    "Ignoring Atlas plugin '%s' because a core tool already owns that name",
                    plugin["name"],
                )
                continue
            tools.append(plugin)
        return tools

    def tool_names(self) -> List[str]:
        return [tool["name"] for tool in self.list_tools()]

    def build_openai_tools_block(self) -> str:
        """Emit all tools as OpenAI-compatible JSON lines for the <tools> block.

        Each line is a JSON object matching the format Nemotron-Orchestrator
        was trained on::

            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        lines: List[str] = []
        for tool in self.list_tools():
            entry = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("input_schema") or {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            lines.append(json.dumps(entry, ensure_ascii=True))
        return "\n".join(lines)

    async def invoke(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.core_tools.has_tool(tool_name):
            return await self.core_tools.invoke(tool_name, arguments, context)
        return await self.plugins.invoke(tool_name, arguments, context)

    def is_exclusive_gpu(self, tool_name: str) -> bool:
        """Return True if dispatching this tool should evict the orchestrator.

        Core tools never hold exclusive GPU. Plugins opt in via
        ``resource_requirements.exclusive_gpu`` in their manifest.
        """
        if self.core_tools.has_tool(tool_name):
            return False
        return self.plugins.is_exclusive_gpu(tool_name)


_tool_catalog: Optional[ToolCatalog] = None


def get_tool_catalog() -> ToolCatalog:
    """Return the Atlas Framework tool catalog singleton."""
    global _tool_catalog
    if _tool_catalog is None:
        _tool_catalog = ToolCatalog()
    return _tool_catalog
