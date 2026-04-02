"""Discovery OS Plugin Manager.

Manages lifecycle of deterministic plugins. All plugins run on CPU.
Models are lazy-loaded on first tool invocation and cached.
Memory can be reclaimed by calling unload().
"""
import logging
from typing import Any, Dict, List, Optional

from app.services.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


class PluginManager:
    """Registry and lifecycle manager for deterministic plugins."""

    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}
        self._loaded: Dict[str, Any] = {}  # name -> loaded model/session handle
        self._unavailable: Dict[str, str] = {}  # name -> reason

    def register(self, name: str, plugin: BasePlugin):
        """Register a plugin (does NOT load the model yet)."""
        self._plugins[name] = plugin
        self._unavailable.pop(name, None)
        logger.info(f"Plugin registered: {name}")

    def mark_unavailable(self, name: str, reason: str):
        """Track plugin imports that failed so frontend can display them."""
        self._unavailable[name] = reason
        logger.warning("Plugin unavailable: %s (%s)", name, reason)

    def get_registered_names(self) -> List[str]:
        """Return list of registered plugin names."""
        return list(self._plugins.keys())

    def get_all_names(self) -> List[str]:
        """Return all plugin names, including unavailable ones."""
        names = set(self._plugins.keys()) | set(self._unavailable.keys())
        return sorted(names)

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a plugin instance by name."""
        return self._plugins.get(name)

    def get_unavailable_reason(self, name: str) -> Optional[str]:
        """Return reason plugin is unavailable, if any."""
        return self._unavailable.get(name)

    async def invoke(self, name: str, **kwargs) -> dict:
        """Lazy-load plugin if needed, then execute deterministically."""
        if name not in self._plugins:
            raise ValueError(
                f"Unknown plugin: {name}. Registered: {list(self._plugins.keys())}"
            )
        if name not in self._loaded:
            logger.info(f"Lazy-loading plugin: {name}")
            self._loaded[name] = await self._plugins[name].load()
        return await self._plugins[name].execute(self._loaded[name], **kwargs)

    def unload(self, name: str):
        """Free memory for a specific plugin."""
        if name in self._loaded:
            del self._loaded[name]
            logger.info(f"Plugin unloaded: {name}")

    def unload_all(self):
        """Free all plugin memory (e.g., between workflow phases)."""
        self._loaded.clear()
        logger.info("All plugins unloaded")

    def get_tool_descriptions(self, available_tools: List[str]) -> str:
        """Build a tool description block for the LLM system prompt.

        Only includes tools in the available_tools list (dynamic per phase).
        """
        lines = []
        for name in available_tools:
            plugin = self._plugins.get(name)
            if plugin:
                lines.append(f"- {name}: {plugin.description}")
        return "\n".join(lines)


# Singleton
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get or create the singleton PluginManager with Phase 1 plugins."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()

        # Register Phase 1 plugins
        try:
            from app.services.plugins.properties import PropertyPredictorPlugin
            _plugin_manager.register("predict_properties", PropertyPredictorPlugin())
        except ImportError as exc:
            _plugin_manager.mark_unavailable("predict_properties", str(exc))

        try:
            from app.services.plugins.toxicity import ToxicityCheckerPlugin
            _plugin_manager.register("check_toxicity", ToxicityCheckerPlugin())
        except ImportError as exc:
            _plugin_manager.mark_unavailable("check_toxicity", str(exc))

        try:
            import importlib.util
            if importlib.util.find_spec("aizynthfinder") is None:
                raise ImportError("aizynthfinder not installed — run: pip install aizynthfinder")
            from app.services.plugins.retrosynthesis import RetrosynthesisPlugin
            _plugin_manager.register("plan_synthesis", RetrosynthesisPlugin())
        except ImportError as exc:
            _plugin_manager.mark_unavailable("plan_synthesis", str(exc))

        try:
            from app.services.plugins.strategy import StrategyPlugin
            _plugin_manager.register("evaluate_strategy", StrategyPlugin())
        except ImportError as exc:
            _plugin_manager.mark_unavailable("evaluate_strategy", str(exc))

        # Phase 3: Spectroscopy
        try:
            from app.services.plugins.spectrum import SpectrumVerifierPlugin
            _plugin_manager.register("verify_spectrum", SpectrumVerifierPlugin())
        except ImportError as exc:
            _plugin_manager.mark_unavailable("verify_spectrum", str(exc))

        # Chemistry Engine Plan plugins (Phase 2)
        try:
            from app.services.plugins.standardize import StandardizePlugin
            _plugin_manager.register("standardize_smiles", StandardizePlugin())
        except Exception as exc:
            _plugin_manager.mark_unavailable("standardize_smiles", str(exc))

        try:
            from app.services.plugins.admet_predict import ADMETPredictPlugin
            _plugin_manager.register("predict_admet", ADMETPredictPlugin())
        except Exception as exc:
            _plugin_manager.mark_unavailable("predict_admet", str(exc))

        try:
            from app.services.plugins.sa_scorer import SAScorePlugin
            _plugin_manager.register("score_synthesizability", SAScorePlugin())
        except Exception as exc:
            _plugin_manager.mark_unavailable("score_synthesizability", str(exc))

        try:
            from app.services.plugins.fragment_enumeration import FragmentEnumerationPlugin
            _plugin_manager.register("enumerate_fragments", FragmentEnumerationPlugin())
        except Exception as exc:
            _plugin_manager.mark_unavailable("enumerate_fragments", str(exc))

        logger.info(
            f"PluginManager initialized with plugins: {_plugin_manager.get_registered_names()}"
        )
    return _plugin_manager
