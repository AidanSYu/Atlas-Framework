"""AiZynthFinder Retrosynthesis Plugin.

Wraps the AiZynthFinder ONNX model to provide retrosynthesis routes.
Runs on CPU via ONNX Runtime to avoid GPU VRAM contention with the LLM.
"""
import asyncio
import logging
from typing import Any

from app.services.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


class RetrosynthesisPlugin(BasePlugin):

    @property
    def name(self) -> str:
        return "plan_synthesis"

    @property
    def description(self) -> str:
        return (
            "Plan a synthesis route for a target molecule using the AiZynthFinder "
            "retrosynthesis engine. Returns a list of possible routes, each containing "
            "the required steps and starting materials."
        )

    async def load(self) -> Any:
        """Lazy-load the AiZynthFinder ONNX model into memory.

        Returns None (skip sentinel) if aizynthfinder is not installed so the
        pipeline continues without hard-erroring on an optional dependency.
        """
        try:
            import importlib.util
            if importlib.util.find_spec("aizynthfinder") is None:
                raise ImportError("aizynthfinder not installed")
        except ImportError:
            logger.warning(
                "aizynthfinder is not installed — plan_synthesis stage will be skipped. "
                "Install with: pip install aizynthfinder"
            )
            return None  # skip sentinel

        logger.info("Loading AiZynthFinder model (this may take a moment)...")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._load_model_sync)

    def _load_model_sync(self) -> Any:
        from aizynthfinder.aizynthfinder import AiZynthFinder
        
        # We assume download_public_data created config.yml in the current dir
        # or we can point to it explicitly. For now, try default config.
        import os
        config_path = os.path.join(os.getcwd(), "data", "config.yml")
        if not os.path.exists(config_path):
            # Fallback to current dir if run from there
            config_path = os.path.join(os.getcwd(), "config.yml")
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"AiZynthFinder config not found at {config_path}. Did you run download_public_data?")

        finder = AiZynthFinder(configfile=config_path)
        return finder

    async def execute(self, model: Any, **kwargs) -> dict:
        """Run retrosynthesis planning for a SMILES string.

        Args (in kwargs):
            smiles: str -- SMILES representation of the target molecule.

        Returns:
            Dict containing the top synthesis routes, or a skip dict if
            aizynthfinder is not installed.
        """
        if model is None:
            return {
                "skipped": True,
                "reason": "aizynthfinder is not installed",
                "summary": (
                    "Retrosynthesis skipped — aizynthfinder not installed. "
                    "Run: pip install aizynthfinder"
                ),
            }
        smiles = kwargs.get("smiles", "")
        if not smiles:
            return {"valid": False, "error": "Missing 'smiles' argument."}

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._compute, model, smiles)

    def _compute(self, finder: Any, smiles: str) -> dict:
        try:
            # Set the target molecule
            finder.target_smiles = smiles
            
            # Ensure policies are selected if config is loaded correctly
            try:
                finder.stock.select("zinc")
                finder.expansion_policy.select("uspto")
                finder.filter_policy.select("uspto")
            except Exception as e:
                logger.warning(f"Failed to explicitly select policies, fallback to defaults: {e}")
            
            # Run tree search
            finder.tree_search()
            finder.build_routes()
            
            # Extract top routes
            routes_data = []
            stats = finder.extract_statistics()
            
            if not finder.routes:
                return {
                    "smiles": smiles,
                    "valid": True,
                    "routes_found": 0,
                    "message": "No synthesis routes found within search limits.",
                    "routes": []
                }
                
            for i, route in enumerate(finder.routes.dicts[:3]):  # Return top 3 routes to avoid context bloat
                # A route dict contains the full reaction tree
                # We extract a simplified summary for the LLM
                
                # Try to extract the list of starting materials and reactions
                # The exact structure depends on AiZynthFinder version, but typically we can get:
                nodes = getattr(finder.routes[i], 'nodes', []) 
                if not nodes and "nodes" in route:
                    nodes = route["nodes"]
                    
                # Very simplified summary
                # Extracting the full tree into a clean linear text format can be complex
                # We'll rely on the built-in properties if available
                
                routes_data.append({
                    "route_index": i + 1,
                    "score": route.get('score', 0),
                    "is_solved": route.get('is_solved', False),
                    "starting_materials": [sm for sm in route.get('starting_materials', [])] if 'starting_materials' in route else [],
                    "number_of_steps": route.get('length', 0) if 'length' in route else len(nodes) // 2 if isinstance(nodes, list) else "Unknown",
                    # Try to get a reaction summary if available
                    "reaction_summary": route.get('reaction_summary', "Detailed reactions omitted for brevity")
                })
                
            return {
                "smiles": smiles,
                "valid": True,
                "routes_found": len(finder.routes),
                "routes": routes_data,
                "search_stats": {
                    "iterations": stats.get('number_of_iterations', 0),
                    "time_s": round(stats.get('time', 0), 2)
                }
            }
        except Exception as e:
            return {"smiles": smiles, "valid": False, "error": str(e)}

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES representation of the target molecule to synthesize",
                }
            },
            "required": ["smiles"],
        }

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string"},
                "valid": {"type": "boolean"},
                "routes_found": {"type": "integer"},
                "message": {"type": "string"},
                "routes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "route_index": {"type": "integer"},
                            "score": {"type": "number"},
                            "is_solved": {"type": "boolean"},
                            "starting_materials": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "number_of_steps": {"type": ["integer", "string"]},
                            "reaction_summary": {"type": "string"}
                        }
                    }
                },
                "search_stats": {
                    "type": "object",
                    "properties": {
                        "iterations": {"type": "integer"},
                        "time_s": {"type": "number"}
                    }
                }
            },
        }
