"""Synthesis strategy evaluation plugin — fully self-contained, no app imports.

Evaluates retrosynthesis routes and scores them for feasibility, difficulty,
and likely bottlenecks.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EvaluateStrategyWrapper:

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        routes = args.get("routes", [])
        if not routes or not isinstance(routes, list):
            return {"valid": False, "error": "Missing or invalid 'routes' argument."}

        evaluated_routes: List[dict] = []
        for route in routes:
            steps = route.get("number_of_steps", 0)
            materials_count = len(route.get("starting_materials", []))
            base_score = route.get("score", 0.5)

            step_penalty = steps * 0.1
            material_penalty = max(0, materials_count - 2) * 0.05
            feasibility = max(0.1, min(0.99, base_score - step_penalty - material_penalty + 0.3))

            if steps > 5 or feasibility < 0.4:
                difficulty = "Hard"
            elif steps > 3 or feasibility < 0.7:
                difficulty = "Medium"
            else:
                difficulty = "Easy"

            evaluated_routes.append({
                "route_index": route.get("route_index", 0),
                "original_score": base_score,
                "feasibility_score": round(feasibility, 2),
                "difficulty": difficulty,
                "estimated_yield": "Unknown (needs reaction review)",
                "bottlenecks": f"Route has {steps} steps. Review literature for typical yields." if steps > 4 else "None obvious",
            })

        evaluated_routes.sort(key=lambda x: x["feasibility_score"], reverse=True)
        for r in evaluated_routes:
            r["engine_used"] = "heuristic_step_material_penalty"
        return {
            "valid": False,
            "engine_used": "heuristic_step_material_penalty",
            "routes_evaluated": len(evaluated_routes),
            "recommendation": f"Route {evaluated_routes[0]['route_index']} appears most feasible (HEURISTIC ONLY).",
            "scored_routes": evaluated_routes,
            "summary": (
                f"Evaluated {len(evaluated_routes)} route(s) with a step+material heuristic. "
                "WARNING: feasibility scores are not derived from real reaction data or yield models; "
                "treat as a rough sort, not a quantitative prediction."
            ),
        }


PLUGIN = EvaluateStrategyWrapper()
