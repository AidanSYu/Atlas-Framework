"""
Manufacturability Agent
Assesses whether a compound/protein is feasible for mass production (scale-up).
Analyzes cost, complexity, regulatory path, and production risks.

Key difference from retrosynthesis:
- Retrosynthesis: Lab-scale synthesis (how to make in the lab?)
- Manufacturability: Commercial scale (can we make 1M+ doses/year profitably?)
"""

import re
from typing import Dict, List, Any, Optional
from .retrosynthesis import RetrosynthesisEngine


class ManufacturabilityAgent:
    """
    Predicts manufacturability and scalability for commercial production.
    
    Assessment factors:
    - Cost of goods (COGS per unit at scale)
    - Production complexity and yield
    - Regulatory path and CMC (Chemistry, Manufacturing, Controls)
    - Supply chain risks
    - Timeline to commercial scale
    - Facility/equipment requirements
    
    Example scenario:
    - Researcher finds promising antibody treatment (easy to express in lab)
    - Manufacturability agent predicts: CHO cell culture at 1000L scale = $50M CapEx
      + $20M/year OpEx + $150/dose COGS → Too expensive for mass market
    """
    
    def __init__(self):
        """Initialize with retrosynthesis engine for chemical analysis."""
        self.retro_engine = RetrosynthesisEngine()
        self.chem_client = self.retro_engine.chem_client  # Reuse ChemLLM instance
    
    def _llm_assessment(self, prompt: str) -> str:
        """Query ChemLLM for manufacturability analysis."""
        if self.chem_client is None:
            return "[ChemLLM unavailable]"
        
        try:
            if hasattr(self.chem_client, 'generate'):
                response = self.chem_client.generate(prompt=prompt, max_tokens=512)
                # Ensure response is a string
                if response is None:
                    return "[ChemLLM returned None]"
                return str(response) if not isinstance(response, str) else response
            else:
                return "[ChemLLM client missing generate method]"
        except Exception as e:
            return f"[Error: {str(e)[:150]}]"
    
    def assess_scalability(self, 
                          compound_name: str, 
                          compound_type: str = "small_molecule",
                          synthesis_complexity: str = "moderate",
                          smiles: str = None,
                          researcher_context: str = None) -> Dict[str, Any]:
        """
        Assess whether a compound/protein is scalable to mass production.
        
        Args:
            compound_name: Drug name (e.g., "aspirin", "insulin")
            compound_type: "small_molecule", "protein", "antibody", "nucleic_acid"
            synthesis_complexity: Lab synthesis difficulty ("simple", "moderate", "complex")
            smiles: SMILES string (small molecules only)
            researcher_context: Additional context from researcher agent
        
        Returns:
            Dict with manufacturing feasibility metrics:
            - scalability_score (0-100): Higher = more feasible for production
            - manufacturability_risks: Top risks ranked by severity
            - cost_estimate: Predicted COGS per unit at different scales
            - production_timeline: Est. years from clinical → commercial
            - regulatory_hurdles: Key CMC/FDA challenges
            - recommendations: Actions to improve manufacturability
        """
        
        result = {
            "compound_name": compound_name,
            "compound_type": compound_type,
            "synthesis_complexity": synthesis_complexity,
            "scalability_score": 50,  # Default neutral
            "manufacturability_risks": [],
            "cost_estimate": "",
            "production_timeline": "",
            "regulatory_hurdles": [],
            "manufacturing_route": "",
            "supply_chain_risks": [],
            "recommendations": [],
            "detailed_assessment": ""
        }
        
        # Build manufacturing assessment prompt
        context_str = f"\nContext from research: {researcher_context[:200]}" if researcher_context else ""
        smiles_str = f"\nSMILES: {smiles}" if smiles else ""
        
        prompt = f"""You are a pharma manufacturing and process development expert.
Assess MASS PRODUCTION feasibility for commercial scale:

Drug: {compound_name}
Type: {compound_type}
Lab synthesis complexity: {synthesis_complexity}{smiles_str}{context_str}

Provide a manufacturing assessment with:

1. **SCALABILITY SCORE (0-100)**: Is this feasible for commercial production?
   (100 = highly scalable, 0 = not viable)

2. **MANUFACTURING COSTS (per dose)**:
   - At 100K doses/year: $___
   - At 1M doses/year: $___
   - At 10M doses/year: $___
   
3. **TOP 3 MANUFACTURING RISKS** (rank by severity):
   - [Risk 1]: Impact? Mitigation?
   - [Risk 2]: Impact? Mitigation?
   - [Risk 3]: Impact? Mitigation?

4. **REGULATORY/CMC HURDLES**:
   - Process validation required?
   - Analytical methods challenges?
   - Stability/degradation concerns?

5. **TIMELINE TO COMMERCIAL SCALE**:
   - Months: Phase I → Phase III: ___
   - Months: Phase III → NDA approval: ___
   - Months: NDA approval → Full production: ___
   - Total: ___ months

6. **SUPPLY CHAIN**: Any critical material bottlenecks?

7. **PRODUCTION ROUTE RECOMMENDATION**:
   - Batch size? Equipment? Facility size?
   - Annual capacity feasible?

Be specific with costs and timelines."""

        detailed = self._llm_assessment(prompt)
        result["detailed_assessment"] = detailed
        
        # Parse scalability score
        score_match = re.search(r'SCALABILITY\s+SCORE[:\s]+(\d+)', detailed, re.IGNORECASE)
        if score_match:
            try:
                result["scalability_score"] = min(100, max(0, int(score_match.group(1))))
            except (ValueError, IndexError):
                pass
        
        # Extract cost estimates
        cost_match = re.search(r'MANUFACTURING\s+COSTS.*?10M.*?\$([0-9.]+)', detailed, re.IGNORECASE | re.DOTALL)
        if cost_match:
            result["cost_estimate"] = f"Est. ${cost_match.group(1)}/dose at 10M doses/year"
        
        # Extract risks
        risks_match = re.search(r'TOP\s+3.*?RISKS(.*?)REGULATORY', detailed, re.IGNORECASE | re.DOTALL)
        if risks_match:
            risk_text = risks_match.group(1)
            risks = re.findall(r'\[Risk\s+\d+\][:\s]*([^[\n]+)', risk_text)
            result["manufacturability_risks"] = [r.strip()[:100] for r in risks[:3]]
        
        # Extract timeline
        timeline_match = re.search(r'Total:\s*([0-9]+)\s*(months|years)', detailed, re.IGNORECASE)
        if timeline_match:
            result["production_timeline"] = f"{timeline_match.group(1)} {timeline_match.group(2)}"
        
        # Extract recommendations
        rec_match = re.search(r'PRODUCTION\s+ROUTE(.*?)$', detailed, re.IGNORECASE | re.DOTALL)
        if rec_match:
            rec_text = rec_match.group(1)
            recs = re.findall(r'[-•]\s*(.+?)(?=[-•]|$)', rec_text)
            result["recommendations"] = [r.strip()[:100] for r in recs[:3]]
        
        return result
