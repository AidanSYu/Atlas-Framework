"""
Retrosynthesis Engine
Uses RDKit + LLM to provide retrosynthesis analysis
"""

import requests
from typing import Dict, List, Any, Optional

# Try to import chemllm package; if unavailable, we'll fall back to a HF wrapper below
try:
    from chemllm import ChemLLMClient
except Exception:
    ChemLLMClient = None


class HuggingFaceChemLLM:
    """Minimal Hugging Face wrapper for ChemLLM models.

    Loads a ChemLLM model from Hugging Face and exposes `generate(prompt, ...)`.
    This is a best-effort fallback — loading requires `transformers` and `torch`.
    """

    def __init__(self, model_id: str = "AI4Chem/ChemLLM-7B-Chat-1.5-DPO"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except Exception as e:
            raise RuntimeError(f"HuggingFace dependencies missing: {e}")

        self.model_id = model_id
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        try:
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model {model_id}: {e}")

    def generate(self, prompt: str, model: str = None, max_tokens: int = 512):
        """Generate response using simple, direct generation without hanging issues."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to device
            try:
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:
                pass

            # Generate with minimal, stable parameters (no generation_config to avoid hanging)
            try:
                with self.torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=min(max_tokens, 512),
                        do_sample=False,  # Greedy decoding - most stable
                        temperature=None,  # Ignore when do_sample=False
                        top_p=None,  # Ignore when do_sample=False
                        top_k=None,  # Ignore when do_sample=False
                        use_cache=False,  # Disable cache to avoid hanging
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            except RuntimeError as gen_err:
                if "out of memory" in str(gen_err).lower():
                    return "[ChemLLM: Out of memory - try smaller prompt]"
                else:
                    return f"[ChemLLM generation error: {str(gen_err)[:80]}]"
            except Exception as gen_err:
                return f"[ChemLLM generation error: {type(gen_err).__name__}: {str(gen_err)[:60]}]"
            
            if outputs is None:
                return "[ChemLLM: No output generated]"
            
            # Decode output safely
            try:
                # Handle batch output (usually just 1 sequence)
                if len(outputs.shape) > 1:
                    text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    text = self.tokenizer.decode(outputs, skip_special_tokens=True)
            except Exception as decode_err:
                return f"[ChemLLM decode error: {str(decode_err)[:50]}]"
            
            return text if text else "[ChemLLM: Empty response]"
        except Exception as e:
            import traceback
            return f"[ChemLLM error: {type(e).__name__}: {str(e)[:80]}]"

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class RetrosynthesisEngine:
    """
    Open-source retrosynthesis engine using RDKit + LLM.
    Provides IBM RXN-like capabilities without needing paid API.
    """
    
    def __init__(self):
        self.common_reactions = self._load_reaction_templates()
        # Initialize a chemistry LLM client when possible
        self.chem_client: Optional[object] = None
        if ChemLLMClient is not None:
            try:
                self.chem_client = ChemLLMClient()
            except Exception:
                self.chem_client = None

        if self.chem_client is None:
            try:
                self.chem_client = HuggingFaceChemLLM()
            except Exception:
                self.chem_client = None
    
    def _load_reaction_templates(self) -> List[Dict[str, str]]:
        """Common organic chemistry reaction templates."""
        return [
            {
                "name": "Ester Hydrolysis",
                "smarts": "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[OH][C:4]",
                "description": "Break ester to carboxylic acid + alcohol"
            },
            {
                "name": "Amide Formation",
                "smarts": "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OH].[NH2:3]",
                "description": "Break amide to carboxylic acid + amine"
            },
            {
                "name": "Grignard Addition",
                "smarts": "[C:1]([OH:2])([C:3])[C:4]>>[C:1]=[O:2].[MgBr][C:4]",
                "description": "Disconnect alcohol to ketone + Grignard"
            },
            {
                "name": "Aldol Condensation",
                "smarts": "[C:1]=[C:2][C:3](=[O:4])[C:5]>>[C:1]=[O].[C:3](=[O:4])[C:5]",
                "description": "Disconnect α,β-unsaturated carbonyl"
            },
            {
                "name": "Wittig Reaction",
                "smarts": "[C:1]=[C:2]>>[C:1]=[O].[C:2]",
                "description": "Disconnect alkene to carbonyl + ylide"
            },
            {
                "name": "SN2 Substitution",
                "smarts": "[C:1][N,O,S:2]>>[C:1][Br,Cl,I].[N,O,S:2]",
                "description": "Disconnect C-heteroatom bond"
            },
            {
                "name": "Friedel-Crafts",
                "smarts": "[c:1][C:2]>>[c:1][H].[C:2]",
                "description": "Disconnect aromatic C-C bond"
            },
            {
                "name": "Reductive Amination",
                "smarts": "[C:1][NH:2][C:3]>>[C:1]=[O].[NH2:2][C:3]",
                "description": "Disconnect C-N to carbonyl + amine"
            }
        ]
    
    def _local_llm(self, prompt: str, timeout: int = 180) -> str:
        """Call the project's chemistry LLM for retrosynthesis prompts.

        Preference order:
        1. `chemllm` package client (if installed)
        2. Hugging Face ChemLLM model via `transformers` (if available)

        Note: Ollama/Mistral fallback has been removed. If no local ChemLLM
        is available, this method returns a clear error string instead of
        attempting to call external Mistral services.
        """

        client = getattr(self, 'chem_client', None)
        if client is None:
            return "[LLM unavailable: ChemLLM client not configured]"

        try:
            if hasattr(client, 'generate'):
                return client.generate(prompt=prompt)
            elif hasattr(client, 'create'):
                return client.create(prompt=prompt)
            else:
                return str(client)
        except Exception as e:
            return f"[LLM error: {type(e).__name__}: {str(e)[:200]}]"
    
    def _analyze_molecule(self, smiles: str) -> Dict[str, Any]:
        """Analyze molecular properties using RDKit."""
        if not RDKIT_AVAILABLE or not smiles:
            return {"error": "RDKit not available or invalid SMILES"}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"error": "Invalid SMILES structure"}
            
            return {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "h_bond_donors": Descriptors.NumHDonors(mol),
                "h_bond_acceptors": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "complexity_score": self._calculate_complexity(mol)
            }
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _calculate_complexity(self, mol) -> float:
        """Calculate synthetic complexity score (0-100)."""
        # Higher = more complex
        score = 0
        score += mol.GetNumHeavyAtoms() * 2  # More atoms = more complex
        score += Descriptors.NumRotatableBonds(mol) * 3  # Flexibility
        score += Descriptors.NumAromaticRings(mol) * 5  # Aromatic systems
        score += len([a for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]) * 10  # Stereocenters
        
        # Normalize to 0-100
        return min(score, 100)
    
    def _identify_functional_groups(self, smiles: str) -> List[str]:
        """Identify key functional groups in molecule."""
        if not RDKIT_AVAILABLE or not smiles:
            return []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            functional_groups = []
            
            # Common functional group SMARTS patterns
            patterns = {
                "Carboxylic Acid": "C(=O)[OH]",
                "Ester": "C(=O)O[C]",
                "Amide": "C(=O)N",
                "Amine": "[NH2,NH1,NH0]",
                "Alcohol": "[OH]",
                "Ketone": "[C]C(=O)[C]",
                "Aldehyde": "[CH](=O)",
                "Ether": "[C]O[C]",
                "Nitro": "N(=O)=O",
                "Halide": "[F,Cl,Br,I]",
                "Aromatic": "c",
                "Alkene": "C=C",
                "Alkyne": "C#C"
            }
            
            for name, smarts in patterns.items():
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    functional_groups.append(name)
            
            return functional_groups
        except Exception:
            return []
    
    def _suggest_disconnections(self, smiles: str) -> List[Dict[str, str]]:
        """Suggest strategic bond disconnections using reaction templates."""
        if not RDKIT_AVAILABLE or not smiles:
            return []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            suggestions = []
            
            for rxn_template in self.common_reactions:
                try:
                    rxn = AllChem.ReactionFromSmarts(rxn_template["smarts"])
                    if rxn:
                        products = rxn.RunReactants((mol,))
                        if products:
                            suggestions.append({
                                "reaction": rxn_template["name"],
                                "description": rxn_template["description"],
                                "precursors": len(products[0])
                            })
                except Exception:
                    continue
            
            return suggestions[:5]  # Top 5 suggestions
        except Exception:
            return []
    
    def retrosynthesis_analysis(self, compound_name: str, smiles: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive retrosynthesis analysis.
        This is the free alternative to IBM RXN.
        """
        
        result = {
            "compound_name": compound_name,
            "smiles": smiles,
            "molecular_properties": {},
            "functional_groups": [],
            "strategic_disconnections": [],
            "retrosynthetic_routes": [],
            "forward_synthesis": "",
            "complexity_assessment": "",
            "commercial_availability": ""
        }
        
        # Step 1: Analyze molecular properties
        if smiles:
            result["molecular_properties"] = self._analyze_molecule(smiles)
            result["functional_groups"] = self._identify_functional_groups(smiles)
            result["strategic_disconnections"] = self._suggest_disconnections(smiles)
        
        # Step 2: Generate retrosynthetic analysis with LLM
        properties_text = ""
        if result["molecular_properties"] and "error" not in result["molecular_properties"]:
            props = result["molecular_properties"]
            mw = props.get('molecular_weight', 'N/A')
            logp = props.get('logp', 'N/A')
            h_donors = props.get('h_bond_donors', 'N/A')
            h_acceptors = props.get('h_bond_acceptors', 'N/A')
            aromatic = props.get('aromatic_rings', 'N/A')
            complexity = props.get('complexity_score', 'N/A')
            
            mw_str = f"{mw:.2f}" if isinstance(mw, (int, float)) else str(mw)
            logp_str = f"{logp:.2f}" if isinstance(logp, (int, float)) else str(logp)
            complexity_str = f"{complexity:.1f}" if isinstance(complexity, (int, float)) else str(complexity)
            
            properties_text = f"""
Molecular Properties:
- MW: {mw_str} g/mol
- LogP: {logp_str}
- H-bond donors: {h_donors}
- H-bond acceptors: {h_acceptors}
- Aromatic rings: {aromatic}
- Complexity score: {complexity_str}/100
"""
        
        functional_groups_text = ""
        if result["functional_groups"]:
            functional_groups_text = f"\nFunctional Groups Present: {', '.join(result['functional_groups'])}"
        
        disconnections_text = ""
        if result["strategic_disconnections"]:
            disconnections_text = "\nPossible Disconnections:\n"
            for disc in result["strategic_disconnections"]:
                disconnections_text += f"- {disc['reaction']}: {disc['description']}\n"
        
        retro_prompt = f"""You are an expert synthetic organic chemist. Provide detailed retrosynthetic analysis for: {compound_name}
{f"SMILES: {smiles}" if smiles else ""}
{properties_text}
{functional_groups_text}
{disconnections_text}

Provide THREE different retrosynthetic routes with the following structure:

**ROUTE 1: [Name the strategy, e.g., "Convergent Synthesis"]**
Key Disconnection: [Main strategic bond to break]
Retrosynthetic Steps:
1. [First disconnection and logic]
2. [Second disconnection]
3. [Continue until commercially available starting materials]

Starting Materials: [List 2-4 commercially available compounds]
Estimated Steps: [Number]
Expected Yield: [Percentage]
Key Challenges: [1-2 main issues]

**ROUTE 2: [Alternative strategy]**
[Same structure as Route 1]

**ROUTE 3: [Third strategy]**
[Same structure as Route 1]

Then provide:
**RECOMMENDED ROUTE:** [Which one and why]
**COMPLEXITY ASSESSMENT:** [Easy/Moderate/Difficult and rationale]

Be specific about reagents, conditions, and protecting groups where relevant."""

        retro_analysis = self._local_llm(retro_prompt)
        result["retrosynthetic_routes"] = retro_analysis
        
        # Skip forward synthesis and availability for speed - can be added later if needed
        result["forward_synthesis"] = "[Skipped for performance - use recommended route from retrosynthetic analysis]"
        result["commercial_availability"] = "[Skipped for performance - focus on route feasibility above]"
        
        return result
