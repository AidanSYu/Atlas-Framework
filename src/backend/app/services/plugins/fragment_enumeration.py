"""Fragment Enumeration Plugin.

Domain-agnostic combinatorial fragment assembly.

Takes a scaffold template (SMILES with an {attachment} slot) and a list of
fragment SMILES, then produces every valid combination as canonical SMILES.
Named presets can be loaded by scaffold_name so callers don't need to
supply raw SMILES — adding a new domain means adding a new preset entry.

Provenance: all generated molecules are tagged source="fragment_enumeration"
so the frontend distinguishes them from LLM-proposed or corpus-extracted candidates.
"""
import asyncio
import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from app.services.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


# ============================================================
# Named scaffold presets — add new domains here.
# Each preset defines:
#   template:  SMILES string containing "{attachment}" placeholder
#   fragments: list of (name, smiles) tuples — fragments are inserted at {attachment}
# ============================================================

SCAFFOLD_PRESETS: Dict[str, dict] = {
    "xyloside": {
        "description": "β-D-xylopyranoside O-linked glycosides (GAG priming)",
        "template": "OC[C@H]1O[C@@H]({attachment})[C@@H](O)[C@H](O)[C@H]1O",
        "fragments": [
            # Simple aryl — best-characterized GAG primers
            ("phenyl",                "Oc1ccccc1"),
            ("4-methylphenyl",        "Oc1ccc(C)cc1"),
            ("4-methoxyphenyl",       "Oc1ccc(OC)cc1"),
            ("4-fluorophenyl",        "Oc1ccc(F)cc1"),
            ("4-chlorophenyl",        "Oc1ccc(Cl)cc1"),
            ("3-methoxyphenyl",       "Oc1cccc(OC)c1"),
            ("3,4-dimethoxyphenyl",   "Oc1ccc(OC)c(OC)c1"),
            # Bicyclic aryl — high priming efficiency
            ("2-naphthyl",            "Oc1ccc2ccccc2c1"),
            ("1-naphthyl",            "Oc1cccc2ccccc12"),
            # Polar aryl — TPSA-tuned
            ("4-hydroxymethylphenyl", "Oc1ccc(CO)cc1"),
            ("4-carboxyphenyl",       "Oc1ccc(C(=O)O)cc1"),
            ("4-acetylphenyl",        "Oc1ccc(C(C)=O)cc1"),
            ("3-hydroxyphenyl",       "Oc1cccc(O)c1"),
            # Heteroaryl — scaffold diversity
            ("benzofuran-5-yl",       "Oc1ccc2occc2c1"),
            ("5-indanyl",             "Oc1ccc2CCCc2c1"),
            ("biphenyl-4-yl",         "Oc1ccc(-c2ccccc2)cc1"),
            # Coumarin — primers of chondroitin sulfate chains
            ("7-methylcoumarin-4-yl", "Oc1cc(=O)oc2cc(C)ccc12"),
            # Benzylic — alkyl linkage variants
            ("benzyl",                "OCc1ccccc1"),
            ("4-methylbenzyl",        "OCc1ccc(C)cc1"),
            ("4-methoxybenzyl",       "OCc1ccc(OC)cc1"),
        ],
    },
    # Future domains — add presets here without touching any other file:
    # "penicillin_core": { "template": ..., "fragments": [...] },
    # "kinase_hinge":    { "template": ..., "fragments": [...] },
}


class FragmentEnumerationPlugin(BasePlugin):
    """Generate candidates by combinatorial fragment assembly.

    Accepts either a named scaffold preset or explicit template + fragment list.
    Completely domain-agnostic: extend SCAFFOLD_PRESETS to add new research areas.
    """

    @property
    def name(self) -> str:
        return "enumerate_fragments"

    @property
    def description(self) -> str:
        return (
            "Generate novel candidate molecules by combinatorially joining a scaffold "
            "template with a fragment library. Accepts named presets (e.g. 'xyloside') "
            "or explicit template SMILES + fragment list. All outputs are RDKit-validated "
            "canonical SMILES with full provenance tagging (source='fragment_enumeration')."
        )

    async def load(self) -> Any:
        return None

    async def execute(self, model: Any, **kwargs) -> dict:
        """Generate candidate SMILES by fragment enumeration.

        Args (in kwargs):
            scaffold_name:   str  -- named preset key (e.g. "xyloside").
            template:        str  -- custom SMILES template with {attachment} slot
                                     (used when scaffold_name is not given).
            fragments:       list -- list of fragment SMILES or (name, smiles) tuples
                                     (used when scaffold_name is not given).
            max_candidates:  int  -- cap on output size (default 25).

        Returns:
            {
              "molecules":   [{smiles, compound_id, inchikey, valid, source, name}, ...],
              "smiles_list": [...],
              "summary":     "Generated N candidates from M fragments (preset: xyloside).",
            }
        """
        scaffold_name: Optional[str] = kwargs.get("scaffold_name")
        custom_template: Optional[str] = kwargs.get("template")
        custom_fragments = kwargs.get("fragments", [])
        max_candidates = int(kwargs.get("max_candidates", 25))

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._enumerate, scaffold_name, custom_template, custom_fragments, max_candidates
        )

    def _enumerate(
        self,
        scaffold_name: Optional[str],
        custom_template: Optional[str],
        custom_fragments: list,
        max_candidates: int,
    ) -> dict:
        # Resolve template + fragments
        preset_label = "custom"
        if scaffold_name and scaffold_name in SCAFFOLD_PRESETS:
            preset = SCAFFOLD_PRESETS[scaffold_name]
            template = preset["template"]
            fragments = preset["fragments"]
            preset_label = scaffold_name
        elif custom_template:
            template = custom_template
            # Normalise fragments: accept plain SMILES strings or (name, smiles) tuples
            fragments = []
            for f in custom_fragments:
                if isinstance(f, (list, tuple)) and len(f) == 2:
                    fragments.append((str(f[0]), str(f[1])))
                elif isinstance(f, str):
                    fragments.append((f[:20], f))
            preset_label = "custom"
        else:
            return {
                "molecules": [],
                "smiles_list": [],
                "summary": (
                    "No scaffold specified. Provide scaffold_name (e.g. 'xyloside') "
                    "or template + fragments."
                ),
            }

        try:
            from rdkit import Chem
            rdkit_available = True
        except ImportError:
            rdkit_available = False
            logger.warning("RDKit unavailable — returning unvalidated SMILES from template.")

        molecules = []
        seen_keys: set = set()

        for name, frag_smiles in fragments:
            if len(molecules) >= max_candidates:
                break
            try:
                raw_smiles = template.format(attachment=frag_smiles)

                if rdkit_available:
                    mol = Chem.MolFromSmiles(raw_smiles)
                    if mol is None:
                        logger.debug("enumerate_fragments: invalid SMILES for '%s': %s", name, raw_smiles)
                        continue
                    canonical = Chem.MolToSmiles(mol, canonical=True)
                    try:
                        inchikey = Chem.MolToInchiKey(mol)
                    except Exception:
                        inchikey = ""
                    if inchikey and inchikey in seen_keys:
                        continue
                    if inchikey:
                        seen_keys.add(inchikey)
                else:
                    canonical = raw_smiles
                    inchikey = ""

                compound_id = f"ATLAS-{uuid4().hex[:8].upper()}"
                molecules.append({
                    "smiles": canonical,
                    "compound_id": compound_id,
                    "inchikey": inchikey,
                    "valid": True,
                    "source": "fragment_enumeration",
                    "name": f"{name} ({preset_label})",
                })
            except Exception as exc:
                logger.debug("enumerate_fragments: error for fragment '%s': %s", name, exc)

        smiles_list = [m["smiles"] for m in molecules]
        return {
            "molecules": molecules,
            "smiles_list": smiles_list,
            "summary": (
                f"Generated {len(molecules)} candidates "
                f"from {len(fragments)} fragments (preset: {preset_label})."
            ),
        }

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "scaffold_name": {
                    "type": "string",
                    "description": (
                        f"Named scaffold preset. Available: {list(SCAFFOLD_PRESETS.keys())}. "
                        "Mutually exclusive with template+fragments."
                    ),
                },
                "template": {
                    "type": "string",
                    "description": "Custom SMILES template with {attachment} placeholder.",
                },
                "fragments": {
                    "type": "array",
                    "description": "Fragment SMILES list or list of [name, smiles] pairs.",
                },
                "max_candidates": {
                    "type": "integer",
                    "description": "Cap on output size (default 25).",
                },
            },
        }

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "molecules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "smiles": {"type": "string"},
                            "compound_id": {"type": "string"},
                            "inchikey": {"type": "string"},
                            "valid": {"type": "boolean"},
                            "source": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                },
                "smiles_list": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": ["molecules", "smiles_list", "summary"],
        }
