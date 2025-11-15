#!/usr/bin/env python
"""
Test script for RetrosynthesisEngine

Run this to test the retrosynthesis agent with various compounds.
"""

import sys
sys.path.insert(0, '.')

from backend.agents.retrosynthesis import RetrosynthesisEngine

def test_retrosynthesis():
    """Test the retrosynthesis engine with example compounds."""
    
    engine = RetrosynthesisEngine()
    
    # Test compounds with SMILES
    test_compounds = [
        {
            "name": "Aspirin (Acetylsalicylic Acid)",
            "smiles": "CC(=O)Oc1ccccc1C(=O)O"
        },
        {
            "name": "Ibuprofen",
            "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
        },
        {
            "name": "Paracetamol (Acetaminophen)",
            "smiles": "CC(=O)Nc1ccc(O)cc1"
        },
        {
            "name": "Caffeine",
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        },
        {
            "name": "Amphetamine",
            "smiles": "CC(N)Cc1ccccc1"
        }
    ]
    
    print("=" * 80)
    print("RETROSYNTHESIS ENGINE TEST")
    print("=" * 80)
    
    for compound in test_compounds:
        print(f"\n{'=' * 80}")
        print(f"Testing: {compound['name']}")
        print(f"SMILES: {compound['smiles']}")
        print(f"{'=' * 80}")
        
        # Run analysis
        result = engine.retrosynthesis_analysis(
            compound_name=compound['name'],
            smiles=compound['smiles']
        )
        
        # Display results
        print("\n[MOLECULAR PROPERTIES]")
        props = result.get('molecular_properties', {})
        if props and 'error' not in props:
            print(f"  Molecular Weight: {props.get('molecular_weight', 'N/A'):.2f} g/mol")
            print(f"  LogP: {props.get('logp', 'N/A'):.2f}")
            print(f"  H-bond Donors: {props.get('h_bond_donors', 'N/A')}")
            print(f"  H-bond Acceptors: {props.get('h_bond_acceptors', 'N/A')}")
            print(f"  Aromatic Rings: {props.get('aromatic_rings', 'N/A')}")
            print(f"  Complexity Score: {props.get('complexity_score', 'N/A'):.1f}/100")
        else:
            print("  (RDKit not available)")
        
        print("\n[FUNCTIONAL GROUPS]")
        fg = result.get('functional_groups', [])
        if fg:
            print(f"  {', '.join(fg)}")
        else:
            print("  (None detected)")
        
        print("\n[STRATEGIC DISCONNECTIONS]")
        disc = result.get('strategic_disconnections', [])
        if disc:
            for d in disc:
                print(f"  - {d['reaction']}: {d['description']}")
        else:
            print("  (None available)")
        
        print("\n[RETROSYNTHETIC ROUTES]")
        routes = result.get('retrosynthetic_routes', [])
        if routes:
            print(routes)
        else:
            print("  (Generating with LLM...)")
        
        print("\n[FORWARD SYNTHESIS]")
        fwd = result.get('forward_synthesis', '')
        if fwd:
            print(fwd[:500] + "..." if len(fwd) > 500 else fwd)
        else:
            print("  (Will be generated)")
    
    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    test_retrosynthesis()
