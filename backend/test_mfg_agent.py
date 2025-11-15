#!/usr/bin/env python
"""
Quick test of ManufacturabilityAgent with ChemLLM integration.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Test import
print("Testing ManufacturabilityAgent import...")
try:
    from agents.manufacturer import ManufacturabilityAgent
    print("✓ Import successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Initialize agent
print("Initializing ManufacturabilityAgent...")
try:
    agent = ManufacturabilityAgent()
    print(f"✓ Agent initialized")
    print(f"  ChemLLM available: {agent.chem_client is not None}\n")
except Exception as e:
    print(f"✗ Init failed: {e}")
    sys.exit(1)

# Test scalability assessment
print("=" * 70)
print("TEST: Scalability Assessment for Aspirin")
print("=" * 70)
try:
    print("\nCalling assess_scalability()...")
    result = agent.assess_scalability(
        compound_name="Acetylsalicylic acid (Aspirin)",
        compound_type="small_molecule",
        synthesis_complexity="simple",
        smiles="CC(=O)Oc1ccccc1C(=O)O"
    )
    
    print(f"\n✓ Assessment complete")
    print(f"\nResults:")
    print(f"  Scalability Score: {result['scalability_score']}/100")
    print(f"  Cost Estimate: {result['cost_estimate']}")
    print(f"  Production Timeline: {result['production_timeline']}")
    
    if result['manufacturability_risks']:
        print(f"\n  Top Risks:")
        for i, risk in enumerate(result['manufacturability_risks'][:2], 1):
            print(f"    {i}. {risk}")
    
    if result['recommendations']:
        print(f"\n  Recommendations:")
        for i, rec in enumerate(result['recommendations'][:2], 1):
            print(f"    {i}. {rec}")
    
    print(f"\n  Assessment (first 300 chars):")
    print(f"  {result['detailed_assessment'][:300]}")
    
except Exception as e:
    print(f"✗ Assessment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("SUCCESS: ManufacturabilityAgent works!")
print("=" * 70)
