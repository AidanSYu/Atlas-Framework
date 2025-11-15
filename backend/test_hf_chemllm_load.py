import traceback
print('Starting Hugging Face ChemLLM load test (this may download large model files)...')
import os, sys
# Ensure project root is on sys.path so `backend` package can be imported
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from backend.agents.retrosynthesis import HuggingFaceChemLLM
except Exception as e:
    print('Failed to import HuggingFaceChemLLM from retrosynthesis:', e)
    traceback.print_exc()
    raise SystemExit(1)

try:
    # Instantiate wrapper (this will try to download/load model weights)
    print('Instantiating HuggingFaceChemLLM (model load may begin)...')
    client = HuggingFaceChemLLM()
    print('Model object created:', type(client))

    # Try a short generation to validate
    print('Running small generation...')
    out = client.generate('Confirm model active: reply OK', max_tokens=32)
    print('Generation output:\n', out[:1000])
except Exception as e:
    print('Error during model load/generate:', e)
    traceback.print_exc()
    raise SystemExit(1)

print('Hugging Face ChemLLM load test completed successfully.')
