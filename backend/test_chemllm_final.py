#!/usr/bin/env python
"""
Test ChemLLM model load and generation.
Downloads the model from Hugging Face if not cached locally.
"""
import sys

print("=" * 60)
print("ChemLLM Model Test")
print("=" * 60)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
except Exception as e:
    print(f"✗ PyTorch error: {e}")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    print(f"✓ Transformers loaded")
except Exception as e:
    print(f"✗ Transformers error: {e}")
    sys.exit(1)

# Download and load model
model_id = "AI4Chem/ChemLLM-7B-Chat-1.5-DPO"
print(f"\nDownloading model: {model_id}")
print("(This may take 5-15 minutes on first run)")

try:
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"   ✓ Tokenizer loaded")
    
    print("\n2. Loading model...")
    if torch.cuda.is_available():
        print(f"   Using GPU: {torch.cuda.get_device_name(0)} (float16)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("   Using CPU (float32, low memory)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    print(f"   ✓ Model loaded")
    
    # Test a simple generation (optimized for speed)
    print("\n3. Testing generation with chemistry prompt...")
    prompt = "What is aspirin? Give a 1-sentence answer."
    
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Fast generation: greedy, short output, no cache
    print("   Generating (this may take 10-30 seconds on first run)...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,  # Short output for speed
        do_sample=False,  # Greedy decoding (fastest)
        use_cache=False,  # Avoid cache issue
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"   ✓ Generation successful!")
    print(f"\n   Response:")
    print(f"   {response}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: ChemLLM is ready to use!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
