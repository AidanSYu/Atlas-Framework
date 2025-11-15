import sys, traceback

print('Running chemllm check...')

try:
    import chemllm
    ver = getattr(chemllm, '__version__', None)
    print('IMPORT_OK', ver if ver is not None else 'version-unknown')
except Exception as e:
    print('IMPORT_ERROR', str(e))
    traceback.print_exc()
    sys.exit(0)

try:
    # Try several common client entrypoints to be robust
    Client = None
    if hasattr(chemllm, 'ChemLLMClient'):
        Client = chemllm.ChemLLMClient
    elif hasattr(chemllm, 'Client'):
        Client = chemllm.Client
    elif hasattr(chemllm, 'create_client'):
        Client = chemllm.create_client

    if Client is None:
        print('GEN_ERROR', 'No known client class found in chemllm package')
        sys.exit(0)

    client = Client()
    # try both generate and create APIs
    if hasattr(client, 'generate'):
        out = client.generate(prompt='Say OK', model='chemfast', max_tokens=20)
    elif hasattr(client, 'create'):
        out = client.create(prompt='Say OK', model='chemfast', max_tokens=20)
    else:
        out = repr(client)
    print('GEN_OK', out)
except Exception as e:
    print('GEN_ERROR', str(e))
    traceback.print_exc()
