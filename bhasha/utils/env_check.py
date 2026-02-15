import importlib

packages = ['datasets', 'pandas', 'numpy', 'transformers', 'huggingface_hub', 'sklearn']
print("Checking packages:")
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg} is installed.")
    except ImportError:
        print(f"[MISSING] {pkg} is NOT installed.")
