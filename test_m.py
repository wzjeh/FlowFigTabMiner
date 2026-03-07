import sys
import os
import traceback

sys.path.insert(0, os.getcwd())

print("Attempting to import MolNexTRSingleton...", flush=True)
try:
    from src.extraction.common.molnextr.molnextr import MolNexTRSingleton
    print("Import successful", flush=True)
    m = MolNexTRSingleton.get_instance()
    print(m, flush=True)
except Exception as e:
    traceback.print_exc()
