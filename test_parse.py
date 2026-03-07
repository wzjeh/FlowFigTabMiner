import sys
import os
sys.path.insert(0, os.getcwd())
from src.flow_dev_miner.input_layer.document_parser import DocumentInputManager
manager = DocumentInputManager()
res = manager.process_pdf("data/input/example.pdf", "data/intermediate/example")
print("Crops Saved:", res['crops_saved'])
