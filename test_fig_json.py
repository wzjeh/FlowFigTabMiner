import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, os.getcwd())
from src.flow_dev_miner.processing_layer.figure_processor import FigureProcessor

fp = FigureProcessor()
# Check if Step 1 output exists
pdf_id = "example"
figures_dir = os.path.join("data/intermediate", pdf_id, "figures")
test_img = os.path.join(figures_dir, "page_6_figure_0.png")

if not os.path.exists(test_img):
    print(f"[Prep] Running Step 1 for {pdf_id}...")
    from src.flow_dev_miner.input_layer.document_manager import DocumentInputManager
    dim = DocumentInputManager()
    dim.process_pdf("data/input/example.pdf", os.path.join("data/intermediate", pdf_id))

print(f"[Test] Processing {test_img}...")
res = fp.process_figure(test_img, os.path.join("data/intermediate", pdf_id))
print(f"[Result] Status: {res.get('status')}")
if 'evidence_path' in res:
   print(f"[Result] Evidence: {res['evidence_path']}")
   import json
   with open(res['evidence_path'], 'r') as f:
       data = json.load(f)
       print(f"[Result] is_relevant: {data.get('is_relevant')}")
EOF
