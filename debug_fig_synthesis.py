import os
import sys
import json
sys.path.insert(0, os.getcwd())
from src.flow_dev_miner.algorithm_layer.semantic_assembler import SemanticAlgorithmCore
from src.adjudication.pdf_parser import PDFParser
from src.adjudication.data_synthesizer import DataSynthesizer

def debug():
    fig_json = "data/intermediate/example/macro_cleaned/page_6_figure_0_t0_evidence.json"
    if not os.path.exists(fig_json):
        print(f"Error: {fig_json} not found")
        return
        
    with open(fig_json, 'r') as f:
        item = json.load(f)
    
    parser = PDFParser()
    full_text = parser.extract_text('data/input/example.pdf')
    
    algo = SemanticAlgorithmCore()
    print(f"--- Extracting GVP ---")
    gvp = algo.extract_gvp(item, full_text=full_text)
    print(f"GVP Summary: {list(gvp.keys())}")
    for k in ['candidates', 'resolved']:
        print(f"  {k}: {gvp.get(k)}")
    
    print(f"\n--- Running Assembly ---")
    synthesizer = DataSynthesizer()
    results = synthesizer.synthesize(item, gvp.get('candidates', {}), gvp.get('resolved', {}))
    print(f"\n--- Final Results ({len(results)} rows) ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    debug()
