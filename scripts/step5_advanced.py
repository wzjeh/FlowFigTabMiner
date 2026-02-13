
import os
import sys
import argparse
import json
import glob
from tqdm import tqdm

# Fix OpenMP Conflict & Force Single Threading
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.adjudication.llm_engine import LLMEngine
from src.adjudication.pdf_parser import PDFParser
from src.adjudication.global_context import GlobalInfoExtractor
from src.adjudication.context_resolver import ContextResolver
from src.adjudication.data_synthesizer import DataSynthesizer

def load_evidence(intermediate_dir, pdf_id):
    """
    Gather all *_evidence.json files from figures and tables.
    Reuse logic from step5_global_single.py but return flat list of items.
    """
    evidence_items = []
    
    # 1. Figures
    ev_dir = "data/evidence"
    if os.path.exists(ev_dir):
        pattern = os.path.join(ev_dir, f"*{pdf_id}*_evidence.json")
        for f in glob.glob(pattern):
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    if data.get('is_relevant', True):
                         # Add source metadata
                         data['source_type'] = 'figure'
                         data['source_file'] = f
                         evidence_items.append(data)
            except: pass

    # 2. Tables
    tables_dir = os.path.join(intermediate_dir, "tables")
    if os.path.exists(tables_dir):
        for root, dirs, files in os.walk(tables_dir):
            for file in files:
                if file.endswith("_evidence.json"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r') as fp:
                            data = json.load(fp)
                            if data.get('is_relevant', False):
                                data['source_type'] = 'table'
                                data['source_file'] = path
                                # Inject CSV content for synthesis
                                csv_path = data.get('csv_path')
                                if csv_path and os.path.exists(csv_path):
                                    with open(csv_path, 'r') as f:
                                        data['data_content'] = f.read()
                                
                                evidence_items.append(data)
                    except: pass
                    
    return evidence_items

def main():
    parser = argparse.ArgumentParser(description="Step 5: Advanced Global Assembly (Context-Aware)")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument("--output_dir", help="Output directory", default="data/final_output")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    intermediate_dir = os.path.join("data/intermediate", basename)
    
    print(f"--- Step 5 (Advanced): Context-Aware Assembly for {basename} ---")
    
    # 1. Parse PDF Text
    print("[1/5] Parsing PDF Text...")
    pdf_parser = PDFParser()
    full_text = pdf_parser.extract_text(pdf_path)
    
    # 2. Gather Evidence
    print("[2/5] Gathering Evidence...")
    evidence_items = load_evidence(intermediate_dir, basename)
    print(f"      Found {len(evidence_items)} items.")
    
    if not evidence_items:
        print("No evidence found. Exiting.")
        return

    # Load Agentic Context (if available)
    agentic_context_path = os.path.join(intermediate_dir, "selected_assets.json")
    master_abbreviations = {}
    
    if os.path.exists(agentic_context_path):
        print(f"[Info] Loading Agentic Context from {agentic_context_path}")
        try:
            with open(agentic_context_path, 'r') as f:
                ac_data = json.load(f)
                rich_list = ac_data.get("rich_metadata", [])
                for item in rich_list:
                    # Aggregate abbreviations
                    abbrevs = item.get("abbreviations", {})
                    if abbrevs:
                        for k, v in abbrevs.items():
                            master_abbreviations[k] = v
            print(f"      Loaded {len(master_abbreviations)} abbreviations from Agentic Brain.")
            print(f"      (Sample: {list(master_abbreviations.items())[:3]}...)")
        except Exception as e:
            print(f"      [Warning] Failed to load agentic context: {e}")

    # Initialize Modules
    llm = LLMEngine()
    extractor = GlobalInfoExtractor(llm)
    # Pass master_abbreviations to resolver
    resolver = ContextResolver(full_text, external_context=master_abbreviations)
    synthesizer = DataSynthesizer(llm)
    
    final_results = []
    
    # 3. Process Each Item (Context -> Resolve -> Synthesize)
    print("[3/5] Processing Items...")
    
    for i, item in enumerate(tqdm(evidence_items)):
        src = item.get('source_file')
        print(f"\nProcessing {os.path.basename(src)}...")
        
        # Add to debug info
        sk = os.path.basename(src).split('_evidence')[0]
        
        # Step 5a: Global Candidates
        global_info = extractor.extract(item)
        candidates = global_info.get("global_candidates", {})
        unknowns = global_info.get("unknown_terms", [])
        
        print(f"  -> Candidates: {candidates}")
        print(f"  -> Unknowns: {unknowns}")
        
        # Step 5b: Resolve Unknowns
        resolved = resolver.resolve(unknowns)
        print(f"  -> Resolved: {resolved}")
        
        # Step 5c: Synthesize Data
        synthesized_data = synthesizer.synthesize(item, candidates, resolved)
        
        # Add source info to each row
        for row in synthesized_data:
            row['_source_key'] = sk
            # Embed debug info into the first row of each source block, or all rows? 
            # Or better: Save a separate debug object in the list? 
            # Actually, let's keep the list pure rows for CSV export.
            # We will attach the debug info to a separate list or dictionary and save it as well.
            
        final_results.extend(synthesized_data)
        
        # Store debug info for UI
        item_debug = {
            "source": sk,
            "step5a_candidates": candidates,
            "step5b_unknowns": unknowns,
            "step5b_resolved": resolved
        }
        # We can save this to a sidecar file for valid JSON, 
        # OR we can append a special metadata object to the main list (might break consumers).
        # Let's save a separate _debug.json
        debug_out_path = os.path.join(output_dir, f"{basename}_step5_debug.json")
        
        # Append to a list
        if 'debug_log' not in locals(): debug_log = []
        debug_log.append(item_debug)

    # 4. Save Final Output
    out_path = os.path.join(output_dir, f"{basename}_final_summary.json")
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Save Debug Log
    debug_out_path = os.path.join(output_dir, f"{basename}_step5_debug.json")
    with open(debug_out_path, 'w') as f:
        json.dump(debug_log, f, indent=2)
        
    print(f"[Success] Saved {len(final_results)} rows to {out_path}")
    print(f"[Debug] Saved debug info to {debug_out_path}")
    
    # UI Output
    print("---JSON_START---")
    print(json.dumps({
        "status": "success", 
        "output_path": out_path, 
        "debug_path": debug_out_path,
        "rows": len(final_results)
    }))
    print("---JSON_END---")

if __name__ == "__main__":
    main()
