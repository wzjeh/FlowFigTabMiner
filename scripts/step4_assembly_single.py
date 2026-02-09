import os
import sys
import json
import argparse

# Fix OpenMP Conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.assembly.evidence_assembler import EvidenceAssembler

def run_step4_single():
    parser = argparse.ArgumentParser(description="Step 4 Assembly Single")
    parser.add_argument("figure_id", help="Figure ID (e.g. page_5_figure_0_t0)")
    parser.add_argument("intermediate_dir", help="Path to macro_cleaned dir with crops")
    parser.add_argument("step3_json", help="Path to JSON file containing Step 3 mapped_data")
    parser.add_argument("--output_dir", default="data/evidence", help="Directory to save evidence")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.intermediate_dir):
        print(json.dumps({"error": f"Intermediate dir not found: {args.intermediate_dir}"}))
        return

    if not os.path.exists(args.step3_json):
        print(json.dumps({"error": f"Step 3 data not found: {args.step3_json}"}))
        return
        
    try:
        # Load Step 3 Data
        with open(args.step3_json, 'r') as f:
            step3_data = json.load(f)
            
        # Extract 'mapped_data' list if it's wrapped in the result object
        extraction_data = step3_data
        if isinstance(step3_data, dict):
            extraction_data = step3_data.get('mapped_data', [])
            
        # Init Assembler
        assembler = EvidenceAssembler(output_dir=args.output_dir)
        
        # Run Assembly
        # Logic matches FigurePipeline loop
        json_path = assembler.assemble(args.figure_id, extraction_data, args.intermediate_dir)
        
        if json_path:
            # Read back the saved JSON to return it
            with open(json_path, 'r') as f:
                final_packet = json.load(f)
                
            print("---JSON_START---")
            print(json.dumps({
                "status": "success",
                "output_path": json_path,
                "evidence": final_packet
            }))
            print("---JSON_END---")
        else:
            print(json.dumps({"status": "filtered", "message": "Discarded by relevance filter."}))

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    run_step4_single()
