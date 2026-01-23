import os
import sys
import json
import argparse

# Fix OpenMP Conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.assembly.evidence_assembler import EvidenceAssembler

def run_step4_single(figure_id, intermediate_dir, extraction_json_path=None, forced_caption=None):
    try:
        assembler = EvidenceAssembler()
        
        # Load extraction data if provided
        extraction_data = []
        if extraction_json_path and os.path.exists(extraction_json_path):
            with open(extraction_json_path, 'r') as f:
                extraction_data = json.load(f)
        
        # Assemble
        json_path = assembler.assemble(figure_id, extraction_data, intermediate_dir)
        
        if json_path:
            # Read content
            with open(json_path, 'r') as f:
                content = json.load(f)
            
            # INJECT FORCED CAPTION (from Upload Mode)
            if forced_caption:
                content['meta']['caption'] = forced_caption
                # Save back
                with open(json_path, 'w') as f:
                    json.dump(content, f, indent=2)
                
            print("---JSON_START---")
            print(json.dumps({"path": json_path, "content": content}))
            print("---JSON_END---")
        else:
             print(json.dumps({"error": "Evidence discarded by filter."}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    # Usage: python step4_single.py figure_id intermediate_dir [extraction.json] [caption]
    if len(sys.argv) > 2:
        fid = sys.argv[1]
        idir = sys.argv[2]
        ext_json = sys.argv[3] if len(sys.argv) > 3 else None
        caption = sys.argv[4] if len(sys.argv) > 4 else None
        run_step4_single(fid, idir, ext_json, caption)
