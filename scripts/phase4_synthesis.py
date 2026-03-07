import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.getcwd())
from src.flow_dev_miner.algorithm_layer.semantic_assembler import SemanticAlgorithmCore
from src.flow_dev_miner.output_layer.data_exporter import DataExporter
from src.flow_dev_miner.output_layer.visual_assembler import VisualAssembler

def gather_evidence_items(intermediate_dir):
    """Gathers raw evidence outputs from Phase 2 and 3"""
    evidence_items = []
    
    if os.path.exists(intermediate_dir):
        for root, dirs, files in os.walk(intermediate_dir):
            for file in files:
                if file.endswith("_evidence.json"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r') as fp:
                            data = json.load(fp)
                            if data.get('is_relevant', False):
                                is_table = 'csv_path' in data
                                data['source_type'] = 'table' if is_table else 'figure'
                                data['source_file'] = path
                                if is_table:
                                    csv_path = data.get('csv_path')
                                    if csv_path and os.path.exists(csv_path):
                                        with open(csv_path, 'r') as f:
                                            data['data_content'] = f.read()
                                evidence_items.append(data)
                                print(f"  [Gather] Included: {path} (type: {data['source_type']})")
                    except Exception as e:
                        print(f"Error loading evidence {path}: {e}")
                        pass
    return evidence_items


def run_phase4(pdf_path, intermediate_dir, output_dir):
    print(f"--- Phase 4: Text & Semantic Assembly ---")
    start_time = time.time()
    
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Read full text extracted in Phase 1
    input_cache = os.path.join(intermediate_dir, "input_cache.json")
    full_text = ""
    if os.path.exists(input_cache):
        with open(input_cache, 'r') as f:
            full_text = json.load(f).get('full_text', "")
    
    algo_core = SemanticAlgorithmCore()
    exporter = DataExporter()
    
    visuals_dir = os.path.join(output_dir, f"{basename}_visuals")
    visualizer = VisualAssembler(output_dir=visuals_dir)
    
    evidence_items = gather_evidence_items(intermediate_dir)
    print(f"  --> Processing {len(evidence_items)} evidence items.")
    
    final_results = []
    debug_log = []
    
    for item in evidence_items:
        src = item.get('source_file')
        sk = os.path.basename(src).split('_evidence')[0]
        
        # Build GVP
        gvp = algo_core.extract_gvp(item, full_text=full_text)
        
        # Assemble
        synthesized_data = algo_core.semantic_assembly(item, gvp)
        
        # Inject metadata & Build Visuals
        for idx, row in enumerate(synthesized_data):
            row['_source_key'] = sk
            row_id = f"{sk}_row{idx}"
            
            orig_img = None
            if item.get('source_type') == 'table':
                 orig_img = src.replace('_evidence.json', '_body_main.png')
            elif item.get('source_type') == 'figure':
                 orig_img = src.replace('_evidence.json', '_cleaned.png')
            
            if orig_img and os.path.exists(orig_img):
                 v_path = visualizer.create_verification_image(orig_img, row, row_id)
                 if v_path: row['_visual_verification'] = v_path
            
        final_results.extend(synthesized_data)
        
        debug_log.append({
            "source": sk,
            "step5a_candidates": gvp.get("candidates"),
            "step5b_unknowns": gvp.get("raw_unknowns"),
            "step5b_resolved": gvp.get("resolved")
        })

    print(f"[Phase 4] Success! Exporting data...")
    summary_path = os.path.join(output_dir, f"{basename}_final_summary.json")
    excel_path = os.path.join(output_dir, f"{basename}_final_summary.xlsx")
    debug_path = os.path.join(output_dir, f"{basename}_step5_debug.json")
    
    exporter.export_to_json(final_results, summary_path)
    exporter.export_to_excel(final_results, excel_path)
    exporter.export_debug_info(debug_log, debug_path)

    print("---JSON_START---")
    print(json.dumps({
        "status": "success", 
        "output_path": summary_path,
        "debug_path": debug_path,
        "rows": len(final_results),
        "time": time.time() - start_time
    }))
    print("---JSON_END---")
    print(f"--- Phase 4 Completed in {time.time() - start_time:.2f}s ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Synthesize Data")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument("intermediate_dir", help="Path to intermediate dir")
    parser.add_argument("output_dir", help="Path to final output dir")
    
    args = parser.parse_args()
    run_phase4(args.pdf_path, args.intermediate_dir, args.output_dir)
