import os
import sys
import json
import argparse
import time

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.flow_dev_miner.processing_layer.figure_processor import FigureProcessor
from src.flow_dev_miner.processing_layer.table_processor import TableProcessor
from src.flow_dev_miner.algorithm_layer.semantic_assembler import SemanticAlgorithmCore
from src.flow_dev_miner.input_layer.document_parser import DocumentInputManager
from src.flow_dev_miner.output_layer.data_exporter import DataExporter
from src.flow_dev_miner.output_layer.visual_assembler import VisualAssembler

def gather_evidence_items(intermediate_dir, pdf_id):
    """Temporary helper to gather raw evidence outputs"""
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


def run_pipeline(pdf_path, output_dir="data/final_output"):
    print(f"--- FlowDevMiner Full Pipeline API execution for {pdf_path} ---")
    start_time = time.time()
    
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    intermediate_dir = os.path.join("data/intermediate", basename)
    os.makedirs(output_dir, exist_ok=True)
    
    print("[1/5] Input Layer: Parsing PDF...")
    input_manager = DocumentInputManager()
    input_res = input_manager.process_pdf(pdf_path, intermediate_dir)
    print(f"  --> Extracted {input_res['crops_saved']} image regions.")
    full_text = input_res.get('full_text', "")
    
    # Check figures and tables 
    figures_dir = os.path.join(intermediate_dir, "figures")
    tables_dir = os.path.join(intermediate_dir, "tables")
    
    import glob
    figures = glob.glob(os.path.join(figures_dir, "*.png"))
    tables = [f for f in glob.glob(os.path.join(tables_dir, "*.png")) 
              if "_body" not in f and "_smiles" not in f and "_crop" not in f]
    
    print(f"  --> Found {len(figures)} raw figures and {len(tables)} raw tables.")
    
    # Select Phase could go here
    
    print("[2/5] Figure Processing...")
    fig_proc = FigureProcessor()
    fig_results = fig_proc.process_batch(figures, intermediate_dir)
    print(f"  --> Processed {len([r for r in fig_results if r['status'] == 'success'])} figures successfully.")
    
    print("[3/5] Table Processing...")
    tab_proc = TableProcessor()
    tab_results = tab_proc.process_batch(tables, tables_dir, intermediate_dir)
    print(f"  --> Processed {len([r for r in tab_results if r['status'] == 'success'])} tables successfully.")
    
    print("[4/5] Text & Semantic Assembly (Using New Architecture)")
    algo_core = SemanticAlgorithmCore()
    exporter = DataExporter()
    
    visuals_dir = os.path.join(output_dir, f"{basename}_visuals")
    visualizer = VisualAssembler(output_dir=visuals_dir)
    
    evidence_items = gather_evidence_items(intermediate_dir, basename)
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
            
            # Find the best original image for verification stitching
            orig_img = None
            if item.get('source_type') == 'table':
                 # Evidence is page_8_table_0_evidence.json, image is page_8_table_0_body_main.png
                 orig_img = src.replace('_evidence.json', '_body_main.png')
            elif item.get('source_type') == 'figure':
                 # Evidence is page_8_figure_0_evidence.json, image is page_8_figure_0_cleaned.png
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

    print(f"[5/5] Success! Pipeline finished. Exporting data...")
    summary_path = os.path.join(output_dir, f"{basename}_final_summary.json")
    debug_path = os.path.join(output_dir, f"{basename}_step5_debug.json")
    
    exporter.export_to_json(final_results, summary_path)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full FlowDevMiner Pipeline")
    parser.add_argument("pdf_path", help="Path to input PDF")
    args = parser.parse_args()
    run_pipeline(args.pdf_path)
