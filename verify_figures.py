import os
import sys
import json
sys.path.insert(0, os.getcwd())
from scripts.run_pipeline_api import gather_evidence_items
from src.flow_dev_miner.algorithm_layer.semantic_assembler import SemanticAlgorithmCore
from src.flow_dev_miner.output_layer.data_exporter import DataExporter
from src.flow_dev_miner.processing_layer.text_processor import TextProcessor

def verify():
    pdf_id = 'example'
    int_dir = f'data/intermediate/{pdf_id}'
    
    # Bypass Florence-2 connectivity check for verification
    os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
    
    items = gather_evidence_items(int_dir, pdf_id)
    ids = [i.get("meta", {}).get("figure_id") or i.get("table_name") for i in items]
    print(f"[Gather] Found {len(items)} items: {ids}")
    
    if 'page_6_figure_0_t0' not in ids:
        print("[Warning] page_6_figure_0_t0 is STILL MISSING from gathered items!")
        # Force a check of why
        fig_json = "data/intermediate/example/macro_cleaned/page_6_figure_0_t0_evidence.json"
        if os.path.exists(fig_json):
            with open(fig_json, 'r') as f:
                d = json.load(f)
                print(f"[Check] {fig_json} is_relevant: {d.get('is_relevant')}")
    
    from src.adjudication.pdf_parser import PDFParser
    parser = PDFParser()
    full_text = parser.extract_text(f'data/input/{pdf_id}.pdf')
    
    algo = SemanticAlgorithmCore()
    exporter = DataExporter()
    
    final_results = []
    for it in items:
        source_id = it.get("meta", {}).get("figure_id") or it.get("table_name")
        print(f"[Synthesis] Processing {source_id}...")
        gvp = algo.extract_gvp(it, full_text=full_text)
        synthed = algo.semantic_assembly(it, gvp)
        for row in synthed:
            row['_source_key'] = source_id
        final_results.extend(synthed)
        
    summary_path = f'data/final_output/{pdf_id}_final_summary_verified.json'
    exporter.export_to_json(final_results, summary_path)
    print(f"[Result] Exported {len(final_results)} rows to {summary_path}")

if __name__ == "__main__":
    verify()
