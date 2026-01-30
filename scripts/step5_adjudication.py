import sys
import os
import argparse
import glob
import json

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.adjudication.pdf_parser import PDFParser
from src.adjudication.meta_aggregator import MetaAggregator
from src.adjudication.llm_engine import LLMEngine
from src.adjudication.data_fusion import DataFusion

def run_step5(pdf_path, evidence_dir="data/evidence", output_dir="data/output"):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"--- Step 5: Semantic Adjudication for {pdf_name} ---")

    # 1. Parse PDF Text
    parser = PDFParser()
    full_text = parser.extract_text(pdf_path)
    if not full_text:
        print("Error: Could not extract text from PDF.")
        return

    # 2. Identify Unknown Terms (from Evidence)
    # We need to find evidence files related to this PDF.
    # Pattern: evidence_dir/{pdf_name}_page_*_evidence.json 
    # OR if evidence filenames don't strictly start with pdf_name (e.g. just page_2_...), 
    # we might need to filter by content or assume standard naming "pdfname_page_...". 
    # Current naming in Step 4: "{figure_id}_evidence.json". 
    # User's current figure_id is "page_2_figure_2...". It MISSES the PDF name prefix if the user just ran "page_2...".
    # BUT Step 4 usually implies we are processing a specific PDF's intermediate results.
    # Let's assume standard flow: intermediate_dir name matches PDF name.
    # evidence files might just be in flat dir.
    # Let's try to match by checking "source_intermediate_dir" inside JSONs or grep filenames?
    # Simple heuristic: Look for files containing 'page_' in evidence dir for now, 
    # or better: The caller should probably know which figures belong to this PDF.
    # FOR NOW: Let's assume we want to process ALL evidence files that match the "page_X" pattern if we are in single-pdf mode?
    # Actually, MetaAggregator logic scans *all* evidence.json. That might be too broad if we have multiple PDFs.
    # Let's refine: Filter files by those whose 'source_intermediate_dir' contains pdf_name.
    
    # Pre-scan all evidence to find relevant ones
    all_jsons = glob.glob(os.path.join(evidence_dir, "*_evidence.json"))
    relevant_jsons = []
    for jf in all_jsons:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                if pdf_name in data.get("meta", {}).get("source_intermediate_dir", ""):
                    relevant_jsons.append(jf)
                # Fallback: if filename matches specific pattern?
                elif os.path.basename(jf).startswith(pdf_name):
                    relevant_jsons.append(jf)
        except: pass
    
    if not relevant_jsons:
        # Fallback for dev: Just take everything if only 1 PDF context
        print(f"Warning: No evidence explicitly linked to {pdf_name} found. checking standard 'page_' files...")
        relevant_jsons = glob.glob(os.path.join(evidence_dir, "page_*_evidence.json"))
    
    if not relevant_jsons:
        print("No evidence files found to adjudicate.")
        return

    print(f"Found {len(relevant_jsons)} relevant evidence files.")

    # 3. Aggregate Terms
    # Temporarily instantiate Aggregator just for these files
    # We assume Aggregator expects a directory, but our relevant_jsons are list.
    # Let's modify logic or just pass directory? Aggregator scans dir.
    # For now let's pass the directory, but Aggregator will scan ALL. 
    # Optimization: Refactor Aggregator later. For now, we trust it or filter later.
    aggregator = MetaAggregator(evidence_dir) 
    # We manually filtered above, but aggregator scans everything. 
    # Let's just use the terms from our relevant list to avoid polluting with other PDFs
    
    unique_terms = set()
    for jf in relevant_jsons:
         with open(jf, 'r') as f:
             data = json.load(f)
             # Reuse Aggregator's private method logic? Or just duplicate simple extraction
             # Quick hack: use aggregator._extract_from_text on content
             text_ev = data.get("text_evidence", {})
             for key in ['x_axis_title', 'y_axis_title', 'chart_text', 'legend_text']:
                 for item in text_ev.get(key, []):
                     aggregator._extract_from_text(item.get("text", ""), unique_terms)
             aggregator._extract_from_text(data.get("meta", {}).get("caption", ""), unique_terms)
    
    filtered_terms = aggregator._filter_terms(unique_terms)
    term_list = sorted(list(filtered_terms))
    print(f"Terms to resolve: {term_list}")

    # 4. LLM Adjudication
    llm = LLMEngine() # provider/config from .env
    knowledge_graph = llm.adjucate(full_text, term_list, pdf_name)
    
    print("\n[Knowledge Extracted]")
    print(json.dumps(knowledge_graph, indent=2))

    # 5. Data Fusion
    fusion = DataFusion(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    for jf in relevant_jsons:
        csv_path = fusion.fuse(jf, knowledge_graph)
        print(f"Saved: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to source PDF")
    args = parser.parse_args()
    
    run_step5(args.pdf_path)
