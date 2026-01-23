import os
import sys
import argparse
import glob

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.adjudication.pdf_parser import PDFParser
from src.adjudication.meta_aggregator import MetaAggregator
from src.adjudication.llm_engine import LLMEngine
from src.adjudication.data_fusion import DataFusion

def main():
    parser = argparse.ArgumentParser(description="Step 5: Semantic Adjudication")
    parser.add_argument("pdf_path", help="Path to the source PDF file")
    parser.add_argument("evidence_dir", help="Directory containing Step 4 evidence JSONs")
    parser.add_argument("--output_dir", default="data/output", help="Directory for final CSVs")
    args = parser.parse_args()

    pdf_name = os.path.basename(args.pdf_path).replace(".pdf", "")
    
    print(f"=== Step 5: Semantic Adjudication for {pdf_name} ===")
    
    # 1. Parse PDF
    pdf_parser = PDFParser()
    full_text = pdf_parser.extract_text(args.pdf_path)
    if not full_text:
        print("Error: Could not extract text from PDF.")
        return

    # 2. Aggregate Meta
    aggregator = MetaAggregator(args.evidence_dir)
    meta_data = aggregator.collect_terms()
    unique_terms = meta_data["terms"]
    print(f"Collected {len(unique_terms)} unique terms for adjudication.")
    
    # 3. LLM Adjudication
    llm = LLMEngine() # Loads API key from .env automatically
    knowledge = llm.adjucate(full_text, unique_terms, pdf_name)
    
    print("Knowledge Graph Constructed:")
    print(knowledge)
    
    # 4. Data Fusion
    fusion = DataFusion(output_dir=args.output_dir)
    json_pattern = os.path.join(args.evidence_dir, "*_evidence.json")
    evidence_files = glob.glob(json_pattern)
    
    results = []
    for ev_file in evidence_files:
        try:
            csv_path = fusion.fuse(ev_file, knowledge)
            results.append(csv_path)
        except Exception as e:
            print(f"Error fusing {ev_file}: {e}")
            
    print(f"=== Step 5 Complete. Generated {len(results)} Rich CSVs in {args.output_dir} ===")

if __name__ == "__main__":
    main()
