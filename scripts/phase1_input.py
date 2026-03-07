import os
import sys
import argparse
import time

sys.path.insert(0, os.getcwd())
from src.flow_dev_miner.input_layer.document_parser import DocumentInputManager

def run_phase1(pdf_path, intermediate_dir):
    print(f"--- Phase 1: Input & Parsing for {pdf_path} ---")
    start_time = time.time()
    
    os.makedirs(intermediate_dir, exist_ok=True)
    
    input_manager = DocumentInputManager()
    input_res = input_manager.process_pdf(pdf_path, intermediate_dir)
    
    print(f"  --> Extracted {input_res['crops_saved']} image regions.")
    print(f"--- Phase 1 Completed in {time.time() - start_time:.2f}s ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Parse PDF")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument("intermediate_dir", help="Path to intermediate output directory")
    args = parser.parse_args()
    run_phase1(args.pdf_path, args.intermediate_dir)
