import os
import sys
import argparse
import glob
import time

sys.path.insert(0, os.getcwd())
from src.flow_dev_miner.processing_layer.figure_processor import FigureProcessor

def run_phase2(intermediate_dir):
    print(f"--- Phase 2: Figure Processing ---")
    start_time = time.time()
    
    figures_dir = os.path.join(intermediate_dir, "figures")
    figures = glob.glob(os.path.join(figures_dir, "*.png"))
    
    print(f"  --> Found {len(figures)} raw figures.")
    if not figures:
        print("  --> Skipping Figure Processing (no figures found).")
        return
        
    fig_proc = FigureProcessor()
    fig_results = fig_proc.process_batch(figures, intermediate_dir)
    
    success_count = len([r for r in fig_results if r['status'] == 'success'])
    print(f"  --> Processed {success_count} figures successfully.")
    print(f"--- Phase 2 Completed in {time.time() - start_time:.2f}s ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Process Figures")
    parser.add_argument("intermediate_dir", help="Path to intermediate output directory")
    args = parser.parse_args()
    run_phase2(args.intermediate_dir)
