import os
import sys
import argparse
import glob
import time

sys.path.insert(0, os.getcwd())
from src.flow_dev_miner.processing_layer.table_processor import TableProcessor

def run_phase3(intermediate_dir):
    print(f"--- Phase 3: Table Processing ---")
    start_time = time.time()
    
    tables_dir = os.path.join(intermediate_dir, "tables")
    tables = [f for f in glob.glob(os.path.join(tables_dir, "*.png")) 
              if "_body" not in f and "_smiles" not in f and "_crop" not in f]
    
    print(f"  --> Found {len(tables)} raw tables.")
    if not tables:
        print("  --> Skipping Table Processing (no tables found).")
        return
        
    tab_proc = TableProcessor()
    tab_results = tab_proc.process_batch(tables, tables_dir, intermediate_dir)
    
    success_count = len([r for r in tab_results if r['status'] == 'success'])
    print(f"  --> Processed {success_count} tables successfully.")
    print(f"--- Phase 3 Completed in {time.time() - start_time:.2f}s ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Process Tables")
    parser.add_argument("intermediate_dir", help="Path to intermediate output directory")
    args = parser.parse_args()
    run_phase3(args.intermediate_dir)
