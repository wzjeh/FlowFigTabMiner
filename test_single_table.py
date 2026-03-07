import os
import sys
import json
import argparse

sys.path.insert(0, os.getcwd())
from src.flow_dev_miner.processing_layer.table_processor import TableProcessor

def run_test(img_path):
    print(f"--- Testing Table Processor on {img_path} ---", flush=True)
    
    basename = os.path.splitext(os.path.basename(img_path))[0]
    intermediate_dir = "data/intermediate/example"
    
    # Process
    tab_proc = TableProcessor()
    # process_batch expects list of image paths
    res = tab_proc.process_batch([img_path], os.path.dirname(img_path), intermediate_dir)
    
    print(f"Result: {json.dumps(res, indent=2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="Path to image")
    args = parser.parse_args()
    run_test(args.img)
