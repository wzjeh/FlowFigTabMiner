import argparse
import os
import sys
import json

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.batch_table_pipeline import BatchTablePipeline

def main():
    parser = argparse.ArgumentParser(description="Run Batch Table Extraction")
    parser.add_argument("--input_dir", required=True, help="Root directory to scan for tables (e.g., data/intermediate)")
    parser.add_argument("--output_dir", default=None, help="Optional output root (default: same as input)")
    args = parser.parse_args()

    pipeline = BatchTablePipeline()
    summary = pipeline.process_directory(args.input_dir, args.output_dir)
    
    # Save summary
    if args.output_dir:
        summary_path = os.path.join(args.output_dir, "batch_summary.json")
    else:
        summary_path = os.path.join(args.input_dir, "batch_summary.json")
        
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"Batch Processing Complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
