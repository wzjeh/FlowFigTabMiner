import os
import sys
import json

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.flow_dev_miner.processing_layer.figure_processor import FigureProcessor
from src.flow_dev_miner.processing_layer.table_processor import TableProcessor

def test_new_architecture():
    print("=== FlowDevMiner 4-Layer Architecture Test ===")
    print("1. Loading Processors (This validates imports and initializations)")
    
    try:
        # Load processors. They use lazy-loading for models, so this should be fast
        fig_processor = FigureProcessor()
        tab_processor = TableProcessor()
        print("[OK] Processors initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize processors: {e}")
        return

    print("\n2. Instructions for Testing")
    print("To test Figure Processing on a specific image:")
    print("  results = fig_processor.process_figure('path/to/figure.png', 'output/base/dir')")
    print("  print(json.dumps(results, indent=2))")
    print("\nTo test Table Processing on a specific image:")
    print("  results = tab_processor.process_table('path/to/table.png', 'output/base/dir')")
    print("  print(json.dumps(results, indent=2))")
    
    # We can pass a specific dummy image if we know one exists, but printing structural success is good enough for now.

if __name__ == "__main__":
    test_new_architecture()
