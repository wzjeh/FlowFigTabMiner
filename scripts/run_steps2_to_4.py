import os
import sys
import argparse

# Fix OpenMP Conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.pipeline.figure_pipeline import FigurePipeline

def run_steps2_4(input_pdf):
    pipeline = FigurePipeline()
    results = pipeline.process_pdf_figures(input_pdf)
    print(f"\n--- Done. Generated {len(results)} valid evidence packets. ---")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_steps2_4(sys.argv[1])
    else:
        # Default behavior or help
        print("Usage: python scripts/run_steps2_to_4.py <path_to_pdf>")

