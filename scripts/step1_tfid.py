import os
import argparse
import sys
import json
import time

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.parsing.active_area_detector import ActiveAreaDetector

def run_step1(input_pdf):
    if not os.path.exists(input_pdf):
        print(f"Error: File not found: {input_pdf}")
        return

    print(f"--- Step 1: TF-ID Detection for {input_pdf} ---")
    try:
        detector = ActiveAreaDetector()
        detections = detector.process_pdf(input_pdf)
        intermediate_dir = "data/intermediate"
        saved_paths = detector.save_crops(input_pdf, detections, intermediate_dir)
        
        print(f"Saved {len(saved_paths)} crops (figures/tables) to {intermediate_dir}")
        
    except Exception as e:
        print(f"Error during TF-ID step: {e}")
        return

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_step1(sys.argv[1])
    else:
        run_step1("data/input/example.pdf")
