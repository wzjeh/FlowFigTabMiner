import os
import sys
import json
import argparse

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.parsing.yolo_detector import YoloDetector

def run_step2_single(image_path):
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        return

    try:
        # Initialize YOLO Macro
        # Suppress prints to keep stdout clean for JSON output
        # But YoloDetector prints "Loading model...". We can't easily suppress without modifying class.
        # We'll just print final JSON at the very end and tell GUI to parse last line.
        
        yolo_macro = YoloDetector(model_path="models/bestYOLOn-2-1.pt")
        
        # Output directory is same as image parent? Or macro_cleaned subdir.
        # Let's match typical pipeline: parent/macro_cleaned
        parent_dir = os.path.dirname(image_path)
        # But if image is in data/intermediate/doc/figures, we want data/intermediate/doc/macro_cleaned?
        # Actually YoloDetector `process_images` takes `output_base_dir` and `output_subdir_name`.
        # typically output_base_dir="data/intermediate", subdir="macro_cleaned".
        # But here image_path might be absolute.
        
        # Let's use image_path's directory as base if possible, or explicit arg.
        # To be safe, we'll just put it in a "cleaned" folder relative to image.
        
        output_dir = os.path.join(os.path.dirname(image_path), "..", "macro_cleaned") # Go up one level from 'figures/'?
        # Assuming input is in .../figures/
        
        # Actually, let's just pass output_base_dir as parent of parent of image
        # e.g. data/intermediate/doc/figures -> data/intermediate/doc
        doc_dir = os.path.dirname(os.path.dirname(image_path))
        
        results = yolo_macro.process_images([image_path], output_base_dir=doc_dir, output_subdir_name="macro_cleaned")
        
        if results:
            item = results[0]
            # Output result structure
            print("---JSON_START---")
            print(json.dumps(item))
            print("---JSON_END---")
        else:
            print(json.dumps({"error": "No results returned"}))
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_step2_single(sys.argv[1])
