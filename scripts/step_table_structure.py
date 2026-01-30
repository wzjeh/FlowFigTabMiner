import os
import argparse
import json
import cv2
import numpy as np
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extraction.table_structure import TableStructureRecognizer

def main():
    parser = argparse.ArgumentParser(description="Step 3: Structure Recognition (TATR)")
    parser.add_argument("image_path", help="Path to cropped table body image")
    args = parser.parse_args()

    image_path = args.image_path
    
    print(f"Processing Structure for: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: File not found {image_path}")
        return

    # Initialize Model
    recognizer = TableStructureRecognizer()
    
    # Run Inference
    structure = recognizer.recognize_structure(image_path)
    
    # Visualizer
    # structure dict keys: 'cells', 'rows', 'columns'
    # Each item has 'box': [x1, y1, x2, y2]
    
    original_img = cv2.imread(image_path)
    vis_img = original_img.copy() if original_img is not None else None
    
    log_data = []

    if vis_img is not None:
        # Draw Rows (Green)
        for r in structure.get('rows', []):
            box = list(map(int, r['box']))
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            log_data.append({"type": "row", "box": box, "score": float(r.get('score', 0.0))})

        # Draw Columns (Orange)
        for c in structure.get('columns', []):
            box = list(map(int, c['box']))
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 165, 255), 2)
            log_data.append({"type": "column", "box": box, "score": float(c.get('score', 0.0))})

        # Draw Cells (Red, thinner)
        cells = structure.get('cells', [])
        for i, c in enumerate(cells):
            box = list(map(int, c['box']))
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
            # Put index
            # cv2.putText(vis_img, str(i), (box[0]+2, box[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            log_data.append({
                "type": "cell", 
                "box": box, 
                "score": float(c.get('score', 0.0)),
                "row_idx": c.get('row_index'), # Fixed key
                "col_idx": c.get('col_index')  # Fixed key
            })

        # Save Visualization
        output_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(output_dir, f"{base_name}_structure_viz.png")
        cv2.imwrite(vis_path, vis_img)
        print(f"Saved visualization to: {vis_path}")

        # Save full structure log (including scores)
        
        output_json = {
            "num_cells": len(structure.get('cells', [])),
            "num_rows": len(structure.get('rows', [])),
            "num_columns": len(structure.get('columns', [])),
            "viz_path": vis_path,
            "logs": log_data,
            "raw_structure": structure # Valid JSON structure? 'box' might be np array or tensor
        }

        # Need to handle serialization of raw_structure if it has arrays
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return super().default(obj)

        print("---JSON_START---")
        print(json.dumps(output_json, cls=NumpyEncoder))
        print("---JSON_END---")
    else:
        print("Error: Could not read image for visualization.")

if __name__ == "__main__":
    main()
