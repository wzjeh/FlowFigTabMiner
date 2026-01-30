import os
import argparse
import json
import cv2
import numpy as np
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parsing.table_filter import TableFilter

def main():
    parser = argparse.ArgumentParser(description="Step 2: Table Segmentation (YOLO)")
    parser.add_argument("image_path", help="Path to input table image")
    parser.add_argument("--output_dir", help="Directory to save extracted components", required=True)
    args = parser.parse_args()

    image_path = args.image_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing Image: {image_path}")
    
    # Initialize Filter
    # Assuming standard model path or configured one
    filter_model = TableFilter()
    
    # Run Inference
    results = filter_model.filter_tables([image_path], conf_threshold=0.4)
    res = results[0]

    output_data = {
        "is_table": res['is_table'],
        "confidence": float(res.get('conf', 0.0)),
        "components_found": [],
        "best_body_crop_path": None,
        "logs": [] # List of raw detections
    }

    if not res['is_table']:
        print("No table body detected.")
        print("---JSON_START---")
        print(json.dumps(output_data))
        print("---JSON_END---")
        return

    table_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save Components
    components = res.get('components', {})
    
    # We want to log ALL raw detections for debugging
    # components dict is {label: [{'crop': ..., 'box': ..., 'conf': ...}]}
    for label, items in components.items():
        for i, item in enumerate(items):
            # Save crop
            fname = f"{table_basename}_{label}_{i}.png"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, item['crop'])
            
            # Log
            log_entry = {
                "label": label,
                "confidence": float(item['conf']),
                "box": [int(x) for x in item['box']], # x1, y1, x2, y2
                "saved_path": fpath
            }
            output_data["logs"].append(log_entry)
            output_data["components_found"].append(label)

    # Save the special "Best Body" with full width logic
    if res['table_body_crop'] is not None:
        body_fname = f"{table_basename}_body_main.png"
        body_path = os.path.join(output_dir, body_fname)
        
        # Apply the WHITE PADDING (30px) here as requested in previous session tasks?
        # The user requested "Refine Cropping Strategy: ... + White Border 30px" in the task list.
        # Let's ensure we do that here.
        body_crop = res['table_body_crop']
        padded_crop = cv2.copyMakeBorder(body_crop, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        cv2.imwrite(body_path, padded_crop)
        output_data["best_body_crop_path"] = body_path
        print(f"Saved main body crop to: {body_path}")

    # Sort logs by confidence
    output_data["logs"].sort(key=lambda x: x['confidence'], reverse=True)
    
    print("---JSON_START---")
    print(json.dumps(output_data))
    print("---JSON_END---")

if __name__ == "__main__":
    main()
