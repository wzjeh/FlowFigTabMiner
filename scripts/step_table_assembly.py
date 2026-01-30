import os
import argparse
import json
import cv2
import pandas as pd
import sys
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extraction.content_recognizer import ContentRecognizer

def remove_lines(image):
    """
    Remove horizontal and vertical lines from the image to improve OCR.
    Args:
        image: cv2 image (BGR or Gray)
    Returns:
        Cleaned image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = image.copy()
    else:
        gray = image.copy()
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # Convert to BGR to draw white on 3 channels matches orig

    # Threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 3) # White out lines

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 3)

    return result

def main():
    parser = argparse.ArgumentParser(description="Step 4: Assembly & Content Extraction")
    parser.add_argument("body_image_path", help="Path to cropped table body image")
    parser.add_argument("structure_json_path", help="Path to structure JSON from Step 3 (contains 'log_data' or raw 'cells')")
    parser.add_argument("--output_dir", help="Directory to save CSV and cell crops", required=True)
    parser.add_argument("--padding", type=int, default=14, help="Padding pixels for OCR optimization")
    args = parser.parse_args()

    body_path = args.body_image_path
    struct_path = args.structure_json_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Body Image
    if not os.path.exists(body_path):
        print(f"Error: Body image not found at {body_path}")
        return
    
    original_img = cv2.imread(body_path)
    h_img, w_img = original_img.shape[:2]

    # 2. Load Structure Data
    with open(struct_path, 'r') as f:
        structure_data = json.load(f)
        
    # Extract cells list. 
    # Structure JSON format from step_table_structure.py: { "logs": [...], "num_cells": ..., "raw_structure": ... }
    # We should look for "logs" where type="cell" OR "raw_structure"['cells'].
    # The 'logs' contain 'box', 'row_idx', 'col_idx' which is what we need.
    
    cells = [c for c in structure_data.get('logs', []) if c['type'] == 'cell']
    
    if not cells:
        print("Error: No cells found in structure JSON.")
        return

    print(f"Loaded {len(cells)} cells. Initializing Recognizer...")

    # 3. Initialize Recognizer (PaddleOCR)
    recognizer = ContentRecognizer()

    # 4. Process Cells
    extracted_data = [] # List of {row, col, text}
    
    # Optional: Save cell debug images
    cells_dir = os.path.join(output_dir, "cells_debug")
    os.makedirs(cells_dir, exist_ok=True)

    for i, cell in enumerate(cells):
        try:
            # Box: [x1, y1, x2, y2]
            box = list(map(int, cell['box']))
            x1, y1, x2, y2 = box
            
            # Use strict box extraction (handle bounds)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            
            crop = original_img[y1:y2, x1:x2]
            
            # --- CLEAN LINES ---
            cleaned_crop = remove_lines(crop)
            
            # --- ADD WHITE PADDING ---
            # User requested white background padding, not original image context
            pad = args.padding
            padded_crop = cv2.copyMakeBorder(
                cleaned_crop, 
                pad, pad, pad, pad, 
                cv2.BORDER_CONSTANT, 
                value=(255, 255, 255)
            )
            
            # Save PADDED crop for debug
            cell_fname = f"cell_{cell.get('row_idx', 'x')}_{cell.get('col_idx', 'x')}_{i}.png"
            cv2.imwrite(os.path.join(cells_dir, cell_fname), padded_crop)
            
            # ContentRecognizer expects image path or numpy array
            # PaddleOCR (and most DL models) usually expect RGB. cv2 reads BGR.
            crop_rgb = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB)
            
            # We assume Text for everything for now 
            text = recognizer._recognize_text(crop_rgb)
            print(f"Cell {i} [r={cell.get('row_idx')}, c={cell.get('col_idx')}]: '{text}'", flush=True) # Debug Log
            
            # Row/Col indices
            # If Step 3 assigned them (Grid intersection logic), use them.
            r_idx = cell.get('row_idx')
            c_idx = cell.get('col_idx')
            
            extracted_data.append({
                'row': r_idx,
                'col': c_idx,
                'text': text,
                'original_box': box
            })
            
        except Exception as e:
            print(f"Error processing cell {i}: {e}")

    # 5. Assemble DataFrame
    if not extracted_data:
        print("No data extracted.")
        return

    # Determine Grid Size
    # Filter out None indices
    valid_data = [d for d in extracted_data if d['row'] is not None and d['col'] is not None]
    
    if not valid_data:
         print("Error: Cells missing row/col indices. Extraction failed.")
         return

    max_row = max(d['row'] for d in valid_data)
    max_col = max(d['col'] for d in valid_data)
    
    # Initialize empty grid
    grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    
    for item in valid_data:
        grid[item['row']][item['col']] = item['text']
        
    df = pd.DataFrame(grid)
    
    # 6. Save CSV
    base_name = os.path.splitext(os.path.basename(body_path))[0]
    csv_filename = f"{base_name}_extracted.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    df.to_csv(csv_path, index=False, header=False)
    print(f"Saved CSV to: {csv_path}")

    # Output JSON for UI
    output_info = {
        "csv_path": csv_path,
        "num_extracted": len(valid_data),
        "dataframe_preview": df.head().to_dict(orient='split') # easier for quick debug
    }
    
    print("---JSON_START---")
    print(json.dumps(output_info))
    print("---JSON_END---")

if __name__ == "__main__":
    main()
