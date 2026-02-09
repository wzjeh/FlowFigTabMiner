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
    from src.utils.config import load_config
    cfg = load_config()
    molscribe_path = cfg.get("tables", {}).get("content", {}).get("molscribe_path", "models/molscribe.ckpt")
    
    recognizer = ContentRecognizer(molscribe_path=molscribe_path)

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
                'row_span': cell.get('row_span', 1),
                'col_span': cell.get('col_span', 1),
                'text': text,
                'original_box': box
            })
            
        except Exception as e:
            print(f"Error processing cell {i}: {e}")

    # 5. Assemble DataFrame
    if not extracted_data:
        print("No data extracted.")
        return

    # --- NEW: Extract Context (Caption/Note) ---
    # Look for {base_name}_table_caption_*.png and {base_name}_table_note_*.png
    # Step 2 usually saves them in output_dir (which is passed as args.output_dir here)
    # But wait, Step 2 output_dir might be different from Step 4 output_dir?
    # In viz_app, we pass t_out for both. So it should be fine.
    
    base_name = os.path.splitext(os.path.basename(body_path))[0]
    # base_name might be {pdf}_table_{i}_body_main
    # We need the original table base name. 
    # Usually: {pdf}_table_{i}
    
    # Heuristic: try to reconstruct prefix
    # If body_path is .../page_5_figure_3_body_main.png -> prefix is page_5_figure_3
    if "_body" in base_name:
        table_prefix = base_name.split("_body")[0]
    else:
        table_prefix = base_name
        
    context_data = {
        "caption": [],
        "table_note": []
    }
    
    import glob
    # Search patterns
    cap_pattern = os.path.join(output_dir, f"{table_prefix}_table_caption_*.png")
    note_pattern = os.path.join(output_dir, f"{table_prefix}_table_note_*.png")
    
    for c_path in sorted(glob.glob(cap_pattern)):
        try:
             # Reuse recognizer's OCR
             # We assume _recognize_text handles simple image path
             # Need RGB for paddleocr? _recognize_text does cvtColor if input is array.
             # If input is path, it reads it.
             # Let's read it here to be safe and consistent with cell logic
             c_img = cv2.imread(c_path)
             if c_img is not None:
                 c_rgb = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
                 txt = recognizer._recognize_text(c_rgb)
                 if txt.strip():
                     context_data["caption"].append(txt)
        except Exception as e:
            print(f"Error OCR caption {c_path}: {e}")

    for n_path in sorted(glob.glob(note_pattern)):
        try:
             n_img = cv2.imread(n_path)
             if n_img is not None:
                 n_rgb = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
                 txt = recognizer._recognize_text(n_rgb)
                 if txt.strip():
                     context_data["table_note"].append(txt)
        except Exception as e:
            print(f"Error OCR note {n_path}: {e}")

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
        r_start = item['row']
        c_start = item['col']
        r_span = item.get('row_span', 1)
        c_span = item.get('col_span', 1)
        
        text = item['text']
        
        # Fill all spanned cells
        for r in range(r_start, r_start + r_span):
            for c in range(c_start, c_start + c_span):
                # Boundary check just in case
                if r < len(grid) and c < len(grid[0]):
                    grid[r][c] = text
        
    df = pd.DataFrame(grid)
    
    # 6. Save CSV
    base_name = os.path.splitext(os.path.basename(body_path))[0]
    csv_filename = f"{base_name}_extracted.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    df.to_csv(csv_path, index=False, header=False)
    print(f"Saved CSV to: {csv_path}")

    # --- NEW: Relevance Filter ---
    # Load keywords from external file
    import yaml
    try:
        with open("keywords.yaml", 'r') as f:
            kw_config = yaml.safe_load(f)
            keywords = kw_config.get('keywords', [])
    except Exception as e:
        print(f"Warning: Could not load keywords.yaml: {e}. Using defaults.")
        keywords = ['yield', 'conversion', 'selectivity', 'product', 'composition'] # minimal fallback
    
    # Text source: Caption + Note + Headers (first 2 rows of df)
    # Combine text
    check_text = (
        " ".join(context_data["caption"]) + " " + 
        " ".join(context_data["table_note"])
    ).lower()
    
    # Add dataframe headers/content (first 3 rows)
    # Flatten first few rows
    if not df.empty:
        head_text = df.head(3).to_string(index=False, header=False)
        check_text += " " + head_text.lower()
        
    import re
    # Normalize
    norm_text = re.sub(r'[^a-z0-9]', '', check_text)
    
    is_relevant = False
    for kw in keywords:
        if kw in norm_text:
            is_relevant = True
            break
            
    if not is_relevant:
        print("Creating Table Evidence but marked as IRRELEVANT (No keywords found).")

    # Output JSON for UI
    output_info = {
        "csv_path": csv_path,
        "num_extracted": len(valid_data),
        "dataframe_preview": df.head().to_dict(orient='split'),
        "cell_logs": extracted_data, # List of {row, col, text, original_box}
        "caption_text": " ".join(context_data["caption"]),
        "table_note_text": " ".join(context_data["table_note"]),
        "is_relevant": is_relevant
    }
    
    # Save Evidence JSON to disk (Clean version for LLM)
    evidence_data = output_info.copy()
    if 'cell_logs' in evidence_data:
        del evidence_data['cell_logs'] # Remove verbose logs
    if 'dataframe_preview' in evidence_data:
         del evidence_data['dataframe_preview'] # Remove preview
         
    json_filename = f"{base_name}_evidence.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved Evidence JSON to: {json_path}")
    
    # Add json_path to output for UI
    output_info["json_path"] = json_path
    
    print("---JSON_START---")
    print(json.dumps(output_info))
    print("---JSON_END---")

if __name__ == "__main__":
    main()
