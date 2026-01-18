
import os
import sys
import cv2
import pandas as pd
import traceback
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.parsing.stage2_detector import Stage2Detector
from src.extraction.legend_matcher import LegendMatcher
from src.extraction.coordinate_mapper import CoordinateMapper

def show_img(path, title="Image"):
    """Helper to show image or print path if in non-GUI env"""
    print(f"[Visualizing] {title}: {path}")
    # cv2.imshow(title, cv2.imread(path))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def display(df):
    """Helper to print dataframe"""
    print(df)

def main():
    # Paths from finding
    base_dir = "data/intermediate_validation/macro_cleaned"
    cleaned_img_path = os.path.join(base_dir, "page_5_figure_1_t0_cleaned.png")
    raw_img_path = os.path.join(base_dir, "page_5_figure_1_t0_raw.png")
    
    # Find legend crops dynamically based on pattern
    legend_crops = []
    if os.path.exists(base_dir):
        legend_crops = [os.path.join(base_dir, f) for f in os.listdir(base_dir) 
                       if "legend" in f and f.endswith(".png") and "page_5_figure_1_t0" in f]

    selected_result = {
        'cleaned_image': cleaned_img_path,
        'raw_image': raw_img_path,
        'elements': {
            'legend': legend_crops
        }
    }
    
    if not os.path.exists(cleaned_img_path):
        print(f"Error: File not found {cleaned_img_path}")
        return

    print("=== Initialize Models ===")
    try:
        yolo_micro = Stage2Detector(model_path="models/bestYOLOm-2-2.pt")
        legend_matcher = LegendMatcher(yolo_model=yolo_micro)
        coord_mapper = CoordinateMapper()
        print("✅ Models initialized.")
    except Exception as e:
        print(f"❌ Model Init Failed: {e}")
        traceback.print_exc()
        return

    print("\n=== 1. Micro Detection ===")
    data_points = []
    micro_detections = []
    try:
        # Use lower confidence and higher resolution to find missing points
        micro_detections = yolo_micro.detect(cleaned_img_path, conf=0.15, imgsz=1280) 
        
        from collections import Counter
        labels = [d['label'] for d in micro_detections]
        print(f"      [DEBUG] All Detections: {Counter(labels)}")
        
        data_points = [d for d in micro_detections if 'data_point' in d['label'] or 'marker' in d['label']]
        print(f"Detected {len(data_points)} Data Points (filtered).")
    except Exception as e:
        print(f"❌ Micro Detection Failed: {e}")
        traceback.print_exc()
        return

    # Logic from user snippet
    if selected_result and data_points:
        # 1. Legend Matching
        legend_crops = selected_result['elements'].get('legend', [])
        prototypes = {}
        
        print("\n--- Legend Analysis ---")
        print(f"Found {len(legend_crops)} legend crops: {legend_crops}")
        
        # Pass ALL crops to be stitched together
        prototypes = legend_matcher.parse_legend_crops(legend_crops)
        
        print(f"Extracted Series Prototypes: {list(prototypes.keys())}")
        
        # Match points
        matched_points = legend_matcher.match_points(micro_detections, prototypes, selected_result['cleaned_image'])
        
        # 2. Coordinate Mapping
        # Use RAW image for OCR of ticks (as they are masked in cleaned image!)
        raw_path = selected_result.get('raw_image', selected_result['cleaned_image'])
        
        print(f"\n--- Mapping Coordinates (using {os.path.basename(raw_path)}) ---")
        try:
            df = coord_mapper.map_coordinates(matched_points, raw_path)
            
            print("\n--- Final Extracted Data ---")
            if not df.empty:
                display(df.head(20))
            else:
                print("No data extracted (Regression failed or no labels found).")
        except Exception as e:
            print(f"❌ Coordinate Mapping Failed: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
