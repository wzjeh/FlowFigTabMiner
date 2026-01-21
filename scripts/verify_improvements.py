import os
import sys
import cv2
import glob
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.parsing.stage2_detector import Stage2Detector
from src.extraction.legend_matcher import LegendMatcher

def verify():
    # Paths
    base_dir = "data/intermediate/macro_cleaned"
    # Using page_7_figure_2 as identified
    target_img_name = "page_7_figure_2_t0_cleaned.png"
    target_img_path = os.path.join(base_dir, target_img_name)
    
    legend_pattern = os.path.join(base_dir, "page_7_figure_2_t0_legend_*.png")
    legend_crops = glob.glob(legend_pattern)
    
    print(f"Target Image: {target_img_path}")
    print(f"Legend Crops: {legend_crops}")
    
    if not os.path.exists(target_img_path):
        print("Error: Target image not found.")
        return

    # Check for models
    model_path = "models/bestYOLOm-2-2.pt"
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Trying absolute path or alternative.")
        # Try to find it if not there
        found_models = glob.glob("**/bestYOLOm-2-2.pt", recursive=True)
        if found_models:
            model_path = found_models[0]
            print(f"Found model at: {model_path}")
        else:
            print("Error: Could not find model file.")
            return

    # Initialize Models
    print("Loading Models...")
    detector = Stage2Detector(model_path=model_path)
    matcher = LegendMatcher(yolo_model=detector)
    
    # 1. Run Detection (Tiled)
    print("Running Detection (Tiled)...")
    # Using low conf + tiling as per new logic
    detections = detector.detect(target_img_path, conf=0.10, use_tiling=True)
    print(f"Found {len(detections)} raw detections.")
    
    # Filter for data points
    points = [d for d in detections if d['label'] in ['data_point', 'marker']]
    print(f"Filtered to {len(points)} data points.")
    
    # 2. Parse Legends
    print("Parsing Legends...")
    prototypes = matcher.parse_legend_crops(legend_crops)
    print(f"Prototypes: {prototypes.keys()}")
    
    # 3. Match
    print("Matching Points...")
    matched_points = matcher.match_points(points, prototypes, target_img_path)
    
    # 4. Visualize
    print("Visualizing...")
    img = cv2.imread(target_img_path)
    if img is None:
        print("Error loading image for visualization.")
        return

    # Colors for series (BGR)
    colors = [
        (255, 0, 0),   # Blue
        (0, 255, 0),   # Green
        (0, 0, 255),   # Red
        (255, 255, 0), # Cyan
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
        (128, 0, 0),   # Dark Blue
        (0, 128, 0),   # Dark Green
        (0, 0, 128)    # Dark Red
    ]
    
    sorted_labels = sorted(prototypes.keys())
    series_map = {label: colors[i%len(colors)] for i, label in enumerate(sorted_labels)}
    
    # Draw Legends on the side or just print the color map
    print("Color Map:")
    for l, c in series_map.items():
        print(f"  {l}: {c}")

    for p in matched_points:
        box = p['box']
        series = p.get('series', 'Unknown')
        color = series_map.get(series, (128, 128, 128))
        
        x1, y1, x2, y2 = map(int, box)
        
        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Center Dot
        cx, cy = p['center']
        cv2.circle(img, (int(cx), int(cy)), 3, (255, 255, 255), -1) # White center
        cv2.circle(img, (int(cx), int(cy)), 2, color, -1)

    # Save
    out_path = "verification_result.png"
    cv2.imwrite(out_path, img)
    print(f"Saved result to {out_path}")

if __name__ == "__main__":
    verify()
