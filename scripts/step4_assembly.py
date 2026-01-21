import os
import sys
import pandas as pd
import json

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.parsing.stage2_detector import Stage2Detector
from src.extraction.legend_matcher import LegendMatcher
from src.extraction.coordinate_mapper import CoordinateMapper
from src.assembly.evidence_assembler import EvidenceAssembler

def run_step4(figure_id="page_3_figure_0_t0"):
    print(f"=== Running Step 4 (Assembly) for {figure_id} ===")
    
    # Paths
    base_dir = "data/intermediate/macro_cleaned"
    # Find the 'cleaned' image for detection
    cleaned_img_path = os.path.join(base_dir, f"{figure_id}_cleaned.png")
    
    if not os.path.exists(cleaned_img_path):
        print(f"Error: Image {cleaned_img_path} not found.")
        return

    # --- 1. Re-Run Extraction Pipeline (to get fresh data) ---
    print(">>> 1. Running Extraction Pipeline...")
    
    # A. Detection (Stage 2)
    model_path = "models/bestYOLOm-2-2.pt"
    detector = Stage2Detector(model_path=model_path)
    # Using optimized parameters from Step 3 verification
    detections = detector.detect(cleaned_img_path, conf=0.10, use_tiling=True)
    
    # Filter for data points
    points = [d for d in detections if d['label'] in ['data_point', 'marker']]
    print(f"    Detected {len(points)} data points.")
    
    # B. Classification (Legend Matcher)
    matcher = LegendMatcher(yolo_model=detector)
    # Get legend crops
    import glob
    legend_pattern = os.path.join(base_dir, f"{figure_id}_legend_*.png")
    legend_crops = glob.glob(legend_pattern)
    
    prototypes = matcher.parse_legend_crops(legend_crops)
    matched_points = matcher.match_points(points, prototypes, cleaned_img_path)
    
    # C. Coordinate Mapping
    mapper = CoordinateMapper()
    
    try:
        # Note: map_coordinates signature is (detections, plot_img_path)
        # It needs BOTH the matched points (with series info) AND the original non-point detections (ticks/labels) to work.
        
        # 1. Identify non-point detections from original set
        other_detections = [d for d in detections if d['label'] not in ['data_point', 'marker']]
        
        # 2. Combine
        full_detections_for_mapping = other_detections + matched_points
        
        data_df = mapper.map_coordinates(full_detections_for_mapping, cleaned_img_path)
        print(f"    Mapped {len(data_df)} points to coordinates.")
    except Exception as e:
        print(f"    Warning: Coordinate Mapping failed ({e}). Proceeding with raw pixels.")
        # Fallback format if mapping fails
        data = []
        for p in matched_points:
            data.append({
                "series": p.get('series', 'Unknown'),
                "x_pixel": p['center'][0],
                "y_pixel": p['center'][1],
                "error": str(e)
            })
        data_df = pd.DataFrame(data)

    # Convert DataFrame to Dict (Records format)
    extraction_data = data_df.to_dict(orient='records')

    # --- 2. Evidence Assembly ---
    print(">>> 2. Assembling Evidence...")
    assembler = EvidenceAssembler()
    json_path = assembler.assemble(figure_id, extraction_data, base_dir)
    
    print(f"=== Success! Evidence saved to: {json_path} ===")
    
    # Print preview
    with open(json_path) as f:
        data = json.load(f)
        print(json.dumps(data, indent=2)[:500] + "...")

if __name__ == "__main__":
    # Allow passing figure_id
    if len(sys.argv) > 1:
        run_step4(sys.argv[1])
    else:
        run_step4() # Default
