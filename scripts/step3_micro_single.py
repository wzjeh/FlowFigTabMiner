import os
import sys
import json
import glob
import pandas as pd

# Fix OpenMP Conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.parsing.stage2_detector import Stage2Detector
from src.extraction.legend_matcher import LegendMatcher
from src.extraction.coordinate_mapper import CoordinateMapper

def run_step3_single(cleaned_image_path):
    if not os.path.exists(cleaned_image_path):
        print(json.dumps({"error": f"Image not found: {cleaned_image_path}"}))
        return

    try:
        # Initialize Models (Lazy to save memory/startup)
        yolo_micro = Stage2Detector(model_path="models/bestYOLOm-2-2.pt")
        legend_matcher = LegendMatcher(yolo_model=yolo_micro)
        coord_mapper = CoordinateMapper()
        
        # 3A. Detection
        micro_detections = yolo_micro.detect(cleaned_image_path, conf=0.10, use_tiling=True)
        points = [d for d in micro_detections if d['label'] in ['data_point', 'marker']]
        
        # 3B. Legend Matching
        # Infer legend crops from directory
        dirname = os.path.dirname(cleaned_image_path)
        basename = os.path.splitext(os.path.basename(cleaned_image_path))[0]
        # Basename usually has '_cleaned' suffix if it came from Step 2 output
        # e.g. 'page_5_figure_0_t0_cleaned' -> original id 'page_5_figure_0_t0'
        # But crops are named '{original_id}_legend_x.png' 
        # So we need to strip '_cleaned'
        search_base = basename.replace("_cleaned", "")
        legend_crops = glob.glob(os.path.join(dirname, f"{search_base}_legend_*.png"))
        
        prototypes = legend_matcher.parse_legend_crops(legend_crops)
        matched_points = legend_matcher.match_points(points, prototypes, cleaned_image_path)
        
        # 3C. Coordinate Mapping
        other_detections = [d for d in micro_detections if d['label'] not in ['data_point', 'marker']]
        full_detections = other_detections + matched_points
        
        try:
            df = coord_mapper.map_coordinates(full_detections, cleaned_image_path)
        except Exception as e:
            # print(f"Mapping error: {e}", file=sys.stderr)
            df = pd.DataFrame()
        
        result = {
            "num_points_detected": len(points),
            "num_legends_found": len(legend_crops),
            "mapped_data": df.to_dict(orient='records') if not df.empty else [],
            "detections": micro_detections  # Send back boxes for visualization
        }
        
        print("---JSON_START---")
        print(json.dumps(result))
        print("---JSON_END---")
            
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_step3_single(sys.argv[1])
