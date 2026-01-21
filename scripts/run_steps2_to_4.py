import os
import argparse
import sys
import json
import time
import pandas as pd

# Fix OpenMP Conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.parsing.yolo_detector import YoloDetector
from src.parsing.stage2_detector import Stage2Detector
from src.extraction.legend_matcher import LegendMatcher
from src.extraction.coordinate_mapper import CoordinateMapper
from src.assembly.evidence_assembler import EvidenceAssembler

def run_steps2_4(input_pdf):
    # Determine intermediate path based on pdf name
    basename = os.path.splitext(os.path.basename(input_pdf))[0]
    # TF-ID saves to data/intermediate/{basename}/figures
    # We need to find the specific crops.
    # Actually, we can just glob the directory.
    
    intermediate_dir = "data/intermediate"
    figures_dir = os.path.join(intermediate_dir, basename, "figures")
    
    if not os.path.exists(figures_dir):
        print(f"Error: Figures directory not found: {figures_dir}. Did Step 1 run?")
        return

    import glob
    figure_images = glob.glob(os.path.join(figures_dir, "*.png"))
    print(f"--- Steps 2-4: Processing {len(figure_images)} figures from {figures_dir} ---")

    if not figure_images:
        return

    # Initialize Models
    try:
        print("Loading YOLO & Method Models...")
        yolo_macro = YoloDetector(model_path="models/bestYOLOn-2-1.pt")
        yolo_micro = Stage2Detector(model_path="models/bestYOLOm-2-2.pt")
        legend_matcher = LegendMatcher(yolo_model=yolo_micro)
        coord_mapper = CoordinateMapper()
        assembler = EvidenceAssembler()
    except Exception as e:
        print(f"Models Init Failed: {e}")
        return

    extracted_results = []

    # Step 2: Macro Cleaning
    print("\nStep 2: Macro Cleaning...")
    # NOTE: yolo_macro.process_images output to intermediate_dir/macro_cleaned
    # We might want to organize by PDF basename too? currently it dumps to same dir.
    # But that's fine for now as long as filenames are unique.
    
    macro_results = yolo_macro.process_images(figure_images, output_base_dir=intermediate_dir, output_subdir_name="macro_cleaned")

    # Step 3 & 4
    print("\nStep 3 & 4: Micro Detection, Extraction & Assembly...")
    
    for item in macro_results:
        original_source = item['original_source']
        cleaned_plot_path = item['cleaned_image']
        elements = item['elements']
        
        figure_id = os.path.splitext(os.path.basename(cleaned_plot_path))[0]
        if figure_id.endswith("_cleaned"):
            figure_id = figure_id.replace("_cleaned", "")
        
        print(f"   >>> Processing Chart: {figure_id}")
        
        try:
            # 3A
            micro_detections = yolo_micro.detect(cleaned_plot_path, conf=0.10, use_tiling=True)
            points = [d for d in micro_detections if d['label'] in ['data_point', 'marker']]
            print(f"      [Step 3a] Detected {len(points)} data points.")
            
            # 3B
            legend_crops = elements.get('legend', [])
            prototypes = legend_matcher.parse_legend_crops(legend_crops)
            matched_points = legend_matcher.match_points(points, prototypes, cleaned_plot_path)
            
            # 3C
            other_detections = [d for d in micro_detections if d['label'] not in ['data_point', 'marker']]
            full_detections = other_detections + matched_points
            
            try:
                df = coord_mapper.map_coordinates(full_detections, cleaned_plot_path)
            except:
                df = pd.DataFrame()
            
            extraction_data = []
            if not df.empty:
                extraction_data = df.to_dict(orient='records')
            else:
                 for p in matched_points:
                         extraction_data.append({
                            "series": p.get('series', 'Unknown'),
                            "x_pixel": p['center'][0],
                            "y_pixel": p['center'][1],
                            "note": "CoordMapping Failed"
                        })
            
            # Step 4 Assembly & Filtering
            macro_cleaned_dir = os.path.dirname(cleaned_plot_path)
            json_path = assembler.assemble(figure_id, extraction_data, macro_cleaned_dir)
            
            if json_path:
                print(f"      -> EVIDENCE SAVED: {json_path}")
                extracted_results.append(json_path)
            else:
                print(f"      -> DISCARDED (Irrelevant).")
                
        except Exception as e:
            print(f"      ! Error: {e}")

    print(f"\n--- Done. Generated {len(extracted_results)} valid evidence packets. ---")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_steps2_4(sys.argv[1])
    else:
        run_steps2_4("data/input/example.pdf")
