import os
import argparse
import json
import time
import sys
import pandas as pd

# Fix OpenMP Conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure src is importable
sys.path.insert(0, os.getcwd())

# Import Torch-based modules at top level (ActiveAreaDetector uses Transformers/Torch)
from src.parsing.active_area_detector import ActiveAreaDetector
from src.parsing.yolo_detector import YoloDetector
from src.parsing.stage2_detector import Stage2Detector
from src.extraction.legend_matcher import LegendMatcher

# NOTE: Paddle-based modules (CoordinateMapper, EvidenceAssembler) are imported LOCALLY
# to avoid crashes on init if Torch is already loaded/active, or simply to manage memory.

def run_pipeline(input_pdf):
    if not os.path.exists(input_pdf):
        print(f"Error: File not found: {input_pdf}")
        return

    print(f"--- Starting Pipeline Steps 1-4 for {input_pdf} ---")
    start_time = time.time()
    
    # --- Step 1: TF-ID Detection (Transformers/Torch) ---
    print("\nStep 1: Detecting Tables and Figures (TF-ID)...")
    try:
        print("   Loading TF-ID Model...")
        tf_id_detector = ActiveAreaDetector()
        
        detections = tf_id_detector.process_pdf(input_pdf)
        intermediate_dir = "data/intermediate"
        saved_paths = tf_id_detector.save_crops(input_pdf, detections, intermediate_dir)
        
        # Free up memory (Optional, but good practice before loading Paddle)
        del tf_id_detector
        import gc
        gc.collect()
        
        figure_images = [p for p in saved_paths if 'figure' in os.path.basename(p)]
        print(f"   Found {len(figure_images)} figures.")
    except Exception as e:
        print(f"Error during TF-ID step: {e}")
        return

    extracted_results = []

    # --- Step 2: Figure Processing (YOLO/Torch) ---
    if figure_images:
        print("\nStep 2: Processing Figures (Macro Cleaning)...")
        try:
            print("   Loading YOLO Models...")
            yolo_macro = YoloDetector(model_path="models/bestYOLOn-2-1.pt")
            yolo_micro = Stage2Detector(model_path="models/bestYOLOm-2-2.pt")
            # LegendMatcher uses YOLO model internally but also clustering
            legend_matcher = LegendMatcher(yolo_model=yolo_micro)
            
            macro_results = yolo_macro.process_images(figure_images, output_base_dir=intermediate_dir, output_subdir_name="macro_cleaned")
        except Exception as e:
            print(f"Error initializing/running YOLO: {e}")
            return
            
        # --- Step 3 & 4: Extraction & Assembly (PaddleOCR) ---
        print("\nStep 3 & 4: Micro Detection, Extraction & Assembly...")
        
        # Lazy Import Paddle Modules NOW
        print("   Importing PaddleOCR modules...")
        try:
            from src.extraction.coordinate_mapper import CoordinateMapper
            from src.assembly.evidence_assembler import EvidenceAssembler
            
            # Instantiate
            coord_mapper = CoordinateMapper()
            assembler = EvidenceAssembler()
        except Exception as e:
             print(f"Failed to import/init Paddle modules: {e}")
             return

        for item in macro_results:
            original_source = item['original_source']
            cleaned_plot_path = item['cleaned_image']
            raw_plot_path = item.get('raw_image', cleaned_plot_path)
            elements = item['elements'] # dict of paths by label
            
            # Figure ID for Assembly
            figure_id = os.path.splitext(os.path.basename(cleaned_plot_path))[0]
            if figure_id.endswith("_cleaned"):
                figure_id = figure_id.replace("_cleaned", "")
            
            print(f"   >>> Processing Chart: {figure_id}")
            
            try:
                # 3A. Micro Detection
                micro_detections = yolo_micro.detect(cleaned_plot_path, conf=0.10, use_tiling=True)
                
                points = [d for d in micro_detections if d['label'] in ['data_point', 'marker']]
                print(f"      [Step 3a] Detected {len(points)} data points.")
                
                # 3B. Legend Matching
                legend_crops = elements.get('legend', [])
                print(f"      [Step 3b] Parsing {len(legend_crops)} legend crops...")
                prototypes = legend_matcher.parse_legend_crops(legend_crops)
                
                matched_points = legend_matcher.match_points(points, prototypes, cleaned_plot_path)
                
                # 3C. Coordinate Mapping
                print(f"      [Step 3c] Mapping coordinates...")
                
                other_detections = [d for d in micro_detections if d['label'] not in ['data_point', 'marker']]
                full_detections_for_mapping = other_detections + matched_points
                
                try:
                    df = coord_mapper.map_coordinates(full_detections_for_mapping, cleaned_plot_path)
                except Exception as e:
                    print(f"      ! Coordinate Mapping Error: {e}")
                    df = pd.DataFrame()

                # Prepare extraction data list
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

                # Step 4: Evidence Assembly
                print(f"      [Step 4] Assembling Evidence...")
                macro_cleaned_dir = os.path.dirname(cleaned_plot_path)
                
                json_path = assembler.assemble(figure_id, extraction_data, macro_cleaned_dir)
                
                if json_path:
                    print(f"      -> Evidence saved to: {json_path}")
                    extracted_results.append(json_path)
                else:
                    print(f"      -> Evidence discarded per filtering rules.")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"      -> Error processing figure {figure_id}: {e}")

    elapsed = time.time() - start_time
    print(f"\n--- Pipeline Steps 1-4 Complete in {elapsed:.1f}s! ---")
    print(f"Generated {len(extracted_results)} evidence packets.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = "data/input/example.pdf"
        
    run_pipeline(pdf_path)
