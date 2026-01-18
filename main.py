import os
import argparse
import json
import time
import pandas as pd
from src.parsing.active_area_detector import ActiveAreaDetector
from src.parsing.yolo_detector import YoloDetector
from src.parsing.stage2_detector import Stage2Detector
from src.extraction.legend_matcher import LegendMatcher
from src.extraction.coordinate_mapper import CoordinateMapper
from src.extraction.table_agent import extract_table

def main():
    parser = argparse.ArgumentParser(description="FlowFigTabMiner: Extract data from Flow Chemistry Papers")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("--output_dir", default="data/output", help="Directory to save final results")
    
    args = parser.parse_args()
    input_pdf = args.pdf_path
    
    if not os.path.exists(input_pdf):
        print(f"Error: File not found: {input_pdf}")
        return

    print(f"--- Starting Pipeline for {input_pdf} ---")
    start_time = time.time()
    
    # Initialize Models
    try:
        # Stage 1 (TF-ID)
        tf_id_detector = ActiveAreaDetector()
        
        # Stage 1.5 (YOLOv11n - Clean & Mask)
        yolo_macro = YoloDetector(model_path="models/bestYOLOn-2-1.pt")
        
        # Stage 2 (YOLOv11m - Micro Detection)
        yolo_micro = Stage2Detector(model_path="models/bestYOLOm-2-2.pt")
        
        # Module 2 & 3
        legend_matcher = LegendMatcher()
        coord_mapper = CoordinateMapper()
        
    except Exception as e:
        print(f"Failed to initialize models: {e}")
        return

    # Step 1: TF-ID Detection (Figures & Tables)
    print("\nStep 1: Detecting Tables and Figures (TF-ID)...")
    try:
        detections = tf_id_detector.process_pdf(input_pdf)
        intermediate_dir = "data/intermediate"
        # Saves to data/intermediate/{pdf_name}/...
        saved_paths = tf_id_detector.save_crops(input_pdf, detections, intermediate_dir)
        
        table_images = [p for p in saved_paths if 'table' in os.path.basename(p)]
        figure_images = [p for p in saved_paths if 'figure' in os.path.basename(p)]
        print(f"   Found {len(table_images)} tables and {len(figure_images)} figures.")
    except Exception as e:
        print(f"Error during TF-ID step: {e}")
        return

    extracted_data = {
        "metadata": {
            "source_pdf": input_pdf,
            "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "tables": [],
        "figures": []
    }

    # Step 2: Figure Processing Pipeline
    if figure_images:
        print("\nStep 2: Processing Figures (Macro -> Micro -> Map)...")
        
        # 2A. Macro Cleaning (YOLOv11n)
        macro_results = yolo_macro.process_images(figure_images, output_base_dir=intermediate_dir, output_subdir_name="macro_cleaned")
        
        for item in macro_results:
            original_source = item['original_source']
            cleaned_plot_path = item['cleaned_image']
            elements = item['elements'] # dict of paths by label
            
            print(f"   Processing Plot: {os.path.basename(original_source)}")
            
            try:
                # 2B. Micro Detection (YOLOv11m) on Cleaned Plot
                # Resize/Pad to 1024 handled by YOLO internally usually, but user mentioned "Scaling to 1024".
                # Ultralytics YOLO handles resizing.
                micro_detections = yolo_micro.detect(cleaned_plot_path, conf=0.15)
                
                # Check if we have data points
                points = [d for d in micro_detections if d['label'] == 'data_point']
                print(f"      -> Detected {len(points)} data points.")
                
                if not points:
                    print("      -> No data points found. Skipping mapping.")
                    continue
                
                # 2C. Legend Matching (Module 2)
                # Parse Legends from Stage 1 crops
                legend_crops = elements.get('legend', [])
                prototypes = {}
                for lc in legend_crops:
                    print(f"      -> Parsing legend: {lc}")
                    protos = legend_matcher.parse_legend(lc)
                    prototypes.update(protos)
                
                # Match points to series
                print("      -> Matching points...")
                matched_points = legend_matcher.match_points(micro_detections, prototypes, cleaned_plot_path)
                
                # 2D. Coordinate Mapping (Module 3)
                raw_plot_path = item.get('raw_image')
                if not raw_plot_path:
                    raw_plot_path = cleaned_plot_path
                
                print(f"      -> Mapping coordinates using {os.path.basename(raw_plot_path)}...")
                df = coord_mapper.map_coordinates(matched_points, raw_plot_path)
                
                # Save Data
                if not df.empty:
                     csv_name = os.path.splitext(os.path.basename(cleaned_plot_path))[0] + ".csv"
                     csv_path = os.path.join(args.output_dir, csv_name)
                     df.to_csv(csv_path, index=False)
                     print(f"      -> Extracted {len(df)} rows. Saved to {csv_path}")
                     
                     extracted_data["figures"].append({
                         "figure_path": original_source,
                         "cleaned_path": cleaned_plot_path,
                         "data_csv": csv_path
                     })
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"      -> Error extraction figure: {e}")

    # Step 3: Table Extraction
    if table_images:
        print("\nStep 3: Processing Tables...")
        for img_path in table_images:
             # Basic table agent wrapper
             try:
                 result = extract_table(img_path)
                 if result.get("is_valid"):
                      extracted_data["tables"].append(result)
                      print(f"   Extracted table: {os.path.basename(img_path)}")
             except Exception as e:
                 print(f"   Error extracting table {os.path.basename(img_path)}: {e}")


    # Final Save
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_pdf))[0]
    output_json = os.path.join(args.output_dir, f"{basename}_results.json")
    
    with open(output_json, "w") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - start_time
    print(f"\n--- Pipeline Complete in {elapsed:.1f}s! Results saved to {output_json} ---")

if __name__ == "__main__":
    main()
