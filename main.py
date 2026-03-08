import os
import argparse
import json
import time
import pandas as pd
from src.parsing.active_area_detector import ActiveAreaDetector
from src.parsing.yolo_detector import YoloDetector
from src.parsing.stage2_detector import Stage2Detector
from src.extraction.figure.legend_matcher import LegendMatcher
from src.extraction.figure.coordinate_mapper import CoordinateMapper
from src.extraction.table.pipeline import TablePipeline
from src.utils.config import load_config

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

    # Load configuration
    cfg = load_config()
    figures_cfg = cfg.get("figures", {})

    # Initialize Models
    try:
        # Stage 1 (TF-ID)
        tf_id_detector = ActiveAreaDetector()

        # Stage 2 Macro (Figure Segmentation) - from config.yaml
        macro_model_path = figures_cfg.get("step2_macro", {}).get("model_path", "models/yolo11m-fig-seg-0207-nobreaknocharttext/runs/detect/train/weights/best.pt")
        macro_conf = figures_cfg.get("step2_macro", {}).get("confidence_threshold", 0.5)
        yolo_macro = YoloDetector(model_path=macro_model_path)

        # Stage 3 Micro (Scatter Point Detection) - from config.yaml
        micro_model_path = figures_cfg.get("step3_micro", {}).get("model_path", "models/yolo11m-fig-scatter-0208/runs/detect/train/weights/best.pt")
        micro_conf = figures_cfg.get("step3_micro", {}).get("confidence_threshold", 0.15)
        yolo_micro = Stage2Detector(model_path=micro_model_path)
        
        # Module 2 & 3
        legend_matcher = LegendMatcher(yolo_model=yolo_micro)
        coord_mapper = CoordinateMapper()
        
        # Table Pipeline
        table_pipeline = TablePipeline()
        
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
                legend_crops = elements.get('legend', [])
                print(f"      -> Parsing {len(legend_crops)} legend crops...")
                prototypes = legend_matcher.parse_legend_crops(legend_crops)
                
                # Match points to series
                print("      -> Matching points...")
                matched_points = legend_matcher.match_points(micro_detections, prototypes, cleaned_plot_path)
                
                # 2D. Coordinate Mapping (Module 3)
                raw_plot_path = item.get('raw_image')
                if not raw_plot_path:
                    raw_plot_path = cleaned_plot_path
                
                print(f"      -> Mapping coordinates using {os.path.basename(raw_plot_path)}...")
                df, debug_log = coord_mapper.map_coordinates(matched_points, raw_plot_path)

                # Print debug log for diagnosis
                if df.empty:
                    print(f"      -> WARNING: No data extracted. Debug log:")
                    for log_line in debug_log[-10:]:  # Last 10 lines
                        print(f"         {log_line}")

                # Save Data
                if not df.empty:
                     csv_name = os.path.splitext(os.path.basename(cleaned_plot_path))[0] + ".csv"
                     csv_path = os.path.join(args.output_dir, csv_name)
                     # Round numeric columns to 2 decimal places
                     df = df.round({'X': 2, 'Y_Left': 2, 'Y_Right/Data_Value': 2})
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
             try:
                 result = table_pipeline.process_table(img_path, output_dir=os.path.join(args.output_dir, "tables"))
                 if result.get("is_valid"):
                      # Flatten 'dataframe' to list of lists or whatever format preferred for JSON
                      # But extracted_data expects dicts.
                      # We can store the CSV path mainly.
                      extracted_data["tables"].append({
                          "table_path": img_path,
                          "csv_path": result['csv_path']
                      })
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
