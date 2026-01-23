import os
import shutil
import glob
import sys
import argparse

# Check if src is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parsing.active_area_detector import ActiveAreaDetector

def main():
    parser = argparse.ArgumentParser(description="Extract tables for YOLO annotation")
    parser.add_argument("--output_dir", default="data/tab-for-annotation", help="Output directory for tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0

    # 1. Process data/input PDFs
    input_dir = "data/input"
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "*.pdf")))
    
    print(f"Found {len(pdf_files)} PDFs in {input_dir}. Initializing TF-ID Detector...")
    
    try:
        detector = ActiveAreaDetector()
    except Exception as e:
        print(f"Failed to initialize ActiveAreaDetector: {e}")
        return

    for pdf_path in pdf_files:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Processing {pdf_name}...")
        
        try:
            # Detect
            detections = detector.process_pdf(pdf_path)
            
            # Save Crops directly or check if already extracted?
            # detector.save_crops usually saves to data/intermediate/{pdf_name}
            # Let's use it to extraction first to intermediate, then copy.
            
            intermediate_dir = "data/intermediate"
            saved_paths = detector.save_crops(pdf_path, detections, intermediate_dir)
            
            # Identify tables
            table_paths = [p for p in saved_paths if 'table' in os.path.basename(p)]
            
            # Copy to annotation dir
            for tp in table_paths:
                # Rename to ensure uniqueness: {pdf_name}_{original_name}
                orig_name = os.path.basename(tp)
                new_name = f"{pdf_name}_{orig_name}"
                dest = os.path.join(args.output_dir, new_name)
                shutil.copy2(tp, dest)
                count += 1
                
        except Exception as e:
            print(f"  Error processing {pdf_name}: {e}")

    # 2. Copy from data/intermediate/lab pub/tables
    lab_pub_tables_dir = "data/intermediate/lab pub/tables"
    if os.path.exists(lab_pub_tables_dir):
        print(f"Processing lab pub tables from {lab_pub_tables_dir}...")
        # Recursively find images? Or just top level
        # Check files
        files = glob.glob(os.path.join(lab_pub_tables_dir, "*"))
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img in images:
            # Prefix with lab_pub_ to avoid collision?
            basename = os.path.basename(img)
            new_name = f"lab_pub_{basename}"
            dest = os.path.join(args.output_dir, new_name)
            shutil.copy2(img, dest)
            count += 1
            
    print(f"Done! Collected {count} tables in {args.output_dir}")

if __name__ == "__main__":
    main()
