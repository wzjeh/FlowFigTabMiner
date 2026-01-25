import os
import cv2
import argparse
import sys
import glob
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parsing.table_filter import TableFilter

def main():
    parser = argparse.ArgumentParser(description="Extract table bodies for TARA fine-tuning")
    parser.add_argument("--input_dir", default="data/tab-for-annotation", help="Input directory")
    parser.add_argument("--output_dir", default="data/table-bodies-for-tara", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing Table Filter (YOLOv11m, Full Width Cut, Bottom Padding=30)...")
    table_filter = TableFilter()
    
    images = glob.glob(os.path.join(args.input_dir, "*.png"))
    # Also support jpg/jpeg
    images.extend(glob.glob(os.path.join(args.input_dir, "*.jpg")))
    images.extend(glob.glob(os.path.join(args.input_dir, "*.jpeg")))
    
    print(f"Found {len(images)} source images. Processing...")
    
    # Process in batches to avoid OOM or slow startup, though filter_tables is serial loop
    # Let's verify filter_tables batch support. It iterates.
    
    extracted_count = 0
    skipped_count = 0
    
    # Process one by one to show progress bar properly
    for img_path in tqdm(images):
        try:
            # Pass single image as list
            res = table_filter.filter_tables([img_path])[0]
            
            base_name = os.path.basename(img_path)
            
            if res['is_table'] and res['table_body_crop'] is not None:
                out_name = f"body_{base_name}"
                out_path = os.path.join(args.output_dir, out_name)
                
                # Add 30px white padding
                crop = res['table_body_crop']
                # borderType=cv2.BORDER_CONSTANT, value=[255,255,255] (White)
                padded_crop = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                
                cv2.imwrite(out_path, padded_crop)
                extracted_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            skipped_count += 1

    print(f"Done. Extracted {extracted_count} table bodies to {args.output_dir}. Skipped {skipped_count}.")

if __name__ == "__main__":
    main()
