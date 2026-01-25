import os
import cv2
import sys
import glob
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parsing.table_filter import TableFilter

def main():
    tables_dir = "data/tab-for-annotation"
    output_dir = "data/debug_crops"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Filter
    print("Initializing Table Filter with new logic...")
    table_filter = TableFilter()
    
    # Pick random images
    all_images = glob.glob(os.path.join(tables_dir, "*.png"))
    if not all_images:
        print("No images found to test.")
        return
        
    test_images = random.sample(all_images, min(5, len(all_images)))
    
    print(f"Testing on {len(test_images)} images...")
    
    results = table_filter.filter_tables(test_images)
    
    for res in results:
        base_name = os.path.basename(res['path'])
        if res['is_table']:
            out_path = os.path.join(output_dir, f"crop_{base_name}")
            cv2.imwrite(out_path, res['table_body_crop'])
            print(f"Saved crop: {out_path} (Original Size: {cv2.imread(res['path']).shape}, Crop Size: {res['table_body_crop'].shape})")
        else:
            print(f"Skipped {base_name} (No table body found)")
            
    print(f"\nDone. Please check {output_dir} to verify padding and full-width cuts.")

if __name__ == "__main__":
    main()
