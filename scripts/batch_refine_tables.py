import os
import glob
import cv2
import argparse
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parsing.table_filter import TableFilter

def main():
    parser = argparse.ArgumentParser(description="Batch Refine Tables (Step 2 Logic) - Extract Bodies Only")
    parser.add_argument("--input_dir", default="data/tab-for-annotation", help="Directory containing raw table images")
    parser.add_argument("--output_dir", default="data/tab-for-annotation/bodies", help="Directory to save clean table bodies")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Gather Images
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif')
    image_paths = sorted([
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(input_dir, f))
    ])
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images. Initializing TableFilter (YOLOv11)...")
    
    # 2. Initialize Model
    try:
        filter_model = TableFilter()
    except Exception as e:
        print(f"Failed to load TableFilter: {e}")
        return

    # 3. Process in Batches
    batch_size = 16
    success_count = 0
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing Batches"):
        batch_paths = image_paths[i : i + batch_size]
        
        try:
            # Batch Inference
            results = filter_model.filter_tables(batch_paths, conf_threshold=0.4)
            
            for j, res in enumerate(results):
                orig_path = batch_paths[j]
                basename = os.path.splitext(os.path.basename(orig_path))[0]
                
                if res['is_table'] and res.get('table_body_crop') is not None:
                    # Save "Best Body Crop"
                    body_crop = res['table_body_crop']
                    
                    # Apply White Padding (30px) - Consistent with Step 2 UI logic
                    padded_crop = cv2.copyMakeBorder(
                        body_crop, 
                        30, 30, 30, 30, 
                        cv2.BORDER_CONSTANT, 
                        value=[255, 255, 255]
                    )
                    
                    # Save to output_dir
                    # Use original name to keep trace
                    save_path = os.path.join(output_dir, f"{basename}.png")
                    cv2.imwrite(save_path, padded_crop)
                    success_count += 1
                else:
                    # Optional: Copy original if detection fails? 
                    # User asked to "remove caption", implying filtering. 
                    # If Model says "No Table" or "No Body", maybe we shouldn't save it as a training sample for table structure?
                    # Or maybe fail-safe save original? 
                    # Let's STRICTLY save only what receives a body detection valid index.
                    pass
                    
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

    print(f"Processing Complete.")
    print(f"Successfully extracted {success_count} table bodies to {output_dir}")

if __name__ == "__main__":
    main()
