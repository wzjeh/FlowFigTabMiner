import os
import cv2
import argparse
import sys
import glob
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parsing.table_filter import TableFilter
from src.extraction.table_structure import TableStructureRecognizer

def main():
    parser = argparse.ArgumentParser(description="Extract cell crops for ResNet training")
    parser.add_argument("--tables_dir", default="data/tab-for-annotation", help="Directory containing table images")
    parser.add_argument("--output_dir", default="data/cells-for-annotation", help="Output directory for cell crops")
    parser.add_argument("--max_tables", type=int, default=50, help="Max number of tables to process (to avoid too many files)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models
    print("Initializing models...")
    table_filter = TableFilter()
    table_structure = TableStructureRecognizer()
    
    # Find table images
    exts = ['*.png', '*.jpg', '*.jpeg']
    table_images = []
    for ext in exts:
        table_images.extend(glob.glob(os.path.join(args.tables_dir, ext)))
    
    # Filter out lab_pub ones if you want standard dataset, or keep all
    # For diversity, keep all.
    # Shuffle?
    import random
    random.shuffle(table_images)
    
    print(f"Found {len(table_images)} tables. Processing {min(len(table_images), args.max_tables)}...")
    
    count_cells = 0
    
    for i, img_path in enumerate(tqdm(table_images[:args.max_tables])):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        try:
            # 1. Filter / Segment Body
            res = table_filter.filter_tables([img_path], conf_threshold=0.4)[0]
            
            if not res['is_table']:
                continue
            
            # Prefer cropped body
            if 'table_body_crop' in res:
                body_img = res['table_body_crop']
            else:
                body_img = cv2.imread(img_path)
            
            if body_img is None or body_img.size == 0:
                continue

            # Save body to temp for structure
            import tempfile
            fd, temp_body_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(temp_body_path, body_img)
            
            # 2. Structure
            struct = table_structure.recognize_structure(temp_body_path)
            # Use our improved logic (which is inside recognize_structure / private methods)
            # The tool call updated table_structure.py, so it should return 'cells' preferred from grid
            cells = struct.get('cells', [])
            
            # 3. Crop Cells
            for j, cell in enumerate(cells):
                # Ensure box is valid
                full_box = cell['box'] # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, full_box)
                
                h, w = body_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Filter tiny crops (noise)
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                
                cell_crop = body_img[y1:y2, x1:x2]
                
                # Save just flat for now, user will sort
                cell_filename = f"{base_name}_cell_{j}.png"
                cv2.imwrite(os.path.join(args.output_dir, cell_filename), cell_crop)
                count_cells += 1
            
            # Cleanup
            os.remove(temp_body_path)
            
        except Exception as e:
            # print(f"Error processing {base_name}: {e}")
            pass

    print(f"Extraction complete. {count_cells} cell images saved to {args.output_dir}")
    print(f"Please manually sort them into {args.output_dir}/train/Text and {args.output_dir}/train/Structure")

if __name__ == "__main__":
    main()
