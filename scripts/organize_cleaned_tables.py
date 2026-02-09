import os
import glob
import shutil
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Organize cleaned table bodies matching data/papers PDFs")
    parser.add_argument("--papers_dir", default="data/papers", help="Root directory of papers")
    parser.add_argument("--source_dir", default="data/tab-for-annotation/bodies", help="Directory with cleaned table bodies")
    parser.add_argument("--output_dir", default="data/papers/cleaned", help="Target directory for filtered images")
    args = parser.parse_args()

    papers_dir = args.papers_dir
    source_dir = args.source_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning for PDFs in {papers_dir}...")
    pdf_files = glob.glob(os.path.join(papers_dir, "**", "*.pdf"), recursive=True)
    
    # Create set of valid prefixes (basenames)
    valid_prefixes = set()
    for p in pdf_files:
        basename = os.path.splitext(os.path.basename(p))[0]
        valid_prefixes.add(basename)
        
    print(f"Found {len(valid_prefixes)} unique PDF basenames.")

    print(f"Scanning source images in {source_dir}...")
    source_images = glob.glob(os.path.join(source_dir, "*.png"))
    
    moved_count = 0
    
    for img_path in tqdm(source_images, desc="Filtering Images"):
        img_name = os.path.basename(img_path)
        
        # Check if img_name starts with any valid prefix followed by underscore or exact match
        # Filename format: {pdf_name}_{page}_{table}.png
        # We need to be careful about substrings (e.g. PMC1 vs PMC10)
        # So we look for prefix + '_'
        
        # Optimization: iterate prefixes? No, too slow if many prefixes.
        # Improve: The prefix is the part before the first/second/etc underscore?
        # Actually pdf filenames can contain underscores.
        # But we know `prefix` is in `valid_prefixes`.
        # Let's iterate valid_prefixes? Or sort?
        
        # Better: Check if `img_name` starts with `prefix + "_"`.
        # Since we have ~100 PDFs, iterating is fine. 
        # For 1000s, trie is better, but here simple loop OK.
        
        is_match = False
        for prefix in valid_prefixes:
            if img_name.startswith(prefix + "_"):
                is_match = True
                break
        
        if is_match:
            dest_path = os.path.join(output_dir, img_name)
            shutil.copy2(img_path, dest_path)
            moved_count += 1
            
    print(f"Organization Complete.")
    print(f"Filtered {moved_count} images matching 'data/papers' PDFs to {output_dir}")

if __name__ == "__main__":
    main()
