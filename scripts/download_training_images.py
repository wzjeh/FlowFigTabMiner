import os
import argparse
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

def main():
    parser = argparse.ArgumentParser(description="Download images for YOLO training")
    parser.add_argument("--output_dir", default="data/yolo_training_images", help="Root directory for downloaded images")
    parser.add_argument("--max_num", type=int, default=1000, help="Total number of images to download")
    args = parser.parse_args()

    root_dir = args.output_dir
    os.makedirs(root_dir, exist_ok=True)

    keywords = [
        "organic molecules structure",
        "organic chemistry table",
        "organic molecules structure table"
    ]
    
    # Split total count roughly equally
    num_per_keyword = (args.max_num // len(keywords)) + 1
    
    print(f"Goal: Download ~{args.max_num} images ({num_per_keyword} per keyword) to {root_dir}")

    for kw in keywords:
        safe_kw = kw.replace(" ", "_")
        kw_dir = os.path.join(root_dir, safe_kw)
        os.makedirs(kw_dir, exist_ok=True)
        
        print(f"--- Downloading for keyword: '{kw}' into {kw_dir} ---")
        
        # Try Bing first (often more reliable for bulk without API barriers)
        try:
            crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': kw_dir})
            crawler.crawl(keyword=kw, filters=None, offset=0, max_num=num_per_keyword)
        except Exception as e:
            print(f"Bing Crawler failed for {kw}: {e}")
            
            # Fallback to Google? Google usually blocks quickly without proxy/API
            # crawler = GoogleImageCrawler(downloader_threads=4, storage={'root_dir': kw_dir})
            # crawler.crawl(keyword=kw, max_num=num_per_keyword)

    # Post-process: flatten? or keep in subdirs?
    # User said "give me 1000 images", implies a collection.
    # We can leave them in subdirs for now, easy to merge later if needed.
    
    print("Download complete.")

if __name__ == "__main__":
    main()
