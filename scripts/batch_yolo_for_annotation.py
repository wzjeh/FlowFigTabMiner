
import os
import sys
import glob

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parsing.yolo_detector import YoloDetector
from config import Config

def main():
    # 1. Setup Directories
    input_dir = os.path.join(Config.BASE_DIR, "data", "for_annotation")
    
    # 2. Find Images
    image_pattern = os.path.join(input_dir, "*.png")
    image_paths = glob.glob(image_pattern)
    
    # Also look for jpg/jpeg
    image_paths.extend(glob.glob(os.path.join(input_dir, "*.jpg")))
    image_paths.extend(glob.glob(os.path.join(input_dir, "*.jpeg")))
    
    # Sort for consistent processing order
    image_paths.sort()
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images in {input_dir}. Starting YOLO batch processing...")

    # 3. Initialize YOLO
    # Use Config to ensure absolute path
    model_path = os.path.join(Config.MODELS_DIR, "best425.pt")
    detector = YoloDetector(model_path=model_path)
    
    # 4. Run Processing
    # Output will be inside: input_dir / "masked"
    # We batch them all in one call, process_images iterates internally.
    try:
        results = detector.process_images(
            image_paths, 
            output_base_dir=input_dir, 
            output_subdir_name="masked"
        )
        print(f"\nProcessing Complete!")
        print(f"Processed: {len(results)} cleaned images successfully.")
        print(f"Output Directory: {os.path.join(input_dir, 'masked')}")
        
    except Exception as e:
        print(f"\nDetailed Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
