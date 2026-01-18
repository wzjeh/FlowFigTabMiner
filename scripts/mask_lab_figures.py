import os
import sys
import glob
from tqdm import tqdm

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.parsing.yolo_detector import YoloDetector

def main():
    # Define directories
    input_dir = os.path.join(project_root, "data", "intermediate", "lab pub", "figures")
    output_dir = os.path.join(project_root, "data", "intermediate", "lab pub", "masked figure")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find images
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process.")
    print(f"Output directory: {output_dir}")

    # Initialize YOLO Detector
    try:
        # Uses default model="models/best.pt" as updated
        detector = YoloDetector()
    except Exception as e:
        print(f"Failed to initialize YoloDetector: {e}")
        return

    # Process images using the detector's method
    # Note: YoloDetector.process_images takes a list of paths and an output BASE dir and SUBDIR name.
    # It creates {output_base_dir}/{output_subdir_name}/...
    # We want output to be strictly `data/intermediate/lab pub/masked figure`.
    # So we pass output_base_dir="data/intermediate/lab pub" and output_subdir_name="masked figure"
    
    base_output = os.path.join(project_root, "data", "intermediate", "lab pub")
    subdir_name = "masked figure"
    
    detector.process_images(image_files, output_base_dir=base_output, output_subdir_name=subdir_name)

    print("\n--- Masking Complete ---")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
