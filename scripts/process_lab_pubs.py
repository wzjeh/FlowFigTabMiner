import os
import sys
import glob
from tqdm import tqdm

# Add project root to sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.parsing.active_area_detector import ActiveAreaDetector

def main():
    # Define input and output directories
    input_dir = os.path.join(project_root, "data", "input", "lab pub")
    output_base_dir = os.path.join(project_root, "data", "intermediate", "lab pub")

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    # Find all PDF files
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDFs to process in {input_dir}")
    print(f"Output directory: {output_base_dir}")

    # Initialize Detector
    try:
        detector = ActiveAreaDetector()
    except Exception as e:
        print(f"Failed to initialize ActiveAreaDetector: {e}")
        return

    # Process each PDF
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            filename = os.path.basename(pdf_path)
            # print(f"\nProcessing: {filename}")

            # 1. Detect
            detections = detector.process_pdf(pdf_path)
            
            if not detections:
                # print(f"   No tables or figures detected in {filename}")
                continue

            # 2. Save Crops
            # ActiveAreaDetector.save_crops expects the base output_dir and creates a subdir with the pdf name
            # So passing output_base_dir is correct. It will create output_base_dir/{pdf_name_no_ext}/...
            saved_paths = detector.save_crops(pdf_path, detections, output_base_dir)
            
            # print(f"   Saved {len(saved_paths)} crops.")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("\n--- Batch Processing Complete ---")
    print(f"Results saved to: {output_base_dir}")

if __name__ == "__main__":
    main()
