import os
import sys
import shutil

# Disable PaddleOCR connectivity checks globally
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["PADDLEPD_DISABLE_MODEL_SOURCE_CHECK"] = "1"

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extraction.table.pipeline import TablePipeline

def verify_table_fix():
    print("Verifying Table Structure Fix...")
    
    # Paths
    # Input Image (from TF-ID output, not cropped body yet)
    # The file found was "data/intermediate/lab-Methyllithium Chemistry/tables/page_4_table_2.png" 
    # But wait, find_by_name returned "page_4_table_2.png" inside `.../tables/`? 
    # Or maybe it was `.../tables/page_4_table_2.png`. Let's assume the latter.
    
    input_image = "data/intermediate/lab-Methyllithium Chemistry/tables/page_4_table_2.png"
    if not os.path.exists(input_image):
        print(f"Error: Input image not found at {input_image}")
        # Try finding anywhere
        import glob
        matches = glob.glob("data/intermediate/**/tables/page_4_table_2.png", recursive=True)
        if matches:
            input_image = matches[0]
            print(f"Found at: {input_image}")
        else:
            return

    output_dir = "data/intermediate/lab-Methyllithium Chemistry/tables/verify_fix"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Pipeline (with defaults)
    pipeline = TablePipeline()
    
    # Run Process
    print(f"Processing {input_image} -> {output_dir}")
    result = pipeline.process_table(input_image, output_dir=output_dir)
    
    if not result['is_valid']:
        print(f"Verification Failed: {result.get('reason')}")
        return

    # Check for _masked.png
    basename = os.path.splitext(os.path.basename(input_image))[0]
    # table_pipeline creates a subdirectory for the table: {output_dir}/{basename}/
    table_subdir = os.path.join(output_dir, basename)
    
    masked_path = os.path.join(table_subdir, f"{basename}_body_main_masked.png")
    smiles_path = os.path.join(table_subdir, f"{basename}_body_main_smiles.png")
    
    if os.path.exists(masked_path):
        print(f"SUCCESS: Found masked image at {masked_path}")
    else:
        print(f"FAILURE: Masked image not found at {masked_path}")
        
    if os.path.exists(smiles_path):
        print(f"FAILURE: Found old SMILES image at {smiles_path} (Should be renamed)")
    else:
        print("SUCCESS: Old SMILES image not created.")

    # Check CSV Content for SMILES
    csv_path = result['csv_path']
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            content = f.read()
            # Check for known SMILES part or just look for molecule-like strings
            # The problematic molecule was "COC(=O)..."
            if "COC(=O)" in content or "Structure" in content or "C=O" in content:
                print("SUCCESS: CSV contains potential SMILES.")
            else:
                print("WARNING: CSV might be empty of molecules.")
            print(f"CSV Content Preview:\n{content[:500]}")
    else:
        print("FAILURE: CSV not created.")

if __name__ == "__main__":
    verify_table_fix()
