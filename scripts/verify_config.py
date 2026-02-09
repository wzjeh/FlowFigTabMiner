import os
import sys
import yaml

# Add project root needed for src import if run from scripts/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config

def main():
    print("--- Verifying Configuration ---")
    cfg = load_config()
    if not cfg:
        print("Error: Could not load config.yaml")
        return

    print("Config loaded successfully.")
    
    # Check Paths
    paths_to_check = []
    
    # Figures
    paths_to_check.append(("figures.step2_macro.model_path", cfg.get("figures", {}).get("step2_macro", {}).get("model_path")))
    paths_to_check.append(("figures.step3_micro.model_path", cfg.get("figures", {}).get("step3_micro", {}).get("model_path")))
    
    # Tables
    paths_to_check.append(("tables.segmentation.model_path", cfg.get("tables", {}).get("segmentation", {}).get("model_path")))
    paths_to_check.append(("tables.molecule_detection.model_path", cfg.get("tables", {}).get("molecule_detection", {}).get("model_path")))
    
    # Structure model is a HF hub name usually, but check if it looks like a path
    struct_model = cfg.get("tables", {}).get("structure", {}).get("model_path")
    if struct_model and (os.path.sep in struct_model or os.path.exists(struct_model)):
         paths_to_check.append(("tables.structure.model_path", struct_model))
    else:
         print(f"[INFO] tables.structure.model_path '{struct_model}' assumed to be HF Hub ID.")

    paths_to_check.append(("tables.cell_classification.model_path", cfg.get("tables", {}).get("cell_classification", {}).get("model_path")))
    paths_to_check.append(("tables.content.molscribe_path", cfg.get("tables", {}).get("content", {}).get("molscribe_path")))
    
    # Check existence
    all_valid = True
    for label, path in paths_to_check:
        if not path:
            print(f"[WARN] {label} is not defined.")
            continue
            
        if os.path.exists(path):
            print(f"[OK] {label}: Found at {path}")
        else:
            print(f"[FAIL] {label}: Not found at {path}")
            all_valid = False
            
    if all_valid:
        print("\nAll checked paths are valid!")
    else:
        print("\nSome model paths are missing. Please check config.yaml or download models.")

if __name__ == "__main__":
    main()
