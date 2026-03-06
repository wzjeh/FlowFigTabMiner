import os
import sys
import cv2
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.extraction.common.content_recognizer import ContentRecognizer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_molscribe_single.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        sys.exit(1)
        
    print(f"Loading image: {image_path}")
    img_bgr = cv2.imread(image_path)
    
    # Check if loaded
    if img_bgr is None:
        print(f"Error: Failed to load {image_path}")
        sys.exit(1)
        
    print(f"Image Size: {img_bgr.shape}")
    
    # Initialize Content Recognizer
    # Assuming molscribe weights are in default location or passed via constructor if needed
    cr = ContentRecognizer()
    
    # Test 1: As Is
    print("\n--- Test 1: Original Size ---")
    smiles_orig = cr.recognize_content(img_bgr, "Structure")
    print(f"Result (Original): {smiles_orig}")
    
    # Test 2: Downscale 50% (Maybe it was too big?)
    # The failed crop is likely already upscaled by MoleculeProcessor. 
    # Let's see if shrinking helps.
    h, w = img_bgr.shape[:2]
    new_h, new_w = int(h * 0.5), int(w * 0.5)
    img_down = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    print("\n--- Test 2: Downscaled (50%) ---")
    smiles_down = cr.recognize_content(img_down, "Structure")
    print(f"Result (Downscaled): {smiles_down}")
    
    # Test 3: Upscale 2x (Maybe it needs MORE?)
    new_h2, new_w2 = int(h * 2), int(w * 2)
    img_up = cv2.resize(img_bgr, (new_w2, new_h2), interpolation=cv2.INTER_CUBIC)
    
    print("\n--- Test 3: Upscaled (200%) ---")
    smiles_up = cr.recognize_content(img_up, "Structure")
    print(f"Result (Upscaled): {smiles_up}")

    # Test 4: OCR Fallback
    print("\n--- Test 4: OCR Fallback ---")
    ocr_text = cr.recognize_content(img_bgr, "Text")
    print(f"Result (OCR): '{ocr_text}'")
