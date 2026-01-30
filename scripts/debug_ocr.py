import os
import sys
import cv2
import numpy as np
from paddleocr import PaddleOCR

def main():
    # Use one of the found cell images
    img_path = "data/intermediate/example/tables/page_7_table_0/cells_debug/cell_0_0_1.png"
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found")
        return

    print(f"Testing OCR on: {img_path}")
    
    # Init
    ocr = PaddleOCR(use_angle_cls=False, lang='en')
    
    # Read Image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("-" * 30)
    print("TEST 1: FULL PIPELINE (det=True, rec=True, cls=False)")
    # Removed cls=False to avoid TypeError
    res_full = ocr.ocr(img_rgb, det=True, rec=True)
    print(f"Type: {type(res_full)}")
    print(f"Result: {res_full}")
    
    print("-" * 30)
    print("TEST 2: REC ONLY (det=False, rec=True, cls=False)")
    # Removed cls=False
    res_rec = ocr.ocr(img_rgb, det=False, rec=True)
    print(f"Type: {type(res_rec)}")
    print(f"Result: {res_rec}")
    
    if isinstance(res_rec, list):
        if len(res_rec) > 0:
            print(f"Item 0 type: {type(res_rec[0])}")
    
    print("-" * 30)

if __name__ == "__main__":
    main()
