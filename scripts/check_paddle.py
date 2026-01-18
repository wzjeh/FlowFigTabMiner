
import cv2
import numpy as np
from paddleocr import PaddleOCR
import sys

def check_paddle():
    print("Initializing PaddleOCR...")
    try:
        # Configuration from LegendMatcher
        ocr1 = PaddleOCR(use_angle_cls=True, lang='en')
        print("LegendMatcher OCR initialized.")
        
        # Configuration from CoordinateMapper
        ocr2 = PaddleOCR(use_angle_cls=False, lang='en')
        print("CoordinateMapper OCR initialized.")
        
        # Create dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print("Running OCR1 (cls=True)...")
        res1 = ocr1.ocr(img, cls=True)
        print("OCR1 Result:", res1)
        
        print("Running OCR2 (cls=False)...")
        res2 = ocr2.ocr(img, cls=False)
        print("OCR2 Result:", res2)
        
    except Exception as e:
        print(f"Error checking PaddleOCR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_paddle()
