import sys
import os
from paddleocr import PaddleOCR

def debug_ocr(img_path):
    print(f"Testing OCR on: {img_path}")
    if not os.path.exists(img_path):
        print("Image not found.")
        return

    # initialize without use_angle_cls just to see
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    result = ocr.ocr(img_path)
    
    print("Raw Result:", result)
    
    # Try parsing
    txts = [line[1][0] for line in result[0]] if result and result[0] else []
    print("Parsed Text:", txts)

if __name__ == "__main__":
    path = "data/intermediate/macro_cleaned/page_7_figure_2_t0_x_axis_title_1.png"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    debug_ocr(path)
