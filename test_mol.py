import cv2
import os
import sys

from ultralytics import YOLO

def main():
    model_path = "models/yolo11s-tab-molecule-0207/runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = YOLO(model_path)
    print("Model loaded.", flush=True)

    paths = [
        "data/intermediate/example/page_7_table_0/page_7_table_0_body_main.png",
        "data/intermediate/example/page_4_table_0/page_4_table_0_body_main.png",
        "data/intermediate/example/page_8_table_0/page_8_table_0_body_main.png"
    ]
    
    for p in paths:
        if os.path.exists(p):
            print(f"Testing {p}...")
            img = cv2.imread(p)
            res = model(img, conf=0.1, imgsz=1024, verbose=False)
            print(f"Boxes found in {os.path.basename(p)}: {len(res[0].boxes)}", flush=True)
        else:
            print(f"File not found: {p}", flush=True)

if __name__ == '__main__':
    main()
