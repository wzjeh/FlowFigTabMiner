
import os
import sys

# Ensure src can be imported
sys.path.append(os.getcwd())

from src.parsing.stage2_detector import Stage2Detector

def check_image():
    image_path = "data/intermediate_validation/macro_cleaned/page_5_figure_0_t0_cleaned.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Checking Micro Detection on: {image_path}")
    
    # Load model with current settings (imgsz=1024 implicitly)
    detector = Stage2Detector(model_path="models/bestYOLOm-2-2.pt")
    
    # Run detection with explicit matching of current config
    # The default detector.detect uses imgsz=1024, rect=False, augment=True as updated
    detections = detector.detect(image_path, conf=0.25)
    
    # Categorize
    points = [d for d in detections if d['label'] == 'data_point']
    ticks = [d for d in detections if 'tick' in d['label']]
    others = [d for d in detections if d not in points and d not in ticks]
    
    print(f"\n--- Detection Results ---")
    print(f"Total Objects: {len(detections)}")
    print(f"Data Points: {len(points)}")
    print(f"Tick Marks/Labels: {len(ticks)}")
    print(f"Other Labels: {len(others)}")
    
    if points:
        print("\nSample Data Points (Top 3 by Conf):")
        sorted_points = sorted(points, key=lambda x: x['conf'], reverse=True)[:3]
        for p in sorted_points:
            print(f"  {p['label']} - Conf: {p['conf']:.4f} - Box: {p['box']}")

if __name__ == "__main__":
    check_image()
