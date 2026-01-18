
import os
import sys

# Ensure src can be imported
sys.path.append(os.getcwd())

try:
    from src.parsing.stage2_detector import Stage2Detector
except ImportError:
    print("Could not import Stage2Detector. Run from project root.")
    sys.exit(1)

def verify():
    image_path = "data/intermediate/lab pub/figures/c5cy02098k_page_2_figure_1.png"
    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}")
        # Try to find any png
        import glob
        pngs = glob.glob("**/*.png", recursive=True)
        if pngs:
            image_path = pngs[0]
            print(f"Using alternative image: {image_path}")
        else:
            print("No images found to test.")
            return

    print(f"Testing Micro Detection on: {image_path}")
    
    detector = Stage2Detector(model_path="models/bestYOLOm-2-2.pt")
    
    # Test with explicit imgsz
    try:
        print("\n--- Running with conf=0.25, imgsz=1024 ---")
        detections = detector.detect(image_path, conf=0.25, imgsz=1024)
        points = [d for d in detections if d['label'] == 'data_point']
        print(f"Success! Detected {len(detections)} total objects, {len(points)} data points.")
        print("Detections sample:", detections[:2])
    except Exception as e:
        print(f"Error with explicit imgsz: {e}")
        import traceback
        traceback.print_exc()

    # Test with default (which should also be 1280 now)
    try:
        print("\n--- Running with default settings (conf=0.15, default imgsz) ---")
        detections_default = detector.detect(image_path, conf=0.15)
        points_default = [d for d in detections_default if d['label'] == 'data_point']
        print(f"Default run: Detected {len(detections_default)} total objects, {len(points_default)} data points.")
    except Exception as e:
        print(f"Error with default settings: {e}")

if __name__ == "__main__":
    verify()
