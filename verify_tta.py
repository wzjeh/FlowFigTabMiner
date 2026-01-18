
import os
import sys

# Ensure src can be imported
sys.path.append(os.getcwd())

from src.parsing.stage2_detector import Stage2Detector

def verify_tta():
    image_path = "data/intermediate/lab pub/figures/c5cy02098k_page_2_figure_1.png"
    if not os.path.exists(image_path):
        import glob
        pngs = glob.glob("**/*.png", recursive=True)
        if pngs: image_path = pngs[0]
        else: return

    detector = Stage2Detector(model_path="models/bestYOLOm-2-2.pt")
    
    print(f"Testing Micro Detection with TTA (augment=True) on: {image_path}")
    detections = detector.detect(image_path, conf=0.25, imgsz=1024)
    
    points = [d for d in detections if d['label'] == 'data_point']
    ticks = [d for d in detections if 'tick' in d['label']]
    labels = [d for d in detections if 'label' in d['label']] # Axis labels etc
    
    print(f"Results: {len(points)} points, {len(ticks)} ticks, {len(labels)} labels.")
    print("Class Summary:", set([d['label'] for d in detections]))

if __name__ == "__main__":
    verify_tta()
