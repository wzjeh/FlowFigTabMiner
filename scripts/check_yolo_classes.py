import sys
import os
# Fix OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

model_path = "models/yolo11m-fig-seg-0207-nobreaknocharttext/runs/detect/train/weights/best.pt"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    sys.exit(1)

print(f"Loading {model_path}...")
model = YOLO(model_path)
print("--- Model Classes ---")
print(model.names)
print("---------------------")
