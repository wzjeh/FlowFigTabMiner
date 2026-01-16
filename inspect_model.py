from ultralytics import YOLO

model = YOLO("models/best425.pt")
print("Model Classes:", model.names)
