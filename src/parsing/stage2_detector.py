import os
from ultralytics import YOLO

class Stage2Detector:
    def __init__(self, model_path="models/bestYOLOm-2-2.pt"):
        print(f"Loading Stage 2 YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        # Class Mapping (User needs to verify this matches bestYOLOm-2-2.pt training)
        # Assuming typical ordering: data_point, tick_mark, etc.
        # This mapping *must* match the training data.yaml.
        # For now, we'll store everything and let the consumer decide based on names if available or IDs.
        # We'll assume the model has names embedded.

    def detect(self, image_path, conf=0.15, imgsz=1024):
        """
        Runs detection on a single image (Cleaned Plot).
        Returns list of detections: {'cls': int, 'label': str, 'conf': float, 'box': [x1, y1, x2, y2], 'center': (cx, cy)}
        """
        results = self.model(image_path, conf=conf, imgsz=imgsz, rect=False, augment=True, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = results.names[cls_id]
            
            detections.append({
                "label": label,
                "cls_id": cls_id,
                "conf": conf,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "center": (float(cx), float(cy))
            })
            
        return detections
