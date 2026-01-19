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
        
        Optimized Strategy:
        1. Run inference with Low Confidence (0.15) to catch everything.
        2. Dynamic Thresholding: If detection explosion (>500), raise threshold to suppress noise.
        3. [Disabled] Spatial NMS: Merge duplicate data points that are spatially too close.
        """
        # 1. Run inference with low confidence (ignore user 'conf' if it's the default, or force low)
        # We use strict 0.15 as base to retrieve weak candidates.
        base_conf = 0.10
        results = self.model(image_path, conf=base_conf, imgsz=imgsz, rect=False, augment=True, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            cls_id = int(box.cls[0].item())
            conf_score = float(box.conf[0].item())
            label = results.names[cls_id]
            
            detections.append({
                "label": label,
                "cls_id": cls_id,
                "conf": conf_score,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "center": (float(cx), float(cy))
            })
            
        # 2. Dynamic Thresholding
        detections = self._apply_dynamic_threshold(detections)
        
        # 3. Spatial NMS (Only for data_points) - DISABLED per user request
        # detections = self._spatial_nms(detections)
        
        return detections

    def _apply_dynamic_threshold(self, detections):
        """
        If density is too high (>500 detections), assume noisy image and raise threshold.
        """
        total_count = len(detections)
        if total_count < 10:
             return detections # Sparse, keep everything
             
        # Heuristic: If count > 500, unlikely to be a clean plot with 500 series points.
        # It's likely grid lines or noise.
        if total_count > 500:
             print(f"      [DynamicConf] High density detected ({total_count} objects). Raising confidence threshold to 0.4.")
             new_thresh = 0.4
             filtered = [d for d in detections if d['conf'] >= new_thresh]
             print(f"      [DynamicConf] Filtered to {len(filtered)} objects.")
             return filtered
             
        return detections

    # def _spatial_nms(self, detections, radius=10.0):
    #     """
    #     Aggressive NMS based on Euclidean distance for 'data_point' class.
    #     Merges points closer than `radius` pixels.
    #     """
    #     points = [d for d in detections if 'point' in d['label'].lower()]
    #     others = [d for d in detections if 'point' not in d['label'].lower()]
        
    #     if not points: return detections
        
    #     # Sort by confidence descending (keep best point)
    #     points.sort(key=lambda x: x['conf'], reverse=True)
        
    #     kept_points = []
    #     import math
        
    #     for p in points:
    #         cx, cy = p['center']
    #         is_duplicate = False
    #         for kp in kept_points:
    #             kcx, kcy = kp['center']
    #             dist = math.sqrt((cx - kcx)**2 + (cy - kcy)**2)
    #             if dist < radius:
    #                 is_duplicate = True
    #                 break
            
    #         if not is_duplicate:
    #             kept_points.append(p)
                
    #     # Debug info
    #     # if len(points) != len(kept_points):
    #     #      print(f"      [SpatialNMS] Merged {len(points)} points into {len(kept_points)} unique points.")
             
    #     return others + kept_points

