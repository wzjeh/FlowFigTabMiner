import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import nms

class Stage2Detector:
    def __init__(self, model_path="models/bestYOLOm-2-2.pt"):
        print(f"Loading Stage 2 YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        # Class Mapping (User needs to verify this matches bestYOLOm-2-2.pt training)
        # Assuming typical ordering: data_point, tick_mark, etc.
        # This mapping *must* match the training data.yaml.
        # For now, we'll store everything and let the consumer decide based on names if available or IDs.
        # We'll assume the model has names embedded.

    def detect(self, image_path, conf=0.15, imgsz=1024, use_tiling=True):
        """
        Runs detection on a single image (Cleaned Plot).
        Returns list of detections: {'cls': int, 'label': str, 'conf': float, 'box': [x1, y1, x2, y2], 'center': (cx, cy)}
        
        Optimized Strategy:
        1. Tiled Inference (Optional): Split image into crops to detect small targets.
        2. Global Inference: Run on full image for context.
        3. Merge Results: Use NMS to combine tiled and global detections.
        """
        # Load image once
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = image_path # Assume it's a numpy array
            
        if img is None:
            print(f"      [Stage2] Error reading image: {image_path}")
            return []

        detections = []
        
        # 1. Tiled Inference
        if use_tiling:
            tiled_dets = self._detect_tiled(img, conf=conf, tile_size=640, overlap=0.25)
            detections.extend(tiled_dets)
            # print(f"      [Stage2] Tiled Detections: {len(tiled_dets)}")
        
        # 2. Global Inference (Standard Resized)
        # Run inference with low confidence to catch weak signals
        # We assume the user wants high recall
        try:
            results = self.model(img, conf=conf, imgsz=imgsz, rect=False, augment=True, verbose=False)[0]
            
            global_dets = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                conf_score = float(box.conf[0].item())
                label = results.names[cls_id]
                
                global_dets.append({
                    "label": label,
                    "cls_id": cls_id,
                    "conf": conf_score,
                    "box": [float(x1), float(y1), float(x2), float(y2)]
                })
            detections.extend(global_dets)
            # print(f"      [Stage2] Global Detections: {len(global_dets)}")
            
        except Exception as e:
            print(f"      [Stage2] Global inference failed: {e}")

        # 3. Validation & Merging (NMS)
        if not detections:
            return []
            
        final_dets = self._nms_merge(detections, iou_thresh=0.50) # Lowered to 0.50 to merge more aggressively
        
        # Format Output
        formatted_output = []
        for d in final_dets:
            x1, y1, x2, y2 = d['box']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            formatted_output.append({
                "label": d['label'],
                "cls_id": d['cls_id'],
                "conf": d['conf'],
                "box": [x1, y1, x2, y2],
                "center": (cx, cy)
            })

        # 4. Dynamic Thresholding Check (Post-Merge)
        formatted_output = self._apply_dynamic_threshold(formatted_output)

        return formatted_output

    def _detect_tiled(self, img, conf=0.15, tile_size=640, overlap=0.25):
        """
        Slice image into overlapping tiles and detect on each.
        Returns detections with coordinates mapped back to full image.
        """
        h, w = img.shape[:2]
        step = int(tile_size * (1 - overlap))
        
        detections = []
        
        y_steps = range(0, h, step)
        x_steps = range(0, w, step)
        
        for y in y_steps:
            for x in x_steps:
                # Define tile box
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                
                # Adjust start to ensure tile is full size if possible (at edges)
                y1 = max(0, y2 - tile_size)
                x1 = max(0, x2 - tile_size)
                
                tile = img[y1:y2, x1:x2]
                if tile.size == 0: continue
                
                # Run Inference on Tile
                try:
                    results = self.model(tile, conf=conf, imgsz=tile_size, verbose=False)[0]
                    
                    for box in results.boxes:
                        local_x1, local_y1, local_x2, local_y2 = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0].item())
                        conf_score = float(box.conf[0].item())
                        label = results.names[cls_id]
                        
                        # Map back to global
                        global_x1 = local_x1 + x1
                        global_y1 = local_y1 + y1
                        global_x2 = local_x2 + x1
                        global_y2 = local_y2 + y1
                        
                        detections.append({
                            "label": label,
                            "cls_id": cls_id,
                            "conf": conf_score,
                            "box": [float(global_x1), float(global_y1), float(global_x2), float(global_y2)]
                        })
                except Exception as e:
                    pass # Tile failure should not break everything
                    
        return detections

    def _nms_merge(self, detections, iou_thresh=0.6):
        """
        Merge overlapping detections from tiles and global pass.
        Uses Torch NMS for efficiency.
        """
        if not detections:
            return []
            
        # Group by class (to perform class-specific NMS)
        # OR perform class-agnostic NMS if we suspect label confusion.
        # Here we do class-specific.
        
        final_list = []
        
        # Helper to convert to tensor
        classes = set(d['cls_id'] for d in detections)
        
        for c in classes:
            class_dets = [d for d in detections if d['cls_id'] == c]
            
            if not class_dets: continue
            
            boxes = torch.tensor([d['box'] for d in class_dets], dtype=torch.float32)
            scores = torch.tensor([d['conf'] for d in class_dets], dtype=torch.float32)
            
            # Apply NMS
            keep_indices = nms(boxes, scores, iou_thresh)
            
            for idx in keep_indices:
                final_list.append(class_dets[idx])
                
        return final_list

    def _apply_dynamic_threshold(self, detections):
        """
        If density is too high (>500 detections), assume noisy image and raise threshold.
        """
        total_count = len(detections)
        if total_count < 10:
             return detections # Sparse, keep everything
             
        # Heuristic: If count > 150, unlikely to be a clean plot with so many distinct points.
        # It's likely grid lines or noise.
        if total_count > 150:
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

