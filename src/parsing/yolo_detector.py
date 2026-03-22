import os
import cv2
import numpy as np
import glob
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="models/bestYOLOn-2-1.pt"):
        print(f"Loading YOLO model from {model_path}...")
        # ultralytics uses torch. Force torch to respect limits if not already
        import torch
        if torch.get_num_threads() > 1:
            torch.set_num_threads(1)
            
        self.model = YOLO(model_path)
        _torch_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(_torch_device)
        print(f"   -> Device: {_torch_device.upper()}")
        # Dynamic Class Mapping from Model
        if hasattr(self.model.names, 'items'):
            self.CLASS_MAP = self.model.names
        else:
            # Assume list-like
            self.CLASS_MAP = {i: n for i, n in enumerate(self.model.names)}
        
        print(f"   -> Classes: {self.CLASS_MAP}")
        
        # Identify Target Class ID dynamically if possible, or fallback to 'target_image'
        # We search for 'target_image' in names
        self.TARGET_CLASS_ID = 3
        
        # Mask Everything except Target
        self.MASK_CLASSES = [k for k in self.CLASS_MAP.keys() if k != self.TARGET_CLASS_ID]

    def process_images(self, image_paths, output_base_dir="data/intermediate", output_subdir_name="yolo_cleaned", imgsz=640):
        """
        Process a list of image paths.
        Returns a list of dictionaries with metadata about processed charts.
        """
        results = []
        yolo_out_dir = os.path.join(output_base_dir, output_subdir_name)
        os.makedirs(yolo_out_dir, exist_ok=True)

        for img_path in image_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            print(f"   YOLO Processing: {basename}...")
            
            # Predict
            try:
                prediction_result = self.model(img_path, imgsz=imgsz, rect=False, verbose=False)[0]
            except Exception as e:
                print(f"      -> prediction failed: {e}")
                continue

            # Load original image for cropping/masking
            original_img = cv2.imread(img_path)
            if original_img is None:
                print(f"      -> failed to read image: {img_path}")
                continue
                
            h, w = original_img.shape[:2]

            # 1. Find Targets
            boxes = prediction_result.boxes
            target_boxes = []
            
            # Store all detections for lookup
            all_detections = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                xyxy = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                conf = float(box.conf[0].item())
                label = self.CLASS_MAP.get(cls_id, f"unknown_{cls_id}")
                
                all_detections.append({
                    "cls_id": cls_id,
                    "label": label,
                    "box": xyxy,
                    "conf": conf
                })
                
                if cls_id == self.TARGET_CLASS_ID:
                    target_boxes.append(xyxy)

            # If no target found, maybe the whole image is the target? 
            if not target_boxes:
                # If no target, skip for now to follow 'target_image' rule strictly.
                print(f"      -> No target_image detected. Skipping.")
                continue

            # 2. Process each target
            for i, t_box in enumerate(target_boxes):
                tx1, ty1, tx2, ty2 = map(int, t_box)
                # Clamp
                tx1, ty1 = max(0, tx1), max(0, ty1)
                tx2, ty2 = min(w, tx2), min(h, ty2)
                
                # Crop Target
                target_crop = original_img[ty1:ty2, tx1:tx2].copy()
                target_h, target_w = target_crop.shape[:2]
                if target_h == 0 or target_w == 0: continue
                
                target_out_name = f"{basename}_t{i}"
                
                # CLEANUP: Remove detailed crops from previous runs for THIS target
                # Pattern: {basename}_t{i}_*
                existing_files = glob.glob(os.path.join(yolo_out_dir, f"{target_out_name}_*"))
                for f in existing_files:
                    try: os.remove(f)
                    except: pass

                # Save RAW Target (for downstream OCR)
                target_raw = target_crop.copy()
                raw_filename = f"{target_out_name}_raw.png"
                raw_path = os.path.join(yolo_out_dir, raw_filename)
                cv2.imwrite(raw_path, target_raw)

                # Identify elements INSIDE or OVERLAPPING this target
                # We check intersection.
                elements_found = {}
                
                # Prepare Mask Overlay
                # Mask colors? White for general cleaning.
                
                # 3. Identify & Mask Elements
                for det in all_detections:
                    if det["cls_id"] in self.MASK_CLASSES:
                        dx1, dy1, dx2, dy2 = map(int, det["box"])
                        label = det["label"]
                        
                        # Intersection with Target
                        ix1 = max(tx1, dx1); iy1 = max(ty1, dy1)
                        ix2 = min(tx2, dx2); iy2 = min(ty2, dy2)
                        
                        intersect_w = max(0, ix2 - ix1)
                        intersect_h = max(0, iy2 - iy1)
                        is_intersecting = (intersect_w * intersect_h) > 0
                        
                        include_element = False
                        
                        # Logic: Include if intersecting target area
                        if is_intersecting:
                            include_element = True
                        
                        # Special logic for Caption? usually below.
                        # If label == 'caption' and intersection is tiny (just touching), maybe ignore?
                        # But if we want to extract it, we should include it.
                        
                        if include_element:
                            # Save Crop (of the element itself)
                            elem_crop = original_img[dy1:dy2, dx1:dx2]
                            if elem_crop.size > 0:
                                elem_filename = f"{target_out_name}_{label}_{len(elements_found.get(label, []))}.png"
                                elem_path = os.path.join(yolo_out_dir, elem_filename)
                                cv2.imwrite(elem_path, elem_crop)
                                
                                if label not in elements_found: elements_found[label] = []
                                elements_found[label].append(elem_path)
                            
                            # Mask (White out on target_crop)
                            # Dont mask caption usually? Or mask if it protrudes into image?
                            # Standard Logic: Clean the image for data extraction. 
                            # If caption is inside, mask it. If outside, masking does nothing (detected outside crop).
                            # If partially inside, mask the part inside.
                            # Exception: subfigure_marker -> MASK IT (it's inside).
                            # Legend -> MASK IT.
                            # Titles -> MASK IT.
                            
                            # Convert global intersection to relative target coords
                            rx1 = ix1 - tx1
                            ry1 = iy1 - ty1
                            rx2 = ix2 - tx1
                            ry2 = iy2 - ty1
                            
                            cv2.rectangle(target_crop, (rx1, ry1), (rx2, ry2), (255, 255, 255), -1)

                # Save Cleaned Target
                clean_filename = f"{target_out_name}_cleaned.png"
                clean_path = os.path.join(yolo_out_dir, clean_filename)
                cv2.imwrite(clean_path, target_crop)
                
                results.append({
                    "original_source": img_path,
                    "cleaned_image": clean_path,
                    "raw_image": raw_path,
                    "elements": elements_found
                })
                print(f"      -> Saved {clean_filename} (masked {sum(len(v) for v in elements_found.values())} elements)")

        return results
