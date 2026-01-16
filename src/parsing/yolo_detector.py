import os
import cv2
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="models/best425.pt"):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        # Class Mapping based on inspection
        self.CLASS_MAP = {
            0: "axis_break",
            1: "chart_text",
            2: "legend",
            3: "other_image",
            4: "target_image",
            5: "x_axis_title",
            6: "y_axis_title"
        }
        self.TARGET_CLASS_ID = 4  # target_image
        # Elements to Mask (and Cut)
        self.MASK_CLASSES = [0, 1, 2, 5, 6] 

    def process_images(self, image_paths, output_base_dir="data/intermediate", output_subdir_name="yolo_cleaned"):
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
            # conf=0.25 is default, maybe lower/higher depending on needs
            try:
                prediction_result = self.model(img_path, verbose=False)[0]
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
                all_detections.append({
                    "cls_id": cls_id,
                    "label": self.CLASS_MAP.get(cls_id, f"unknown_{cls_id}"),
                    "box": xyxy,
                    "conf": conf
                })
                
                if cls_id == self.TARGET_CLASS_ID:
                    target_boxes.append(xyxy)

            # If no target found, maybe the whole image is the target? 
            # User instruction: "Crop target... if one image has (a)(b), crop two frames."
            # If nothing detected, we might optionally skip or treat whole as target if no 'other_image'
            if not target_boxes:
                # heuristic: if 'other_image' (3) is dominant, skip. Else assume full image.
                # For now, let's be strict: if no target_image detected, maybe it's not a chart.
                # But let's check if there are parts like 'x_axis' etc. 
                # If parts exist but no target box, we use the bounding box of all parts?
                # Simpler: If no target, skip for now to follow 'target_image' rule strictly.
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
                
                # Identify elements INSIDE or OVERLAPPING this target
                # We check intersection.
                elements_found = {}
                
                # Prepare Mask Overlay
                # We want to mask on the target_crop.
                # Need to map global coords to crop coords.
                
                for det in all_detections:
                    if det["cls_id"] in self.MASK_CLASSES:
                        dx1, dy1, dx2, dy2 = map(int, det["box"])
                        
                        # Check intersection with target box
                        ix1 = max(tx1, dx1)
                        iy1 = max(ty1, dy1)
                        ix2 = min(tx2, dx2)
                        iy2 = min(ty2, dy2)
                        
                        if ix1 < ix2 and iy1 < iy2:
                            # Intersection exists. 
                            # 1. Cut (Save separate crop of the element)
                            # Use un-clamped global coords for the element crop itself?
                            # Or intersection? Better to take the whole element crop if possible, 
                            # but sometimes element sticks out? Let's crop the element from original.
                            elem_crop = original_img[dy1:dy2, dx1:dx2]
                            elem_label = det["label"]
                            elem_filename = f"{target_out_name}_{elem_label}_{len(elements_found)}.png"
                            elem_path = os.path.join(yolo_out_dir, elem_filename)
                            if elem_crop.size > 0:
                                cv2.imwrite(elem_path, elem_crop)
                                if elem_label not in elements_found: elements_found[elem_label] = []
                                elements_found[elem_label].append(elem_path)
                            
                            # 2. Mask (White out on target_crop)
                            # Convert global intersection to relative target coords
                            rx1 = ix1 - tx1
                            ry1 = iy1 - ty1
                            rx2 = ix2 - tx1
                            ry2 = iy2 - ty1
                            
                            # Draw white rectangle filled
                            cv2.rectangle(target_crop, (rx1, ry1), (rx2, ry2), (255, 255, 255), -1)

                # Save Cleaned Target
                clean_filename = f"{target_out_name}_cleaned.png"
                clean_path = os.path.join(yolo_out_dir, clean_filename)
                cv2.imwrite(clean_path, target_crop)
                
                results.append({
                    "original_source": img_path,
                    "cleaned_image": clean_path,
                    "elements": elements_found
                })
                print(f"      -> Saved {clean_filename} (masked {sum(len(v) for v in elements_found.values())} elements)")

        return results
