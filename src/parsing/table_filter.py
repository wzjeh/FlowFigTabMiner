import os
import cv2
import numpy as np
from ultralytics import YOLO

class TableFilter:
    def __init__(self, model_path="models/bestYOLOl.pt"):
        """
        Initialize the YOLO-based table filter/segmenter.
        Args:
            model_path (str): Path to the trained YOLOv11 large weights.
        """
        self.model_path = model_path
        self.model = None
        # Class mapping based on user info
        self.CLASS_MAP = {
            0: "table_body",
            1: "table_caption",
            2: "table_note",
            3: "table_scheme" # or whatever the 4th class is
        }
        
        if os.path.exists(model_path):
            print(f"Loading Table Segmentation YOLO model from {model_path}...")
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"Warning: Failed to load Table Filter model: {e}")
        else:
            print(f"Warning: Table Filter model not found at {model_path}.")

    def filter_tables(self, image_paths, conf_threshold=0.5):
        """
        Segment table bodies from images.
        Returns a list of dicts: 
        {
            'path': str, 
            'is_table': bool, 
            'table_body_crop': np.ndarray (or None), 
            'table_caption_crop': np.ndarray,
            'conf': float
        }
        """
        results = []
        
        if self.model is None:
            # Fallback: Treat whole image as table body if model missing
            for p in image_paths:
                img = cv2.imread(p)
                results.append({
                    'path': p, 
                    'is_table': True, 
                    'conf': 0.0, 
                    'table_body_crop': img, # Full image
                    'reason': 'Model not loaded'
                })
            return results

        for img_path in image_paths:
            try:
                # Run inference with resizing to 1440 as requested/trained
                # rect=False might be needed if trained on squares? User said "fit 1440x1440" which usually means letterbox. 
                # Ultralytics handles this by default with imgsz argument.
                prediction = self.model(img_path, imgsz=1440, verbose=False)[0]
                
                original_img = cv2.imread(img_path)
                if original_img is None:
                     results.append({'path': img_path, 'is_table': False, 'reason': 'Read Error'})
                     continue

                h, w = original_img.shape[:2]
                
                # Check for "table_body"
                best_body_box = None
                max_conf = 0.0
                
                # Also store caption for potential future use
                caption_box = None

                components = {}
                
                if len(prediction.boxes) > 0:
                    for box in prediction.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.CLASS_MAP.get(cls_id, "unknown")
                        
                        if conf >= conf_threshold:
                            # Save cop
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                            bx1, by1 = max(0, bx1), max(0, by1)
                            bx2, by2 = min(w, bx2), min(h, by2)
                            comp_crop = original_img[by1:by2, bx1:bx2]
                            
                            # Store in components dict
                            # We might have multiple captions/notes? Use list.
                            if class_name not in components:
                                components[class_name] = []
                            components[class_name].append({
                                'crop': comp_crop,
                                'box': [bx1, by1, bx2, by2],
                                'conf': conf
                            })
                            
                            # Identify best table_body for main processing
                            if class_name == "table_body":
                                # Strategy: Largest area or highest conf? Usually highest conf.
                                if conf > max_conf:
                                    max_conf = conf
                                    best_body_box = box.xyxy[0].cpu().numpy()

                # Process results
                is_table = False
                body_crop = None
                crop_coords = None
                
                # Check if valid table body exists
                if "table_body" in components:
                    is_table = True
                    # Get the body corresponding to max_conf found above (or just take the first/best)
                    # Let's re-extract strictly the max_conf one for 'table_body_crop' key
                    # Or simpler: The loop above already accumulated them.
                    # Just pick the "best" one to be the PRIMARY body for downstream.
                    best_body_item = max(components["table_body"], key=lambda x: x['conf'])
                    body_crop = best_body_item['crop']
                    crop_coords = best_body_item['box']
                else:
                    # HEURISTIC: If no body detected, but 'table_scheme' exists, maybe treat as valid but different type?
                    # User filter preference: "filter out fake tables".
                    # Let's strictly require table_body for now, OR valid structure.
                    pass

                results.append({
                    'path': img_path, 
                    'is_table': is_table, 
                    'conf': max_conf,
                    'table_body_crop': body_crop,
                    'crop_coords': crop_coords,
                    'components': components, # All raw crops
                    'reason': 'No table_body detected' if not is_table else None
                })
                
            except Exception as e:
                print(f"Error filtering {img_path}: {e}")
                results.append({'path': img_path, 'is_table': False, 'error': str(e)})

        return results
