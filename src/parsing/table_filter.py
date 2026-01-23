import os
import cv2
from ultralytics import YOLO

class TableFilter:
    def __init__(self, model_path="models/table_filter_yolo.pt"):
        """
        Initialize the YOLO-based table filter.
        Args:
            model_path (str): Path to the trained YOLOv11nano weights.
        """
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            print(f"Loading Table Filter YOLO model from {model_path}...")
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"Warning: Failed to load Table Filter model: {e}")
        else:
            print(f"Warning: Table Filter model not found at {model_path}. Filtering may be disabled or mocked.")

    def filter_tables(self, image_paths, conf_threshold=0.5):
        """
        Filter a list of table images. 
        Returns a list of dicts: {'path': str, 'is_table': bool, 'conf': float}
        """
        results = []
        
        # If model is not loaded, assume all are tables (fallback) or none?
        # Let's assume all are tables for now to avoid blocking development if weights are missing.
        if self.model is None:
            for p in image_paths:
                results.append({'path': p, 'is_table': True, 'conf': 0.0, 'reason': 'Model not loaded'})
            return results

        for img_path in image_paths:
            try:
                # Run inference
                # Assuming class 0 is "table" and class 1 is "not_table" OR 
                # simply detection of "table" object confirms it's a table.
                # Adjust logic based on actual training strategy. 
                # STRATEGY: Classification model (YOLO-Classify) or Detection? 
                # User said: "detect tables... filter out fake tables". 
                # Let's assume it's a detection model that detects "real_table".
                # If a "real_table" is detected with high confidence, keep it.
                
                prediction = self.model(img_path, verbose=False)[0]
                
                # Check for "table" class detection with high confidence
                is_table = False
                max_conf = 0.0
                
                # If it's a classification model:
                if hasattr(prediction, 'probs') and prediction.probs is not None:
                     # e.g. names: {0: 'false_positive', 1: 'true_table'}
                     # We need to know the class mapping. 
                     # For now, let's assume detection logic: Box exists = True.
                     pass 
                
                # If it's a detection model:
                if len(prediction.boxes) > 0:
                    for box in prediction.boxes:
                        conf = float(box.conf[0])
                        if conf > max_conf:
                            max_conf = conf
                        if conf >= conf_threshold:
                            is_table = True
                            
                results.append({
                    'path': img_path, 
                    'is_table': is_table, 
                    'conf': max_conf
                })
                
            except Exception as e:
                print(f"Error filtering {img_path}: {e}")
                results.append({'path': img_path, 'is_table': False, 'conf': 0.0, 'error': str(e)})

        return results
