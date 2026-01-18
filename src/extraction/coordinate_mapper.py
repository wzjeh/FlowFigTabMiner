import numpy as np
import cv2
from paddleocr import PaddleOCR
from sklearn.linear_model import LinearRegression
import pandas as pd
import re

class CoordinateMapper:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en')

    def map_coordinates(self, detections, plot_img_path):
        """
        detections: list from Stage2Detector (contains data_points, tick_marks, tick_labels?)
        plot_img_path: path to full image for OCR cropping
        
        Returns: DataFrame with [Series, X, Y]
        """
        img = cv2.imread(plot_img_path)
        img_h, img_w = img.shape[:2]

        # 1. Separate Detections
        ticks = [d for d in detections if d['label'] == 'tick_mark'] # Check strict label name from YOLO
        # Note: If YOLO detects "tick_label" specifically, use that.
        # If not, we might need to OCR around tick marks?
        # Assuming YOLO detects 'tick_mark' generally.
        
        # NOTE: The user's YOLOv11m might need to be checked for exact classes.
        # "Stage 2 Model - Medium... 识别所有 tick_mark (刻度线) 和 tick_label (刻度数字)"
        # Assuming classes: 'tick_mark', 'tick_label', 'data_point'.
        
        tick_labels = [d for d in detections if 'label' in d['label']] # 'x_tick_label' / 'y_tick_label' or just 'tick_label'
        # Fallback: if 'tick_label' not present, look for text near tick_marks using purely OCR if needed.
        # Let's assume YOLO gives us 'tick_label'.
        
        x_ticks, y_ticks = [], []
        
        # 2. Heuristic Separation of X/Y Ticks
        # X Ticks: Usually near the bottom 20% of image? Or grouped by Y-coordinate?
        # Y Ticks: Usually near left 20%?
        # Better: Group by alignment.
        
        # Let's try to OCR the 'tick_label' boxes first to get values
        x_pairs = [] # (pixel_x, value)
        y_pairs = [] # (pixel_y, value)

        for tl in tick_labels:
            bbox = tl['box'] # x1,y1,x2,y2
            # Crop and OCR
            crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            if crop.size == 0: continue
            
            result = self.ocr.ocr(crop)
            if not result or not result[0]: continue
            
            try:
                text = result[0][0][1][0]
            except Exception as e:
                print(f"DEBUG: CoordinateMapper OCR result error: {type(result)} -> {result}")
                continue
            val = self._parse_number(text)
            if val is None: continue
            
            # Determine if X or Y axis based on position relative to image center?
            # Or aspect ratio of box?
            # Or YOLO class name (x_tick_label vs y_tick_label)?
            
            label_name = tl['label']
            cx, cy = tl['center']
            
            if 'x' in label_name:
                x_pairs.append([cx, val])
            elif 'y' in label_name:
                y_pairs.append([cy, val])
            else:
                # Heuristic: 
                # If box is wider -> X? 
                # If Cy > height * 0.8 -> X? Cx < width * 0.2 -> Y?
                if cy > img_h * 0.7: # Bottom
                    x_pairs.append([cx, val])
                else: # Assume left/Y
                    y_pairs.append([cy, val])

        # 3. Linear Regression
        model_x = self._fit_axis(x_pairs)
        model_y = self._fit_axis(y_pairs)
        
        # 4. Transform Points
        data_rows = []
        data_points = [d for d in detections if d['label'] == 'data_point']
        
        for p in data_points:
            cx, cy = p['center']
            series = p.get('series', 'Default')
            
            val_x = model_x.predict([[cx]])[0][0] if model_x else None
            val_y = model_y.predict([[cy]])[0][0] if model_y else None
            
            data_rows.append({
                "Series": series,
                "X": val_x,
                "Y": val_y
            })
            
        return pd.DataFrame(data_rows)

    def _parse_number(self, text):
        clean = re.sub(r'[^\d\.\-eE]', '', text)
        try:
            return float(clean)
        except:
            return None

    def _fit_axis(self, pairs):
        if len(pairs) < 2:
            return None
        data = np.array(pairs)
        X = data[:, 0].reshape(-1, 1) # Pixels
        y = data[:, 1].reshape(-1, 1) # Values
        model = LinearRegression()
        model.fit(X, y)
        return model
