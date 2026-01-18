import numpy as np
import cv2
from paddleocr import PaddleOCR
from sklearn.linear_model import RANSACRegressor, LinearRegression
import pandas as pd
import re

class CoordinateMapper:
    def __init__(self):
        # Initialize OCR - suppress logs
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en')

    def map_coordinates(self, detections, plot_img_path):
        """
        detections: list from Stage2Detector (contains data_points, tick_marks, tick_labels?)
        plot_img_path: path to full image for OCR cropping
        
        Returns: DataFrame with [Series, X, Y]
        Strategy (Tick-less):
        1. Find all 'tick_label' / text-like detections.
        2. OCR them to get values.
        3. Separate X-axis labels vs Y-axis labels based on position.
        4. Use RANSAC to fit Pixel->Value model for X and Y separately.
        5. Predict values for all data_points.
        """
        img = cv2.imread(plot_img_path)
        if img is None: return pd.DataFrame()
        img_h, img_w = img.shape[:2]

        # 1. Gather all potential axis labels
        # Helper: parse numbers
        def parse_val(txt):
            # Remove non-numeric except . - e E
            clean = re.sub(r'[^\d\.\-eE]', '', txt)
            try:
                return float(clean)
            except:
                return None

        # Collect data for regression
        x_candidates = [] # (center_x, val, raw_text)
        y_candidates = [] # (center_y, val, raw_text)
        
        # We can use YOLO 'tick_label' if available, otherwise we might need to OCR
        # existing text regions or use the 'tick_label' boxes provided by Stage 2.
        # Assuming Stage 2 outputs 'tick_label' class.
        
        labels = [d for d in detections if 'label' in d['label']] # simplistic check
        
        print(f"      [DEBUG] CoordMapper found {len(labels)} potential label detections.")
        
        # If no explicit labels from YOLO, might need to run OCR on the whole image?
        # NO, user said "identify each label's number". 
        # So we assume YOLO gives us boxes for labels.
        
        for d in labels:
            bbox = d['box']
            cx, cy = d['center']
            
            # Crop & OCR
            x1, y1, x2, y2 = map(int, bbox)
            # Pad slightly
            pad = 2
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(img_w, x2+pad), min(img_h, y2+pad)
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # Upscale for better OCR on small text
            if crop.shape[0] < 50 or crop.shape[1] < 50:
                scale = 3
                crop = cv2.resize(crop, (crop.shape[1]*scale, crop.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

            res = self.ocr.ocr(crop)
            if not res or not res[0]: continue
            
            # Robust parsing of OCR result
            # Robust parsing of OCR result
            text = ""
            if res and res[0]:
                # Case 1: PaddleX Dict Format
                if isinstance(res[0], dict) and 'rec_texts' in res[0]:
                    rec_texts = res[0].get('rec_texts', [])
                    if rec_texts:
                        text = rec_texts[0]
                # Case 2: List of Lists Format
                elif isinstance(res[0], list):
                    lines = res[0]
                    for line in lines:
                        if isinstance(line, list) and len(line) >= 2:
                            text_info = line[1]
                            if isinstance(text_info, (tuple, list)) and len(text_info) > 0:
                                text = text_info[0]
                                break
            val = parse_val(text)
            
            if val is not None:
                # Store tuple: (coord, val, raw_text)
                
                # Check specific class first if available (e.g., 'x_tick_label')
                cls_name = d['label']
                if 'x' in cls_name:
                    x_candidates.append([cx, val, text])
                elif 'y' in cls_name:
                    y_candidates.append([cy, val, text])
                else: 
                    # Fallback Position Heuristic
                    if cy > img_h * 0.8: # Bottom area
                        x_candidates.append([cx, val, text])
                    elif cx < img_w * 0.25: # Left area
                        y_candidates.append([cy, val, text])
                    else:
                        pass
                        
        
        # 2. Detect Scale Type & Fit Models using RANSAC
        
        # Helper to decide Log vs Linear
        def check_log_scale(candidates):
            if len(candidates) < 2: return False
            
            # User Heuristic: If x labels start with '10' (scientific notation 10^x), it's log scale.
            # Example OCR: "100", "101", "102" or "10 0", "10 1"
            # We check the raw text prefix.
            
            matches = 0
            for item in candidates:
                raw_txt = item[2].strip()
                # Remove common OCR noise like leading dot
                if raw_txt.startswith('.'): raw_txt = raw_txt[1:]
                
                if raw_txt.startswith("10"):
                    matches += 1
            
            # If significant portion starts with 10, assume Log
            if len(candidates) > 0 and (matches / len(candidates)) > 0.5:
                 print(f"      -> Detected LOG Scale (Prefix '10' heuristic: {matches}/{len(candidates)})")
                 return True
            
            return False

        is_x_log = check_log_scale(x_candidates)
        is_y_log = check_log_scale(y_candidates)

        # Fit X
        if is_x_log:
            # Transform values to log domain for fitting
            x_log_cand = [[p[0], np.log10(p[1])] for p in x_candidates if p[1] > 0]
            model_x = self._fit_ransac(x_log_cand)
        else:
            # Strip text for regression
            model_x = self._fit_ransac([[p[0], p[1]] for p in x_candidates])

        # Fit Y
        if is_y_log:
             y_log_cand = [[p[0], np.log10(p[1])] for p in y_candidates if p[1] > 0]
             model_y = self._fit_ransac(y_log_cand)
        else:
             # Strip text for regression
             model_y = self._fit_ransac([[p[0], p[1]] for p in y_candidates])
        
        # 3. Transform Points
        data_rows = []
        points = [d for d in detections if d['label'] == 'data_point']
        
        for p in points:
            cx, cy = p['center']
            series = p.get('series', 'Default')
            
            # X Calculation
            real_x = None
            if model_x:
                try:
                    pred = model_x.predict([[cx]])[0]
                    real_x = np.power(10, pred) if is_x_log else pred
                except: pass
                
            # Y Calculation
            real_y = None
            if model_y:
                try:
                    pred = model_y.predict([[cy]])[0]
                    real_y = np.power(10, pred) if is_y_log else pred
                except: pass
            
            if real_x is not None and real_y is not None:
                data_rows.append({
                    "Series": series,
                    "X": float(real_x),
                    "Y": float(real_y),
                    "Pixel_X": cx,
                    "Pixel_Y": cy
                })
            
        return pd.DataFrame(data_rows)

    def _fit_ransac(self, pairs):
        """
        Fit a robust linear regression model (RANSAC).
        pairs: list of [pixel_coord, value]
        """
        if len(pairs) < 3: # Need at least a few points for RANSAC
            # Fallback to simple Linear Regression if 2 points
            if len(pairs) == 2:
                data = np.array(pairs)
                X = data[:, 0].reshape(-1, 1)
                y = data[:, 1]
                model = LinearRegression()
                model.fit(X, y)
                return model
            return None
            
        data = np.array(pairs)
        X = data[:, 0].reshape(-1, 1) # Pixel
        y = data[:, 1]                # Value
        
        # RANSAC
        # Note: residual_threshold might need tuning based on pixel noise
        ransac = RANSACRegressor(random_state=42, min_samples=2, residual_threshold=10.0)
        try:
            ransac.fit(X, y)
            print(f"      -> RANSAC Fit: Inliers={np.sum(ransac.inlier_mask_)} / {len(pairs)}")
            return ransac
        except Exception as e:
            print(f"      ! RANSAC failed: {e}")
            return None

