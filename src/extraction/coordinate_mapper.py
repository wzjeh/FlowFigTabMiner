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
        
        Returns: DataFrame with [Series, X, Y_Left, Y_Right]
        Strategy (Dual Y-Axis Support):
        1. Find all 'tick_label' / text-like detections.
        2. OCR them to get values.
        3. Separate X vs Y-Left vs Y-Right based on position (Left/Right of image center).
        4. Fit RANSAC models for X, Y-Left, and Y-Right (if detected).
        5. Predict dual values for all data_points.
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
        x_candidates = []      # (center_x, val, raw_text)
        y_left_candidates = [] # (center_y, val, raw_text)
        y_right_candidates = [] # (center_y, val, raw_text)
        
        labels = [d for d in detections if 'label' in d['label']] # simplistic check
        
        print(f"      [DEBUG] CoordMapper found {len(labels)} potential label detections.")
        
        for d in labels:
            bbox = d['box']
            cx, cy = d['center']
            
            # Crop & OCR
            x1, y1, x2, y2 = map(int, bbox)
            pad = 12 # Increased to 12 to fix severe clipping (98->8)
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
            text = ""
            if res and res[0]:
                if isinstance(res[0], dict) and 'rec_texts' in res[0]:
                    rec_texts = res[0].get('rec_texts', [])
                    if rec_texts:
                        text = rec_texts[0]
                elif isinstance(res[0], list):
                    lines = res[0]
                    for line in lines:
                        if isinstance(line, list) and len(line) >= 2:
                            text_info = line[1]
                            if isinstance(text_info, (tuple, list)) and len(text_info) > 0:
                                text = text_info[0]
                                text = text_info[0]
                                break
            
            # DEBUG
            if text:
                 pass # print(f"      [DEBUG] OCR Text for {d['label']}: '{text}'")
            else:
                 pass # print(f"      [DEBUG] OCR Failed for {d['label']}")
                 
            val = parse_val(text)
            # print(f"      [DEBUG] Parsed Val: {val}")
            
            if val is not None:
                # Heuristic Separation of Axes
                cls_name = d['label']
                mid_x = img_w / 2.0
                
                # Note: Start storing (cx, cy) so we can filter by alignment later
                if 'x' in cls_name:
                    x_candidates.append([cx, val, text, cx, cy]) # Added cx, cy
                elif 'y' in cls_name:
                    if cx < mid_x:
                        y_left_candidates.append([cy, val, text, cx, cy])
                    else:
                        y_right_candidates.append([cy, val, text, cx, cy])
                else: 
                    # Fallback Position Heuristic
                    if cy > img_h * 0.8: # Bottom area -> X
                        x_candidates.append([cx, val, text, cx, cy])
                    else:
                        # Side areas -> Y
                        if cx < mid_x:
                            y_left_candidates.append([cy, val, text, cx, cy])
                        else:
                            y_right_candidates.append([cy, val, text, cx, cy])
                        
        
        # 1.5 Alignment Filtering (Refinement Phase 6)
        def filter_aligned_candidates(candidates, axis_type):
            """
            Filters candidates that deviate spatially from the median axis line.
            axis_type='x': Candidates should have same Y (secondary coord is index 4).
            axis_type='y': Candidates should have same X (secondary coord is index 3).
            """
            if not candidates or len(candidates) < 3: return candidates # Too few to filter
            
            # Extract Secondary Coordinate
            # x_cand: [cx, val, text, cx, cy] -> idx=4 (cy)
            # y_cand: [cy, val, text, cx, cy] -> idx=3 (cx)
            sec_idx = 4 if axis_type == 'x' else 3
            
            coords = [c[sec_idx] for c in candidates]
            
            # Clustering to find Main Axis Line
            bins = {}
            bin_size = 20
            for c in coords:
                b = int(c / bin_size)
                if b not in bins: bins[b] = []
                bins[b].append(c)
                
            # Find Best Bin
            # Logic: 
            # 1. Identify "Major" bins (e.g. at least 50% of the max count)
            # 2. Among major bins, pick the one closest to the image/plot center (Coordinate: img_size/2)
            # This avoids picking footers (far bottom) or titles (far top/left) even if they have many detections.
            
            max_count = max(len(v) for v in bins.values())
            major_bins = [b for b, v in bins.items() if len(v) >= max(3, max_count * 0.5)] # At least 3 or half max
            
            if not major_bins:
                 # Fallback to simple max
                 best_bin = max(bins, key=lambda k: len(bins[k]))
            else:
                # Coordinate of the bin is `b * bin_size`.
                # We want the one closest to center of image.
                # Center val depends on axis_type.
                # If axis_type='x', we are looking at Y-coordinates (idx 4). Center is img_h/2.
                # If axis_type='y', we are looking at X-coordinates (idx 3). Center is img_w/2.
                # Note: We don't have img_h/img_w in this scope easily.
                # We can estimate median of all candidates as center? No.
                # But typically:
                # X-Axis: Median Y ~ 80-90% of height. Footer ~ 95% height.
                # We want the Smaller Y (Upper). 
                # Y-Axis: Median X ~ 10-20%width. Title ~ 5%. 
                # We want the Larger X (Inner) for Left Axis? 
                # Let's just assume "Closer to Median of ALL candidates" is a proxy for "Center of Plot"?
                # No, candidates includes the noise.
                
                # Robust Heuristic:
                # X-Axis: Pick the one with SMALLER Y (Top-most among candidates, i.e. closer to plot).
                # Y-Left: Pick the one with LARGER X (Right-most among candidates, i.e. closer to plot).
                # Y-Right: Pick the one with SMALLER X (Left-most among candidates, i.e. closer to plot).
                
                # Wait. "Major Bins" handles the count.
                # Among Major Bins:
                sorted_bins = sorted(major_bins) # Sort by coordinate value
                
                if axis_type == 'x':
                    # Bottom Axis -> Candidates are Y-coords. Plot is above. Axis is top-most of the bottom items.
                    # Pick Smallest Y (first bin).
                    best_bin = sorted_bins[0]
                elif axis_type == 'y':
                    # This is tricky. Y-Left vs Y-Right filters are separate.
                    # For Y-Left candidates (Left side of image): Axis is usually to the Right of the Title. So Pick Largest X.
                    # For Y-Right candidates (Right side): Axis is usually to the Left of the Title. So Pick Smallest X.
                    
                    # How do we know if we are filtering Y-Left or Y-Right?
                    # We pass `axis_type='y'`, but we don't know the side.
                    # We can infer from the values in the bin.
                    # If bin values are small (< img_w/2), it's Left.
                    # If bin values are large (> img_w/2), it's Right.
                    
                    avg_val = np.mean(bins[sorted_bins[0]])
                    # We assume 1000px width typically.
                    # Actually we can check the first item.
                    if avg_val < 500: # Left side
                         # Pick Largest X (Closest to plot) -> Last Bin
                         best_bin = sorted_bins[-1]
                    else: # Right side
                         # Pick Smallest X (Closest to plot) -> First Bin
                         best_bin = sorted_bins[0]
                         
            cluster = bins[best_bin]
            median_val = np.median(cluster)
            
            # Threshold: 30 pixels (More generous as we are centering on the cluster)
            threshold = 30.0 
            
            filtered = []
            for c in candidates:
                if abs(c[sec_idx] - median_val) < threshold:
                    filtered.append(c)
                else:
                    print(f"      [DEBUG] Dropped outlier label: '{c[2]}' at {c[3]:.1f},{c[4]:.1f} (Dist to Cluster {median_val:.1f}: {abs(c[sec_idx]-median_val):.1f})")
            return filtered

        # Apply Filtering
        # print(f"      [DEBUG] Filtering X candidates: {len(x_candidates)} -> ", end="")
        x_candidates = filter_aligned_candidates(x_candidates, 'x')
        # print(len(x_candidates))
        
        # print(f"      [DEBUG] Filtering YL candidates: {len(y_left_candidates)} -> ", end="")
        y_left_candidates = filter_aligned_candidates(y_left_candidates, 'y')
        # print(len(y_left_candidates))
        
        # print(f"      [DEBUG] Filtering YR candidates: {len(y_right_candidates)} -> ", end="")
        y_right_candidates = filter_aligned_candidates(y_right_candidates, 'y')
        # print(len(y_right_candidates))

        # 2. Detect Scale Type & Fit Models using RANSAC
        
        def check_log_scale(candidates):
            if len(candidates) < 2: return False
            matches = 0
            for item in candidates:
                raw_txt = item[2].strip()
                if raw_txt.startswith('.'): raw_txt = raw_txt[1:]
                if raw_txt.startswith("10"):
                    matches += 1
            if len(candidates) > 0 and (matches / len(candidates)) > 0.5:
                 print(f"      -> Detected LOG Scale (Prefix '10' heuristic: {matches}/{len(candidates)})")
                 return True
            return False

        is_x_log = check_log_scale(x_candidates)
        is_yl_log = check_log_scale(y_left_candidates)
        is_yr_log = check_log_scale(y_right_candidates)

        # Helper to fit model
        def fit_axis(candidates, is_log):
            if not candidates: return None
            if is_log:
                pairs = [[p[0], np.log10(p[1])] for p in candidates if p[1] > 0]
            else:
                pairs = [[p[0], p[1]] for p in candidates]
            return self._fit_ransac(pairs)

        model_x = fit_axis(x_candidates, is_x_log)
        model_yl = fit_axis(y_left_candidates, is_yl_log)
        model_yr = fit_axis(y_right_candidates, is_yr_log)
        
        # 3. Data Point Cleaning (Remove Ghost Points) & Transform
        data_rows = []
        points = [d for d in detections if d['label'] == 'data_point']
        
        # Helper to get median axis line from candidates
        def get_axis_line(candidates, axis_type):
            if not candidates: return None
            # axis_type='x' -> Want Y-coord (Horizontal Line). Candidates have [cx, val, text, cx, cy] -> idx 4
            # axis_type='y' -> Want X-coord (Vertical Line). Candidates have [cy, val, text, cx, cy] -> idx 3
            idx = 4 if axis_type == 'x' else 3
            coords = [c[idx] for c in candidates]
            
            # Use Histogram/Mode instead of Median filtering to handle dual-lines
            # Bin every 10 pixels
            if not coords: return None
            
            bins = {}
            bin_size = 20
            for c in coords:
                b = int(c / bin_size)
                if b not in bins: bins[b] = []
                bins[b].append(c)
                
            # Find bin with most items
            best_bin = max(bins, key=lambda k: len(bins[k]))
            
            # If tie, pick based on heuristic?
            # For X-axis, pick the BOTTOM one? (Higher Y index)
            # For Y-axis, pick the OUTER one? (Extreme Left or Right)
            # But robust mode is usually enough. If tie, we check further.
            
            # Refine line by taking median of the best cluster
            cluster = bins[best_bin]
            return np.median(cluster)

        # Calculate axis lines
        line_x = get_axis_line(x_candidates, 'x')      # Median Y
        line_yl = get_axis_line(y_left_candidates, 'y') # Median X
        line_yr = get_axis_line(y_right_candidates, 'y') # Median X
        
        # Filter ghost points
        cleaned_points = []
        ghost_threshold = 10.0 # Pixel distance threshold for being "on the axis line"
        
        for p in points:
            cx, cy = p['center']
            is_ghost = False
            
            # Check Y-Left Ghost (Vertical Line)
            if line_yl is not None and abs(cx - line_yl) < ghost_threshold:
                # print(f"      [DEBUG] Marking GhostPoint at {cx},{cy} (Near YL {line_yl})")
                is_ghost = True
            
            # Check Y-Right Ghost (Vertical Line)
            if line_yr is not None and abs(cx - line_yr) < ghost_threshold:
                # print(f"      [DEBUG] Marking GhostPoint at {cx},{cy} (Near YR {line_yr})")
                is_ghost = True
                
            # Check X-Axis Ghost (Horizontal Line) - Usually less common for labels to look like points, but possible
            # Note: X-axis labels are usually BELOW the plot, so their Y is > Data Y. 
            # But if a data point is exactly on the X-axis line, it might be an axis tick/label residue?
            # User said: "Non-existent data point close to Y-axis label X-coord is essentially a label detecetd as data point"
            # So primarily we filter based on X-coordinates of Y-axes.
            
            if not is_ghost:
                cleaned_points.append(p)
            else:
               # print(f"      [DEBUG] Ghost Point Dropped: {p}")
               pass # Dropping ghost point
               
        points = cleaned_points # Use filtered list

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
                
            # Y Left Calculation
            real_yl = None
            if model_yl:
                try:
                    pred = model_yl.predict([[cy]])[0]
                    real_yl = np.power(10, pred) if is_yl_log else pred
                except: pass

            # Y Right Calculation
            real_yr = None
            if model_yr:
                try:
                    pred = model_yr.predict([[cy]])[0]
                    real_yr = np.power(10, pred) if is_yr_log else pred
                except: pass
            
            # If explicit right model missing, maybe fallback? 
            # Ideally, if no right axis detected, real_yr remains None. 
            # But the user asked for "Redundant Extraction".
            # If only one axis exists, Y_Left == Y_Right is NOT correct if scales differ.
            # If only left axis exists, Y_Right should be None or NaN to indicate absence.
            
            if real_x is not None and (real_yl is not None or real_yr is not None):
                # We need at least one Y value to be valid
                data_rows.append({
                    "Series": series,
                    "X": float(real_x),
                    "Y_Left": float(real_yl) if real_yl is not None else None,
                    "Y_Right": float(real_yr) if real_yr is not None else None,
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

