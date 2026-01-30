import numpy as np
import cv2
import os
from paddleocr import PaddleOCR
from sklearn.linear_model import RANSACRegressor, LinearRegression
import pandas as pd
import re

class CoordinateMapper:
    def __init__(self):
        # Initialize OCR - suppress logs
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en')

    def map_coordinates(self, detections, plot_img_path, force_log_x=False, extract_point_labels=False):
        """
        detections: list from Stage2Detector
        force_log_x: bool, if True, treat small integer ticks as Exponents (10^x).
        extract_point_labels: bool, if True, OCR near points for values.
        Returns: (DataFrame, debug_log_list)
        """
        debug_log = []
        def log(msg): debug_log.append(msg)
        
        import traceback
        try:
            log(f"Starting Map Coordinates on {os.path.basename(plot_img_path)}")
            img = cv2.imread(plot_img_path)
            if img is None: 
                log("Error: Image load failed")
                return pd.DataFrame(), debug_log
            img_h, img_w = img.shape[:2]

            # 0. Filter Nested Boxes (NMS-like for nested ticks)
            # User Logic: If small label inside large label (overlap > 85%), keep Large.
            # Actually logic says "keep Large, discard Small".
            def filter_nested_boxes(dets):
                if not dets: return []
                keep = [True] * len(dets)
                # Sort by area descending (Largest first)
                # No, lets just do O(N^2) it is small.
                
                for i in range(len(dets)):
                    if not keep[i]: continue
                    box_i = dets[i]['box']
                    area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                    
                    for j in range(len(dets)):
                        if i == j: continue
                        if not keep[j]: continue
                        
                        box_j = dets[j]['box']
                        area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                        
                        # Compute Intersection
                        xx1 = max(box_i[0], box_j[0])
                        yy1 = max(box_i[1], box_j[1])
                        xx2 = min(box_i[2], box_j[2])
                        yy2 = min(box_i[3], box_j[3])
                        
                        w = max(0, xx2 - xx1)
                        h = max(0, yy2 - yy1)
                        inter = w * h
                        
                        if inter == 0: continue
                        
                        # Check coverage of small box
                        if area_j < area_i:
                             # j is smaller
                             coverage = inter / area_j
                             if coverage > 0.8:
                                 keep[j] = False
                        else:
                             # i is smaller (or equal)
                             coverage = inter / area_i
                             if coverage > 0.8:
                                 keep[i] = False
                                 break # i is removed, stop checking i
                
                return [dets[i] for i in range(len(dets)) if keep[i]]

            # Count before
            count_before = len(detections)
            detections = filter_nested_boxes(detections)
            log(f"Nested Filter: {count_before} -> {len(detections)}")

            # [NEW] Pre-calculate Plot Bounds using Tick Marks to filter candidates
            # This prevents heatmap numbers from being treated as axis labels
            tick_marks = [d for d in detections if d['label'] == 'tick_mark']
            
            plot_x_min = 0 # Left Axis X
            plot_x_max = img_w # Right Axis X (Default to edge)
            plot_y_max = img_h # Bottom Axis Y
            
            has_right_axis = False # Is there a Right Axis?

            if tick_marks:
                txs = [d['center'][0] for d in tick_marks]
                tys = [d['center'][1] for d in tick_marks]
                
                # --- Y-Axis Detection (Vertical Columns) ---
                bins_x = {}
                bin_size = 20 # px
                for x in txs:
                    b = int(x / bin_size)
                    bins_x[b] = bins_x.get(b, []) + [x]
                
                major_cols = [b for b, v in bins_x.items() if len(v) >= 2]
                if major_cols:
                    sorted_cols = sorted(major_cols)
                    # Leftmost -> plot_x_min
                    left_cluster = bins_x[sorted_cols[0]]
                    if len(left_cluster) >= 2:
                        plot_x_min = np.median(left_cluster)
                    
                    # Check for Rightmost (Right Axis)
                    if len(sorted_cols) >= 2:
                        right_cluster = bins_x[sorted_cols[-1]]
                        x_right = np.median(right_cluster)
                        # Check separation
                        if (x_right - plot_x_min) > (img_w * 0.4): # >40% width
                            has_right_axis = True
                            plot_x_max = x_right
                            log(f"Detected Dual Vertical Axes (Scatter Mode). Width={plot_x_max - plot_x_min:.1f}")

                # --- X-Axis Detection (Horizontal Rows) ---
                bins_y = {}
                for y in tys:
                    b = int(y / bin_size)
                    bins_y[b] = bins_y.get(b, []) + [y]
                
                if bins_y:
                    # Bottom axis usually has LARGEST Y
                    best_y = max(bins_y, key=lambda k: len(bins_y[k]))
                    if len(bins_y[best_y]) >= 2:
                        plot_y_max = np.median(bins_y[best_y]) # Bottom X-Axis Y-pos
            
            log(f"Estimated Plot Boundaries: LeftAxis_X={plot_x_min:.1f}, RightAxis_X={plot_x_max:.1f}, BottomAxis_Y={plot_y_max:.1f}")
            log(f"Has Right Axis: {has_right_axis}")

            # 1. Gather all potential axis labels
            def parse_val(txt):
                # Handle scientific notation like '10^-2', '10-2', '10^2'
                # Replace '10^' or '10' followed by '-' as '1e'
                if '10' in txt:
                    # Try to clean up common OCR issues for scientific
                    # 10^-2 -> 1e-2
                    # 10-2 -> 1e-2
                    sub = txt.replace('10^', '1e').replace('10', '1e')
                    # Ensure structure is 1e...
                    if '1e' in sub:
                         # Check if it looks valid
                         try: return float(sub)
                         except: pass
                
                clean = re.sub(r'[^\d\.\-eE]', '', txt)
                try: return float(clean)
                except: return None

            x_candidates = []      # (center_x, val, raw_text, cx, cy)
            y_left_candidates = [] 
            y_right_candidates = []
            
            # FIX: Relax filter to include everything text-like
            text_dets = [d for d in detections if d['label'] not in ['data_point', 'marker', 'legend_marker']]
            log(f"Processing {len(text_dets)} text/tick detections")
            log(f"(Classes: {list(set(d['label'] for d in text_dets))})")
            
            for i, d in enumerate(text_dets):
                bbox = d['box']
                cx, cy = d['center']
                
                # FILTER: Must be OUTSIDE the plot area to be an axis label
                # X-Label: Should be BELOW plot_y_max (plus some margin? No, strictly greater usually)
                # Y-Label: Should be LEFT of plot_x_min OR RIGHT of plot_x_max (if dual axis)
                
                is_potential_axis = False
                margin = 2 # FIX: Strict margin
                
                # Bottom X-Axis
                if cy > plot_y_max - margin: is_potential_axis = True
                
                # Left Y-Axis
                if cx < plot_x_min + margin: is_potential_axis = True
                
                # Right Y-Axis (Allow if dual axis OR if simply to the right?)
                # If plot_x_max is close to img_w (default), this condition is hard to satisfy unless cx > img_w.
                # If dual axis detected, plot_x_max is inward.
                if has_right_axis:
                    if cx > plot_x_max - margin: is_potential_axis = True
                else:
                    # Standard Single Axis -> Fallback check if simple scatter
                    # To be safe, allow anything > img_w * 0.9? No, heatmap values are there.
                    # Stick to logic: Only allow Right Axis candidates if we explicitly detected ticks there.
                    pass
                
                # Check bounds existence
                has_bounds = (plot_x_min > 20 and plot_y_max < img_h - 20)
                
                if has_bounds and not is_potential_axis:
                    # Log rejection (limit volume)
                    if i < 10: log(f"Rejected Candidate [{d['label']}] at ({cx:.1f},{cy:.1f}) - Inside Plot")
                    continue 

                x1, y1, x2, y2 = map(int, bbox)
                # FIX: Padding reduced to 12
                pad = 12 
                x1, y1 = max(0, x1-pad), max(0, y1-pad)
                x2, y2 = min(img_w, x2+pad), min(img_h, y2+pad)
                crop = img[y1:y2, x1:x2]
                
                if crop.size == 0: continue
                if crop.shape[0] < 50 or crop.shape[1] < 50:
                    scale = 3
                    crop = cv2.resize(crop, (crop.shape[1]*scale, crop.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

                res = self.ocr.ocr(crop)
                text = ""
                # Handle PaddleX Dict vs List
                if res and isinstance(res, list) and len(res) > 0:
                    first_item = res[0]
                    if isinstance(first_item, dict):
                        if 'rec_texts' in first_item and first_item['rec_texts']:
                            text = first_item['rec_texts'][0]
                    elif isinstance(first_item, list):
                        for line in first_item:
                            if isinstance(line, list) and len(line) >= 2:
                                 # line: [box, (text, conf)]
                                 txt_obj = line[1]
                                 if isinstance(txt_obj, (list, tuple)) and len(txt_obj) > 0:
                                     text = txt_obj[0]
                                     break
                
                if i < 5: log(f"Top 5 AxisCand OCR [{i}]: Text='{text}'")

                val = parse_val(text)
                
                if val is not None:
                    cls_name = d['label']
                    mid_x = img_w / 2.0
                    item = [cx, val, text, cx, cy]
                    
                    # Strict Geographic Sorting based on Bounds
                    if cy > plot_y_max - margin:
                        x_candidates.append(item)
                    elif cx < plot_x_min + margin:
                        y_left_candidates.append(item)
                    else:
                        # Fallback to simple split
                        if cy > img_h * 0.8: x_candidates.append(item)
                        else:
                            if cx < mid_x: y_left_candidates.append(item)
                            else: y_right_candidates.append(item)
                            
            log(f"Candidates Found: X={len(x_candidates)}, YL={len(y_left_candidates)}, YR={len(y_right_candidates)}")
            if x_candidates: log(f"Sample X: {[c[2] for c in x_candidates[:5]]}")
            if y_left_candidates: log(f"Sample YL: {[c[2] for c in y_left_candidates[:5]]}")

            # 1.5 Alignment Filtering
            def filter_aligned_candidates(candidates, axis_type):
                if not candidates or len(candidates) < 3: return candidates
                sec_idx = 4 if axis_type == 'x' else 3
                coords = [c[sec_idx] for c in candidates]
                
                bins = {}
                bin_size = 20
                for c in coords:
                    b = int(c / bin_size)
                    if b not in bins: bins[b] = []
                    bins[b].append(c)
                    
                max_count = max(len(v) for v in bins.values())
                major_bins = [b for b, v in bins.items() if len(v) >= max(3, max_count * 0.5)]
                
                if not major_bins:
                     best_bin = max(bins, key=lambda k: len(bins[k]))
                else:
                    sorted_bins = sorted(major_bins)
                    if axis_type == 'x': best_bin = sorted_bins[0]
                    elif axis_type == 'y':
                        # If "Left" Y-axis, we want the LEFT-most column (Smallest X)
                        best_bin = sorted_bins[0] 
                             
                cluster = bins[best_bin]
                median_val = np.median(cluster)
                threshold = 30.0 
                
                filtered = []
                for c in candidates:
                    if abs(c[sec_idx] - median_val) < threshold: filtered.append(c)
                    else: 
                         if len(filtered) < 5: # log sample drops
                             pass # log(f"Dropped outlier aligned: {c[2]}")
                return filtered

            x_candidates = filter_aligned_candidates(x_candidates, 'x')
            y_left_candidates = filter_aligned_candidates(y_left_candidates, 'y')
            y_right_candidates = filter_aligned_candidates(y_right_candidates, 'y')
            
            log(f"After Filter: X={len(x_candidates)}, YL={len(y_left_candidates)}, YR={len(y_right_candidates)}")
            if x_candidates: log(f"Sample X: {[c[2] for c in x_candidates[:3]]}")
            if y_left_candidates: log(f"Sample YL: {[c[2] for c in y_left_candidates[:3]]}")

            # 2. Detect Scale Type & Fit Models
            def check_log_scale(candidates):
                if len(candidates) < 2: return False
                matches = 0
                for item in candidates:
                    raw_txt = item[2].strip()
                    # support 10^ or 1e
                    if raw_txt.startswith("10") or "e" in raw_txt: matches += 1
                if len(candidates) > 0 and (matches / len(candidates)) > 0.5:
                     return True
                return False

            is_x_log = force_log_x or check_log_scale(x_candidates)
            
            # User says: "Only heatmap X is log, others are normal".
            # So trigger Linear Y if heatmap mode.
            if extract_point_labels:
                 is_yl_log = False
                 is_yr_log = False
            else:
                 is_yl_log = check_log_scale(y_left_candidates)
                 is_yr_log = check_log_scale(y_right_candidates)
            
            log(f"Log Scale Detect: X={is_x_log}, YL={is_yl_log}")

            def prepare_pairs_and_fit(candidates, is_log, axis_type):
                 # Candidates: [coord_primary, val, text, cx, cy]
                 # idx 3 is cx, idx 4 is cy
                 pairs = []
                 for p in candidates:
                     # FIX: Select correct pixel coord based on axis
                     pixel = p[3] if axis_type == 'x' else p[4]
                     val = p[1]
                     target = val
                     if is_log:
                         # Heuristic for Exponent vs Value
                         if abs(val) < 15 and float(val).is_integer():
                             target = val # Exponent
                         elif val > 0:
                             target = np.log10(val)
                         else: continue
                     pairs.append([pixel, target])
                 
                 if not pairs: return None
                 if len(pairs) > 0:
                      # Debug Log for pairs
                      subset = pairs[:5]
                      log(f"Fitting Model ({axis_type}) with {len(pairs)} pairs. Sample: {subset}")
                       
                 return self._fit_ransac(pairs)

            model_x = prepare_pairs_and_fit(x_candidates, is_x_log, 'x')
            model_yl = prepare_pairs_and_fit(y_left_candidates, is_yl_log, 'y')
            model_yr = prepare_pairs_and_fit(y_right_candidates, is_yr_log, 'y')
            
            log(f"Models Fit: X={'OK' if model_x else 'FAIL'}, YL={'OK' if model_yl else 'FAIL'}")
            
            # 3. Data Point Cleaning & Transform
            data_rows = []
            points = [d for d in detections if d['label'] == 'data_point']
            log(f"Data Points to Map: {len(points)}")
            
            # Calculate axis lines (Ghost filtering)
            def get_axis_line_pos(candidates, axis_type):
                if not candidates: return None
                idx = 4 if axis_type == 'x' else 3
                coords = [c[idx] for c in candidates]
                if not coords: return None
                return np.median(coords) # Simply median for now

            line_yl = get_axis_line_pos(y_left_candidates, 'y') 
            line_yr = get_axis_line_pos(y_right_candidates, 'y') 
            
            cleaned_points = []
            ghost_threshold = 10.0
            
            for p in points:
                cx, cy = p['center']
                is_ghost = False
                if line_yl and abs(cx - line_yl) < ghost_threshold: is_ghost = True
                if line_yr and abs(cx - line_yr) < ghost_threshold: is_ghost = True
                if not is_ghost: cleaned_points.append(p)
                    
            points = cleaned_points

            # 3.5 Inside-Plot Label Promotion (Refined Quadrant Logic)
            # User Logic: Find the intersection of the Left Y-Axis and Bottom X-Axis.
            # The "Plot Area" is the quadrant to the Top-Right of this intersection.
            # i.e., x > axis_x and y < axis_y (since image Y grows down).
            
            # 3.5 Determine Chart Type (Scatter/Dual vs Heatmap) Logic
            # Detect Columns of Vertical Ticks (Left vs Right Axis)
            tick_marks = [d for d in detections if d['label'] == 'tick_mark']
            has_right_axis = False
            
            if tick_marks:
                txs = [d['center'][0] for d in tick_marks]
                bins_x = {}
                bin_size = 20 # px
                for x in txs:
                    b = int(x / bin_size)
                    bins_x[b] = bins_x.get(b, []) + [x]
                
                # Count major columns (>=3 ticks)
                major_cols = [b for b, v in bins_x.items() if len(v) >= 3]
                
                # If we have at least 2 distinct major columns separated by plot width
                # Let's check spread
                if len(major_cols) >= 2:
                    sorted_cols = sorted(major_cols)
                    min_x = np.median(bins_x[sorted_cols[0]])
                    max_x = np.median(bins_x[sorted_cols[-1]])
                    if (max_x - min_x) > (img_w * 0.4): # Significant width
                        has_right_axis = True
                        log(f"Detected Dual Vertical Axes (Scatter Mode). Width={max_x - min_x:.1f}")
            
            # Quadrant Logic / Data Value Promotion
            # Only enabled if Heatmap likely (No Right Axis) AND Requested
            # OR simplistic: If Right Axis, Disable. If No Right Axis, check flag.
            
            use_data_value_promotion = False
            
            if has_right_axis:
                use_data_value_promotion = False # Always disable for dual-axis scatter
                log("Dual Axis Scatter detected -> Disabling Data Value Promotion.")
            else:
                # Ambiguous (Single Left Axis). Could be Simple Scatter or Heatmap.
                # Default to Scatter (Safe) unless extracting labels explicitly.
                if extract_point_labels:
                    use_data_value_promotion = True
                    log("Single Axis + Extract Labels -> Enabling Data Value Promotion (Heatmap Mode).")
                else:
                    use_data_value_promotion = False
                    log("Single Axis + No Extract Flag -> Disabling Data Value Promotion (Simple Scatter Mode).")
            
            if use_data_value_promotion:
                # Reuse pre-calculated boundaries
                axis_x_line = plot_x_min
                axis_y_line = plot_y_max
                
                if axis_x_line > 0 and axis_y_line < img_h:
                     # ... existing Quadrant Logic ...
                     margin = 15.0 # px padding
                     quad_x = axis_x_line + margin
                     quad_y = axis_y_line - margin
                     
                     promoted_count = 0
                     for d in detections:
                          lbl = d['label']
                          if lbl in ['tick_label', 'text', 'value']:
                              cx, cy = d['center']
                              
                              if cx > quad_x and cy < quad_y:
                                  # Promote!
                                  d['label'] = 'data_value'
                                  promoted_count += 1
                     
                     if promoted_count > 0:
                         log(f"Promoted {promoted_count} labels to 'data_value' (Quadrant Logic)")

            # 4. Point Label Extraction (YOLO-guided)
            value_labels = ['data_value', 'value_label', 'point_value', 'floating_value']
            value_dets = [d for d in detections if d['label'] in value_labels]
            
            # Fallback Logic adjusted: If extract_point_labels is True, we already promoted text.
            # But if detections were initially 'value_label', we keep them.
            
            should_extract = len(value_dets) > 0 or extract_point_labels
            point_labels = {} 
            
            if should_extract and value_dets:
                log(f"Extracting Point Labels from {len(value_dets)} value boxes")
                # ... extraction loop ...
                for idx, p in enumerate(points):
                        px, py = p['center']
                        best_det = None
                        min_dist = float('inf')
                        search_radius = 120.0 
                        for v in value_dets:
                            vx, vy = v['center']
                            dist = np.sqrt((vx - px)**2 + (vy - py)**2)
                            if dist < search_radius and dist < min_dist:
                                min_dist = dist
                                best_det = v
                        
                        if best_det:
                            # ... OCR crop logic ...
                            bx1, by1, bx2, by2 = map(int, best_det['box'])
                            pad = 14
                            bx1, by1 = max(0, bx1-pad), max(0, by1-pad)
                            bx2, by2 = min(img_w, bx2+pad), min(img_h, by2+pad)
                            crop = img[by1:by2, bx1:bx2]
                            if crop.shape[0] < 60 or crop.shape[1] < 60:
                                scale = 4
                                crop = cv2.resize(crop, (crop.shape[1]*scale, crop.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
                            try:
                                res = self.ocr.ocr(crop)
                                val = None
                                txt = ""
                                if res and isinstance(res, list) and len(res) > 0:
                                    first_item = res[0]
                                    if isinstance(first_item, dict): 
                                        if 'rec_texts' in first_item: txt = first_item['rec_texts'][0]
                                    elif isinstance(first_item, list):
                                        line_list = first_item
                                        if line_list and isinstance(line_list, list) and len(line_list) > 0:
                                             first_line = line_list[0]
                                             if len(first_line) >= 2:
                                                 txt_obj = first_line[1]
                                                 if isinstance(txt_obj, (list, tuple)) and len(txt_obj) > 0:
                                                     txt = txt_obj[0]
                                    val = parse_val(txt)
                            except Exception: pass
                            if val is not None:
                                point_labels[idx] = val
            
            log(f"Point Labels Extracted: {len(point_labels)}")

            # 5. Prediction
            for idx, p in enumerate(points):
                cx, cy = p['center']
                series = p.get('series', 'Default')
                
                real_x = None
                if model_x:
                     try:
                        pred = model_x.predict([[cx]])[0]
                        real_x = np.power(10, pred) if is_x_log else pred
                     except: pass
                
                real_yl = None
                if model_yl:
                    try:
                        pred = model_yl.predict([[cy]])[0]
                        real_yl = np.power(10, pred) if is_yl_log else pred
                    except: pass

                # Y_Right Logic / Data Value Logic
                # Column "Y_Right/Data_Value" semantics:
                # If Heatmap -> Data Value (Z)
                # If Scatter -> Y-Right Axis Value
                
                real_yr = None
                
                if has_right_axis and model_yr:
                    # Case A: Dual Axis Scatter -> Use Axis Model
                     try:
                        pred = model_yr.predict([[cy]])[0]
                        real_yr = np.power(10, pred) if is_yr_log else pred
                     except: pass
                
                else:
                    # Case B: Single Axis / Heatmap -> Use Point Label if exists
                    label_val = point_labels.get(idx)
                    if label_val is not None:
                        real_yr = label_val
                
                if real_x is not None:
                    row = {
                        "Series": series,
                        "X": float(real_x),
                        "Y_Left": float(real_yl) if real_yl is not None else None,
                        "Y_Right/Data_Value": float(real_yr) if real_yr is not None else None,
                    }
                    data_rows.append(row)
            
            return pd.DataFrame(data_rows), debug_log
        
        except Exception as e:
            msg = f"CRITICAL ERROR in MapCoordinates:\n{traceback.format_exc()}"
            log(msg)
            return pd.DataFrame(), debug_log

    def _fit_ransac(self, pairs):
        # Require at least 3 points for robust regression to avoid wild extrapolation
        if len(pairs) < 3: return None
        
        data = np.array(pairs)
        X = data[:, 0].reshape(-1, 1)
        y = data[:, 1]
        
        # For >2 points, try RANSAC for robustness
        ransac = RANSACRegressor(random_state=42, min_samples=2, residual_threshold=10.0)
        try:
            ransac.fit(X, y)
            return ransac
        except: 
            try:
                model = LinearRegression()
                model.fit(X, y)
                return model
            except: return None
