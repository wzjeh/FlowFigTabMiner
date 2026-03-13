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
                
                # Sort by area ascending (Smallest first) to safely handle
                # But actually O(N^2) comparison is fine.
                
                for i in range(len(dets)):
                    if not keep[i]: continue
                    box_i = dets[i]['box']
                    area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                    conf_i = dets[i].get('conf', 0.0)
                    
                    for j in range(len(dets)):
                        if i == j: continue
                        if not keep[j]: continue
                        
                        box_j = dets[j]['box']
                        area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
                        conf_j = dets[j].get('conf', 0.0)
                        
                        # Compute Intersection
                        xx1 = max(box_i[0], box_j[0])
                        yy1 = max(box_i[1], box_j[1])
                        xx2 = min(box_i[2], box_j[2])
                        yy2 = min(box_i[3], box_j[3])
                        
                        w = max(0, xx2 - xx1)
                        h = max(0, yy2 - yy1)
                        inter = w * h
                        
                        if inter == 0: continue
                        
                        # Coverage Check
                        # If small inside large
                        if area_j < area_i:
                             # j is smaller
                             coverage = inter / area_j
                             if coverage > 0.8:
                                 # j is inside i.
                                 # Previously: kept i (Large).
                                 # FIX: Usually individual ticks are better than a cluster.
                                 # But sometimes a "text_block" is better than "char".
                                 # Let's rely on Confidence AND Size.
                                 
                                 # If i is HUGE (>3x j) and j is tick_label, keep j (Small).
                                 if area_i > 3 * area_j:
                                     keep[i] = False # Kill Large
                                     # break? No, i might cover others.
                                 elif conf_j > conf_i:
                                     keep[i] = False # Kill i (Lower conf)
                                 else:
                                     keep[j] = False # Kill j (Default logic if similar size/conf)

                        else:
                             # i is smaller (or equal)
                             coverage = inter / area_i
                             if coverage > 0.8:
                                 # i is inside j.
                                 if area_j > 3 * area_i:
                                     keep[j] = False # Kill Large (j)
                                 elif conf_i > conf_j:
                                     keep[j] = False # Kill j
                                 else:
                                     keep[i] = False # Kill i
                                     break
                
                return [dets[i] for i in range(len(dets)) if keep[i]]

            # Count before
            count_before = len(detections)
            detections = filter_nested_boxes(detections)
            log(f"Nested Filter: {count_before} -> {len(detections)}")

            # [NEW] Pre-calculate Plot Bounds using Labels (since tick_mark is gone)
            # Use clusters of x/y labels to estimate axis lines.
            
            x_label_dets = [d for d in detections if d['label'] == 'x_tick_label']
            y_label_dets = [d for d in detections if d['label'] == 'y_tick_label']
            
            plot_x_min = 0 
            plot_x_max = img_w
            plot_y_max = img_h
            
            has_right_axis = False

            # Estimate Bottom X-Axis Y-pos from x_tick_labels
            if x_label_dets:
                ys = [d['center'][1] for d in x_label_dets]
                # The axis line is usually just above the labels (min y of labels? or median - height?)
                # Actually, simply use the median Y of the labels as the "Axis Area" 
                # and assume the plot ends slightly above needed.
                # Let's take the Top of the bounding boxes?
                tops = [d['box'][1] for d in x_label_dets]
                plot_y_max = np.median(tops) if tops else img_h

            # Estimate Left/Right Y-Axis X-pos from y_tick_labels
            if y_label_dets:
                xs = [d['center'][0] for d in y_label_dets]
                
                # Cluster Y-labels by X-coordinate
                bins_x = {}
                bin_size = 50 # Larger bin for text labels
                for x in xs:
                    b = int(x / bin_size)
                    bins_x[b] = bins_x.get(b, []) + [x]
                
                major_bins = [b for b, v in bins_x.items() if len(v) >= 2]
                if major_bins:
                    sorted_bins = sorted(major_bins)
                    
                    # Leftmost -> Left Axis
                    left_cluster = bins_x[sorted_bins[0]]
                    plot_x_min = np.median(left_cluster) # Actually this is center of text. Axis is to the right?
                    # For Left Axis, text is to the Left of line. Line ~ max(box.x2)?
                    
                    # Refine plot_x_min using box right edges
                    # But generic center is robust enough for exclusion logic.
                    
                    # Check for Right Axis
                    if len(sorted_bins) >= 2:
                        right_cluster = bins_x[sorted_bins[-1]]
                        x_right = np.median(right_cluster)
                        if (x_right - plot_x_min) > (img_w * 0.4):
                            has_right_axis = True
                            plot_x_max = x_right
                            log(f"Detected Dual Vertical Axes (Labels). Width={plot_x_max - plot_x_min:.1f}")

            log(f"Estimated Plot Bounds (Labels): LeftX~{plot_x_min:.0f}, RightX~{plot_x_max:.0f}, BottomY~{plot_y_max:.0f}")
            log(f"Has Right Axis: {has_right_axis}")

            # 1. Gather candidates based on Class
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
            
            # Filter specifically for tick labels
            # If model is trusted, we just use the label.
            # But we still run OCR to get value.
            
            # Helper to process a list of dets
            def process_candidates(dets, cand_list):
                 for i, d in enumerate(dets):
                    bbox = d['box']
                    cx, cy = d['center']
                    
                    x1, y1, x2, y2 = map(int, bbox)
                    # FIX: Padding increased to handle tight Y-labels
                    pad = 15 
                    x1, y1 = max(0, x1-pad), max(0, y1-pad)
                    x2, y2 = min(img_w, x2+pad), min(img_h, y2+pad)
                    crop = img[y1:y2, x1:x2]
                    
                    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5: continue
                    
                    # Upscale small
                    if crop.shape[0] < 50:
                        scale = 3
                        crop = cv2.resize(crop, (crop.shape[1]*scale, crop.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

                    res = self.ocr.ocr(crop)
                    text = ""
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
                                     if isinstance(txt_obj, (list, tuple)): text = txt_obj[0]; break

                    if i < 5: log(f"OCR [{d['label']}]: '{text}'")
                    val = parse_val(text)
                    if val is not None:
                        # [cx, val, text, cx, cy]
                        cand_list.append([cx, val, text, cx, cy])

            # Helper: Validates monotonic sequence (Longest Monotonic Subsequence)
            def filter_monotonic(candidates, direction='decreasing'):
                if not candidates: return []
                # candidates: list of [cx, val, text, cx, cy]
                # Sort by Y-coordinate (Top to Bottom)
                # Note: valid Y-axis usually has values DECREASING as Y-pixel increases (Top->Bottom)
                
                # Sort by cy (pixel) asc
                cands_sorted = sorted(candidates, key=lambda x: x[4]) 
                
                # Extract values
                vals = [c[1] for c in cands_sorted]
                
                # We want the longest subsequence that is strictly 'decreasing' (or 'increasing' if axis inverted)
                # Standard plot: Y-axis 100 (top) -> 0 (bottom). So vals should be DECREASING.
                
                n = len(vals)
                if n < 2: return cands_sorted
                
                # Simple LIS/LDS O(N^2) dynamic programming
                # dp[i] = length of substring ending at i
                # parent[i] = index of previous element
                
                dp = [1] * n
                parent = [-1] * n
                
                for i in range(n):
                    for j in range(i):
                        is_valid = False
                        if direction == 'decreasing':
                            if vals[i] < vals[j]: is_valid = True # Top(j) > Bottom(i) -> 100 > 90
                        else: # increasing (rare, inverted Y)
                            if vals[i] > vals[j]: is_valid = True
                        
                        if is_valid:
                            if dp[j] + 1 > dp[i]:
                                dp[i] = dp[j] + 1
                                parent[i] = j
                
                # Backtrack best path
                max_len = 0
                end_idx = -1
                for i in range(n):
                    if dp[i] > max_len:
                        max_len = dp[i]
                        end_idx = i
                
                if end_idx == -1: return [] # Should not happen
                
                keep_indices = []
                curr = end_idx
                while curr != -1:
                    keep_indices.append(curr)
                    curr = parent[curr]
                
                keep_indices.reverse()
                return [cands_sorted[i] for i in keep_indices]

            # Process X Labels
            process_candidates(x_label_dets, x_candidates)
            
            # Process Y Labels
            # ... (Split code unchanged) ...
            mid_x = img_w / 2
            
            # [Refactored Split Logic]
            y_left_pool = []
            y_right_pool = []
            
            if y_label_dets:
                 if has_right_axis:
                     for d in y_label_dets:
                         cx = d['center'][0]
                         dist_l = abs(cx - plot_x_min)
                         dist_r = abs(cx - plot_x_max)
                         if dist_l < dist_r: y_left_pool.append(d)
                         else: y_right_pool.append(d)
                 else:
                     # Assumption: single axis usually Left.
                     # But some charts have only Right axis? Rare.
                     y_left_pool = y_label_dets

            process_candidates(y_left_pool, y_left_candidates)
            process_candidates(y_right_pool, y_right_candidates)

            # [NEW] Apply Monotonic Filter to Y-Candidates
            # Default assumption: Standard Axis (Values decrease Key Top->Bottom)
            # We can try both directions and keep the one with more points?
            # Or just enforce standard. Most flowcharts are standard.
            
            count_yl_raw = len(y_left_candidates)
            y_left_candidates = filter_monotonic(y_left_candidates, 'decreasing')
            if len(y_left_candidates) < count_yl_raw:
                log(f"Monotonic Filter (YL): {count_yl_raw} -> {len(y_left_candidates)} (Removed outliers)")
            
            count_yr_raw = len(y_right_candidates)
            if has_right_axis:
                y_right_candidates = filter_monotonic(y_right_candidates, 'decreasing')
                if len(y_right_candidates) < count_yr_raw:
                    log(f"Monotonic Filter (YR): {count_yr_raw} -> {len(y_right_candidates)}")

            log(f"Candidates Found: X={len(x_candidates)}, YL={len(y_left_candidates)}, YR={len(y_right_candidates)}")
            if x_candidates: log(f"Sample X: {[c[2] for c in x_candidates[:5]]}")
            if y_left_candidates: log(f"Sample YL: {[c[2] for c in y_left_candidates[:5]]}")
                            
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
            
            # FIX: If we successfully matched a Right Axis model, force dual-axis mode
            if model_yr:
                has_right_axis = True
                log("Right Axis Model Fit Success -> Forcing Dual Axis Mode")
            
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
            # Detect Columns of Vertical Labels (Left vs Right Axis)
            # Already done in Pre-calculate step, can reuse 'has_right_axis'
            
            # If has_right_axis is True, we know it's Dual Axis.
            if has_right_axis:
                 log(f"Dual Vertical Axes Confirmed via Labels.")
            
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
                    # Y_Left (e.g. temperature) can be negative (°C), so no >= 0 constraint.
                    # X (e.g. residence time) is always positive; keep >= 0 guard for it.
                    # Y_Right/Data_Value (yield %) is always >= 0.
                    f_yl = float(real_yl) if real_yl is not None else None
                    f_yr = float(real_yr) if (real_yr is not None and real_yr >= 0) else None
                    f_x = float(real_x) if (real_x is not None and real_x >= 0) else None
                    
                    row = {
                        "Series": series,
                        "X": f_x,
                        "Y_Left": f_yl,
                        "Y_Right/Data_Value": f_yr,
                    }
                    data_rows.append(row)
            
            return pd.DataFrame(data_rows), debug_log
        
        except Exception as e:
            msg = f"CRITICAL ERROR in MapCoordinates:\n{traceback.format_exc()}"
            log(msg)
            return pd.DataFrame(), debug_log

    def _fit_ransac(self, pairs):
        # Require at least 2 points for line fitting (Linear Algebra basic)
        if len(pairs) < 2: return None
        
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
