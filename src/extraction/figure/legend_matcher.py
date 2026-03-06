import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from collections import defaultdict

class LegendMatcher:
    def __init__(self, yolo_model=None):
        """
        yolo_model: Instance of Stage2Detector (or compatible YOLO wrapper) loaded with bestYOLOm weights.
        """
        self.yolo = yolo_model
        # Initialize OCR - suppress logs
        self.ocr = PaddleOCR(use_angle_cls=False, lang='en')

    def parse_legend_crops(self, legend_crops):
        """
        Refined Logic:
        1. Merge multiple legend crops into one master image (vertical stack).
        2. Detect Markers (YOLO) on master image.
        3. Detect Text (OCR) on master image.
        4. Match Markers to Text (Spatial).
        """
        if not legend_crops:
            return {}

        # 1. Merge Crops
        # Read all images
        images = []
        for path in legend_crops:
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
        
        if not images:
            return {}
            
        # Vertical Stack with padding
        max_w = max(img.shape[1] for img in images)
        total_h = sum(img.shape[0] for img in images) + (len(images)-1)*10 # 10px padding
        
        master_img = np.full((total_h, max_w, 3), 255, dtype=np.uint8)
        y_offset = 0
        for img in images:
            h, w = img.shape[:2]
            master_img[y_offset:y_offset+h, 0:w] = img
            y_offset += h + 10
            
        # Save master legend for debugging
        debug_master_path = "debug_master_legend.png"
        cv2.imwrite(debug_master_path, master_img)
        print(f"      [DEBUG] Saved Master Legend to {debug_master_path}")
        
        # 2. Detect Markers (YOLO) on Master Image
        if self.yolo is None: return {}
        
        try:
            # Use high resolution for master legend image
            markers = self.yolo.detect(debug_master_path, conf=0.15, imgsz=1280)
            markers = [m for m in markers if m['label'] in ['data_point', 'marker']]
            print(f"      [DEBUG] Master Legend Markers: {len(markers)}")
        except Exception as e:
            print(f"      ! YOLO failed on Master Legend: {e}")
            return {}

        img_h, img_w = master_img.shape[:2]
        
        # 3. Detect Text (OCR) on Master Image
        # Upscale if small
        ocr_scale = 1
        ocr_img = master_img
        if img_h < 100 or img_w < 200:
            ocr_scale = 3
            ocr_img = cv2.resize(master_img, (img_w*ocr_scale, img_h*ocr_scale), interpolation=cv2.INTER_CUBIC)
            
        ocr_result = self.ocr.ocr(ocr_img)
        
        texts = []
        if ocr_result:
            # Handle PaddleX / New PaddleOCR Dict Format
            if isinstance(ocr_result[0], dict) and 'rec_texts' in ocr_result[0]:
                data = ocr_result[0]
                rec_texts = data.get('rec_texts', [])
                rec_boxes = data.get('rec_boxes', [])
                
                for i, txt in enumerate(rec_texts):
                    box = rec_boxes[i]
                    if hasattr(box, 'tolist'): box = box.tolist()
                    
                    x_coords = []
                    y_coords = []
                    if isinstance(box, (list, tuple)) and len(box) > 0:
                        if isinstance(box[0], (list, tuple)):
                             x_coords = [p[0] for p in box]
                             y_coords = [p[1] for p in box]
                        elif len(box) >= 4:
                             x_coords = [box[0], box[2]]
                             y_coords = [box[1], box[3]]

                    if not x_coords: continue

                    x1, y1 = min(x_coords)/ocr_scale, min(y_coords)/ocr_scale
                    x2, y2 = max(x_coords)/ocr_scale, max(y_coords)/ocr_scale
                    
                    texts.append({
                        'text': txt,
                        'box': [x1, y1, x2, y2],
                        'center': ((x1+x2)/2, (y1+y2)/2)
                    })
                    
            # Handle Classic List Format
            elif isinstance(ocr_result[0], list):
                 for line in ocr_result[0]:
                    box = line[0]
                    txt = line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                    
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    
                    x1, y1 = min(x_coords)/ocr_scale, min(y_coords)/ocr_scale
                    x2, y2 = max(x_coords)/ocr_scale, max(y_coords)/ocr_scale
                     
                    texts.append({
                        'text': txt,
                        'box': [x1, y1, x2, y2],
                        'center': ((x1+x2)/2, (y1+y2)/2)
                    })

        print(f"      [DEBUG] Master Legend Texts: {len(texts)}")

        # 4. Spatial Matching
        prototypes = {}
        for m in markers:
            mbox = m['box']
            mcx, mcy = m['center']
            
            best_text = None
            min_dist = float('inf')
            
            for t in texts:
                tbox = t['box']
                tcx, tcy = t['center']
                
                # Loose constraint: Text roughly right or below
                # But mostly vertically aligned.
                
                # Vertical proximity
                ref_h = max(tbox[3]-tbox[1], mbox[3]-mbox[1]) # Height of text/marker
                
                # FIX: Strict vertical alignment. 
                # Legend text is usually centered vertically with marker.
                # Allow tolerance of ~1.2 lines?
                if abs(tcy - mcy) > ref_h * 1.5: continue

                # FIX: Use Left Edge for horizontal distance
                # Text Center (tcx) biases towards short words.
                # Marker is always to the LEFT of Text.
                # Distance = (Text Left - Marker Right) ideal?
                # or (Text Left - Marker Center)? Let's use Text Left.
                tx1 = tbox[0]
                
                # Check horizontal ordering: Text must be to the right of marker
                # Allow slight overlap (e.g. marker bounding box big)
                if tx1 < mbox[0] - 10: continue

                # Calculate Distance:
                # Vertical diff is critical. Horizontal diff is secondary (closest to the right).
                # Weighted distance
                dy = abs(tcy - mcy)
                dx = abs(tx1 - mcx) # Distance from marker center to text start
                
                # Penalize vertical distance heavily
                dist = dy * 5.0 + dx
                
                if dist < min_dist:
                    min_dist = dist
                    best_text = t['text']
            
            if best_text:
                # Extract Color
                pad_x = max(1, int((mbox[2]-mbox[0])*0.2))
                pad_y = max(1, int((mbox[3]-mbox[1])*0.2))
                
                x1 = max(0, int(mbox[0])+pad_x)
                y1 = max(0, int(mbox[1])+pad_y)
                x2 = min(img_w, int(mbox[2])-pad_x)
                y2 = min(img_h, int(mbox[3])-pad_y)
                
                CONST_DEBUG = True

                if x2 > x1 and y2 > y1:
                    crop = master_img[y1:y2, x1:x2]
                    if crop.size > 0:
                        # Use Weighted Mean or Dominant? 
                        # Dominant is safer for extracted legend key too (often on white bg)
                        mean_color = self.get_dominant_color(crop) 
                        
                        if best_text not in prototypes:
                            prototypes[best_text] = {'colors': []}
                        prototypes[best_text]['colors'].append(mean_color)
                        
                        if CONST_DEBUG:
                            print(f"      [DEBUG] Proto '{best_text}': HSV={mean_color}")

        # Aggregate
        final_protos = {}
        for label, data in prototypes.items():
            colors = np.array(data['colors'])
            if len(colors) == 0: continue
            
            # Smart Filter: Prefer High Saturation/Value samples
            # HSV: H=0-179, S=0-255, V=0-255
            # Low Saturation (<40) or Low Value (<40) = Achromatic (Black/Gray/White)
            # We want the color that distinguishes the series.
            
            s_vals = colors[:, 1]
            v_vals = colors[:, 2]
            
            # Filter for "Colorful" markers (Saturation > 40 and Value > 40)
            mask_colorful = (s_vals > 40) & (v_vals > 40)
            
            if np.sum(mask_colorful) > 0:
                # We have colorful candidates.
                valid_colors = colors[mask_colorful]
                
                # STRATEGY: Pick the "Best" Representative instead of Averaging.
                # Averaging Hues [0, 179] -> 89 (Wrong).
                # Averaging Red [0] and Purple [130] -> 65 (Yellow) (Wrong).
                
                # We pick the sample with the HIGHEST SATURATION.
                # This assumes the most saturated marker found is the "True" legend color
                # and others might be faded, noise, or partial crops.
                
                best_idx = np.argmax(valid_colors[:, 1]) # Max Saturation
                final_mean = valid_colors[best_idx]
                
                if CONST_DEBUG:
                     print(f"      [DEBUG] Proto '{label}' Selected BEST from {len(valid_colors)} candidates (Max Sat: {final_mean[1]}).")

            else:
                # Fallback: All seem gray/black.
                # Just take the one with max Value (Brightest)? or Mean?
                # If truly achromatic, Mean is fine.
                final_mean = np.mean(colors, axis=0)

            final_protos[label] = {
                'hsv': final_mean,
                'count': len(colors)
            }
            if CONST_DEBUG:
                print(f"      [DEBUG] Final Proto '{label}': HSV={final_protos[label]['hsv'].astype(int)}")
            
        return final_protos

    def get_dominant_color(self, img_crop):
        """
        Extract the dominant foreground color, ignoring white/light background.
        This is crucial for hollow markers where the center is white.
        """
        if img_crop.size == 0: return np.array([0, 0, 0])
        
        # Convert to HSV for masking
        hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        
        # Define Background: Low Saturation AND High Value (White/Gray)
        # OpenAI CV: S(0-255), V(0-255)
        # White is S~0, V~255
        mask_bg = (s < 40) & (v > 200)
        mask_fg = ~mask_bg
        
        if np.sum(mask_fg) < 5: 
             # Fallback to Gaussian weighted mean if mostly background (e.g. very thin marker or actually gray)
             return self.get_weighted_mean_color(img_crop)
             
        # Compute mean of FG pixels in BGR space (to avoid Hue wrap issues)
        fg_pixels = img_crop[mask_fg]
        mean_bgr = np.mean(fg_pixels, axis=0).astype(np.uint8)
        
        mean_hsv = cv2.cvtColor(np.array([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        return mean_hsv

    def get_weighted_mean_color(self, img_crop):
        """
        Compute mean color with Gaussian weighting centered in the crop.
        Fallback for when smart masking fails.
        """
        h, w = img_crop.shape[:2]
        if h == 0 or w == 0: return np.array([0, 0, 0])
        
        # Create Gaussian Mask
        sigma_x = w / 2.5
        sigma_y = h / 2.5
        
        x = np.linspace(-w/2, w/2, w)
        y = np.linspace(-h/2, h/2, h)
        X, Y = np.meshgrid(x, y)
        
        gaussian = np.exp(-(X**2 / (2*sigma_x**2) + Y**2 / (2*sigma_y**2)))
        
        # Normalize mask
        if np.sum(gaussian) == 0:
             # Should convert BGR->HSV here?
             # No, return raw HSV? 
             # Let's fix this to be consistent with get_dominant_color
             m_bgr = np.mean(img_crop, axis=(0,1)).astype(np.uint8)
             return cv2.cvtColor(np.array([[m_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
             
        weights = gaussian / np.sum(gaussian)
        
        # 1. Convert to RGB float (Linearize if possible, but standard BGR is fine for now)
        # Input is BGR (OpenCV default)
        
        # We compute weighted mean on BGR channels directly.
        
        weighted_b = np.sum(img_crop[:,:,0] * weights)
        weighted_g = np.sum(img_crop[:,:,1] * weights)
        weighted_r = np.sum(img_crop[:,:,2] * weights)
        
        mean_bgr = np.array([[[weighted_b, weighted_g, weighted_r]]], dtype=np.uint8)
        
        # Convert the final WEIGHTED MEAN BGR to HSV
        mean_hsv = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2HSV)[0][0]
        
        return mean_hsv

    # def hsv_dist(self, c1, c2):
        # Default now handled by custom distance func in match_points
    #     """Weighted HSV distance."""
    #     dh = min(abs(c1[0] - c2[0]), 180 - abs(c1[0] - c2[0])) / 180.0
    #     ds = abs(c1[1] - c2[1]) / 255.0
    #     dv = abs(c1[2] - c2[2]) / 255.0
    #     return 4.0*dh + 1.0*ds + 0.5*dv
    
    def weighted_hsv_dist(self, c1, c2):
         """
         c1, c2: [H, S, V] arrays.
         H is 0-179, S/V 0-255.
         """
         # Cast to float to avoid uint8 overflow
         c1 = c1.astype(float)
         c2 = c2.astype(float)

         # Wrap Hue
         dh = min(abs(c1[0] - c2[0]), 180 - abs(c1[0] - c2[0])) / 180.0
         ds = abs(c1[1] - c2[1]) / 255.0
         dv = abs(c1[2] - c2[2]) / 255.0
         
         # Weights: Hue is most important, then Saturation. Value is least robust (lighting/shadows).
         return 4.0*dh + 1.5*ds + 0.5*dv

    def match_points(self, points, prototypes, plot_img_path):
        """
        Refined Logic: Nearest Neighbor with Weighted Color
        1. Extract weighted mean color for all data points.
        2. Assign each point to the closest Legend Prototype.
        """
        if not points or not prototypes:
            return points

        img = cv2.imread(plot_img_path)
        if img is None: return points
        img_h, img_w = img.shape[:2]
        
        # Filter data points
        data_points = []
        indices = [] 
        for i, p in enumerate(points):
            if p['label'] in ['data_point', 'marker']:
                data_points.append(p)
                indices.append(i)
                
        if not data_points: return points
        
        # 1. Extract Features (Weighted Mean Color)
        features = []
        valid_indices = []
        
        legend_labels = list(prototypes.keys())
        legend_colors = np.array([prototypes[l]['hsv'] for l in legend_labels])
        
        for i, p in enumerate(data_points):
            bbox = p['box']
            # Crop slightly larger to ensure we get the whole marker, 
            # but Gaussian weight will focus on center.
            pad_x = max(1, int((bbox[2]-bbox[0])*0.1)) # Small padding
            pad_y = max(1, int((bbox[3]-bbox[1])*0.1))
            
            x1 = max(0, int(bbox[0])-pad_x)
            y1 = max(0, int(bbox[1])-pad_y)
            x2 = min(img_w, int(bbox[2])+pad_x)
            y2 = min(img_h, int(bbox[3])+pad_y)
            
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    mean = self.get_dominant_color(crop)
                    features.append(mean)
                    valid_indices.append(indices[i])
        
        if not features: return points
        
        X = np.array(features)

        # 2. Nearest Neighbor Assignment
        n_points = len(X)
        n_legends = len(legend_colors)
        
        dists = np.zeros((n_points, n_legends))
        
        for i in range(n_points):
            for j in range(n_legends):
                dists[i, j] = self.weighted_hsv_dist(X[i], legend_colors[j])
        
        # Assign to min distance
        labels_idx = np.argmin(dists, axis=1)
        assigned_labels = [legend_labels[idx] for idx in labels_idx]

        # Debug Stats for first few points
        if n_points > 0:
            print(f"      [DEBUG] Point 0 HSV: {X[0].astype(int)}")
            for j in range(n_legends):
                print(f"        -> Dist to '{legend_labels[j]}': {dists[0,j]:.2f}")
            print(f"        -> Assigned: {assigned_labels[0]}")

        # 3. Apply labels back to points
        for i, idx in enumerate(valid_indices):
            points[idx]['series'] = assigned_labels[i]
            points[idx]['match_dist'] = dists[i, labels_idx[i]]
            
        return points
