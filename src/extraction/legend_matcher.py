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
                if tcx < mcx - 10: continue # Allow slight left drift
                
                # Vertical proximity
                ref_h = max(tbox[3]-tbox[1], mbox[3]-mbox[1])
                if abs(tcy - mcy) > ref_h * 3.0: continue

                dist = np.sqrt((tcx - mcx)**2 + (tcy - mcy)**2)
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
                
                if x2 > x1 and y2 > y1:
                    crop = master_img[y1:y2, x1:x2]
                    if crop.size > 0:
                        # hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        # mean_color = np.mean(hsv_crop, axis=(0,1))
                        
                        # Use Weighted Mean
                        mean_color = self.get_weighted_mean_color(crop) # Crop is BGR
                        
                        if best_text not in prototypes:
                            prototypes[best_text] = {'colors': []}
                        prototypes[best_text]['colors'].append(mean_color)

        # Aggregate
        final_protos = {}
        for label, data in prototypes.items():
            final_protos[label] = {
                'hsv': np.mean(data['colors'], axis=0),
                'count': len(data['colors'])
            }
            
        return final_protos

    def get_weighted_mean_color(self, img_crop):
        """
        Compute mean color with Gaussian weighting centered in the crop.
        This reduces the influence of background pixels at the edges.
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
             return np.mean(img_crop, axis=(0,1))
             
        weights = gaussian / np.sum(gaussian)
        
        # Compute Weighted Mean per channel (in HSV)
        # However, for Hue, weighted mean is tricky due to circularity.
        # Strict way: Convert to cartesian, mean, convert back.
        # Practical way for small hue diffs: Weighted Average is OK.
        # Best way: Use RGB for weighted mean, then convert to HSV.
        
        # 1. Convert to RGB float
        rgb_crop = cv2.cvtColor(img_crop, cv2.COLOR_HSV2BGR) # Wait, input is usually HSV from existing logic? No, extraction is BGR usually. 
        # Ah, update: we should pass BGR here.
        
        # Let's check caller. Caller passes BGR?
        # In current code: `hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)` happens inside the loop.
        # We will move HSV conversion here.
        
        # We process in Linear RGB space to be strictly correct for physical mixing, 
        # but standard Gamma RGB is fine for perceptual center extraction.
        
        weighted_r = np.sum(rgb_crop[:,:,2] * weights)
        weighted_g = np.sum(rgb_crop[:,:,1] * weights)
        weighted_b = np.sum(rgb_crop[:,:,0] * weights)
        
        mean_bgr = np.array([[[weighted_b, weighted_g, weighted_r]]], dtype=np.uint8)
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
                    # mean = self.get_weighted_mean_color(crop)
                    # For speed on many points, we can do simple mean if crop is tight, 
                    # but user requested weighting.
                    mean = self.get_weighted_mean_color(crop)
                    features.append(mean)
                    valid_indices.append(indices[i])
        
        if not features: return points
        
        X = np.array(features)

        # 2. Nearest Neighbor Assignment
        # We calculate distance from every point to every legend color.
        # Custom Metric because of Hue Wrapping.
        
        n_points = len(X)
        n_legends = len(legend_colors)
        
        # Manual cdist with weighted_hsv_dist
        dists = np.zeros((n_points, n_legends))
        
        for i in range(n_points):
            for j in range(n_legends):
                dists[i, j] = self.weighted_hsv_dist(X[i], legend_colors[j])
                
        # Assign to min distance
        labels_idx = np.argmin(dists, axis=1)
        assigned_labels = [legend_labels[idx] for idx in labels_idx]

        # 3. Apply labels back to points
        for i, idx in enumerate(valid_indices):
            points[idx]['series'] = assigned_labels[i]
            # Optional: Store confidence/distance?
            # points[idx]['match_dist'] = dists[i, labels_idx[i]]
            
        return points
