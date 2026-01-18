import cv2
import numpy as np
import os
from paddleocr import PaddleOCR
from sklearn.cluster import KMeans
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
                        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        mean_color = np.mean(hsv_crop, axis=(0,1))
                        
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

    def hsv_dist(self, c1, c2):
        """Weighted HSV distance."""
        dh = min(abs(c1[0] - c2[0]), 180 - abs(c1[0] - c2[0])) / 180.0
        ds = abs(c1[1] - c2[1]) / 255.0
        dv = abs(c1[2] - c2[2]) / 255.0
        return 4.0*dh + 1.0*ds + 0.5*dv

    def match_points(self, points, prototypes, plot_img_path):
        """
        Refined Logic: Cluster-then-Match
        1. Extract colors for all data points.
        2. K-Means Clustering (K = Num Legend Entries).
        3. Match Clusters to Legends using Hungarian Algorithm (linear_sum_assignment).
        """
        from scipy.optimize import linear_sum_assignment
        
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
        
        # 1. Extract Features
        features = []
        valid_indices = []
        for i, p in enumerate(data_points):
            bbox = p['box']
            # Crop center
            pad_x = max(1, int((bbox[2]-bbox[0])*0.25))
            pad_y = max(1, int((bbox[3]-bbox[1])*0.25))
            x1 = max(0, int(bbox[0])+pad_x)
            y1 = max(0, int(bbox[1])+pad_y)
            x2 = min(img_w, int(bbox[2])-pad_x)
            y2 = min(img_h, int(bbox[3])-pad_y)
            
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mean = np.mean(hsv_crop, axis=(0,1))
                    features.append(mean)
                    valid_indices.append(indices[i])
        
        if not features: return points
        
        X = np.array(features)
        legend_labels = list(prototypes.keys())
        legend_colors = [prototypes[l]['hsv'] for l in legend_labels]
        n_legends = len(legend_labels)
        n_points = len(X)
        
        assigned_labels = ["Unknown"] * n_points
        
        # Strategy A: Balanced Assignment (if detected points are divisible by series count)
        # This addresses the user observation: "Each series should have 5 points"
        # We assume detection is perfect (Recall = 100%) for this to work best.
        if n_points > 0 and n_points % n_legends == 0:
            count_per_series = n_points // n_legends
            print(f"      [DEBUG] Balanced Mode TRIGGERED: {n_points} points, {n_legends} series -> {count_per_series} per series.")
            
            # Create targets: Repeat each legend color 'count_per_series' times
            targets = []
            target_map = [] 
            
            for i, proto_hsv in enumerate(legend_colors):
                for _ in range(count_per_series):
                    targets.append(proto_hsv)
                    target_map.append(legend_labels[i])
            
            # Cost Matrix
            cost_matrix = np.zeros((n_points, n_points))
            for i in range(n_points): 
                for j in range(n_points): 
                    cost_matrix[i, j] = self.hsv_dist(X[i], targets[j])
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                assigned_labels[r] = target_map[c]
                
        else:
            # Fallback
            print(f"      [DEBUG] Unbalanced Mode TRIGGERED: {n_points} points / {n_legends} series != Int. Running KMeans.")
            print(f"      [DEBUG] Prototype Keys: {legend_labels}")
            
            # If fewer points than clusters, reduce K
            n_clusters = min(n_legends, n_points)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            cluster_centers = kmeans.cluster_centers_
            
            # Match Clusters to Legends
            cost_matrix = np.zeros((n_clusters, n_legends))
            for i in range(n_clusters):
                for j in range(n_legends):
                    cost_matrix[i, j] = self.hsv_dist(cluster_centers[i], legend_colors[j])
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            cluster_to_legend = {}
            for r, c in zip(row_ind, col_ind):
                cluster_to_legend[r] = legend_labels[c]
                
            for i in range(n_points):
                c_lbl = cluster_labels[i]
                if c_lbl in cluster_to_legend:
                    assigned_labels[i] = cluster_to_legend[c_lbl]

        # 4. Apply labels back to points
        for i, idx in enumerate(valid_indices):
            points[idx]['series'] = assigned_labels[i]
            
        return points
