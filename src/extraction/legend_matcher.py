import cv2
import numpy as np
from paddleocr import PaddleOCR
import math

class LegendMatcher:
    def __init__(self):
        # Initialize OCR once
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def parse_legend(self, legend_image_path):
        """
        Parses a legend image crop to extract series prototypes.
        Returns: {'Series Name': {'color': (h,s,v), 'icon_roi': np.array}}
        """
        img = cv2.imread(legend_image_path)
        if img is None:
            return {}

        # 1. OCR to find text and its bbox
        result = self.ocr.ocr(legend_image_path)
        prototypes = {}

        if not result or not result[0]:
            return prototypes

        for line in result[0]:
            # line structure: [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], (text, conf)]
            try:
                box, (text, conf) = line
            except ValueError:
                print(f"DEBUG: LegendMatcher line structure mismatch: {line} (len={len(line)})")
                continue
            
            # 2. Heuristic: Icon is usually to the LEFT of the text
            # box[0] is top-left, box[3] is bottom-left
            text_x_min = min(p[0] for p in box)
            text_y_min = min(p[1] for p in box)
            text_y_max = max(p[1] for p in box)
            text_height = text_y_max - text_y_min
            
            # Define search area for icon: Left of text, same height, some width
            # Assume icon width approx 2x text height or fixed
            search_w = int(text_height * 2.5) 
            target_x_max = int(text_x_min)
            target_x_min = max(0, target_x_max - search_w)
            
            target_y_min = int(text_y_min)
            target_y_max = int(text_y_max)
            
            # Crop icon candidate
            icon_crop = img[target_y_min:target_y_max, target_x_min:target_x_max]
            
            if icon_crop.size == 0:
                continue

            # 3. Extract Dominant Color (HSV)
            # Remove white background
            hsv = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
            # Mask out white/black to find color
            # Define white/black limits
            # low_S = 30, val > 50?
            # Better: KMeans or simple dominant non-bg color
            
            dominant_color = self._get_dominant_color(icon_crop)
            
            prototypes[text] = {
                "color": dominant_color,
                "icon_crop": icon_crop # Store for shape matching if needed
            }
            
        return prototypes

    def _get_dominant_color(self, img_crop):
        # Reshape to list of pixels
        pixels = img_crop.reshape(-1, 3)
        # Filter out near-white
        # BGR > 230,230,230 is white
        non_white = pixels[np.all(pixels < 230, axis=1)] 
        
        if len(non_white) == 0:
            return (0, 0, 0) # No color found
            
        # Get mean of remaining
        mean_bgr = np.mean(non_white, axis=0)
        
        # Convert to HSV for better comparison
        # np.uint8 wrapper needed for cv2.cvtColor
        mean_hsv = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        return mean_hsv

    def match_points(self, points, prototypes, full_plot_img_path):
        """
        Assigns a series label to each point.
        points: list of dicts from Stage2Detector
        prototypes: dict from parse_legend
        full_plot_img_path: path to the plot image to extract point colors
        """
        if not prototypes:
            # If no legend prototypes, assign all to "Default"
            for p in points:
                p["series"] = "Default"
            return points

        plot_img = cv2.imread(full_plot_img_path)
        if plot_img is None:
            return points

        for p in points:
            if p["label"] != "data_point": # Only match data points
                continue
                
            # Get point crop
            cx, cy = p["center"]
            box_sz = 10 # 10px radius
            x1 = max(0, int(cx - box_sz))
            y1 = max(0, int(cy - box_sz))
            x2 = min(plot_img.shape[1], int(cx + box_sz))
            y2 = min(plot_img.shape[0], int(cy + box_sz))
            
            point_crop = plot_img[y1:y2, x1:x2]
            point_color = self._get_dominant_color(point_crop)
            
            # Find best match in prototypes based on HSV distance
            best_series = None
            min_dist = float('inf')
            
            for name, proto in prototypes.items():
                proto_color = proto["color"] # HSV
                
                # HSV Distance: weight H more
                dh = min(abs(point_color[0] - proto_color[0]), 180 - abs(point_color[0] - proto_color[0]))
                ds = abs(point_color[1] - proto_color[1])
                dv = abs(point_color[2] - proto_color[2])
                
                # Simple weight
                dist = dh * 2 + ds * 1 + dv * 0.5
                
                if dist < min_dist:
                    min_dist = dist
                    best_series = name
            
            p["series"] = best_series
            
        return points
