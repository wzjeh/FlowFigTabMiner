import os
import cv2
import numpy as np
from ultralytics import YOLO
import PIL.Image

class MoleculeProcessor:
    def __init__(self, model_path=None, conf_threshold=0.25):
        """
        Initialize the Molecule Processor.
        Args:
            model_path (str): Path to YOLO11s molecule detection model.
            conf_threshold (float): Confidence threshold for detection.
        """
        self.conf_threshold = conf_threshold
        
        # Load YOLO model
        if model_path and os.path.exists(model_path):
            print(f"Loading Molecule Detection Model from {model_path}...")
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"Error loading Molecule Model: {e}")
                self.model = None
        else:
            print(f"Warning: Molecule Model not found at {model_path}")
            self.model = None

    def process_image(self, image_path_or_array, content_recognizer):
        """
        Detect molecules, convert to SMILES using content_recognizer, 
        and replace them in the image with text.
        
        Args:
            image_path_or_array: Path to image or cv2 image array (BGR).
            content_recognizer: Instance of ContentRecognizer (must have molscribe loaded).
            
        Returns:
            processed_image: cv2 image (BGR) with molecules replaced by SMILES text.
            molecule_data: List of dicts [{'box': [x1,y1,x2,y2], 'smiles': str, 'conf': float}]
        """
        if self.model is None:
            # Pass through if no model
            if isinstance(image_path_or_array, str):
                return cv2.imread(image_path_or_array), []
            return image_path_or_array, []

        # Load Image
        if isinstance(image_path_or_array, str):
            original_img = cv2.imread(image_path_or_array)
        else:
            original_img = image_path_or_array.copy()

        if original_img is None:
            return None, []

        h, w = original_img.shape[:2]
        
        # 1. Detect Molecules
        results = self.model(original_img, conf=self.conf_threshold, verbose=False)[0]
        
        metrics = []
        
        # Prepare for modification
        processed_img = original_img.copy()

        if len(results.boxes) > 0:
            print(f"   -> Detected {len(results.boxes)} potential molecules.")
            
            for box in results.boxes:
                # Get Box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                
                # Boundary checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Crop Molecule
                mol_crop = original_img[y1:y2, x1:x2]
                
                # 2. Convert to SMILES
                # MolScribe expects PIL image or path? 
                # ContentRecognizer._recognize_structure calls molscribe.predict_images([image_path])
                # We need to temporarily save crop or modify recognizer to accept array?
                # MolScribe's predict_images usually takes paths. predict_images_from_arrays takes arrays?
                # Let's check ContentRecognizer. It seems it only has _recognize_structure(image_path).
                # We'll save a temp crop to be safe and consistent with existing interface.
                
                import tempfile
                fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                cv2.imwrite(temp_path, mol_crop)
                
                smiles = ""
                try:
                    # We use 'Structure' type to trigger _recognize_structure
                    # And we need to ensure content_recognizer has molscribe initialized
                    smiles = content_recognizer.recognize_content(temp_path, "Structure")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                logging_smiles = smiles if smiles else "[NoSMILES]"
                
                # 3. Replace in Image
                # A. Fill White
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
                
                # B. Put Text (Centered)
                # If SMILES is very long, it might overflow.
                # Heuristic: Trim or wrap? For structure recognition, just having "some text" might be enough 
                # to be treated as a cell content. 
                # But user said "replace molecules with SMILES strings".
                # Let's try to fit it.
                
                text_to_draw = smiles if smiles else "Structure"
                
                font_scale = 0.5
                thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Calculate size
                (tw, th), _ = cv2.getTextSize(text_to_draw, font, font_scale, thickness)
                
                # Center
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                tx = max(x1, cx - tw // 2)
                ty = cy + th // 2
                
                cv2.putText(processed_img, text_to_draw, (tx, ty), font, font_scale, (0, 0, 0), thickness)
                
                # Store metadata
                metrics.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'smiles': smiles
                })
                
        return processed_img, metrics
