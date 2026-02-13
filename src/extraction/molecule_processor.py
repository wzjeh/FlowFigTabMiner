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
            print(f"Loading Molecule Detection Model from {model_path}...")
            try:
                import torch
                if torch.get_num_threads() > 1:
                    torch.set_num_threads(1)
                self.model = YOLO(model_path)
                print(f"   MoleculeProcessor: YOLO model loaded successfully from {model_path}")
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
        
        h, w = original_img.shape[:2]
        
        # 1. Detect Molecules
        print(f"   MoleculeProcessor: Starting YOLO inference on image {w}x{h}...", flush=True)
        try:
             results = self.model(original_img, conf=self.conf_threshold, verbose=False)[0]
             print(f"   MoleculeProcessor: YOLO inference done. Found {len(results.boxes)} boxes.", flush=True)
        except Exception as e:
             print(f"   MoleculeProcessor: YOLO INFERENCE FAILED: {e}", flush=True)
             return None, []
        
        metrics = []
        
        # Prepare for modification
        processed_img = original_img.copy()

        if len(results.boxes) > 0:
            print(f"   -> Detected {len(results.boxes)} potential molecules.")
            
            # DEBUG: Save visualization of what YOLO sees
            debug_viz = original_img.copy()
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(debug_viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug_viz, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Save to same directory as input (if path known) or temp
            # Since we don't know output path here easily, let's look at image_path_or_array
            if isinstance(image_path_or_array, str):
                 debug_path = os.path.splitext(image_path_or_array)[0] + "_debug_yolo.png"
                 cv2.imwrite(debug_path, debug_viz)
                 print(f"   -> Saved YOLO debug viz to {debug_path}")
            
            for i, box in enumerate(results.boxes):
                # Get Box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                
                # Boundary checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Crop Molecule with Padding logic
                # PREVIOUS: Expanded box in original image (Risk: Includes neighbor tokens)
                # NEW: Tight crop + Synthetic White Padding (Clean isolation)
                
                # 1. Tight Crop
                tc_crop = original_img[y1:y2, x1:x2].copy()
                
                # 2. White Padding
                pad_val = 30
                try:
                    mol_crop = cv2.copyMakeBorder(tc_crop, pad_val, pad_val, pad_val, pad_val, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                except Exception as e:
                    print(f"Warning: Padding failed for box {i}: {e}")
                    mol_crop = tc_crop
                
                # DEBUG: Save failures
                # We need to do this AFTER prediction, but let's prepare the path
                fail_dir = os.path.join(os.path.dirname(image_path_or_array) if isinstance(image_path_or_array, str) else ".", "mol_debug_crops")
                os.makedirs(fail_dir, exist_ok=True)

                # 3. Resolution Upscaling
                # MolScribe works best on larger images. If crop is small, upscale.
                # Threshold: height < 300px
                h_crop, w_crop = mol_crop.shape[:2]
                if h_crop < 300:
                    scale_factor = 300 / h_crop
                    # Limit scale factor to avoid excessive blur (max 3x)
                    scale_factor = min(scale_factor, 3.0)
                    if scale_factor > 1.0:
                        new_w = int(w_crop * scale_factor)
                        new_h = int(h_crop * scale_factor)
                        mol_crop = cv2.resize(mol_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        print(f"   -> [Debug] Upscaled box {i} by {scale_factor:.1f}x ({w_crop}x{h_crop} -> {new_w}x{new_h})")
                
                # 2. Convert to SMILES
                # MolScribe expects PIL image or array. ContentRecognizer now supports array.
                smiles = ""
                try:
                    # We use 'Structure' type to trigger _recognize_structure
                    smiles = content_recognizer.recognize_content(mol_crop, "Structure")
                except Exception as e:
                    print(f"Structure Rec Error: {e}")
                
                logging_smiles = smiles if smiles else "[NoSMILES]"
                
                if not smiles or smiles == "<invalid>":
                     # Save the crop for inspection
                     fail_fname = f"fail_box_{i}_{logging_smiles}.png"
                     fail_path = os.path.join(fail_dir, fail_fname)
                     cv2.imwrite(fail_path, mol_crop)
                     print(f"   -> [Debug] Saved failed crop to {fail_path}")
                
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
                print(f"   -> Box {i}: SMILES='{smiles}' | Drawing Text='{text_to_draw}'", flush=True) # DEBUG
                
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
