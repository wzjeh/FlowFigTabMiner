import os
import cv2
import numpy as np
import torch
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
                _torch_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
                self.model.to(_torch_device)
                print(f"   MoleculeProcessor: YOLO model loaded successfully from {model_path} (device: {_torch_device.upper()})")
            except Exception as e:
                print(f"Error loading Molecule Model: {e}")
                self.model = None
        else:
            print(f"Warning: Molecule Model not found at {model_path}")
            self.model = None

    def process_image(self, image_path_or_array, content_recognizer, mask_only=False, output_path=None):
        """
        Detect molecules, convert to SMILES using content_recognizer, 
        and replace them in the image with text or just mask them.
        
        Args:
            image_path_or_array: Path to image or cv2 image array (BGR).
            content_recognizer: Instance of ContentRecognizer (MolNexTR for SMILES).
            mask_only (bool): If True, only mask the molecule with white box (no text).
            output_path (str): Optional path to save the molecule detection visualization.
            
        Returns:
            processed_image: cv2 image (BGR) with molecules masked/replaced.
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
        print(f"   MoleculeProcessor: Starting YOLO inference on image {w}x{h}...", flush=True)
        try:
             # Run inference with resizing to 1024 as requested (match training args)
             results = self.model(original_img, conf=self.conf_threshold, imgsz=1024, rect=False, verbose=False)[0]
             print(f"   MoleculeProcessor: YOLO inference done. Found {len(results.boxes)} boxes.", flush=True)
        except Exception as e:
             print(f"   MoleculeProcessor: YOLO INFERENCE FAILED: {e}", flush=True)
             return None, []
        
        metrics = []
        
        # Prepare for modification
        processed_img = original_img.copy()

        if len(results.boxes) > 0:
            print(f"   -> Detected {len(results.boxes)} potential molecules.")
            
            # Save visualization of what YOLO sees
            debug_viz = original_img.copy()
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.rectangle(debug_viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(debug_viz, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 1. Use explicit output_path if provided
            if output_path:
                 cv2.imwrite(output_path, debug_viz)
                 print(f"   -> Saved Molecule YOLO debug viz to {output_path}")
            # 2. Fallback to legacy behavior if image_path_or_array is a string
            elif isinstance(image_path_or_array, str):
                 debug_path = os.path.splitext(image_path_or_array)[0] + "_debug_yolo.png"
                 cv2.imwrite(debug_path, debug_viz)
                 print(f"   -> Saved Molecule YOLO debug viz to {debug_path}")
            
            # DEBUG: Save failures (prepare) — once for the whole table.
            fail_dir = os.path.join(os.path.dirname(image_path_or_array) if isinstance(image_path_or_array, str) else ".", "mol_debug_crops")
            os.makedirs(fail_dir, exist_ok=True)

            # ── Phase 1: crop every molecule box (geometry unchanged) ──
            box_crops = []  # (i, x1, y1, x2, y2, conf, mol_crop)
            for i, box in enumerate(results.boxes):
                # Get Box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])

                # Boundary checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Tight Crop with Safety Margin (10px) + Synthetic White Padding.
                # Margin prevents cut-off bonds/labels; MolNexTR is robust to
                # white space but sensitive to cut-offs.
                margin = 10
                tc_x1 = max(0, x1 - margin)
                tc_y1 = max(0, y1 - margin)
                tc_x2 = min(w, x2 + margin)
                tc_y2 = min(h, y2 + margin)
                tc_crop = original_img[tc_y1:tc_y2, tc_x1:tc_x2].copy()

                pad_val = 30
                try:
                    mol_crop = cv2.copyMakeBorder(tc_crop, pad_val, pad_val, pad_val, pad_val, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                except Exception as e:
                    print(f"Warning: Padding failed for box {i}: {e}")
                    mol_crop = tc_crop

                # Resolution upscaling: only if short side < 192px (MolNexTR
                # handles ~200-300px naturally; over-upscaling adds ringing).
                h_crop, w_crop = mol_crop.shape[:2]
                target_upscale_h = 300
                if h_crop < 192:
                    scale_factor = min(target_upscale_h / h_crop, 3.0)
                    if scale_factor > 1.0:
                        new_w = int(w_crop * scale_factor)
                        new_h = int(h_crop * scale_factor)
                        mol_crop = cv2.resize(mol_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        print(f"   -> [Debug] Upscaled TINY box {i} by {scale_factor:.1f}x ({w_crop}x{h_crop} -> {new_w}x{new_h})")
                else:
                    print(f"   -> [Debug] Box {i} sufficient size ({w_crop}x{h_crop}). Skipped Upscaling.")

                box_crops.append((i, x1, y1, x2, y2, conf, mol_crop))

            # ── Phase 2: structure recognition — micro-batch (issue #14) ──
            # Optional local optimization, NOT the headline fix.  SMILES are
            # byte-identical to per-box (deterministic greedy decode); ~2x faster
            # in isolation on MPS once MOLNEXTR_NUM_WORKERS=1 removes the per-box
            # Pool overhead.  Default bs=4 keeps the MPS unified-memory peak safe
            # on 16GB machines.  The real table-stage bottleneck is downstream
            # caption/note OCR, not this step (issue #14 follow-up), so don't
            # expect a full-pipeline speedup here.  MOLNEXTR_BATCH=0 → per-box.
            use_microbatch = os.environ.get("MOLNEXTR_BATCH", "1") != "0"
            batch_size = int(os.environ.get("MOLNEXTR_BATCH_SIZE", "4"))
            crops_only = [bc[6] for bc in box_crops]
            if use_microbatch:
                smiles_list = content_recognizer.recognize_structures_batch(crops_only, batch_size=batch_size)
            else:
                smiles_list = []
                for c in crops_only:
                    try:
                        smiles_list.append(content_recognizer.recognize_content(c, "Structure"))
                    except Exception as e:
                        print(f"Structure Rec Error: {e}")
                        smiles_list.append("")

            # ── Phase 3: per-box OCR fallback + mask + metrics (unchanged) ──
            for (i, x1, y1, x2, y2, conf, mol_crop), smiles in zip(box_crops, smiles_list):
                logging_smiles = smiles if smiles else "[NoSMILES]"

                if not smiles or smiles == "<invalid>":
                     print(f"   -> [Debug] MolNexTR failed or returned <invalid>. Attempting OCR Fallback...", flush=True)
                     # Fallback to OCR (per-box, intentionally kept serial)
                     ocr_text = content_recognizer.recognize_content(mol_crop, "Text")
                     if ocr_text and len(ocr_text.strip()) > 0:
                         smiles = ocr_text.strip()
                         print(f"   -> [Debug] OCR Fallback Successful: '{smiles}'")
                     else:
                         fail_fname = f"fail_box_{i}_{logging_smiles}.png"
                         fail_path = os.path.join(fail_dir, fail_fname)
                         cv2.imwrite(fail_path, mol_crop)
                         print(f"   -> [Debug] OCR also failed. Saved failed crop to {fail_path}")

                # Replace in image: fill white box
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), (255, 255, 255), -1)

                # Put text (centered) — only if not mask_only
                if not mask_only:
                    text_to_draw = smiles if smiles else "Structure"
                    print(f"   -> Box {i}: SMILES='{smiles}' | Drawing Text='{text_to_draw}'", flush=True)
                    font_scale = 0.5
                    thickness = 1
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), _ = cv2.getTextSize(text_to_draw, font, font_scale, thickness)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    tx = max(x1, cx - tw // 2)
                    ty = cy + th // 2
                    cv2.putText(processed_img, text_to_draw, (tx, ty), font, font_scale, (0, 0, 0), thickness)
                else:
                    print(f"   -> Box {i}: SMILES='{smiles}' | Masked (White Box)", flush=True)

                # Store metadata
                metrics.append({
                    'box': [x1, y1, x2, y2],
                    'conf': conf,
                    'smiles': smiles
                })

        return processed_img, metrics
