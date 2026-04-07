import os
# Must be set BEFORE any paddle/paddleocr import
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLEPD_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"
# Disable OneDNN (MKL-DNN) — causes NotImplementedError on Cloud Run CPUs
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_pir_apply_mkldnn_pass"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
import paddle
try:
    paddle.set_flags({'FLAGS_use_mkldnn': False, 'FLAGS_pir_apply_mkldnn_pass': False})
except Exception:
    pass
from src.extraction.common.ocr_backend import get_ocr_instance, get_rec_instance

class ContentRecognizer:
    def __init__(self):
        """
        Initialize OCR for table cells: full-pipeline (det+rec) with rec-only fallback.
        Full pipeline handles multi-line headers and complex cells well.
        Rec-only fallback catches small cells where the detector returns empty.
        """
        print("Loading OCR for table cells (full-pipeline + rec-only fallback)...")
        self.ocr = get_ocr_instance(lang='en', enable_mkldnn=False)
        self.rec = get_rec_instance()

        # MolNexTR for chemical structure recognition
        self.molnextr = None
        try:
            from src.extraction.common.molnextr.molnextr import MolNexTRSingleton
            print("Loading MolNexTR...", flush=True)
            self.molnextr = MolNexTRSingleton.get_instance()
            print("MolNexTR Loaded Successfully.", flush=True)
        except Exception as e:
            print(f"Error initializing MolNexTR: {e}", flush=True)
            print("Warning: MolNexTR failed to load. Chemical structure recognition will fail.", flush=True)

    def recognize_content(self, image_input, content_type):
        """
        Recognize content from a cell image based on type.
        Args:
            image_input (str or np.ndarray): Path to cell image or image array.
            content_type (str): "Text" or "Structure".
        Returns:
            str: Recognized text or SMILES.
        """
        if content_type == "Structure":
            return self._recognize_structure(image_input)
        else:
            return self._recognize_text(image_input)

    def _recognize_text(self, image_input):
        text = ""
        # Try full det+rec pipeline first — better for headers and complex cells
        try:
            result = self.ocr.ocr(image_input)
            if result and result[0]:
                lines = result[0]
                if isinstance(lines, dict):
                    texts = lines.get('rec_texts', [])
                    text = ' '.join(texts)
                else:
                    parts = []
                    for line in lines:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            t = line[1]
                            parts.append(str(t[0]) if isinstance(t, (list, tuple)) else str(t))
                    text = ' '.join(parts)
        except Exception as e:
            pass  # fall through to rec-only

        # Fallback: if full pipeline returned empty or failed, use rec-only
        # (small cells where detector can't find text regions)
        if not text.strip():
            try:
                text, _conf = self.rec.recognize(image_input)
            except Exception as e:
                print(f"OCR Error on {image_input if isinstance(image_input, str) else 'Image Array'}: {e}")

        return text

    def _recognize_structure(self, image_input):
        if self.molnextr is None:
            return "[MolNexTR Missing]"
        
        try:
            # Prepare image for MolNexTR
            # If path, load it. If array, use it.
            import cv2
            import numpy as np
            
            img = image_input
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                     return "[Error: Image Read Failed]"
            
            # Ensure it is RGB. MolNexTR expects RGB (same as MolScribe).
            # OpenCV (cv2.imread) returns BGR.
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # MolNexTR inference
            # predict_images([img]) -> [{'predicted_smiles': '...', ...}]
            # Note: MolNexTR.predict_images takes a list of images or paths.
            # But the underlying model.predict_images expects transformed tensors if passed directly?
            # Wait, MolNexTRSingleton instance is the `molnextr` class from `src/extraction/molnextr/model.py`.
            # Its `predict_images` method (line 97) takes `input_images` list.
            # And it applies `self.transform` inside loop (line 104).
            # `self.transform` from albumentations expects image=...
            # The `predict_images` implementation:
            # images = [self.transform(image=image, keypoints=[])['image'] for image in batch_images]
            # So passing standard RGB numpy arrays is correct.
            
            result = self.molnextr.predict_images([img])
            # print(f"   ContentRecognizer: MolNexTR Raw Result: {result}", flush=True) # DEBUG
            
            if result and len(result) > 0:
                smiles = result[0].get('predicted_smiles', "")
                # Clean up if needed, though MolNexTR usually returns valid SMILES or None
                return smiles
            return ""
        except Exception as e:
            print(f"MolNexTR Error: {e}")
            return "[Error]"
