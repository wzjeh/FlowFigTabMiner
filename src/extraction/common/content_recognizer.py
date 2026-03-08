import os
# Disable connectivity checks to prevent hangs
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["PADDLEPD_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # Disable HuggingFace connectivity checks
from paddleocr import PaddleOCR

class ContentRecognizer:
    def __init__(self):
        """
        Initialize PaddleOCR and MolNexTR models.
        Note: MolScribe has been fully replaced by MolNexTR.
        """
        # PaddleOCR
        # use_angle_cls=True loads the direction classifier
        # lang='en' for English tables
        print("Loading PaddleOCR...")
        # Force single thread for torch interaction (MolScribe uses torch)
        try:
            import torch
            if torch.get_num_threads() > 1:
                torch.set_num_threads(1)
        except:
            pass
            
        # Check if gpu is available
        use_gpu = False # Set to False by default to avoid issues if paddle-gpu not installed
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                use_gpu = True
        except:
            pass

        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en'
        )

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
        try:
            # Use parameter-less call (default full pipeline)
            # This is robust across PaddleOCR / PaddleX versions
            # PaddleOCR supports path or ndarray
            result = self.ocr.ocr(image_input)
            # print(f"DEBUG: OCR Result Type: {type(result)}", flush=True)

            text = ""
            # Handle PaddleX Dict vs List
            if result:
                 # Case 1: List format (Standard PaddleOCR)
                 if isinstance(result, list) and len(result) > 0:
                      first_item = result[0]
                      
                      # Case 1.1: New PaddleX format returns dict
                      if isinstance(first_item, dict): 
                          if 'rec_texts' in first_item and first_item['rec_texts']:
                              text = first_item['rec_texts'][0]

                      # Case 1.2: Standard list of lines
                      elif isinstance(first_item, list):
                           # Concatenate all detected lines
                           texts = []
                           for line in first_item:
                               if isinstance(line, list) and len(line) >= 2:
                                    # line: [box, (text, conf)]
                                    txt_obj = line[1]
                                    if isinstance(txt_obj, (list, tuple)) and len(txt_obj) > 0:
                                        texts.append(txt_obj[0])
                           text = " ".join(texts)
            
            return text

        except Exception as e:
            print(f"OCR Error on {image_input if isinstance(image_input, str) else 'Image Array'}: {e}")
            return ""

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
