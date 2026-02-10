from paddleocr import PaddleOCR
import os
try:
    from molscribe import MolScribe
except ImportError:
    MolScribe = None
    print("Warning: MolScribe not found. Chemical structure recognition will fail.")

class ContentRecognizer:
    def __init__(self, molscribe_path=None):
        """
        Initialize OCR and MolScribe models.
        """
        # PaddleOCR
        # use_angle_cls=True loads the direction classifier
        # lang='en' for English tables
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
            # use_mp=False, # Removed: Causes 'Unknown argument' error in some versions
            # show_log=False, # Removed: Causes 'Unknown argument' error
            # enable_mkldnn=not use_gpu # Removed: Potentially unsafe if show_log failed
        )
        
        # MolScribe
        self.molscribe = None
        if MolScribe:
            # MolScribe loads weights automatically or from a path
            # Assume default checkpoint or download
            try:
                print("Loading MolScribe...")
                # MolScribe expects a valid checkpoint path often, or downloads it.
                # However, the previous error 'NoneType object has no attribute seek' usually implies
                # it tried to load 'None' as a file or similar issue in internal loading.
                # Let's specify the weight path explicitly if available, or force download by handling the init carefully.
                
                # Check if we have a local model
                ckpt_path = molscribe_path or "models/swin_base_char_aux_1m680k.pth"
                
                if os.path.exists(ckpt_path):
                    self.molscribe = MolScribe(model_path=ckpt_path, device='cuda' if use_gpu else 'cpu')
                else:
                    # Try default load but might fail if network restricted or cache issue
                    # The error suggests torch.load(f) where f is None.
                    # Workaround: Use HuggingFace Hub directly if needed or skip if not found.
                    print(f"Debug: No local MolScribe checkpoint found at {ckpt_path}. Attempting default init.")
                    try:
                        self.molscribe = MolScribe(model_path=None, device='cuda' if use_gpu else 'cpu')
                    except AttributeError as ae:
                        if "'NoneType' object has no attribute 'seek'" in str(ae):
                            print("Warning: MolScribe failed to download/load default weights. Please clear cache or manually download 'molscribe.ckpt' to models/.")
                        else:
                            raise ae
            except Exception as e:
                print(f"Error initializing MolScribe: {e}")

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
        if self.molscribe is None:
            return "[MolScribe Missing]"
        
        try:
            # Prepare image for MolScribe
            # If path, load it. If array, use it.
            import cv2
            import numpy as np
            
            img = image_input
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                     return "[Error: Image Read Failed]"
            
            # Ensure it is RGB. MolScribe expects RGB.
            # OpenCV (cv2.imread) returns BGR.
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # MolScribe inference
            # predict_images([img]) -> [{'smiles': '...', 'molfile': '...'}]
            result = self.molscribe.predict_images([img])
            print(f"   ContentRecognizer: MolScribe Raw Result: {result}", flush=True) # DEBUG
            
            if result and len(result) > 0:
                smiles = result[0].get('smiles', "")
                # Some versions might return <invalid> or similar
                if smiles == "<invalid>":
                    print("   ContentRecognizer: MolScribe returned <invalid>.", flush=True)
                return smiles
            return ""
        except Exception as e:
            print(f"MolScribe Error: {e}")
            return "[Error]"
