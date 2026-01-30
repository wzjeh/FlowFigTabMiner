from paddleocr import PaddleOCR
import os
try:
    from molscribe import MolScribe
except ImportError:
    MolScribe = None
    print("Warning: MolScribe not found. Chemical structure recognition will fail.")

class ContentRecognizer:
    def __init__(self):
        """
        Initialize OCR and MolScribe models.
        """
        # PaddleOCR
        # use_angle_cls=True loads the direction classifier
        # lang='en' for English tables
        print("Loading PaddleOCR...")
        # Check if gpu is available
        use_gpu = False # Set to False by default to avoid issues if paddle-gpu not installed
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                use_gpu = True
        except:
            pass

        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
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
                molscribe_ckpt = "models/molscribe.ckpt"
                if os.path.exists(molscribe_ckpt):
                    self.molscribe = MolScribe(model_path=molscribe_ckpt, device='cuda' if use_gpu else 'cpu')
                else:
                    # Try default load but might fail if network restricted or cache issue
                    # The error suggests torch.load(f) where f is None.
                    # Workaround: Use HuggingFace Hub directly if needed or skip if not found.
                    print("Debug: No local MolScribe checkpoint found at models/molscribe.ckpt. Attempting default init.")
                    try:
                        self.molscribe = MolScribe(model_path=None, device='cuda' if use_gpu else 'cpu')
                    except AttributeError as ae:
                        if "'NoneType' object has no attribute 'seek'" in str(ae):
                            print("Warning: MolScribe failed to download/load default weights. Please clear cache or manually download 'molscribe.ckpt' to models/.")
                        else:
                            raise ae
            except Exception as e:
                print(f"Error initializing MolScribe: {e}")

    def recognize_content(self, image_path, content_type):
        """
        Recognize content from a cell image based on type.
        Args:
            image_path (str): Path to cell image.
            content_type (str): "Text" or "Structure".
        Returns:
            str: Recognized text or SMILES.
        """
        if content_type == "Structure":
            return self._recognize_structure(image_path)
        else:
            return self._recognize_text(image_path)

    def _recognize_text(self, image_path):
        try:
            # Use parameter-less call (default full pipeline)
            # This is robust across PaddleOCR / PaddleX versions
            result = self.ocr.ocr(image_path)
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
            print(f"OCR Error on {image_path}: {e}")
            return ""

    def _recognize_structure(self, image_path):
        if self.molscribe is None:
            return "[MolScribe Missing]"
        
        try:
            # MolScribe inference
            # predict_images([path]) -> [{'smiles': '...', 'molfile': '...'}]
            result = self.molscribe.predict_images([image_path])
            if result and len(result) > 0:
                return result[0].get('smiles', "")
            return ""
        except Exception as e:
            print(f"MolScribe Error on {image_path}: {e}")
            return "[Error]"
