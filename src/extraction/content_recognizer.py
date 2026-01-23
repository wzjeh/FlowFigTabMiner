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

        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=use_gpu)
        
        # MolScribe
        self.molscribe = None
        if MolScribe:
            # MolScribe loads weights automatically or from a path
            # Assume default checkpoint or download
            try:
                print("Loading MolScribe...")
                # MolScribe(model_path=None, device=None) - None downloads/uses default
                self.molscribe = MolScribe(model_path=None, device='cuda' if use_gpu else 'cpu')
            except Exception as e:
                print(f"Error loading MolScribe: {e}")

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
            # PaddleOCR expects image path or numpy array
            result = self.ocr.ocr(image_path, cls=True)
            # Result structure: [[[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]],("text", confidence)]...]]
            if not result or result[0] is None:
                return ""
            
            # Concatenate all detected text lines
            texts = [line[1][0] for line in result[0]]
            return " ".join(texts)
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
