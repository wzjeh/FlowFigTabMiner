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
        # issue #17 Phase 1: split timing into inference (the det+rec model call)
        # vs postprocess (text assembly + rec-only fallback).  No-op unless
        # OCR_PROFILE=1; never alters control flow or output.
        from src.extraction.common.ocr_profile import profiler

        text = ""
        # Try full det+rec pipeline first — better for headers and complex cells
        try:
            with profiler.ocr_phase("ocr_inference"):
                result = self.ocr.ocr(image_input)
            with profiler.ocr_phase("ocr_postprocess"):
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
                with profiler.ocr_phase("ocr_postprocess"):
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

    def _to_rgb(self, image_input):
        """Load (if path) and convert BGR->RGB for MolNexTR; None on read fail."""
        import cv2
        img = image_input
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                return None
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _mps_empty_cache(self):
        """Return the MPS allocator cache to the OS (no-op off MPS).

        Apple Silicon MPS uses UNIFIED memory and never frees its allocator
        cache on its own — leaving it to accumulate across batches exhausted
        16GB RAM and triggered a kernel panic (WindowServer watchdog reboot,
        issue #14).  Call this after every batch.
        """
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def recognize_structures_batch(self, image_inputs, batch_size=4):
        """Batch MolNexTR structure recognition (issue #14, optional optimization).

        Runs all crops through ``predict_images`` in one native batch
        (torch.stack + single forward per ``batch_size`` chunk).  SMILES are
        byte-identical to per-box (deterministic greedy decode, no sampling).
        Measured speedup is ~2x in isolation on MPS once MOLNEXTR_NUM_WORKERS=1
        removes the per-box multiprocessing.Pool overhead — NOT the headline
        fix.  At the full-pipeline level the gain is masked by the downstream
        caption/note OCR + evidence step, which dominates table-stage wall time
        (see issue #14 follow-up).  This is a low-risk local optimization, not a
        promised pipeline speedup; MOLNEXTR_BATCH=0 disables it.

        Returns a list of SMILES strings aligned 1:1 with ``image_inputs``.
        A box that the model leaves empty/invalid stays "" so the caller can do
        its per-box OCR fallback unchanged.  If the whole batch raises (e.g. a
        single malformed crop), we retry per-box so one bad crop can't blank the
        whole table.
        """
        if self.molnextr is None:
            return ["[MolNexTR Missing]"] * len(image_inputs)

        rgb = [self._to_rgb(im) for im in image_inputs]
        valid_idx = [i for i, im in enumerate(rgb) if im is not None]
        valid = [rgb[i] for i in valid_idx]
        out = [""] * len(image_inputs)
        if not valid:
            return out

        # Process in chunks of batch_size, freeing the MPS cache after EACH
        # chunk so peak memory stays bounded regardless of table size (a
        # structure-dense 84-box table otherwise accumulates until 16GB RAM is
        # exhausted — issue #14 kernel panic).  Chunking here (not relying on
        # predict_images' internal loop) is what lets us empty_cache per chunk.
        smis = []
        for k in range(0, len(valid), batch_size):
            chunk = valid[k:k + batch_size]
            try:
                res = self.molnextr.predict_images(chunk, batch_size=batch_size)
                smis += [r.get("predicted_smiles", "") for r in res]
            except Exception as e:
                print(f"MolNexTR batch chunk failed ({e}); retrying per-box", flush=True)
                for im in chunk:
                    try:
                        smis.append(self.molnextr.predict_images([im])[0].get("predicted_smiles", ""))
                    except Exception as e2:
                        print(f"MolNexTR per-box retry failed: {e2}", flush=True)
                        smis.append("")
            self._mps_empty_cache()

        for j, i in enumerate(valid_idx):
            out[i] = smis[j]
        return out
