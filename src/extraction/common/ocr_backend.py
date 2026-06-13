"""
OCR backend factory.

Two modes:
  get_ocr_instance()  → full det+rec pipeline (for multi-line text: captions, legends)
  get_rec_instance()  → rec-only model (for pre-cropped single-text regions: tick labels, cells)

USE_EASYOCR=1 → EasyOCR fallback (Cloud Run, avoids PaddlePaddle PIR crash)
"""
import os

import cv2


# Cap on the LONGEST side (px) of a caption/note crop AFTER the OCR upscale.
# issue #17 Phase 1.5: the unconditional 3x upscale blew a multi-line note up to
# ~4000px wide, hitting PaddleOCR's max_side_limit=4000, where the CPU DB text
# detector is superlinear in area (det=287s, plus an OOM on the largest crop).
# Capping the longest side keeps det input small. Env-overridable.
# Default 1600 (issue #17 Phase 2 verify on Nagaki page_3 note): vs the uncapped
# 3x it cut det 287s->2.4s / total 394s->5.1s and peak mem 15.1GB->9.7GB, with
# text strictly better than the (truncated) baseline. 1600 over 2000 for the
# wider memory margin on 16GB machines; text only feeds LLM context + keywords.
OCR_UPSCALE_MAX_SIDE = int(os.environ.get("OCR_UPSCALE_MAX_SIDE", "1600"))


def upscale_for_ocr(img, pad=50, max_factor=3.0, max_side=None):
    """Upscale a small-text crop for OCR, capping the post-resize longest side.

    Keeps the existing behaviour for already-small crops (up to ``max_factor``x,
    INTER_CUBIC, then a white border) but never blows the longest side past
    ``max_side`` — so a multi-line note is not inflated to the ~4000px regime
    where the PaddleOCR DB detector explodes on CPU (issue #17).

    Args:
        img: BGR crop (numpy array).
        pad: white border added on every side after resizing.
        max_factor: maximum upscale factor (the prior hard-coded 3x).
        max_side: cap on the longest side before padding; defaults to
            ``OCR_UPSCALE_MAX_SIDE`` (env ``OCR_UPSCALE_MAX_SIDE``).
    Returns:
        Padded BGR image (caller converts to RGB), longest side
        ``<= max_side + 2*pad``.
    """
    if max_side is None:
        max_side = OCR_UPSCALE_MAX_SIDE
    h, w = img.shape[:2]
    # Never downscale below 1x, never upscale beyond max_factor, never exceed max_side.
    scale = max(1.0, min(max_factor, max_side / max(h, w)))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


# Singleton full det+rec instances, keyed by (lang, enable_mkldnn).
# Without this, every caller (EvidenceAssembler, LegendMatcher,
# ContentRecognizer) builds its OWN PaddleOCR — 3 per run — each
# spawning its own thread pool (the load=30 / repeated "No ccache found"
# culprit).  Memoizing collapses them to one shared instance.  Safe
# because the pipeline runs single-process (single-instance lock) and
# OCR calls are sequential, not concurrent.
_ocr_instances: dict = {}


def get_ocr_instance(lang='en', enable_mkldnn=False, **kwargs):
    """Full det+rec pipeline. Use for multi-line text (captions, legends, table cells).

    Memoized per (lang, enable_mkldnn) — repeat calls return the same
    instance instead of constructing a fresh PaddleOCR each time.
    """
    key = (lang, enable_mkldnn)
    cached = _ocr_instances.get(key)
    if cached is not None:
        return cached

    if os.environ.get('USE_EASYOCR', '0') == '1':
        instance = _EasyOCRWrapper()
    else:
        from paddleocr import PaddleOCR
        instance = PaddleOCR(lang=lang, enable_mkldnn=enable_mkldnn)

    _ocr_instances[key] = instance
    return instance


# Singleton rec-only instance (lazy)
_rec_instance = None


def get_rec_instance(model_name=None):
    """Rec-only model for pre-cropped single-text regions.

    Skips detection — treats entire crop as one text line.
    Default: PP-OCRv4_server_rec (higher accuracy).
    Fallback: en_PP-OCRv5_mobile_rec (lighter, faster).
    """
    global _rec_instance
    if _rec_instance is not None:
        return _rec_instance
    if os.environ.get('USE_EASYOCR', '0') == '1':
        _rec_instance = _EasyOCRRecWrapper()
        return _rec_instance
    _rec_instance = _PaddleRecOnly(model_name=model_name)
    return _rec_instance


class _PaddleRecOnly:
    """Wraps PaddleX rec-only model for single-text-line recognition."""

    def __init__(self, model_name=None):
        from paddlex import create_model
        if model_name is None:
            # Try server (more accurate), fall back to mobile
            for name in ['PP-OCRv4_server_rec', 'en_PP-OCRv5_mobile_rec']:
                try:
                    self._model = create_model(name)
                    self._model_name = name
                    print(f"[RecOnlyOCR] Loaded: {name}")
                    return
                except Exception:
                    continue
            raise RuntimeError("No PaddleX rec model available")
        else:
            self._model = create_model(model_name)
            self._model_name = model_name
            print(f"[RecOnlyOCR] Loaded: {model_name}")

    def recognize(self, crop_image):
        """Recognize text from a single crop image.

        Args:
            crop_image: numpy array (BGR) or file path
        Returns:
            (text: str, confidence: float)
        """
        try:
            result = list(self._model.predict(crop_image))[0]
            return result.get('rec_text', ''), result.get('rec_score', 0.0)
        except Exception:
            return '', 0.0

    def ocr(self, crop_image, **kwargs):
        """Compatibility wrapper matching PaddleOCR .ocr() return format.

        Returns PaddleX dict format: [{'rec_texts': [...], 'rec_scores': [...]}]
        """
        text, score = self.recognize(crop_image)
        return [{'rec_texts': [text], 'rec_scores': [score]}]


class _EasyOCRRecWrapper:
    """EasyOCR fallback for rec-only mode."""

    def __init__(self):
        import easyocr
        self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def recognize(self, crop_image):
        try:
            results = self._reader.readtext(crop_image)
            if not results:
                return '', 0.0
            # Concatenate all detected text, use min confidence
            text = ' '.join(t for _, t, c in results if c > 0.3)
            conf = min(c for _, _, c in results) if results else 0.0
            return text, conf
        except Exception:
            return '', 0.0

    def ocr(self, crop_image, **kwargs):
        text, score = self.recognize(crop_image)
        return [{'rec_texts': [text], 'rec_scores': [score]}]


class _EasyOCRWrapper:
    """Wraps EasyOCR to expose the same .ocr() interface as PaddleOCR (old list format)."""

    def __init__(self):
        import easyocr
        self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def ocr(self, img_or_path, **kwargs):
        """
        Returns old PaddleOCR list format:
          [ [ [bbox_4pts, (text, conf)], ... ] ]
        bbox_4pts = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        """
        try:
            results = self._reader.readtext(img_or_path)
        except Exception:
            return [[]]
        if not results:
            return [[]]
        converted = [
            [bbox, (text, float(conf))]
            for bbox, text, conf in results
        ]
        return [converted]

    def ocr_image(self, path):
        """Simple path-based OCR returning plain text string."""
        try:
            results = self._reader.readtext(path)
            return " ".join(t for _, t, c in results if c > 0.3)
        except Exception:
            return ""
