"""
OCR backend factory.
USE_EASYOCR=1 → EasyOCR (Cloud Run, avoids PaddlePaddle PIR crash)
otherwise      → PaddleOCR (local, default)
"""
import os


def get_ocr_instance(use_angle_cls=True, lang='en', enable_mkldnn=False):
    if os.environ.get('USE_EASYOCR', '0') == '1':
        return _EasyOCRWrapper()
    from paddleocr import PaddleOCR
    return PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, enable_mkldnn=enable_mkldnn)


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
