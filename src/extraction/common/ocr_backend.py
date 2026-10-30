"""
OCR backend factory.

Two modes:
  get_ocr_instance()  → full det+rec pipeline (for multi-line text: captions, legends)
  get_rec_instance()  → rec-only model (for pre-cropped single-text regions: tick labels, cells)

Engine (FFTM_OCR_ENGINE):
  rapid (default) → RapidOCR: PP-OCR models on ONNX Runtime, weights in models/rapidocr/.
                    On flowchem-bench it gives the same scores as PaddleOCR (2026-10-30).
                    The rec-only model runs on CoreML when ONNX Runtime offers it (macOS).
  apple           → Apple Vision text recognition (macOS only).
USE_EASYOCR=1 → EasyOCR (old Cloud Run figure-service image).
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


RAPIDOCR_DIR = os.environ.get("FFTM_RAPIDOCR_DIR", os.path.join("models", "rapidocr"))

# One shared full det+rec instance and one rec-only instance per process.
# Safe because the pipeline runs single-process (single-instance lock) and
# OCR calls are sequential, not concurrent.
_ocr_instance = None
_rec_instance = None


def ocr_engine():
    """Engine name from FFTM_OCR_ENGINE (rapid | apple)."""
    engine = os.environ.get('FFTM_OCR_ENGINE', 'rapid').strip().lower()
    if engine not in ('rapid', 'apple'):
        raise ValueError(f"FFTM_OCR_ENGINE must be rapid or apple, not {engine!r}")
    return engine


def get_ocr_instance():
    """Full det+rec pipeline. Use for multi-line text (captions, legends, table cells)."""
    global _ocr_instance
    if _ocr_instance is None:
        if os.environ.get('USE_EASYOCR', '0') == '1':
            _ocr_instance = _EasyOCRWrapper()
        elif ocr_engine() == 'apple':
            _ocr_instance = _AppleVision()
        else:
            _ocr_instance = _RapidFull()
    return _ocr_instance


def get_rec_instance():
    """Rec-only model for pre-cropped single-text regions (tick labels, cells).

    Skips detection — treats the entire crop as one text line.
    """
    global _rec_instance
    if _rec_instance is None:
        if os.environ.get('USE_EASYOCR', '0') == '1':
            _rec_instance = _EasyOCRRecWrapper()
        elif ocr_engine() == 'apple':
            _rec_instance = _AppleVision()
        else:
            _rec_instance = _RapidRecOnly()
    return _rec_instance


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


def _to_bgr(img_or_path):
    if isinstance(img_or_path, str):
        return cv2.imread(img_or_path)
    return img_or_path


def _xyxy(poly):
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _model(name):
    path = os.path.join(RAPIDOCR_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: hf download wyzhaoc/FlowFigTabMiner-models --local-dir models/")
    return path


def _rapid(params):
    from rapidocr import RapidOCR
    base = {"Global.log_level": "error", "Global.text_score": 0.0}
    base.update(params)
    return RapidOCR(params=base)


def _use_coreml(ort_wrapper, model_path):
    """Re-open one ONNX session with CoreML first (macOS); same model, so the
    output is unchanged (913/913 identical on the benchmark tick crops).
    Only used for the rec-only model: CoreML garbles the PP-OCRv5 detector."""
    import onnxruntime as ort
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        return False
    so = ort.SessionOptions()
    so.log_severity_level = 4
    ort_wrapper.session = ort.InferenceSession(
        model_path, sess_options=so, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    return True


class _RapidFull:
    """RapidOCR det+rec with the models PaddleOCR(lang='en') used:
    PP-OCRv5 server det + en PP-OCRv5 mobile rec, Paddle's det thresholds.
    Paddle's doc preprocessor (orientation + UVDoc unwarping) is not used."""

    def __init__(self):
        from rapidocr import LangDet, LangRec, ModelType, OCRVersion
        self._engine = _rapid({
            "Global.max_side_len": 4000,
            "Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.SERVER, "Det.lang_type": LangDet.CH,
            "Det.model_path": _model("ch_PP-OCRv5_server_det.onnx"),
            "Det.limit_side_len": 64, "Det.limit_type": "min",
            "Det.thresh": 0.3, "Det.box_thresh": 0.6, "Det.unclip_ratio": 1.5,
            "Cls.model_path": _model("ch_ppocr_mobile_v2.0_cls_infer.onnx"),
            "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE, "Rec.lang_type": LangRec.EN,
            "Rec.model_path": _model("en_PP-OCRv5_rec_mobile_infer.onnx"),
        })
        print("[OCR] RapidOCR full: PP-OCRv5 server det + en PP-OCRv5 mobile rec")

    def ocr(self, img_or_path, **kwargs):
        """PaddleOCR 3.x dict format: [{'rec_texts', 'rec_scores', 'rec_boxes'}]."""
        out = self._engine(_to_bgr(img_or_path))
        if out.txts is None:
            return [{'rec_texts': [], 'rec_scores': [], 'rec_boxes': []}]
        return [{'rec_texts': list(out.txts), 'rec_scores': [float(x) for x in out.scores],
                 'rec_boxes': [_xyxy(b) for b in out.boxes]}]


class _RapidRecOnly:
    """RapidOCR rec-only with the model the Paddle path used (PP-OCRv4 server rec)."""

    def __init__(self):
        from rapidocr import LangRec, ModelType, OCRVersion
        path = _model("ch_PP-OCRv4_rec_server_infer.onnx")
        self._engine = _rapid({
            "Global.use_det": False, "Global.use_cls": False,
            "Rec.ocr_version": OCRVersion.PPOCRV4, "Rec.model_type": ModelType.SERVER, "Rec.lang_type": LangRec.CH,
            "Rec.model_path": path,
        })
        coreml = _use_coreml(self._engine.text_rec.session, path)
        print(f"[OCR] RapidOCR rec-only: PP-OCRv4 server rec ({'CoreML' if coreml else 'CPU'})")

    def recognize(self, crop_image):
        try:
            out = self._engine(_to_bgr(crop_image), use_det=False, use_cls=False, use_rec=True)
            if not out.txts:
                return '', 0.0
            return out.txts[0], float(out.scores[0])
        except Exception:
            return '', 0.0

    def ocr(self, crop_image, **kwargs):
        text, score = self.recognize(crop_image)
        return [{'rec_texts': [text], 'rec_scores': [score]}]


class _AppleVision:
    """Apple Vision VNRecognizeTextRequest (accurate, language correction off so
    numbers are not 'corrected').  Serves both the full and the rec-only role:
    Vision has no rec-only mode, so recognize() joins the lines it finds."""

    def __init__(self):
        import Vision  # noqa: F401  (pyobjc-framework-Vision, macOS only)
        print("[OCR] Apple Vision (accurate, no language correction)")

    def _run(self, img_or_path):
        import objc
        import Vision
        from Foundation import NSData
        img = _to_bgr(img_or_path)
        if img is None:
            return [], 0, 0
        h, w = img.shape[:2]
        ok, png = cv2.imencode('.png', img)
        if not ok:
            return [], w, h
        lines = []
        with objc.autorelease_pool():
            req = Vision.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(0)          # accurate
            req.setUsesLanguageCorrection_(False)
            req.setRecognitionLanguages_(["en-US"])
            data = NSData.dataWithBytes_length_(png.tobytes(), len(png))
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
            handler.performRequests_error_([req], None)
            for r in req.results() or []:
                cand = r.topCandidates_(1)
                if not cand:
                    continue
                bb = r.boundingBox()           # normalised, origin bottom-left
                x1 = bb.origin.x * w
                y1 = (1.0 - bb.origin.y - bb.size.height) * h
                lines.append((str(cand[0].string()), float(cand[0].confidence()),
                              [x1, y1, x1 + bb.size.width * w, y1 + bb.size.height * h]))
        return lines, w, h

    def ocr(self, img_or_path, **kwargs):
        lines, _, _ = self._run(img_or_path)
        return [{'rec_texts': [t for t, _, _ in lines], 'rec_scores': [c for _, c, _ in lines],
                 'rec_boxes': [b for _, _, b in lines]}]

    def recognize(self, crop_image):
        try:
            lines, _, _ = self._run(crop_image)
        except Exception:
            return '', 0.0
        if not lines:
            return '', 0.0
        lines.sort(key=lambda t: (t[2][1], t[2][0]))
        return ' '.join(t for t, _, _ in lines), min(c for _, c, _ in lines)
