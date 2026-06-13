"""
Unit tests for upscale_for_ocr (issue #17 Phase 2 — capped caption/note upscale).

Pure logic, no models. Verifies the cap math: small crops still get up to 3x,
large crops are capped so the longest side never enters the ~4000px det-blowup
regime, aspect ratio is preserved, nothing is downscaled, and the env override
is honoured.
"""
import os

import numpy as np
import pytest

from src.extraction.common.ocr_backend import upscale_for_ocr


def _img(w, h):
    return np.full((h, w, 3), 255, dtype=np.uint8)


PAD = 50  # default white border per side


class TestUpscaleForOcr:
    def test_small_crop_gets_full_3x(self):
        # 100x40, longest=100; 3x=300 << max_side -> full 3x then +2*pad.
        out = upscale_for_ocr(_img(100, 40), max_side=2000)
        h, w = out.shape[:2]
        assert w == 100 * 3 + 2 * PAD
        assert h == 40 * 3 + 2 * PAD

    def test_large_crop_capped_at_max_side(self):
        # Nagaki note: 1379x286, longest=1379. 3x would be 4137 (>2000).
        # scale = 2000/1379 -> longest side (pre-pad) <= 2000.
        out = upscale_for_ocr(_img(1379, 286), max_side=2000)
        h, w = out.shape[:2]
        assert w <= 2000 + 2 * PAD
        # height scaled by the same factor (aspect ratio preserved); cv2.resize
        # rounds the target size, so allow ±1px.
        scale = 2000 / 1379
        assert abs(h - (round(286 * scale) + 2 * PAD)) <= 1

    def test_aspect_ratio_preserved(self):
        out = upscale_for_ocr(_img(1227, 44), max_side=2000)
        h, w = out.shape[:2]
        scale = 2000 / 1227
        assert abs(w - (round(1227 * scale) + 2 * PAD)) <= 1
        assert abs(h - (round(44 * scale) + 2 * PAD)) <= 1

    def test_never_downscales(self):
        # Original longest side already exceeds max_side -> scale clamped to 1x.
        out = upscale_for_ocr(_img(3000, 100), max_side=2000)
        h, w = out.shape[:2]
        assert w == 3000 + 2 * PAD  # 1x, only padded
        assert h == 100 + 2 * PAD

    def test_lower_cap_gives_smaller_output(self):
        big = upscale_for_ocr(_img(1379, 286), max_side=2000)
        small = upscale_for_ocr(_img(1379, 286), max_side=1600)
        assert small.shape[1] < big.shape[1]
        assert small.shape[1] <= 1600 + 2 * PAD

    def test_env_default_override(self, monkeypatch):
        import importlib
        monkeypatch.setenv("OCR_UPSCALE_MAX_SIDE", "1600")
        import src.extraction.common.ocr_backend as ob
        importlib.reload(ob)
        try:
            assert ob.OCR_UPSCALE_MAX_SIDE == 1600
            out = ob.upscale_for_ocr(_img(1379, 286))  # uses module default
            assert out.shape[1] <= 1600 + 2 * PAD
        finally:
            monkeypatch.delenv("OCR_UPSCALE_MAX_SIDE", raising=False)
            importlib.reload(ob)  # restore default for other tests
