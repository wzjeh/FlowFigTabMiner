"""VLM second reader: parsing, fusion, contact sheet, reader, mapper regression."""
import json
import numpy as np
import cv2
import pytest

from src.extraction.figure.axis_fit import parse_tick_text, parse_value_text, fuse_tick_readings, fuse_value_readings, robust_fit
from src.extraction.figure.contact_sheet import build_contact_sheet, crop_box
from src.extraction.figure.label_reader import VLMLabelReader, LabelReadingsResponse
from src.llm.types import LLMResponse


# ── parsing (behaviour of the former nested parse_val + the VLM caret form) ──
@pytest.mark.parametrize("txt,expected", [
    ("10-1.5", -1.5), ("10 -2", -2.0), ("100.5", 0.5), ("101", 101.0), ("100", 100.0), ("10^-1.5", -1.5), ("10^0", 0.0), ("10^1", 1.0),
    ("-40", -40.0), ("0.5", 0.5), ("-10 0", None), ("10-1.", None), ("flow", None), ("10", None), ("", None),
])
def test_parse_tick_text(txt, expected):
    assert parse_tick_text(txt) == expected


# ── fusion ──────────────────────────────────────────────────────────────────
def test_fuse_tick_readings_cases():
    assert fuse_tick_readings("10-1.5", "10^-1.5") == [(-1.5, "agree")]
    assert fuse_tick_readings("10-1.", "10^-1.5") == [(-1.5, "vlm")]          # OCR ambiguous → VLM only
    assert fuse_tick_readings("-40", "") == [(-40.0, "ocr")]
    assert fuse_tick_readings("101", "10^-1") == [(101.0, "conflict:ocr"), (-1.0, "conflict:vlm")]
    assert fuse_tick_readings("", "") == []


def test_conflict_is_arbitrated_by_geometry():
    # ticks at 202/286/380/575 px; box at 286 read "101" by OCR, "10^-1" by VLM → both go in, fit keeps -1.0
    pairs = [(202, -1.5), (380, -0.5), (575, 0.5)]
    for v, src in fuse_tick_readings("101", "10^-1"):
        pairs.append((286, v))
    m = robust_fit(pairs, is_log=True)
    assert m.inliers == 4 and abs(m.predict([286])[0] - (-1.0)) < 0.1


def test_fuse_value_readings_policies():
    assert fuse_value_readings(43.0, 43.0) == (43.0, "agree")
    assert fuse_value_readings(4.0, 43.0, "vlm") == (43.0, "conflict:vlm")
    assert fuse_value_readings(4.0, 43.0, "ocr") == (4.0, "conflict:ocr")
    assert fuse_value_readings(4.0, 43.0, "null") == (None, "conflict:null")
    assert fuse_value_readings(133.0, 33.0) == (33.0, "vlm")            # OCR out of range
    assert fuse_value_readings(33.0, None) == (33.0, "ocr")
    assert fuse_value_readings(None, float("nan")) == (None, "none")


# ── contact sheet ───────────────────────────────────────────────────────────
def _img_with_boxes(n):
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    dets = []
    for i in range(n):
        x, y = 20 + (i % 5) * 110, 20 + (i // 5) * 90
        cv2.putText(img, str(i * 7), (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        dets.append({"label": "x_tick_label", "box": [x, y, x + 60, y + 35], "center": (x + 30, y + 17)})
    return img, dets


def test_contact_sheet_indexes_match_dets():
    img, dets = _img_with_boxes(7)
    sheet, cells = build_contact_sheet(img, dets, cols=3)
    assert len(cells) == 7 and all(cells[i]["det"] is dets[i] and cells[i]["index"] == i for i in range(7))
    assert sheet.shape[0] > 3 * 60 and sheet.shape[1] > 3 * 60
    crop, cb = crop_box(img, [0, 0, 10, 10], min_side=60)
    assert min(crop.shape[:2]) >= 60 and cb == [0, 0, 25, 25]


def test_contact_sheet_empty():
    sheet, cells = build_contact_sheet(np.zeros((10, 10, 3), np.uint8), [])
    assert cells == [] and sheet.shape[0] > 0


# ── reader with a stub VLM ──────────────────────────────────────────────────
class _StubVLM:
    def __init__(self, payload=None, raise_exc=None):
        self.payload, self.raise_exc, self.calls = payload, raise_exc, []

    def inspect(self, image, system_prompt, user_prompt, cfg, response_schema=None):
        self.calls.append((image, response_schema))
        if self.raise_exc:
            raise self.raise_exc
        return LLMResponse(text="{}", model="stub", tokens_in=100, tokens_out=20, latency_ms=3.0), self.payload


def test_reader_maps_indices_and_writes_sheet(tmp_path):
    img, dets = _img_with_boxes(3)
    vlm = _StubVLM({"readings": [{"index": 0, "text": "10^-1.5", "kind": "tick"}, {"index": 2, "text": "43", "kind": "value"},
                                 {"index": 9, "text": "ignored", "kind": "tick"}]})
    res = VLMLabelReader(vlm=vlm, cfg=None).read(img, dets, "fig1", str(tmp_path))
    assert res.ok and res.texts == {0: "10^-1.5", 2: "43"} and res.kinds[2] == "value"
    assert (tmp_path / "fig1_labels_sheet.png").exists() and res.model == "stub" and res.tokens_in == 100
    assert vlm.calls[0][1] is LabelReadingsResponse


def test_reader_failure_is_soft(tmp_path):
    img, dets = _img_with_boxes(2)
    res = VLMLabelReader(vlm=_StubVLM(raise_exc=RuntimeError("boom")), cfg=None).read(img, dets, "fig1", str(tmp_path))
    assert not res.ok and res.texts == {} and "boom" in res.notes
    assert VLMLabelReader(vlm=_StubVLM(), cfg=None).read(img, [], "fig1", str(tmp_path)).notes == "no boxes"


@pytest.mark.parametrize("txt,expected", [("10", 10.0), ("43", 43.0), ("_69", 69.0), ("-82", -82.0), ("133", 133.0), ("品", None), ("", None), ("J61", 61.0)])
def test_parse_value_text(txt, expected):
    assert parse_value_text(txt) == expected
