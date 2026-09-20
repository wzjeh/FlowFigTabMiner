"""Series recovery for xy plots without a YOLO legend crop."""
import cv2
import numpy as np

from src.extraction.figure.legend_matcher import LegendMatcher
from src.extraction.figure.series_recovery import (
    NAMED_COLORS_HSV,
    prototypes_from_markers,
    prototypes_from_text_layer,
    recover_series_prototypes,
)


def test_markers_map_colour_words_to_hsv():
    res = prototypes_from_markers([
        {"name": "-78 °C", "color": "red circles", "marker": "circle"},
        {"name": "0 °C", "color": "Blue", "marker": "square"},
        {"name": "24 °C", "color": "black", "marker": "triangle"},
    ])
    assert res is not None and res.source == "vlm_colors"
    assert list(res.prototypes) == ["-78 °C", "0 °C", "24 °C"]
    assert np.array_equal(res.prototypes["0 °C"]["hsv"], NAMED_COLORS_HSV["blue"])


def test_markers_refuse_unknown_or_duplicate_colours():
    assert prototypes_from_markers([{"name": "a", "color": "red"}, {"name": "b", "color": None}]) is None
    assert prototypes_from_markers([{"name": "a", "color": "open"}, {"name": "b", "color": "filled"}]) is None
    assert prototypes_from_markers([{"name": "a", "color": "red"}]) is None


def _synthetic_figure():
    """400x300 crop (bbox 100x75 pt at 4 px/pt) with a red and a blue swatch
    left of two legend strings, plus a plot of 20 red + 20 blue points."""
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 52), (40, 68), (0, 0, 220), -1)      # red swatch (BGR)
    cv2.rectangle(img, (20, 92), (40, 108), (220, 0, 0), -1)     # blue swatch
    rng = np.random.default_rng(0)
    points, truth = [], []
    for i in range(40):
        x, y = int(rng.integers(120, 380)), int(rng.integers(130, 290))
        col = (0, 0, 220) if i % 2 == 0 else (220, 0, 0)
        cv2.circle(img, (x, y), 5, col, -1)
        points.append({"label": "data_point", "box": [x - 6, y - 6, x + 6, y + 6], "center": [x, y]})
        truth.append("-78 °C" if i % 2 == 0 else "0 °C")
    inner_lines = [
        {"text": "−78 °C", "bbox_pt": [12.0, 13.0, 30.0, 17.0]},   # px x 48..120, y 52..68
        {"text": "0 °C", "bbox_pt": [12.0, 23.0, 24.0, 27.0]},
        {"text": "Yield (%)", "bbox_pt": [2.0, 40.0, 6.0, 60.0]},
    ]
    return img, inner_lines, points, truth


def test_text_layer_samples_swatch_colours():
    img, inner_lines, _, _ = _synthetic_figure()
    res = prototypes_from_text_layer(img, [0, 0, 100, 75], inner_lines, ["-78 °C", "0 °C"])
    assert res is not None and res.source == "text_layer"
    assert abs(int(res.prototypes["-78 °C"]["hsv"][0]) - 0) <= 3
    assert abs(int(res.prototypes["0 °C"]["hsv"][0]) - 120) <= 3
    # Missing swatch → refuse (no partial prototypes).
    assert prototypes_from_text_layer(img, [0, 0, 100, 75], inner_lines, ["-78 °C", "Yield (%)"]) is None
    # Name not in the text layer → refuse.
    assert prototypes_from_text_layer(img, [0, 0, 100, 75], inner_lines, ["-78 °C", "-48 °C"]) is None


def test_end_to_end_points_get_series_via_match_points(tmp_path):
    img, inner_lines, points, truth = _synthetic_figure()
    path = tmp_path / "page_1_figure_0.png"
    cv2.imwrite(str(path), img)
    ctx = {"bbox_pt": [0, 0, 100, 75], "inner_lines": inner_lines}
    res = recover_series_prototypes(["-78 °C", "0 °C"], [], str(path), ctx, log=lambda *_: None)
    assert res is not None and res.source == "text_layer"
    matcher = LegendMatcher.__new__(LegendMatcher)      # skip OCR/YOLO loading
    matcher.yolo = None; matcher.ocr = None
    matched = matcher.match_points(points, res.prototypes, str(path))
    correct = sum(1 for p, t in zip(matched, truth) if p.get("series") == t)
    assert correct / len(truth) >= 0.95


def test_recovery_disabled_without_two_names():
    assert recover_series_prototypes(["only"], [], None, None, log=lambda *_: None) is None
    assert recover_series_prototypes([], [{"name": "a", "color": "red"}], None, None, log=lambda *_: None) is None
