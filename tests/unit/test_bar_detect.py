"""Filled-bar detection on a synthetic bar chart."""
import cv2
import numpy as np

from src.extraction.figure.bar_detect import detect_bars


def _chart():
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.line(img, (40, 260), (390, 260), (0, 0, 0), 2)          # x axis (2 px, touches every bar)
    cv2.line(img, (40, 20), (40, 260), (0, 0, 0), 1)            # y axis
    cv2.rectangle(img, (80, 100), (120, 259), (128, 128, 128), -1)    # bar 1, top 100
    cv2.rectangle(img, (180, 150), (220, 259), (128, 128, 128), -1)   # bar 2, top 150
    cv2.rectangle(img, (280, 180), (320, 259), (128, 128, 128), 2)    # outlined bar → ignored
    cv2.putText(img, "80", (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.rectangle(img, (340, 40), (356, 56), (128, 128, 128), -1)     # legend swatch (floating square)
    return img


def test_detects_filled_bars_and_ignores_axes_text_and_outlines():
    bars = detect_bars(_chart(), x_range=(45, 395), y_range=(15, 290), baseline=260)   # ROI includes the axis line
    assert [round(b["cx"]) for b in bars] == [100, 200]                  # swatch at y=40..56 is not on the baseline
    assert [round(b["cx"]) for b in detect_bars(_chart(), (45, 395), (15, 265))] == [100, 200, 348]
    assert [b["top"] for b in bars] == [100.0, 150.0] and all(b["bottom"] >= 259 for b in bars)


def test_empty_or_tiny_plot_area():
    assert detect_bars(np.full((50, 50, 3), 255, dtype=np.uint8), (0, 50), (0, 50)) == []
    assert detect_bars(_chart(), (100, 105), (0, 300)) == []
