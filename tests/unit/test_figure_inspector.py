"""
Unit tests for figure VLM inspector helpers.

Covers _extract_points handling of both Gemini payload shapes:
- dict form: {"points": [...]}
- bare list form: [...] (free-schema responses sometimes return the array directly)
"""
import pytest

from src.llm.inspectors.figure import _extract_points

POINTS = [
    {"series": "yield", "x": 1.0, "y": 52.3, "value": 52.3},
    {"series": "yield", "x": 2.0, "y": 61.0},
]


class TestExtractPoints:
    def test_dict_payload(self):
        out = _extract_points({"points": POINTS})
        assert len(out) == 2
        assert out[0] == {"series": "yield", "x": 1.0, "y": 52.3, "value": 52.3}
        assert out[1] == {"series": "yield", "x": 2.0, "y": 61.0}

    def test_list_payload(self):
        out = _extract_points(POINTS)
        assert len(out) == 2
        assert out[0]["x"] == 1.0
        assert out[1]["y"] == 61.0

    def test_dict_payload_without_points_key(self):
        assert _extract_points({"foo": "bar"}) == []

    def test_dict_payload_with_none_points(self):
        assert _extract_points({"points": None}) == []

    def test_list_payload_skips_invalid_entries(self):
        out = _extract_points([{"x": 1, "y": 2}, {"x": 3}, "junk", None])
        assert out == [{"x": 1, "y": 2}]

    def test_non_dict_non_list_payload(self):
        assert _extract_points(None) == []
        assert _extract_points("oops") == []
