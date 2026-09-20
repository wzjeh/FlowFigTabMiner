"""VLM metadata string hygiene (control characters, schema echoes, over-long)."""
from src.extraction.figure.metadata_vlm import clean_vlm_text, FigureMetadataResponse


def test_clean_vlm_text_drops_junk_keeps_real():
    assert clean_vlm_text("-78 " + "\x08" * 2000) == "-78"          # backspace flood stripped
    assert clean_vlm_text("legend_series_names_null_null") is None
    assert clean_vlm_text("y_axis_unit_%") is None
    assert clean_vlm_text("x" * 121) is None
    assert clean_vlm_text("  3,4-dichloroaniline  ") == "3,4-dichloroaniline"
    assert clean_vlm_text("−78 °C") == "−78 °C"
    assert clean_vlm_text(None) is None
    assert clean_vlm_text(5) == "5"


def test_response_schema_accepts_markers():
    r = FigureMetadataResponse(legend_series_names=["a", None], legend_markers=[{"name": "a", "color": "red"}])
    assert r.legend_markers[0].color == "red" and r.legend_markers[0].marker is None
