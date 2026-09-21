"""PointCountConsistency: unlabelled plots are not an error; heatmaps without labels are."""
from src.extraction.fusion import PointCountConsistency


def test_unlabelled_scatter_is_not_flagged_but_heatmap_is():
    chk = PointCountConsistency()
    assert chk.run({"yolo_point_count": 20, "ocr_value_count": 0}).issues == []
    assert chk.run({"yolo_point_count": 20, "ocr_value_count": 0, "is_heatmap": False}).issues == []
    res = chk.run({"yolo_point_count": 20, "ocr_value_count": 0, "is_heatmap": True})
    assert [i.severity.name for i in res.issues] == ["ERROR"]
    res = chk.run({"yolo_point_count": 20, "ocr_value_count": 3})          # annotated chart, most labels missed
    assert [i.severity.name for i in res.issues] == ["ERROR"]
    assert chk.run({"yolo_point_count": 20, "ocr_value_count": 18}).issues == []
