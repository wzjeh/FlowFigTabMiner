"""Deterministic grid operations for VLM-transcribed tables."""
from src.extraction.table.grid_ops import STRUCTURE_TOKEN as T, align_structures, cluster_rows, text_agreement


def _box(x, y, w=40, h=30):
    return [x, y, x + w, y + h]


def _meta(boxes, smiles=None):
    smiles = smiles or [f"C{i}" for i in range(len(boxes))]
    return [{"box": b, "smiles": s, "conf": 0.9} for b, s in zip(boxes, smiles)]


def test_cluster_rows_by_vertical_gap():
    boxes = [_box(200, 10), _box(20, 12), _box(20, 100), _box(200, 104, h=60)]
    rows = cluster_rows(boxes)
    assert rows == [[1, 0], [2, 3]]


def test_align_ok_row_by_row():
    grid = [["Entry", "Ar1Br", "Ar2X", "Product", "Yield"],
            ["1", T, T, T, "93"],
            ["2", "", T, T, "20"],
            ["3", T, T, T, "32"]]
    boxes = [_box(100, 10), _box(200, 10), _box(300, 10),
             _box(200, 60), _box(300, 60),
             _box(100, 110), _box(200, 110), _box(300, 110)]
    out, rep = align_structures(grid, _meta(boxes))
    assert rep["status"] == "ok" and rep["assigned"] == 8 and rep["unresolved"] == 0
    assert out[1][1:4] == ["C0", "C1", "C2"] and out[2][1:4] == ["", "C3", "C4"] and out[3][1:4] == ["C5", "C6", "C7"]
    assert out[0] == grid[0] and out[2][4] == "20"


def test_align_partial_when_one_row_short():
    grid = [["1", T, T], ["2", T, T]]
    boxes = [_box(100, 10), _box(200, 10), _box(100, 60)]        # row 2 lost a box
    out, rep = align_structures(grid, _meta(boxes))
    assert rep["status"] == "partial" and rep["assigned"] == 2
    assert out[0] == ["1", "C0", "C1"] and out[1] == ["2", T, T]


def test_align_global_when_rows_split_but_totals_match():
    grid = [["1", T], ["2", T], ["3", T]]
    boxes = [_box(100, 10, h=20), _box(100, 60, h=200), _box(100, 400, h=20)]   # tall middle box breaks the gap rule
    out, rep = align_structures(grid, _meta(boxes))
    assert rep["status"] in ("ok", "global") and [r[1] for r in out] == ["C0", "C1", "C2"]


def test_align_failed_keeps_tokens():
    grid = [["1", T], ["2", T]]
    boxes = [_box(100, 10), _box(100, 60), _box(100, 110)]         # stray scheme drawing
    out, rep = align_structures(grid, _meta(boxes))
    assert rep["status"] == "failed" and rep["assigned"] == 0 and out == grid


def test_align_multi_token_cell_and_invalid_smiles():
    grid = [["1", f"{T}; {T}", "50"]]
    boxes = [_box(100, 10), _box(160, 10)]
    out, rep = align_structures(grid, _meta(boxes, ["CCO", ""]))
    assert out[0][1] == f"CCO; {T}" and rep["assigned"] == 1 and rep["unresolved"] == 1 and rep["status"] == "ok"


def test_align_no_boxes():
    grid = [["1", T]]
    out, rep = align_structures(grid, [])
    assert rep["status"] == "none" and out == grid


def test_text_agreement():
    rows = [["1", "MeOH", "87"], ["2", "MeI", "36"], ["3", "MeOTf", "82.5"]]
    assert text_agreement(rows, "1 | MeOH | 87\n2 | MeI | 36\n3 | MeOTf | 82,5") == 1.0
    assert text_agreement(rows, "1 | 87\n2 | 36") == 0.667            # 82.5 and 3 missing
    assert text_agreement(rows, "") is None and text_agreement(rows, None) is None
    assert text_agreement([["a", "b"]], "1 2 3") is None               # too few numbers in the grid
    assert text_agreement([["-28", "0.055", "84"]], "\x01 28 | 0.055 | 84") == 1.0


def test_anchored_alignment_places_boxes_by_row_and_column():
    from src.extraction.table.grid_ops import text_layer_anchors
    grid = [["Entry", "Ar1Br", "Ar2X", "Product", "Yield"],
            ["1", T, T, T, "93"], ["2", "", T, T, "20"], ["3", T, T, T, "32"]]
    # text layer in page points: crop bbox at (100,100) pt, scale 4 → px = (pt-100)*4; body offset y=40 px
    lines = [{"text": "Entry", "bbox_pt": [102, 105, 108, 108]}, {"text": "Ar1Br", "bbox_pt": [120, 105, 130, 108]},
             {"text": "Ar2X", "bbox_pt": [145, 105, 155, 108]}, {"text": "Ar1–Ar2", "bbox_pt": [170, 105, 185, 108]},
             {"text": "Yield [%]", "bbox_pt": [200, 105, 212, 108]},
             {"text": "1", "bbox_pt": [102, 120, 104, 123]}, {"text": "2", "bbox_pt": [102, 140, 104, 143]},
             {"text": "3", "bbox_pt": [102, 160, 104, 163]}, {"text": "1", "bbox_pt": [205, 120, 207, 123]}]
    grid[0][3] = "Ar1–Ar2"
    rows, cols = text_layer_anchors(grid, 1, lines, [100, 100, 220, 170], 4.0, (0.0, 40.0), 480.0)
    assert rows == [None, 46.0, 126.0, 206.0] and cols and len(cols) == 5
    # boxes: row 1 has all three, row 2 misses Ar1Br (ditto) plus a stray extra box in the Yield column, row 3 complete
    boxes = [(90, 40), (190, 40), (290, 40), (190, 120), (290, 120), (420, 120), (90, 200), (190, 200), (290, 200)]
    meta = _meta([_box(x, y) for x, y in boxes])
    out, rep = align_structures(grid, meta, row_centres=rows, col_centres=cols)
    assert rep["status"] == "anchored_partial" and rep["assigned"] == 8 and rep["unplaced"] == 1
    assert out[1][1:4] == ["C0", "C1", "C2"] and out[2][1:4] == ["", "C3", "C4"] and out[3][1:4] == ["C6", "C7", "C8"]


def test_anchors_refuse_when_entry_labels_missing():
    from src.extraction.table.grid_ops import text_layer_anchors
    grid = [["Entry", "P"], ["1", T], ["2", T]]
    lines = [{"text": "1", "bbox_pt": [102, 120, 104, 123]}]          # row 2 has no anchor
    rows, cols = text_layer_anchors(grid, 1, lines, [100, 100, 220, 170], 4.0, (0.0, 0.0), 480.0)
    assert rows is None and cols is None


def test_anchored_rows_only_uses_box_x_clusters_and_drops_header_band_boxes():
    grid = [["Entry", "A", "B", "Yield"], ["1", T, T, "93"], ["2", T, T, "20"], ["3", T, T, "32"]]
    rows = [None, 100.0, 200.0, 300.0]
    boxes = [_box(150, 0, h=20), _box(350, 0, h=20),          # garbage above the first row → unplaced
             _box(140, 85), _box(340, 85),
             _box(150, 185), _box(345, 185), _box(150, 205),  # duplicate detection in row 2 col A → nearest wins
             _box(150, 285), _box(350, 285)]
    smiles = ["G1", "G2", "A1", "B1", "A2", "B2", "A2dup", "A3", "B3"]
    out, rep = align_structures(grid, _meta(boxes, smiles), row_centres=rows)
    assert rep["anchors"] == "rows+box_cols" and rep["assigned"] == 6 and rep["unplaced"] == 3
    assert out[1][1:3] == ["A1", "B1"] and out[2][1:3] == ["A2", "B2"] and out[3][1:3] == ["A3", "B3"]


def test_atom_label_fragments_are_stripped_when_token_is_filled():
    from src.extraction.table.grid_ops import _strip_atom_labels
    assert _strip_atom_labels("MeO-Brc1ccc(C)cc1.CC", ["Brc1ccc(C)cc1.CC"]) == "Brc1ccc(C)cc1.CC"
    assert _strip_atom_labels("Br-Fc1ccc(Br)cc1-F", ["Fc1ccc(Br)cc1"]) == "Fc1ccc(Br)cc1"
    assert _strip_atom_labels("MeO-[STRUCTURE]", [T]) == "[STRUCTURE]"
    assert _strip_atom_labels("c1ccccc1 3a", ["c1ccccc1"]) == "c1ccccc1 3a"       # compound labels stay
    assert _strip_atom_labels("CCO; c1ccccc1", ["CCO", "c1ccccc1"]) == "CCO; c1ccccc1"
    grid = [["1", f"MeO-{T}", "93"]]
    out, rep = align_structures(grid, _meta([_box(100, 10)], ["COc1ccc(Br)cc1"]))
    assert out[0][1] == "COc1ccc(Br)cc1"
