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
    assert rep["status"] == "anchored" and rep["assigned"] == 8 and rep["unresolved"] == 0
    assert out[1][1:4] == ["C0", "C1", "C2"] and out[2][1:4] == ["", "C3", "C4"] and out[3][1:4] == ["C5", "C6", "C7"]
    assert out[0] == grid[0] and out[2][4] == "20"


def test_align_partial_when_one_row_short():
    grid = [["1", T, T], ["2", T, T]]
    boxes = [_box(100, 10), _box(200, 10), _box(100, 60)]        # row 2 lost a box
    out, rep = align_structures(grid, _meta(boxes))
    assert rep["status"] == "anchored_partial" and rep["assigned"] == 3
    assert out[0] == ["1", "C0", "C1"] and out[1] == ["2", "C2", T]      # per-cell: the surviving box still lands


def test_align_global_when_rows_split_but_totals_match():
    grid = [["1", T], ["2", T], ["3", T]]
    boxes = [_box(100, 10, h=20), _box(100, 60, h=200), _box(100, 400, h=20)]   # tall middle box breaks the gap rule
    out, rep = align_structures(grid, _meta(boxes))
    assert rep["status"] in ("ok", "global", "anchored") and [r[1] for r in out] == ["C0", "C1", "C2"]


def test_align_failed_keeps_tokens():
    grid = [["1", T], ["2", T]]
    boxes = [_box(100, 10), _box(100, 60), _box(100, 110)]         # stray scheme drawing
    out, rep = align_structures(grid, _meta(boxes))
    assert rep["status"] == "failed" and rep["assigned"] == 0 and out == grid


def test_align_multi_token_cell_and_invalid_smiles():
    grid = [["1", f"{T}; {T}", "50"]]
    boxes = [_box(100, 10), _box(160, 10)]
    out, rep = align_structures(grid, _meta(boxes, ["CCO", ""]))
    assert out[0][1] == f"CCO; {T}" and rep["assigned"] == 1 and rep["unresolved"] == 1 and rep["status"] == "anchored"


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
    assert rows == [None, 46.0, 126.0, 206.0] and cols and len(cols) == 5   # entry numbers + yields
    # boxes: row 1 has all three, row 2 misses Ar1Br (ditto) plus a stray extra box in the Yield column, row 3 complete
    boxes = [(90, 40), (190, 40), (290, 40), (190, 120), (290, 120), (420, 120), (90, 200), (190, 200), (290, 200)]
    meta = _meta([_box(x, y) for x, y in boxes])
    out, rep = align_structures(grid, meta, row_centres=rows, col_centres=cols)
    assert rep["status"] == "anchored_partial" and rep["assigned"] == 8 and rep["unplaced"] == 1
    assert out[1][1:4] == ["C0", "C1", "C2"] and out[2][1:4] == ["", "C3", "C4"] and out[3][1:4] == ["C6", "C7", "C8"]


def test_anchors_refuse_when_too_few_rows_match():
    from src.extraction.table.grid_ops import text_layer_anchors
    grid = [["Entry", "P"], ["1", T], ["2", T], ["3", T], ["4", T]]
    lines = [{"text": "1", "bbox_pt": [102, 120, 104, 123]}]          # only 1 of 4 rows anchored
    rows, cols = text_layer_anchors(grid, 1, lines, [100, 100, 220, 170], 4.0, (0.0, 0.0), 480.0)
    assert rows is None and cols is None


def test_anchors_from_any_text_cell_with_interpolation():
    from src.extraction.table.grid_ops import text_layer_anchors
    grid = [["Substrate", "E", "Product", "Yield"],
            [T, "MeI", T, "87"], [T, "MeOTf", T, "82"], [T, "MeI", T, "36"], [T, "PhCHO", T, "93"]]
    # rows 1,2,4 anchored by electrophile / yield lines; row 3 has none → interpolated
    lines = [{"text": "MeI", "bbox_pt": [130, 120, 136, 123]}, {"text": "87", "bbox_pt": [200, 120, 204, 123]},
             {"text": "MeOTf", "bbox_pt": [130, 140, 138, 143]}, {"text": "82", "bbox_pt": [200, 140, 204, 143]},
             {"text": "PhCHO", "bbox_pt": [130, 180, 138, 183]}, {"text": "93", "bbox_pt": [200, 180, 204, 183]}]
    rows, cols = text_layer_anchors(grid, 1, lines, [100, 100, 220, 200], 4.0, (0.0, 0.0), 480.0)
    assert rows == [None, 86.0, 166.0, 246.0, 326.0] and cols is None


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


def test_duplicates_and_side_by_side_drawings_do_not_block_anchored_alignment():
    grid = [["Substrate", "E", "Product", "Yield"], [T, "MeI", T, "92"], [T, "MeOTf", T, "88"]]
    rows = [None, 100.0, 300.0]
    boxes = [_box(100, 80), _box(102, 82),                    # duplicate detection of the substrate
             _box(700, 80), _box(900, 85),                     # product + its isomer drawn side by side
             _box(100, 280), _box(700, 280)]
    out, rep = align_structures(grid, _meta(boxes, ["S1", "S1dup", "P1", "P1iso", "S2", "P2"]), row_centres=rows)
    assert rep["n_duplicates"] == 1 and rep["assigned"] == 4 and rep["anchors"] == "rows+box_cols"
    assert out[1][0] in ("S1", "S1dup") and out[1][2] == "P1" and out[2][0] == "S2" and out[2][2] == "P2"


def test_box_row_clusters_become_anchors_without_text_layer():
    grid = [["Substrate", "E", "Product", "Yield"], [T, "MeI", T, "92"], [T, "MeOTf", T, "88"], [T, "PhCHO", T, "70"]]
    boxes = [_box(100, 80), _box(700, 80), _box(900, 84),          # row 1 + an isomer drawing beside the product
             _box(100, 280), _box(700, 280),
             _box(100, 480), _box(700, 480), _box(705, 482)]        # row 3 + duplicate detection
    out, rep = align_structures(grid, _meta(boxes, ["S1", "P1", "P1iso", "S2", "P2", "S3", "P3", "P3dup"]))
    assert rep["anchors"] == "box_rows+box_cols" and rep["assigned"] == 6
    assert [r[0] for r in out[1:]] == ["S1", "S2", "S3"] and [r[2] for r in out[1:]] == ["P1", "P2", "P3"]


def test_column_sequence_fallback_when_rows_do_not_cluster():
    # 4 token rows; substrate drawings are tall and overlap the next row's product drawings vertically,
    # so vertical clustering yields the wrong row count — per-column ordering still works.
    grid = [["S", "E", "P", "Y"]] + [[T, "MeI", T, "9"] for _ in range(4)]
    boxes = [_box(100, 40, h=220), _box(700, 100),
             _box(100, 180, h=220), _box(700, 240), _box(760, 244),     # isomer beside the product
             _box(100, 320, h=220), _box(700, 380),
             _box(100, 460, h=220), _box(700, 520)]
    names = ["S1", "P1", "S2", "P2", "P2iso", "S3", "P3", "S4", "P4"]
    out, rep = align_structures(grid, _meta(boxes, names))
    assert rep["status"] == "columns" and rep["assigned"] == 8
    assert [r[0] for r in out[1:]] == ["S1", "S2", "S3", "S4"] and [r[2] for r in out[1:]] == ["P1", "P2", "P3", "P4"]


def test_drawing_spanning_two_text_rows_lands_in_the_row_with_the_slot():
    # Product drawn once for the MeI / MeOTf pair: the VLM writes the token on
    # the first row and "" (ditto) on the second; the drawing's centre is
    # nearer the second row's text line.
    grid = [["E", "Product", "Yield"], ["tBuOH", T, "93"], ["MeI", T, "62"], ["MeOTf", "", "82"], ["PhCHO", T, "70"]]
    rows = [None, 100.0, 200.0, 240.0, 340.0]
    cols = [30.0, 150.0, 300.0]
    meta = _meta([_box(130, 85), _box(130, 222), _box(130, 325)])      # 2nd box centre y=237 ≈ MeOTf line
    out, rep = align_structures(grid, meta, row_centres=rows, col_centres=cols)
    assert rep["status"] == "anchored" and rep["assigned"] == 3 and rep["unplaced"] == 0
    assert out[2][1] == "C1" and out[3][1] == ""


def test_leftover_drawing_fills_the_empty_cell_it_is_printed_in():
    # Ester 8b is drawn in row 3 but the VLM missed the token (wrote "").
    grid = [["Ester", "E", "Product"], [T + " 8a", "tBuOH", T], ["", "MeI", T], ["", "iPrOH", T], ["", "MeI", T]]
    rows = [None, 100.0, 200.0, 300.0, 400.0]
    cols = [50.0, 150.0, 300.0]
    meta = _meta([_box(30, 85), _box(280, 85), _box(280, 185), _box(30, 285), _box(280, 285), _box(280, 385)])
    out, rep = align_structures(grid, meta, row_centres=rows, col_centres=cols)
    assert rep["assigned"] == 5 and rep["filled_empty"] == 1 and rep["unplaced"] == 0 and rep["status"] == "anchored"
    assert out[3][0] == "C3" and out[2][0] == "" and out[4][0] == ""


def test_tall_merged_cell_drawing_walks_up_to_its_token():
    # Ester 1a drawn once for entries 1–4: token on row 1, "" below, drawing
    # centred at row 3.  It must fill the token, not the empty cell it sits in.
    grid = [["Ester", "E", "Yield"], [T + " 1a", "tBuOH", "93"], ["", "MeI", "88"], ["", "Me3SiCl", "96"], ["", "PhCHO", "82"],
            [T + " 1b", "iPrOH", "87"], ["", "MeI", "62"]]
    rows = [None, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    cols = [50.0, 150.0, 300.0]
    meta = _meta([_box(30, 285, h=40), _box(30, 535, h=40)])
    out, rep = align_structures(grid, meta, row_centres=rows, col_centres=cols)
    assert rep["status"] == "anchored" and rep["assigned"] == 2 and rep["filled_empty"] == 0
    assert out[1][0] == "C0 1a" and out[3][0] == "" and out[5][0] == "C1 1b" and out[6][0] == ""
