"""Robust axis calibration + precedence rules (2026-09-20 audit fixes)."""
import numpy as np
import pytest
from src.extraction.figure.axis_fit import robust_fit, monotonic_subsequence, tick_text_is_ambiguous, is_grid_like
from src.adjudication.post_processor import inherit_conditions, source_blocks_inheritance


def test_robust_fit_rejects_lost_minus_sign_exponent():
    # true axis: 10^-1.5 @202, 10^-0.5 @380, 10^0.5 @575 ; "101" (really 10^-1) @286 mis-read as +1.0
    pairs = [(202, -1.5), (380, -0.5), (575, 0.5), (286, 1.0)]
    m = robust_fit(pairs, is_log=True)
    assert m is not None and m.inliers == 3 and m.total == 4
    assert abs(m.predict([286])[0] - (-1.0)) < 0.1          # outlier position now maps to 10^-1


def test_robust_fit_rejects_minus100_on_linear_axis():
    pairs = [(65, 20.0), (255, 0.0), (352, -100.0)]           # "-10 0" → -100 ; true value -10
    m = robust_fit(pairs, is_log=False)
    assert m is not None and m.inliers == 2
    assert abs(m.predict([352])[0] - (-10.2)) < 1.5


def test_robust_fit_needs_two_consistent_ticks():
    assert robust_fit([(100, 0.0)], is_log=True) is None
    m = robust_fit([(100, 0.0), (200, 1.0)], is_log=True)
    assert m is not None and abs(m.predict([300])[0] - 2.0) < 1e-9


def test_monotonic_subsequence_on_x():
    cands = [[0, -1.5, "10-1.5", 202, 0], [0, 1.0, "101", 286, 0], [0, -0.5, "10-0.5", 380, 0], [0, 0.5, "100.5", 575, 0]]
    kept = monotonic_subsequence(cands, pixel_idx=3, direction="increasing")
    assert [c[2] for c in kept] == ["10-1.5", "10-0.5", "100.5"]


def test_ambiguous_tick_texts():
    assert tick_text_is_ambiguous("-10 0") and tick_text_is_ambiguous("10-1.")
    assert not tick_text_is_ambiguous("-10") and not tick_text_is_ambiguous("10-1.5") and not tick_text_is_ambiguous("10 -2")


def test_grid_like_distinguishes_heatmap_from_scatter():
    grid = [{"X": 10 ** (i * 0.5 - 1.5), "Y_Left": -78 + 20 * j} for i in range(5) for j in range(5)]
    assert is_grid_like(grid)["grid"] is True
    rng = np.random.default_rng(0)
    scatter = [{"X": float(10 ** rng.uniform(-2, 1)), "Y_Left": float(rng.uniform(0, 100))} for _ in range(40)]
    assert is_grid_like(scatter)["grid"] is False


def test_inheritance_blocked_by_source_text():
    gv = {"default_conditions": {"residence_time_s": {"value": 0.055, "quote": "tR = 0.055 s", "scope": "paper"},
                                 "flow_rate_mL_min": {"value": 6.0, "quote": "6.0 mL/min", "scope": "paper"},
                                 "solvent": {"value": "tetrahydrofuran", "quote": "in THF", "scope": "paper"}}}
    rec = {"conditions": {"residence_time_s": None, "flow_rate_mL_min": None, "solvent": None}}
    out = inherit_conditions(dict(rec), {}, gv, {"source_local": 0, "paper_global": 0},
                             source_text="Table 1. Effect of residence time in a conventional macrobatch reactor")
    c = out["conditions"]
    assert c["residence_time_s"] is None and c["flow_rate_mL_min"] is None     # varied / batch → blocked
    assert c["solvent"] == "tetrahydrofuran"                                     # caption silent → inherited
    assert source_blocks_inheritance("temperature_C", "Effects of temperature and residence time")
    assert source_blocks_inheritance("temperature_C", "Scope of electrophiles") is None


def test_flow_rate_not_inherited_when_residence_time_varies():
    gv = {"default_conditions": {"flow_rate_mL_min": {"value": 6.0, "quote": "flow rate: 6.0 mL/min", "scope": "paper"}}}
    rec = {"__synthesized": True, "conditions": {"residence_time_s": 0.3, "flow_rate_mL_min": None}}
    out = inherit_conditions(dict(rec), {}, gv, None, source_text="Figure 2. Yield of 3")
    assert out["conditions"]["flow_rate_mL_min"] is None
    rec2 = {"conditions": {"residence_time_s": None, "flow_rate_mL_min": None}}
    assert inherit_conditions(dict(rec2), {}, gv, None, source_text="Table 3. Scope of electrophiles")["conditions"]["flow_rate_mL_min"] == 6.0


def test_llm_copied_global_flow_rate_removed_on_tr_scan():
    gv = {"default_conditions": {"flow_rate_mL_min": {"value": 6.0, "quote": "flow rate: 6.0 mL/min", "scope": "paper"}}}
    rec = {"__synthesized": True, "conditions": {"residence_time_s": 0.3, "flow_rate_mL_min": 6.0}}
    out = inherit_conditions(dict(rec), {}, gv, None, source_text="Figure 2. Effect of residence time")
    assert out["conditions"]["flow_rate_mL_min"] is None and out["conditions_provenance"]["flow_rate_mL_min"] == "removed_global_default"
    rec2 = {"__synthesized": True, "conditions": {"residence_time_s": 0.3, "flow_rate_mL_min": 2.5}}   # source-specific value stays
    assert inherit_conditions(dict(rec2), {}, gv, None, source_text="Figure 2")["conditions"]["flow_rate_mL_min"] == 2.5


def test_llm_value_kept_when_source_mentions_it():
    gv = {"default_conditions": {"solvent": {"value": "tetrahydrofuran", "quote": "in THF", "scope": "paper"},
                                 "temperature_C": {"value": -78, "quote": "at -78 °C", "scope": "paper"}}}
    rec = {"conditions": {"solvent": "tetrahydrofuran", "temperature_C": -78}}
    out = inherit_conditions(dict(rec), {}, gv, None, source_text="[a] Reactions run in THF at -78 °C")
    assert out["conditions"]["solvent"] == "tetrahydrofuran" and out["conditions"]["temperature_C"] == -78


# ── 2026-09-20 round 3: tick marks, axis direction, dual-axis gate ──────────
@pytest.mark.parametrize("txt,expected", [("100-", 100.0), ("-20-", -20.0), ("80_", 80.0), ("０", 0.0), ("10-", None), ("10°", None), ("60–", 60.0)])
def test_parse_tick_text_strips_tick_marks(txt, expected):
    from src.extraction.figure.axis_fit import parse_tick_text
    assert parse_tick_text(txt) == expected


def test_best_monotonic_keeps_increasing_axis():
    from src.extraction.figure.axis_fit import best_monotonic_subsequence
    cands = [[0, v, str(v), 12, y] for v, y in ((50, 25), (55, 103), (60, 179), (65, 256), (70, 337), (75, 411))]
    kept, direction = best_monotonic_subsequence(cands, pixel_idx=4)
    assert len(kept) == 6 and direction == "increasing"
    dec = [[0, v, str(v), 12, y] for v, y in ((80, 96), (40, 166), (0, 234))]
    kept, direction = best_monotonic_subsequence(dec, pixel_idx=4)
    assert len(kept) == 3 and direction == "decreasing"


def test_decide_dual_axis_needs_three_right_ticks():
    from src.extraction.figure.axis_fit import decide_dual_axis
    assert decide_dual_axis(4, 3) and not decide_dual_axis(4, 2) and not decide_dual_axis(0, 5)


def test_value_boxes_label_at_most_one_point():
    from src.extraction.figure.axis_fit import match_labels_to_points
    pts = [(100, 100), (130, 100), (160, 100), (400, 400)]
    boxes = [(112, 90), (500, 500)]           # one stray legend number near three points
    out = match_labels_to_points(pts, boxes, radius=120.0)
    assert out == {0: 0}                      # nearest point only; box 1 is too far from everything
    # Dense heatmap cells: every point keeps its own label.
    pts = [(x, 50) for x in range(0, 500, 50)]
    boxes = [(x + 5, 40) for x in range(0, 500, 50)]
    out = match_labels_to_points(pts, boxes)
    assert out == {i: i for i in range(10)}


def test_value_boxes_follow_the_figure_offset_convention():
    from src.extraction.figure.axis_fit import match_labels_to_points
    # 3 rows of markers 40 px apart; labels printed 22 px below each marker
    # (i.e. nearer to the marker BELOW than to their own marker).
    pts = [(x, y) for y in (100, 140, 180) for x in (100, 150, 200)]
    boxes = [(x + 6, y + 22) for (x, y) in pts if y < 180] + [(x + 6, 202) for x in (100, 150, 200)]
    out = match_labels_to_points(pts, boxes)
    assert out == {i: i for i in range(9)}
