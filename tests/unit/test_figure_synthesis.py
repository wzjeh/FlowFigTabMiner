"""Design step D — deterministic figure-record synthesis."""
from src.adjudication.figure_synthesis import snap_levels, synthesize_records, validate_template


def test_snap_levels_clusters_heatmap_rows():
    vals = [-77.6, -77.7, -57.9, -0.02, 0.4, 19.8, None, -47.9]
    out = snap_levels(vals)
    assert out == [-78.0, -78.0, -58.0, 0.0, 0.0, 20.0, None, -48.0]


def test_validate_template_rejects_unmapped():
    assert validate_template({"record_template": {}, "axis_map": {"X": "nonsense"}}) is not None
    assert validate_template({"record_template": {}, "axis_map": {"X": "conditions.residence_time_s"}}) is None
    assert validate_template([]) is not None


def test_synthesize_heatmap_records_copies_numbers_and_snaps():
    tpl = {
        "record_template": {"product_name": "product 3", "reaction_class": "halogen-metal exchange",
                            "conditions": {"solvent": "tetrahydrofuran", "temperature_C": None},
                            "other_metrics": {}, "yield_pct": None},
        "axis_map": {"X": "conditions.residence_time_s", "Y_Left": "conditions.temperature_C",
                     "Y_Right/Data_Value": "yield_pct"},
        "series_map": {},
    }
    raw = [{"Series": "Default", "X": 0.0421, "Y_Left": -77.6, "Y_Right/Data_Value": 33.0},
           {"Series": "Default", "X": 8.79, "Y_Left": -0.02, "Y_Right/Data_Value": None}]
    recs = synthesize_records(tpl, raw, "Figure 2 (p.2)", {"chart_type": "heatmap"})
    assert len(recs) == 2
    a, b = recs
    assert a["conditions"]["temperature_C"] == -78.0 and a["conditions"]["residence_time_s"] == 0.0421 and a["yield_pct"] == 33.0
    assert b["conditions"]["temperature_C"] == 0.0 and b["yield_pct"] is None      # missing label stays null
    assert a["product_name"] == "product 3" and a["conditions"]["solvent"] == "tetrahydrofuran"
    assert a["source_table_or_figure"] == "Figure 2 (p.2)" and a["__synthesized"] is True
    assert a is not b and a["conditions"] is not b["conditions"]                  # deep copies


def test_synthesize_series_map_and_transforms():
    tpl = {
        "record_template": {"conditions": {}, "other_metrics": {}},
        "axis_map": {"X": "conditions.residence_time_s", "Y_Left": "yield_pct", "Y_Right/Data_Value": None},
        "axis_transforms": {"X": {"scale": 60, "offset": 0}},               # minutes → seconds
        "series_map": {"-78 °C": {"conditions.temperature_C": -78}, "3a": {"product_label": "3a", "bogus.path": 1}},
    }
    raw = [{"Series": "-78 °C", "X": 2.0, "Y_Left": 55.5}, {"Series": "3a", "X": 1.0, "Y_Left": 10.0},
           {"Series": "unmapped", "X": 1.0, "Y_Left": 1.0}]
    recs = synthesize_records(tpl, raw, "Figure 1", {"chart_type": "xy"})
    assert recs[0]["conditions"]["temperature_C"] == -78 and recs[0]["conditions"]["residence_time_s"] == 120.0
    assert recs[0]["yield_pct"] == 55.5
    assert recs[1]["product_label"] == "3a" and "bogus" not in recs[1]
    assert recs[2]["other_metrics"]["series"] == "unmapped"                    # unknown series kept as metric


def test_snap_to_ticks_and_nan_guard():
    from src.adjudication.figure_synthesis import snap_to_ticks
    assert snap_to_ticks([-76.6, -77.7, -30.0, 19.5, None], [-78, -58, -48, -28, 0, 20]) == [-78.0, -78.0, -28.0, 20.0, None]
    tpl = {"record_template": {"conditions": {}, "other_metrics": {}},
           "axis_map": {"X": "conditions.residence_time_s", "Y_Left": "conditions.temperature_C", "Y_Right/Data_Value": "yield_pct"}}
    raw = [{"Series": "Default", "X": 1.0, "Y_Left": -76.6, "Y_Right/Data_Value": float("nan")}]
    rec = synthesize_records(tpl, raw, "Figure 1", {"chart_type": "heatmap", "y_left_ticks": [-78, -58, 0]})[0]
    assert rec["conditions"]["temperature_C"] == -78.0 and rec.get("yield_pct") is None


def test_series_map_cannot_fabricate_outcomes():
    tpl = {"record_template": {"conditions": {}, "other_metrics": {}},
           "axis_map": {"X": "conditions.residence_time_s", "Y_Left": "conditions.temperature_C", "Y_Right/Data_Value": "yield_pct"},
           "series_map": {"<20%": {"yield_pct": 10, "conditions.catalyst": "Pd"}}}
    raw = [{"Series": "<20%", "X": 1.0, "Y_Left": -78.0, "Y_Right/Data_Value": None}]
    rec = synthesize_records(tpl, raw, "Figure 1", {"chart_type": "heatmap"})[0]
    assert rec.get("yield_pct") is None and rec["conditions"]["catalyst"] == "Pd"
