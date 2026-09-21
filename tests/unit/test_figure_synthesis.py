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


def test_rows_between_ticks_snap_to_cluster_levels():
    tpl = {"record_template": {"conditions": {}, "other_metrics": {}},
           "axis_map": {"X": "conditions.residence_time_s", "Y_Left": "conditions.temperature_C", "Y_Right/Data_Value": "yield_pct"}}
    raw = [{"Series": "Default", "X": 1.0, "Y_Left": v, "Y_Right/Data_Value": 1.0} for v in (-27.7, -27.5, -28.1, -0.3, 0.2)]
    recs = synthesize_records(tpl, raw, "Figure 1", {"chart_type": "heatmap", "y_left_ticks": [0, -20, -40, -60]})
    assert [r["conditions"]["temperature_C"] for r in recs] == [-28.0, -28.0, -28.0, 0.0, 0.0]


def test_numeric_identity_from_series_map_becomes_string():
    from src.adjudication.post_processor import normalize_chem_name
    tpl = {"record_template": {"conditions": {}, "other_metrics": {}},
           "axis_map": {"X": "conditions.residence_time_s", "Y_Left": "yield_pct", "Y_Right/Data_Value": None},
           "series_map": {"3": {"product_label": 3}}}
    rec = synthesize_records(tpl, [{"Series": "3", "X": 1.0, "Y_Left": 5.0}], "Figure 1", {"chart_type": "xy"})[0]
    assert rec["product_label"] == "3"
    assert isinstance(normalize_chem_name(3.0), str) and normalize_chem_name(None) is None and normalize_chem_name({"a": 1}) is None


def test_snap_levels_log_merges_column_jitter_only():
    from src.adjudication.figure_synthesis import snap_levels_log
    vals = [0.310, 0.322, 0.316, 1.05, 0.98, 3.2, None, 0.0]
    out = snap_levels_log(vals)
    assert out[0] == out[1] == out[2] and abs(out[0] - 0.316) < 0.01     # one column
    assert out[3] == out[4] and abs(out[3] - 1.014) < 0.02 and abs(out[5] - 3.2) < 1e-9 and out[6] is None and out[7] == 0.0


def test_heatmap_x_columns_are_snapped_in_synthesis():
    tpl = {"record_template": {"conditions": {}, "other_metrics": {}},
           "axis_map": {"X": "conditions.residence_time_s", "Y_Left": "conditions.temperature_C", "Y_Right/Data_Value": "yield_pct"}}
    raw = [{"Series": "Default", "X": x, "Y_Left": -78.0, "Y_Right/Data_Value": 1.0} for x in (0.310, 0.322, 0.316)]
    recs = synthesize_records(tpl, raw, "Figure 1", {"chart_type": "heatmap"})
    assert len({r["conditions"]["residence_time_s"] for r in recs}) == 1


def test_series_routes_a_shared_axis_to_different_outcome_fields():
    from src.adjudication.figure_synthesis import validate_template
    tpl = {"record_template": {"conditions": {}}, "axis_map": {"X": "conditions.pressure_bar", "Y_Left": None},
           "series_map": {"Conversion": {"conversion_pct": "Y_Left"}, "Selectivity": {"selectivity_pct": "Y_Left"}}}
    assert validate_template(tpl) is None
    raw = [{"X": 5, "Y_Left": 91.0, "Series": "Conversion"}, {"X": 5, "Y_Left": 98.5, "Series": "Selectivity"}]
    a, b = synthesize_records(tpl, raw, "Figure 3", {"chart_type": "xy"})
    assert a["conversion_pct"] == 91.0 and a.get("selectivity_pct") is None and b["selectivity_pct"] == 98.5
    assert "conversion_pct" in a["__data_fields"] and a["conditions"]["pressure_bar"] == 5
    # a template that maps nothing at all is still rejected
    assert validate_template({"record_template": {}, "axis_map": {"X": None}, "series_map": {"A": {"product_label": "3a"}}})


def test_unmapped_column_falls_back_instead_of_being_dropped():
    from src.adjudication.figure_synthesis import axis_fallback
    lv = {"axis_semantics": {"x_axis": {"maps_to_field": "conditions.pressure_bar"},
                             "y_left_axis": {"maps_to_field": "other_metrics.conversion_or_selectivity_pct"},
                             "data_value": {"maps_to_field": "not a path"}}}
    fb = axis_fallback(lv)
    assert fb == {"X": "conditions.pressure_bar", "Y_Left": "other_metrics.conversion_or_selectivity_pct"}
    tpl = {"record_template": {"conditions": {}}, "axis_map": {"X": "conditions.pressure_bar", "Y_Left": None, "Y_Right/Data_Value": None},
           "series_map": {"Conversion": {"conversion_pct": "Y_Left"}}}
    raw = [{"X": 5, "Y_Left": 91.0, "Y_Right/Data_Value": 0.4, "Series": "Default"}]      # legend matching failed
    rec = synthesize_records(tpl, raw, "Figure 3", {"chart_type": "xy"}, fb)[0]
    assert rec["other_metrics"] == {"conversion_or_selectivity_pct": 91.0, "data_value": 0.4} and rec.get("conversion_pct") is None


def test_series_route_by_axis_field_reference():
    tpl = {"record_template": {"conditions": {}},
           "axis_map": {"X": "other_metrics.catalyst_mol_pct", "Y_Left": "other_metrics.yield_or_conversion_pct"},
           "series_map": {"Yield of biphenyl": {"product_name": "biphenyl", "yield_pct": "other_metrics.yield_or_conversion_pct"},
                          "Conversion of 4-NBDT": {"conversion_pct": "other_metrics.yield_or_conversion_pct"}}}
    raw = [{"X": 1.0, "Y_Left": 86.8, "Series": "Conversion of 4-NBDT"}, {"X": 1.0, "Y_Left": 71.0, "Series": "Yield of biphenyl"},
           {"X": 2.0, "Y_Left": 50.0, "Series": "Default"}]
    a, b, c = synthesize_records(tpl, raw, "Figure 9", {"chart_type": "xy"})
    assert a["conversion_pct"] == 86.8 and "yield_or_conversion_pct" not in a["other_metrics"]
    assert b["yield_pct"] == 71.0 and b["product_name"] == "biphenyl"
    assert c["other_metrics"]["yield_or_conversion_pct"] == 50.0 and c.get("yield_pct") is None      # unmatched series keeps the axis field
