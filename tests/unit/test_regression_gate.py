"""Regression gate: per-source diff and batch health are pure functions."""
import json
import os

from scripts.regression_gate import diff_sources, field_counts, health, source_kind, summarise


def _rec(sid, label, **kw):
    r = {"__source_id": sid, "source_table_or_figure": label, "conditions": kw.pop("conditions", {})}
    r.update(kw)
    return r


def test_counts_and_kinds():
    recs = [_rec("page_2_table_0", "Table 1", yield_pct=90, product_smiles="CCO", has_outcome=True,
                 conditions={"temperature_C": -78, "residence_time_s": 0.5, "residence_time_2_s": 2.3, "solvent": "THF"}),
            _rec("page_3_figure_0_t0", "Figure 2", conditions={"reaction_time_s": 60})]
    c = field_counts(recs)
    assert (c["n"], c["T"], c["tR"], c["tR2"], c["t_batch"], c["yield"], c["psm"], c["outcome"], c["solvent"]) == (2, 1, 1, 1, 1, 1, 1, 1, 1)
    assert [source_kind(r) for r in recs] == ["table", "figure"]
    kinds, sources = summarise({"paperA": recs})
    assert kinds["table"]["n"] == 1 and kinds["figure"]["n"] == 1 and set(sources) == {("paperA", "Table 1"), ("paperA", "Figure 2")}


def test_diff_lists_drift_and_vanished_sources_only():
    old = {"p": [_rec("page_2_table_0", "Table 1", product_smiles="CCO") for _ in range(10)]
                + [_rec("page_4_table_0", "Table 2", yield_pct=5) for _ in range(4)]}
    new = {"p": [_rec("page_2_table_0", "Table 1", product_smiles=("CCO" if i < 2 else None), yield_pct=50) for i in range(10)]}
    rows = diff_sources(summarise(old)[1], summarise(new)[1])
    got = {(r["source"], r["field"], r["old"], r["new"], r["note"]) for r in rows}
    assert ("Table 1", "psm", 10, 2, "") in got and ("Table 1", "yield", 0, 10, "") in got
    assert ("Table 2", "n", 4, 0, "source vanished") in got
    assert diff_sources(summarise(old)[1], summarise(old)[1]) == []
    # a one-record wobble on a large source is not reported
    big_a = {"p": [_rec("t", "Table 9", yield_pct=1) for _ in range(50)]}
    big_b = {"p": [_rec("t", "Table 9", yield_pct=(1 if i else None)) for i in range(50)]}
    assert diff_sources(summarise(big_a)[1], summarise(big_b)[1]) == []


def test_health_reports_crashes_failed_sources_and_empty_outputs(tmp_path):
    inter, final = tmp_path / "inter", tmp_path / "final"
    for b, timing in (("ok_paper", {"status": "ok", "total": 100.0, "figure": 60.0}), ("crashed", None), ("empty", {"status": "ok", "total": 5.0})):
        os.makedirs(inter / b / "status")
        if timing:
            json.dump(timing, open(inter / b / "timing.json", "w"))
    json.dump({"source_id": "page_2_table_0", "stage": "table_vlm", "outcome": "failed", "reason": "ParseError"},
              open(inter / "ok_paper" / "status" / "page_2_table_0.json", "w"))
    json.dump({"source_id": "page_3_figure_0", "stage": "assembly", "outcome": "ok"}, open(inter / "ok_paper" / "status" / "page_3_figure_0.json", "w"))
    os.makedirs(final)
    json.dump([{"a": 1}, {"a": 2}], open(final / "ok_paper_normalized.json", "w"))
    json.dump([], open(final / "empty_normalized.json", "w"))
    h = health(["ok_paper", "crashed", "empty", "never_started"], str(inter), str(final))
    assert h["no_intermediate"] == ["never_started"] and [x["paper"] for x in h["not_ok"]] == ["crashed"]
    assert h["zero_records"] == ["empty"] and h["records"] == 2
    assert h["stage_outcomes"] == {"table_vlm:failed": 1, "assembly:ok": 1}
    assert h["failures"] == [{"paper": "ok_paper", "source": "page_2_table_0", "stage": "table_vlm", "reason": "ParseError"}]
