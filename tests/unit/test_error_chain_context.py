"""Unit tests for the deterministic context layer (CaptionLocator, label-anchored
text windows, chart-fact enforcement, condition inheritance, status records)."""
import json
import os

import fitz
import pytest

from src.parsing.caption_locator import locate_contexts, load_context
from src.adjudication.pdf_parser import extract_text_window, _label_patterns
from src.adjudication.post_processor import inherit_conditions
from src.adjudication.local_vars_builder import LocalVarsBuilder
from src.adjudication.source_discovery import infer_legacy_facts
from src.pipeline.status import write_status


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_pdf(tmp_path):
    """One page: a 'figure' box with caption below, a 'table' with caption
    above, rows inside, and a footnote beneath."""
    pdf = tmp_path / "paper.pdf"
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    # figure area 50..300 x 80..300 ; caption below at y=310
    page.insert_text((60, 320), "Figure 2. Effects of temperature and residence time on the yield of 3.", fontsize=9)
    page.insert_text((60, 332), "Conditions: THF, 1.5 equiv BuLi.", fontsize=9)
    # table: caption at y=420, rows 440..500, footnote at 520
    page.insert_text((60, 420), "Table 1: Br/Li exchange of aryl bromides.[a]", fontsize=9)
    page.insert_text((60, 445), "Entry", fontsize=9); page.insert_text((200, 445), "Yield [%]", fontsize=9)
    page.insert_text((60, 460), "1", fontsize=9); page.insert_text((200, 460), "90", fontsize=9)
    page.insert_text((60, 475), "2", fontsize=9); page.insert_text((200, 475), "20 [b]", fontsize=9)
    page.insert_text((60, 520), "[a] All reactions were run at -78 C in THF. [b] GC yield.", fontsize=9)
    doc.save(str(pdf)); doc.close()
    layout = {"pdf": str(pdf), "sources": [
        {"source_id": "page_1_figure_0", "kind": "figure", "page": 1, "bbox_pt": [50, 80, 300, 300]},
        {"source_id": "page_1_table_0", "kind": "table", "page": 1, "bbox_pt": [50, 430, 400, 505]},
    ]}
    return pdf, layout


# ── CaptionLocator ──────────────────────────────────────────────────────────
def test_caption_locator_resolves_labels_captions_footnotes(synthetic_pdf, tmp_path):
    pdf, layout = synthetic_pdf
    ctx = locate_contexts(str(pdf), layout)
    fig, tab = ctx["page_1_figure_0"], ctx["page_1_table_0"]
    assert fig["label"] == "Figure 2" and fig["caption_source"] == "pdf_text"
    assert fig["caption"].startswith("Figure 2. Effects of temperature")
    assert tab["label"] == "Table 1"
    assert tab["footnote"].startswith("[a] All reactions were run at -78 C")
    # in-bbox text layer is row-clustered and excludes caption/footnote
    assert "1 | 90" in tab["inner_text"] and "2 | 20 [b]" in tab["inner_text"]
    assert "Table 1:" not in tab["inner_text"] and "[a] All" not in tab["inner_text"]


def test_load_context_strips_macro_suffix(tmp_path):
    ctx_dir = tmp_path / "context"; ctx_dir.mkdir()
    (ctx_dir / "page_2_figure_1_context.json").write_text(json.dumps({"label": "Figure 3"}))
    assert load_context(str(tmp_path), "page_2_figure_1_t0")["label"] == "Figure 3"
    assert load_context(str(tmp_path), "page_9_figure_9_t0") is None


# ── text window ─────────────────────────────────────────────────────────────
def test_label_patterns_do_not_match_longer_numbers():
    import re
    pat = _label_patterns("Figure 2")[0]
    assert re.search(pat, "as shown in fig. 2,") and re.search(pat, "figure2b")
    assert not re.search(pat, "figure 20") and not re.search(pat, "figure 12")


def test_text_window_collects_every_mention_and_caption():
    filler = "lorem ipsum " * 300           # ~3.6k chars between mentions
    text = ("Intro. " + filler + "First mention of Figure 2 here. " + filler
            + "Figure 2. Effects of temperature on yield (caption). " + filler
            + "Second mention (Fig. 2) later. " + filler + "General procedure: all runs in THF at -78 C. " + filler)
    win = extract_text_window(text, "page_3_figure_0_t0", "figure", label="Figure 2",
                              extra_anchor_keywords=("Figure 2. Effects of temperature",))
    assert "First mention of Figure 2" in win and "Second mention (Fig. 2)" in win
    assert "Effects of temperature on yield (caption)" in win
    assert "General procedure" in win               # experimental anchor kept
    assert len(win) <= 6000 + 2000 + 400            # separators only
    # legacy path (no label): first 6000 chars fallback still works
    legacy = extract_text_window(text, "page_3_figure_0_t0", "figure")
    assert legacy.startswith("Intro.")


# ── chart facts ─────────────────────────────────────────────────────────────
def test_enforce_chart_facts_fixes_swapped_heatmap_axes():
    ev = {"meta": {"facts": {"chart_type": "heatmap"}},
          "raw_data": [{"X": 0.1, "Y_Left": -78.0, "Y_Right/Data_Value": 33.0},
                       {"X": 5.0, "Y_Left": 0.0, "Y_Right/Data_Value": 4.0}]}
    llm = {"figure_type": "scatter", "axis_semantics": {
        "x_axis": {"raw_label": "", "semantic_meaning": "?", "maps_to_field": "conditions.temperature_C"},
        "y_left_axis": {"raw_label": "", "semantic_meaning": "?", "maps_to_field": "conditions.residence_time_s"},
        "y_right_axis": None, "data_value": None}}
    out = LocalVarsBuilder._enforce_chart_facts(llm, ev)
    ax = out["axis_semantics"]
    assert ax["x_axis"]["maps_to_field"] == "conditions.residence_time_s"
    assert ax["y_left_axis"]["maps_to_field"] == "conditions.temperature_C"
    assert ax["data_value"]["maps_to_field"] == "yield_pct"
    assert out["figure_type"] == "heatmap" and "enforced" in out["data_interpretation_notes"]


def test_enforce_chart_facts_leaves_xy_plots_alone():
    ev = {"meta": {"facts": {"chart_type": "xy"}}, "raw_data": []}
    llm = {"figure_type": "line", "axis_semantics": {"x_axis": {"maps_to_field": "conditions.residence_time_s"}}}
    assert LocalVarsBuilder._enforce_chart_facts(dict(llm), ev) == llm


def test_infer_legacy_facts_detects_heatmap_from_point_labels():
    ev = {"meta": {}, "raw_data": [{"Series": "Default", "X": 1, "Y_Left": -78, "Y_Right/Data_Value": 50}] * 4}
    out = infer_legacy_facts(ev)
    assert out["meta"]["facts"]["chart_type"] == "heatmap" and out["meta"]["figure_type"] == "heatmap"
    assert out["meta"]["facts"]["series_matched_ratio"] == 0.0
    ev2 = {"meta": {}, "raw_data": [{"Series": "-78 °C", "X": 1, "Y_Left": 50, "Y_Right/Data_Value": None}]}
    assert "chart_type" not in infer_legacy_facts(ev2)["meta"]["facts"]


# ── inheritance ─────────────────────────────────────────────────────────────
def test_inherit_conditions_fills_empty_only_with_provenance():
    rec = {"conditions": {"temperature_C": None, "solvent": "THF", "residence_time_s": None, "reactor_type": None}}
    local = {"fixed_conditions": {"temperature_C": -78, "solvent": "toluene",
                                  "residence_time_s": {"step1": 2.6, "step2": 94}}}   # dict → not inheritable
    gv = {"default_conditions": {
        "residence_time_s": {"value": 10, "quote": "tR = 10 s for all", "scope": "paper"},
        "reactor_type": {"value": "T-mixer", "quote": "", "scope": "paper"},           # no quote → ignored
        "pressure_bar": {"value": 5, "quote": "5 bar", "scope": "partial"},            # partial → ignored
    }}
    stats = {"source_local": 0, "paper_global": 0}
    out = inherit_conditions(rec, local, gv, stats)
    c, p = out["conditions"], out["conditions_provenance"]
    assert c["temperature_C"] == -78 and p["temperature_C"] == "source_local"
    assert c["solvent"] == "THF" and p["solvent"] == "llm_source"          # never overwritten
    assert c["residence_time_s"] == 10 and p["residence_time_s"] == "paper_global"
    assert c["reactor_type"] is None and c.get("pressure_bar") is None
    assert stats == {"source_local": 1, "paper_global": 1}


# ── status records ──────────────────────────────────────────────────────────
def test_write_status_creates_record(tmp_path):
    write_status(str(tmp_path), "page_1_table_0", "table_filter", "filtered", "rejected", conf=0.1)
    rec = json.load(open(tmp_path / "status" / "page_1_table_0.json"))
    assert rec["outcome"] == "filtered" and rec["conf"] == 0.1 and rec["stage"] == "table_filter"


# ── hygiene + quote validation ──────────────────────────────────────────────
def test_inherit_rejects_placeholders_and_strips_src_tags():
    rec = {"conditions": {"solvent": "THF [src=paddleocr]", "catalyst": "missing", "temperature_C": None}}
    local = {"fixed_conditions": {"temperature_C": "missing", "catalyst": "n/a"}}
    out = inherit_conditions(rec, local, {}, None)
    c = out["conditions"]
    assert c["solvent"] == "THF" and c["catalyst"] is None and c["temperature_C"] is None
    assert "temperature_C" not in out["conditions_provenance"]


def test_global_vars_quote_must_support_value():
    from src.adjudication.global_vars_builder import GlobalVarsBuilder
    res = GlobalVarsBuilder._validate({"default_conditions": {
        "temperature_C": {"value": -60, "quote": "reduced the residence time dramatically", "scope": "paper"},
        "solvent": {"value": "tetrahydrofuran", "quote": "0.10 M in THF", "scope": "paper"},
        "flow_rate_mL_min": {"value": 6.0, "quote": "flow rate: 6.0 mL min-1", "scope": "paper"},
        "residence_time_s": {"value": 94, "quote": "at 50 \u00b0C (tR2 = 94 s)", "scope": "paper"},
        "reactor_type": {"value": "flow microreactor", "quote": "using the integrated flow microreactor system", "scope": "paper"},
        "pressure_bar": {"value": 5, "quote": "", "scope": "paper"},
    }})
    dc = res["default_conditions"]
    assert dc["temperature_C"]["scope"] == "partial"          # quote does not state -60
    assert dc["solvent"]["scope"] == "paper"                  # alias THF accepted
    assert dc["flow_rate_mL_min"]["scope"] == "paper" and dc["residence_time_s"]["scope"] == "paper"
    assert dc["reactor_type"]["scope"] == "paper"
    assert dc["pressure_bar"]["value"] is None                # no quote -> no value
