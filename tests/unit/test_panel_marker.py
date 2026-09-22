"""Sub-panel letter -> caption segment -> one evidence line in the prompts.

Nagaki 2010 Figure 3 has panels (a)-(d), one ester each; the assembler wrote
"alkyl benzoates / 3" for all four because the letter never reached it."""
import copy
import json

from src.adjudication.local_vars_builder import LocalVarsBuilder
from src.adjudication.per_source_prompts import CommonPreamble, FigureTemplateBuilder
from src.adjudication.source_discovery import SourcePacket, apply_context_to_evidence, panel_caption, _human_label_for_figure
from src.adjudication.figure_synthesis import synthesize_records
from src.assembly.evidence_assembler import panel_letter

CAP = ("Figure 3. Temperature-residence time maps for the reaction of (a) tert-butyl p-bromobenzoate (1a), "
       "(b) isopropyl p-bromobenzoate (1b) and (c) methyl p-bromobenzoate (1c) with sBuLi.")


def test_panel_letter_normalises_ocr_markers():
    assert [panel_letter(t) for t in ["(a)", "b)", "C", "(ii)", " d ", "(a", "1a", "Figure", "", None]] == \
           ["a", "b", "c", "ii", "d", "a", None, None, None, None]


def test_panel_caption_returns_the_segment_of_one_letter():
    assert panel_caption(CAP, "a") == "tert-butyl p-bromobenzoate (1a)"
    assert panel_caption(CAP, "b") == "isopropyl p-bromobenzoate (1b)"
    assert panel_caption(CAP, "c") == "methyl p-bromobenzoate (1c) with sBuLi."
    assert panel_caption(CAP, "d") is None
    assert panel_caption("Figure 2. Yield of 3 versus residence time (R1) at -78 °C.", "a") is None   # "(R1)" is not a panel
    assert panel_caption("a) yield; b) conversion", "b") == "conversion"


def _evidence(letter=None):
    ev = {"is_relevant": True, "meta": {"figure_type": "heatmap"}, "raw_data": [{"X": 1.0, "Y_Left": 50.0, "Series": "1a"}],
          "text_evidence": {"legend_text": [], "x_axis_title": [], "y_axis_title": [], "chart_text": []}}
    if letter:
        ev["text_evidence"]["panel_marker"] = letter
    return ev


def test_evidence_and_human_label_carry_the_panel():
    ctx = {"label": "Figure 3", "caption": CAP, "caption_source": "pdf_text"}
    ev = apply_context_to_evidence(_evidence("b"), ctx, "figure")
    assert ev["panel_caption"] == "isopropyl p-bromobenzoate (1b)"
    assert _human_label_for_figure("page_3_figure_0_t1", ev, None, ctx).startswith("Figure 3b (p.3)")
    plain = apply_context_to_evidence(_evidence(), ctx, "figure")
    assert "panel_caption" not in plain and _human_label_for_figure("page_3_figure_0_t1", plain, None, ctx).startswith("Figure 3 (p.3)")


def test_prompts_gain_one_line_only_when_a_panel_is_known():
    ctx = {"label": "Figure 3", "caption": CAP, "caption_source": "pdf_text"}
    with_panel = apply_context_to_evidence(_evidence("b"), ctx, "figure")
    without = apply_context_to_evidence(_evidence(), ctx, "figure")
    b = LocalVarsBuilder(llm=None, llm_cfg=None)
    _, u1 = b._build_figure_prompts("page_3_figure_0_t1", with_panel, "(text)", context=ctx)
    _, u0 = b._build_figure_prompts("page_3_figure_0_t1", without, "(text)", context=ctx)
    line = "Panel (b) [src=caption]: isopropyl p-bromobenzoate (1b)\n"
    assert line in u1 and u1.replace(line, "") == u0
    pk = lambda e: SourcePacket(source_id="page_3_figure_0_t1", source_type="figure", human_label="Figure 3b", evidence=e,
                                local_vars={"reaction_context": "x"}, csv_content="", text_window="(text)")
    tb = FigureTemplateBuilder()
    _, t1 = tb.build(pk(with_panel), CommonPreamble.build({}, ""))
    _, t0 = tb.build(pk(without), CommonPreamble.build({}, ""))
    assert line in t1 and t1.replace(line, "") == t0


def test_synthesized_records_keep_the_legend_name_even_when_mapped():
    tpl = {"record_template": {"product_label": "3"}, "axis_map": {"X": "conditions.residence_time_s", "Y_Left": "yield_pct"},
           "series_map": {"1a": {"reactant1_name": "tert-butyl p-bromobenzoate"}}}
    raw = [{"X": 1.0, "Y_Left": 50.0, "Series": "1a"}, {"X": 2.0, "Y_Left": 60.0, "Series": "Default"}]
    recs = synthesize_records(tpl, raw, "Figure 3")
    assert recs[0]["other_metrics"].get("series") == "1a" and recs[0]["reactant1_name"] == "tert-butyl p-bromobenzoate"
    assert "series" not in recs[1].get("other_metrics", {})


def test_marker_with_its_own_text_feeds_the_panel_line():
    from src.assembly.evidence_assembler import panel_text
    from src.adjudication.source_discovery import panel_line
    assert (panel_letter("d) R = methyl"), panel_text("d) R = methyl")) == ("d", "R = methyl")
    assert (panel_letter("a) R= tert-butyl"), panel_text("a) R= tert-butyl")) == ("a", "R= tert-butyl")
    assert (panel_letter("d）R=methyl"), panel_text("d）R=methyl")) == ("d", "R=methyl")      # full-width paren from OCR
    assert panel_letter("a yield") is None and panel_text("(b)") is None
    ev = {"text_evidence": {"panel_marker": "d", "panel_text": "R = methyl"}, "panel_caption": "methyl ester (1d)"}
    assert panel_line(ev) == "Panel (d) [src=marker]: R = methyl | [src=caption]: methyl ester (1d)\n"
    assert panel_line({"text_evidence": {"panel_marker": "d"}}) == ""
    assert panel_line({"text_evidence": {}, "panel_caption": "x"}) == ""
