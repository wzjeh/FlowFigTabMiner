"""Pre-filter: an unreadable text layer is 'cannot judge', not 'irrelevant'."""
from src.preprocessing import paper_filter
from src.preprocessing.paper_filter import text_layer_readable


def test_text_layer_readable():
    assert text_layer_readable("We report a continuous flow microreactor process for the nitration of toluene. " * 8)
    garbled = '!""# $% "" &    $ &    $ &  ( (   $%&  ' * 300
    assert not text_layer_readable(garbled)
    # the publisher watermark is real text but does not make the paper readable
    assert not text_layer_readable(garbled + " Downloaded from https://onlinelibrary.wiley.com See the Terms and Conditions " * 3)
    assert not text_layer_readable("")


def test_unreadable_text_layer_passes_the_filter(tmp_path, monkeypatch):
    import sys, types

    class _Page:
        def get_textpage(self):
            return types.SimpleNamespace(get_text_range=lambda: '!""# $% "" & $ & ( ( $%& ' * 200)

    class _Doc(list):
        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "pypdfium2", types.SimpleNamespace(PdfDocument=lambda p: _Doc([_Page(), _Page(), _Page()])))
    pdf = tmp_path / "garbled.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fn = next(getattr(paper_filter, n) for n in ("is_flow_chemistry_paper", "check_paper", "filter_paper", "is_relevant_paper") if hasattr(paper_filter, n))
    res = fn(str(pdf))
    assert res["is_relevant"] is True and "unreadable" in res["reason"]


def test_yoshida_microflow_wording_passes_the_prefilter():
    """Nagaki 2009 (BJOC) says 'integrated microflow systems' and never 'microreactor' or 'continuous flow';
    it was skipped as irrelevant until the wording joined keywords.yaml."""
    import yaml
    inc = [k.lower() for k in yaml.safe_load(open("keywords.yaml"))["paper_filter"]["flow_chemistry_include"]]
    text = "synthesis of unsymmetrically substituted biaryls via sequential lithiation using integrated microflow systems"
    assert any(k in text for k in inc)


def test_intermediate_dirs_with_brackets_are_still_discovered(tmp_path):
    """glob treats '[b]' in a paper name as a character class: Asai 2011 'Benzo[b]thiophen' had 5 crops saved and
    0 figures processed. Every glob on a paper-derived directory must escape the directory part."""
    import glob, os, json
    from src.adjudication.source_discovery import discover
    root = tmp_path / "Asai 2011 - Benzo[b]thiophen-3-yl"; (root / "macro_cleaned").mkdir(parents=True); (root / "local_vars").mkdir()
    json.dump({"is_relevant": True, "meta": {}, "text_evidence": {}, "raw_data": [{"X": 1.0, "Y_Left": 50.0}]}, open(root / "macro_cleaned" / "page_2_figure_0_t0_evidence.json", "w"))
    json.dump({"figure_type": "scatter"}, open(root / "local_vars" / "page_2_figure_0_t0_local_vars.json", "w"))
    assert glob.glob(os.path.join(str(root), "macro_cleaned", "*.json")) == []          # the trap
    assert [p.source_id for p in discover(str(tmp_path), root.name, "text")] == ["page_2_figure_0_t0"]
