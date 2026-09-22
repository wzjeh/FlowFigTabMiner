"""A figure whose template only describes its product gets one naming call."""
import json

from src.adjudication.per_source_prompts import CommonPreamble
from src.adjudication.product_namer import ProductNamer, is_generic_product
from src.adjudication.source_discovery import SourcePacket
from src.llm.config import LLMConfig
from src.llm.types import LLMResponse


def test_generic_means_no_structure_no_label_and_a_descriptive_name():
    assert is_generic_product("trapped product with benzaldehyde", None, None)
    assert is_generic_product("benzyllithiums", None, None)
    assert is_generic_product("tridecafluorohexylstannane derivative", None, None)
    assert is_generic_product("", None, None) and is_generic_product(None, None, None)
    assert not is_generic_product("product 3", "3", None)                     # a label is a handle
    assert not is_generic_product("alkyl benzoates", None, "*OC(=O)c1ccccc1")  # a structure, even generic, is not renamed here
    assert not is_generic_product("tributyl(tridecafluorohexyl)stannane", None, None)


class _LLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def chat(self, messages, cfg):
        self.calls += 1
        return LLMResponse(text=self.reply, model="stub", tokens_in=1, tokens_out=1, latency_ms=1.0)


def _packet():
    return SourcePacket(source_id="page_3_figure_0_t0", source_type="figure", human_label="Figure 2",
                        evidence={"meta": {"caption_pdf": "Figure 2. Yield of the stannane."}, "text_evidence": {}},
                        local_vars={"reaction_context": "tridecafluorohexyl iodide + MeLi, then chlorotributylstannane",
                                    "fixed_conditions": {"notes": "quenched with Bu3SnCl"}},
                        text_window="(text)")


def _tpl(name="tridecafluorohexylstannane derivative"):
    return {"record_template": {"reactant1_name": "tridecafluorohexyl iodide", "reactant2_name": "MeLi", "product_name": name},
            "axis_map": {}, "series_map": {}}


def test_generic_product_is_named_and_the_old_name_kept(tmp_path):
    llm = _LLM(json.dumps({"product_name": "tributyl(tridecafluorohexyl)stannane", "product_label": "3", "basis": "text says"}))
    out = ProductNamer(llm, LLMConfig(provider="gemini", model="x", temperature=0.0)).refine(_tpl(), _packet(), CommonPreamble.build({}, ""), str(tmp_path))
    rt = out["record_template"]
    assert rt["product_name"] == "tributyl(tridecafluorohexyl)stannane" and rt["product_name_generic"] == "tridecafluorohexylstannane derivative"
    assert rt["product_label"] == "3" and llm.calls == 1 and (tmp_path / "page_3_figure_0_t0_product_raw.txt").exists()


def test_specific_product_is_left_alone_without_a_call():
    llm = _LLM("{}")
    tpl = _tpl("tributyl(tridecafluorohexyl)stannane")
    assert ProductNamer(llm, LLMConfig(provider="gemini", model="x", temperature=0.0)).refine(tpl, _packet(), CommonPreamble.build({}, "")) is tpl
    assert llm.calls == 0


def test_generic_or_null_or_broken_answers_change_nothing():
    cfg = LLMConfig(provider="gemini", model="x", temperature=0.0)
    for reply in (json.dumps({"product_name": None, "basis": "not determined"}),
                  json.dumps({"product_name": "stannylated product", "basis": "x"}),
                  "not json at all"):
        tpl = _tpl()
        assert ProductNamer(_LLM(reply), cfg).refine(tpl, _packet(), CommonPreamble.build({}, "")) == _tpl()


def test_a_name_no_database_knows_counts_as_generic():
    assert is_generic_product("tridecafluorohexylstannane", None, None, resolves=lambda n: False)
    assert not is_generic_product("tridecafluorohexylstannane", None, None, resolves=lambda n: True)
    assert not is_generic_product("tridecafluorohexylstannane", None, None)          # no resolver: name is trusted
