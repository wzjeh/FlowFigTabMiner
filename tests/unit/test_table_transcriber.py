"""VLM table transcriber: grid normalisation, hygiene, graceful failure."""
import cv2
import numpy as np
import pytest

from src.extraction.table.table_vlm import TableTranscriber, TableTranscriptionResponse, normalize_grid
from src.llm.config import VLMConfig
from src.llm.types import LLMResponse


class _Stub:
    name = "stub"

    def __init__(self, payload=None, exc=None):
        self.payload, self.exc = payload, exc

    def inspect(self, image, system_prompt, user_prompt, cfg, response_schema=None):
        if self.exc:
            raise self.exc
        assert response_schema in (None, TableTranscriptionResponse)
        return LLMResponse(text="{}", model="stub", tokens_in=100, tokens_out=20, latency_ms=3.0, finish_reason="STOP"), self.payload


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "page_1_table_0.png"
    cv2.imwrite(str(p), np.full((40, 60, 3), 255, dtype=np.uint8))
    return p


_CFG = VLMConfig(provider="gemini", model="x", temperature=0.0)


def test_normalize_grid_pads_trims_and_cleans():
    hdr, dat, n, notes = normalize_grid([["Entry", "Yield"]],
                                        [["1", "93\x08"], ["2"], ["3", "71", "extra"], ["", ""]])
    assert n == 2 and hdr == [["Entry", "Yield"]]
    assert dat == [["1", "93"], ["2", ""], ["3", "71"]]          # blank row dropped
    assert any("padded" in x for x in notes) and any("trimmed" in x for x in notes)


def test_transcribe_happy_path(img):
    tr = TableTranscriber(_Stub({"caption": "Table 1. Test", "header_rows": [["Entry", "Product", "Yield"]],
                                 "data_rows": [["1", "[STRUCTURE]", "93"], ["2", "[STRUCTURE]", "20"]],
                                 "footnotes": "[a] GC yield.", "scheme_conditions": None}), _CFG).transcribe(img)
    assert tr.ok and tr.n_cols == 3 and len(tr.data_rows) == 2 and tr.header_rows == [["Entry", "Product", "Yield"]]
    assert tr.caption == "Table 1. Test" and tr.footnotes == "[a] GC yield." and tr.scheme_conditions is None
    assert tr.model == "stub" and tr.tokens_in == 100 and tr.notes is None


def test_transcribe_empty_grid_and_failure(img):
    tr = TableTranscriber(_Stub({"header_rows": [["A"]], "data_rows": []}), _CFG).transcribe(img)
    assert not tr.ok and "empty grid" in tr.notes
    tr = TableTranscriber(_Stub(exc=RuntimeError("boom")), _CFG).transcribe(img)
    assert not tr.ok and "boom" in tr.notes and tr.data_rows == []


def test_transcribe_rejects_schema_echo_caption(img):
    tr = TableTranscriber(_Stub({"caption": "legend_series_names_null_null", "data_rows": [["1", "2"]]}), _CFG).transcribe(img)
    assert tr.ok and tr.caption is None


class _Seq(_Stub):
    """One payload per call (attempt 1, attempt 2 …)."""

    def __init__(self, payloads):
        super().__init__()
        self.payloads, self.temps = list(payloads), []

    def inspect(self, image, system_prompt, user_prompt, cfg, response_schema=None):
        self.temps.append(cfg.temperature)
        self.payload = self.payloads.pop(0)
        return super().inspect(image, system_prompt, user_prompt, cfg, response_schema)


def test_single_column_grid_is_rerolled_once(img):
    one_col = {"header_rows": [["T1"], ["T2"], ["Yield"]], "data_rows": [["-78"], ["-78"], ["84"]]}
    good = {"header_rows": [["T1", "T2", "Yield"]], "data_rows": [["-78", "-78", "84"]]}
    stub = _Seq([one_col, good])
    tr = TableTranscriber(vlm=stub, cfg=_CFG).transcribe(img)
    assert tr.ok and tr.n_cols == 3 and tr.data_rows == [["-78", "-78", "84"]] and stub.temps == [0.0, 0.0]
    # attempt 2 no better → keep attempt 1 and say so
    stub = _Seq([one_col, one_col, one_col])
    tr = TableTranscriber(vlm=stub, cfg=_CFG).transcribe(img)
    assert tr.ok and tr.n_cols == 1 and "single-column" in (tr.notes or "") and stub.temps == [0.0, 0.0, 0.4]
    # a genuine one-column list of two rows is not re-rolled
    stub = _Seq([{"header_rows": [["Item"]], "data_rows": [["a"]]}])
    tr = TableTranscriber(vlm=stub, cfg=_CFG).transcribe(img)
    assert tr.ok and tr.n_cols == 1 and stub.temps == [0.0]


def test_caption_and_footnotes_are_one_line(img):
    payload = {"caption": "Table 5. Br\nLi exchange", "footnotes": "[a] GC.\n[b] NMR.", "header_rows": [["a", "b"]], "data_rows": [["1", "2"]]}
    tr = TableTranscriber(vlm=_Stub(payload), cfg=_CFG).transcribe(img)
    assert tr.caption == "Table 5. Br Li exchange" and tr.footnotes == "[a] GC. [b] NMR."
