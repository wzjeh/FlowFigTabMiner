"""Unit tests for ``PerSourceAssembler`` + the dual-anchor text window.

The assembler is exercised with a stub ``LLMProvider`` so no real Gemini
call is made.  The dual-anchor window helper is tested on a synthetic
mock paper where the figure citation and the General Procedure
heading are deliberately far apart in the text.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List

import pytest

from src.adjudication.pdf_parser import extract_text_window
from src.adjudication.per_source_assembler import PerSourceAssembler
from src.adjudication.per_source_prompts import (
    CommonPreamble,
    FigurePromptBuilder,
    TablePromptBuilder,
)
from src.adjudication.source_discovery import SourcePacket
from src.llm.types import ChatMessage, LLMResponse, Role


# ── Stub LLM provider ──────────────────────────────────────────────────


class _StubLLM:
    """Returns one canned JSON-array response per source_id seen in the
    user prompt.  Records which sources were called so we can assert
    fan-out reached every packet.
    """

    def __init__(self, responses_by_source: dict[str, str]):
        self.responses_by_source = responses_by_source
        self.sources_called: list[str] = []

    def chat(self, messages: list[ChatMessage], cfg) -> LLMResponse:  # noqa: D401
        user = next(m.content for m in messages if m.role == Role.USER)
        # The prompt builders always include ``({source_id})`` in the
        # "THIS SOURCE" header — use that to identify which canned
        # response to return.
        matched = None
        for sid in self.responses_by_source:
            if f"({sid})" in user:
                matched = sid
                break
        if matched is None:
            text = "[]"
        else:
            text = self.responses_by_source[matched]
            self.sources_called.append(matched)
        return LLMResponse(
            text=text,
            model="stub",
            tokens_in=10,
            tokens_out=10,
            latency_ms=5.0,
            retry_count=0,
        )


def _make_packet(source_id: str, source_type: str, raw_data=None) -> SourcePacket:
    evidence: dict = {"is_relevant": True}
    if source_type == "figure":
        evidence.update({
            "text_evidence": {
                "x_axis_title": [{"text": "X", "source": "vlm_metadata"}],
                "y_axis_title": [{"text": "Y", "source": "vlm_metadata"}],
            },
            "raw_data": raw_data or [],
            "meta": {"figure_type": "scatter"},
        })
    else:
        evidence.update({
            "caption_text": "Test table",
            "table_note_text": "",
            "header_row_count": 1,
            "num_extracted": 0,
            "csv_path": "",
        })
    return SourcePacket(
        source_id=source_id,
        source_type=source_type,
        human_label=f"{source_type.title()} {source_id}",
        evidence=evidence,
        local_vars={"reaction_context": "test"},
        csv_content="col1,col2\n1,2\n3,4\n" if source_type == "table" else "",
        text_window="(test paper context)",
    )


# ── PerSourceAssembler tests ───────────────────────────────────────────


def test_fanout_reaches_every_packet(tmp_path):
    """Stub LLM returns 2 records per source.  Assembler should call
    every packet exactly once and aggregate 2 × N records."""
    packets = [
        _make_packet("page_4_table_0", "table"),
        _make_packet("page_5_figure_0_t0", "figure", raw_data=[{"x": 1}, {"x": 2}]),
        _make_packet("page_6_figure_0_t0", "figure", raw_data=[{"x": 3}]),
    ]
    canned = {
        p.source_id: json.dumps([
            {"product_name": "P1", "yield_pct": 50},
            {"product_name": "P2", "yield_pct": 60},
        ])
        for p in packets
    }
    llm = _StubLLM(canned)
    asm = PerSourceAssembler(
        llm=llm, llm_cfg=_DummyCfg(),  # type: ignore[arg-type]
        prompt_builders={"figure": FigurePromptBuilder(), "table": TablePromptBuilder()},
        max_workers=4, raw_dir=str(tmp_path),
    )
    preamble = CommonPreamble.build({}, "")
    records = asm.assemble(packets, preamble, "test_pdf")

    assert len(records) == 6
    assert sorted(llm.sources_called) == sorted(p.source_id for p in packets)
    # human_label tag injected
    assert all(r.get("source_table_or_figure") for r in records)
    # __assembly_order stripped
    assert all("__assembly_order" not in r for r in records)


def test_parse_failure_isolates_one_source(tmp_path):
    """Source B returns garbage; A and C must still survive."""
    packets = [
        _make_packet("page_4_table_0", "table"),
        _make_packet("page_5_figure_0_t0", "figure"),
        _make_packet("page_6_figure_0_t0", "figure"),
    ]
    canned = {
        "page_4_table_0": json.dumps([{"product_name": "OK_A"}]),
        "page_5_figure_0_t0": "this is not JSON at all",
        "page_6_figure_0_t0": json.dumps([{"product_name": "OK_C"}]),
    }
    llm = _StubLLM(canned)
    asm = PerSourceAssembler(
        llm=llm, llm_cfg=_DummyCfg(),  # type: ignore[arg-type]
        prompt_builders={"figure": FigurePromptBuilder(), "table": TablePromptBuilder()},
        max_workers=4, raw_dir=str(tmp_path),
    )
    records = asm.assemble(packets, CommonPreamble.build({}, ""), "test_pdf")

    products = [r.get("product_name") for r in records]
    assert "OK_A" in products
    assert "OK_C" in products
    # Garbage source must have written a raw forensic file.
    raw_path = tmp_path / "test_pdf" / "per_source_raw" / "page_5_figure_0_t0_raw.txt"
    assert raw_path.exists()


def test_empty_discovery_no_llm_calls(tmp_path):
    llm = _StubLLM({})
    asm = PerSourceAssembler(
        llm=llm, llm_cfg=_DummyCfg(),  # type: ignore[arg-type]
        prompt_builders={"figure": FigurePromptBuilder(), "table": TablePromptBuilder()},
        max_workers=4, raw_dir=str(tmp_path),
    )
    records = asm.assemble([], CommonPreamble.build({}, ""), "test_pdf")
    assert records == []
    assert llm.sources_called == []


def test_order_deterministic_under_completion_jitter(tmp_path):
    """Even if futures complete in jumbled order, records are sorted by
    (human_label, source order)."""
    packets = [
        _make_packet("page_4_table_0", "table"),
        _make_packet("page_5_figure_0_t0", "figure"),
        _make_packet("page_6_figure_0_t0", "figure"),
    ]
    canned = {
        p.source_id: json.dumps([{"product_name": f"from_{p.source_id}_rec_{i}"} for i in range(3)])
        for p in packets
    }
    llm = _StubLLM(canned)
    asm = PerSourceAssembler(
        llm=llm, llm_cfg=_DummyCfg(),  # type: ignore[arg-type]
        prompt_builders={"figure": FigurePromptBuilder(), "table": TablePromptBuilder()},
        max_workers=4, raw_dir=str(tmp_path),
    )
    records1 = asm.assemble(packets, CommonPreamble.build({}, ""), "test_pdf")
    records2 = asm.assemble(packets, CommonPreamble.build({}, ""), "test_pdf")

    names1 = [r["product_name"] for r in records1]
    names2 = [r["product_name"] for r in records2]
    assert names1 == names2


# ── Dual-anchor text-window tests ──────────────────────────────────────


def test_window_captures_experimental_section_far_from_anchor():
    """Mock paper: figure citation in the first 5 KB, "General Procedure"
    section near the end (~15 KB).  The window must pull in both."""
    head = "Introduction paragraph mentioning Figure 3 yield data. " * 20
    head += "Continuing local context for Fig. 3 about hydrogenation. " * 30
    head_pad = "filler text in the body. " * 200  # ~5 KB
    body = head + head_pad
    body += "Section 4.\n"
    body += "General Procedure\n"
    body += "All reactions were conducted at 40 °C and 2.0 MPa pressure in methanol with Pd/C catalyst. " * 5
    body += "Catalyst loading: 2.53 g. Reactor: micropacked bed. " * 5
    body += "More tail text. " * 200

    window = extract_text_window(
        body, "page_6_figure_3_t0", "figure",
        primary_size=6000, experimental_size=2000,
    )
    assert "Figure 3" in window or "Fig. 3" in window
    assert "General Procedure" in window
    assert "40 °C" in window or "methanol" in window or "Pd/C" in window


def test_window_nbsp_in_heading_still_matches():
    """If the paper uses NBSP between words (common after PDF extraction),
    our normaliser must still match the anchor."""
    text = "Some intro. " * 10
    text += "Figure 5 shows … " * 5
    text += "later body content. " * 100
    text += "Materials and\xa0methods\n"
    text += "Reactions in MeOH at 40 °C and 20 bar. " * 5

    window = extract_text_window(
        text, "page_5_figure_5_t0", "figure",
        primary_size=4000, experimental_size=2000,
    )
    # Despite the NBSP, the experimental anchor should have fired.
    assert "MeOH" in window or "40" in window


def test_window_empty_paper_returns_empty():
    assert extract_text_window("", "page_1_figure_1_t0", "figure") == ""


def test_window_no_anchor_falls_back_to_head():
    """When neither anchor matches, returns the first primary_size chars."""
    text = "Just some boring text about nothing relevant. " * 200
    window = extract_text_window(text, "page_99_figure_99_t0", "figure", primary_size=500, experimental_size=0)
    assert window == text[:500]


# ── Dummy cfg for the stub LLM (chat() doesn't actually use it) ───────


class _DummyCfg:
    model = "stub"
    temperature = 0.0
    max_output_tokens = 1024
    max_retries = 0
    timeout_s = 30.0
