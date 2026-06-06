"""Source discovery — walk the intermediate dir, return one ``SourcePacket``
per figure / table that GlobalAssembly should ask the LLM about.

Pure I/O; no LLM calls.  Keeps GlobalAssembly orchestrator-thin and lets
``PerSourceAssembler`` consume an already-prepared list of packets.

Output ordering is deterministic — tables first (matches prior log
behaviour) then figures by ``(page_number, instance_index)``.  This
matters because the assembler sorts records by ``source_table_or_figure``
at the end and you want a stable cross-PDF row order.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.adjudication.pdf_parser import extract_text_window

logger = logging.getLogger(__name__)


class SourcePacket(BaseModel):
    """Everything PerSourceAssembler needs to build a per-source LLM prompt."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    source_id: str
    source_type: Literal["figure", "table"]
    human_label: str               # e.g. "Figure 3", "Table 1"; for display + tagging
    evidence: Dict[str, Any]       # raw evidence JSON (caption, raw_data, csv_path, ...)
    local_vars: Optional[Dict[str, Any]] = None
    csv_content: str = ""          # table only — full CSV text
    text_window: str = ""          # dual-anchor 8KB paper excerpt
    # Provenance / debug aids:
    evidence_path: str = ""
    local_vars_path: str = ""


def _page_index(source_id: str) -> int:
    """Parse ``page_{N}`` out of a source_id for ordering."""
    m = re.search(r"page_(\d+)", source_id)
    return int(m.group(1)) if m else 9_999


def _instance_index(source_id: str) -> int:
    """Parse the instance index (the ``_{N}`` after figure/table) for tie-break."""
    m = re.search(r"(?:figure|table)_(\d+)", source_id)
    return int(m.group(1)) if m else 0


def _human_label_for_figure(source_id: str, evidence: Dict[str, Any], local_vars: Optional[Dict[str, Any]]) -> str:
    """Best-effort human label for a figure source.

    Order: local_vars reaction_context (first 80 chars) → first chart_text →
    fallback "Figure (page {N}, instance {M})".
    """
    if local_vars:
        ctx = local_vars.get("reaction_context") or ""
        if ctx:
            return f"Figure (page {_page_index(source_id)}) — {ctx[:80].strip()}"
    chart_texts = evidence.get("text_evidence", {}).get("chart_text", [])
    if chart_texts and isinstance(chart_texts[0], dict):
        t = chart_texts[0].get("text", "")
        if t:
            return f"Figure (page {_page_index(source_id)}) — {t[:80].strip()}"
    return f"Figure (page {_page_index(source_id)}, instance {_instance_index(source_id)})"


def _human_label_for_table(source_id: str, evidence: Dict[str, Any]) -> str:
    caption = evidence.get("caption_text", "") or ""
    if caption:
        return f"Table (page {_page_index(source_id)}) — {caption[:80].strip()}"
    return f"Table (page {_page_index(source_id)})"


def _load_local_vars(out_dir: str, source_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    path = os.path.join(out_dir, f"{source_id}_local_vars.json")
    if not os.path.exists(path):
        return None, ""
    try:
        return json.load(open(path)), path
    except Exception as exc:
        logger.warning("source_discovery local_vars load failed source=%s exc=%s", source_id, exc)
        return None, path


def _read_csv(csv_path: Optional[str]) -> str:
    if not csv_path or not os.path.exists(csv_path):
        return ""
    try:
        with open(csv_path) as f:
            return f.read()
    except Exception as exc:
        logger.warning("source_discovery csv read failed path=%s exc=%s", csv_path, exc)
        return ""


def _figure_anchor_keywords(evidence: Dict[str, Any]) -> Tuple[str, ...]:
    """Pull a few caption fragments out of the figure evidence so the
    text-window primary anchor can find the paper's prose about this figure
    even when the source_id-based ``figure N`` keyword is wrong.
    """
    out: List[str] = []
    chart_texts = evidence.get("text_evidence", {}).get("chart_text", [])
    for item in chart_texts:
        if isinstance(item, dict):
            t = item.get("text", "")
            if t and len(t) >= 5:
                # Use the first 60 chars as one anchor — captions are usually short.
                out.append(t[:60].lower())
    return tuple(out[:3])


def _table_anchor_keywords(evidence: Dict[str, Any]) -> Tuple[str, ...]:
    caption = evidence.get("caption_text", "") or ""
    if caption and len(caption) >= 5:
        return (caption[:60].lower(),)
    return ()


def discover(intermediate_dir: str, basename: str, paper_text: str) -> List[SourcePacket]:
    """Walk the per-PDF intermediate dir; return one packet per relevant source.

    ``intermediate_dir`` is the parent (e.g. ``data/intermediate``);
    sources live under ``intermediate_dir/{basename}/{macro_cleaned,tables}/``.
    """
    pdf_root = os.path.join(intermediate_dir, basename)
    local_vars_dir = os.path.join(pdf_root, "local_vars")
    packets: List[SourcePacket] = []

    # --- Tables: nested layout ``tables/{table_basename}/{...}_evidence.json``
    for ev_path in sorted(glob.glob(os.path.join(pdf_root, "tables", "**", "*_evidence.json"), recursive=True)):
        try:
            evidence = json.load(open(ev_path))
        except Exception as exc:
            logger.warning("source_discovery table evidence load failed path=%s exc=%s", ev_path, exc)
            continue
        source_id = os.path.basename(os.path.dirname(ev_path))
        if not evidence.get("is_relevant", True):
            logger.info("source_discovery skip irrelevant table source=%s", source_id)
            continue
        local_vars, lv_path = _load_local_vars(local_vars_dir, source_id)
        csv_content = _read_csv(evidence.get("csv_path"))
        text_window = extract_text_window(
            paper_text, source_id, "table",
            primary_size=6000, experimental_size=2000,
            extra_anchor_keywords=_table_anchor_keywords(evidence),
        )
        packets.append(SourcePacket(
            source_id=source_id, source_type="table",
            human_label=_human_label_for_table(source_id, evidence),
            evidence=evidence, local_vars=local_vars,
            csv_content=csv_content, text_window=text_window,
            evidence_path=ev_path, local_vars_path=lv_path,
        ))

    # --- Figures: flat layout ``macro_cleaned/{figure_id}_evidence.json``
    for ev_path in sorted(glob.glob(os.path.join(pdf_root, "macro_cleaned", "*_evidence.json"))):
        try:
            evidence = json.load(open(ev_path))
        except Exception as exc:
            logger.warning("source_discovery figure evidence load failed path=%s exc=%s", ev_path, exc)
            continue
        source_id = os.path.basename(ev_path).replace("_evidence.json", "")
        if not evidence.get("is_relevant", True):
            logger.info("source_discovery skip irrelevant figure source=%s", source_id)
            continue
        local_vars, lv_path = _load_local_vars(local_vars_dir, source_id)
        text_window = extract_text_window(
            paper_text, source_id, "figure",
            primary_size=6000, experimental_size=2000,
            extra_anchor_keywords=_figure_anchor_keywords(evidence),
        )
        packets.append(SourcePacket(
            source_id=source_id, source_type="figure",
            human_label=_human_label_for_figure(source_id, evidence, local_vars),
            evidence=evidence, local_vars=local_vars,
            csv_content="", text_window=text_window,
            evidence_path=ev_path, local_vars_path=lv_path,
        ))

    # Deterministic order: tables first (already by glob sort), then figures by page+instance.
    table_packets = [p for p in packets if p.source_type == "table"]
    figure_packets = sorted(
        (p for p in packets if p.source_type == "figure"),
        key=lambda p: (_page_index(p.source_id), _instance_index(p.source_id)),
    )
    return table_packets + figure_packets
