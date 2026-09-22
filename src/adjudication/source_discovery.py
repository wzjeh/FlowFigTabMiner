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
from src.parsing.caption_locator import load_context
from src.extraction.figure.axis_fit import is_grid_like
from src.pipeline.status import write_status

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
    context: Optional[Dict[str, Any]] = None   # CaptionLocator: label / caption / footnote (PDF text layer)
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


def _human_label_for_figure(source_id: str, evidence: Dict[str, Any], local_vars: Optional[Dict[str, Any]],
                            context: Optional[Dict[str, Any]] = None) -> str:
    """Best-effort human label for a figure source.

    Order: CaptionLocator label ("Figure 2 (p.2) — caption…") → local_vars
    reaction_context (first 80 chars) → first chart_text → fallback
    "Figure (page {N}, instance {M})".
    """
    if context and context.get("label"):
        cap = (context.get("caption") or "")[:80].strip()
        letter = (evidence.get("text_evidence") or {}).get("panel_marker") or ""
        return f"{context['label']}{letter} (p.{_page_index(source_id)})" + (f" — {cap}" if cap else "")
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


def _human_label_for_table(source_id: str, evidence: Dict[str, Any],
                           context: Optional[Dict[str, Any]] = None) -> str:
    if context and context.get("label"):
        cap = (context.get("caption") or "")[:80].strip()
        return f"{context['label']} (p.{_page_index(source_id)})" + (f" — {cap}" if cap else "")
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


def _skip_irrelevant_figures() -> bool:
    """Config gate for the figure keyword filter (``adjudication.skip_irrelevant_figures``).

    The filter is SOFT by default: a figure whose OCR text lacks result
    keywords is still assembled, it just carries ``is_relevant=False``.
    """
    try:
        from src.utils.config import load_config
        return bool(load_config().get("adjudication", {}).get("skip_irrelevant_figures", False))
    except Exception:
        return False


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


def apply_context_to_evidence(evidence: Dict[str, Any], context: Optional[Dict[str, Any]], source_type: str) -> Dict[str, Any]:
    """Return a copy of ``evidence`` with PDF-text caption/footnote applied.

    Tables: the VLM transcription's ``caption_text`` / ``table_note_text``
    stand; the PDF text layer's caption / footnote fill in only when empty.
    Figures: ``meta.label/caption_pdf/footnote_pdf`` are filled in when the
    evidence predates the CaptionLocator.  Provenance is kept in
    ``caption_source`` / ``note_source``.
    """
    if not context:
        return infer_legacy_facts(evidence) if source_type == "figure" else evidence
    ev = dict(evidence)
    if source_type == "table":
        # The transcriber read the printed caption / footnotes off the crop;
        # the PDF text layer (hyphenation markers, glued superscripts, Greek
        # letters lost to font mapping) only fills in when the crop showed none.
        if evidence.get("caption_text"):
            ev["caption_source"] = "vlm"
        elif context.get("caption_source") == "pdf_text" and context.get("caption"):
            ev["caption_text"] = context["caption"]
            ev["caption_source"] = "pdf_text"
        if evidence.get("table_note_text"):
            ev["note_source"] = "vlm"
        elif context.get("footnote"):
            ev["table_note_text"] = context["footnote"]
            ev["note_source"] = "pdf_text"
        ev["label"] = context.get("label")
        if context.get("inner_text"):
            ev["inner_text"] = context["inner_text"]
    else:
        meta = dict(ev.get("meta", {}) or {})
        if context.get("inner_text"):
            meta["inner_text"] = context["inner_text"]
        if not meta.get("label"):
            meta["label"] = context.get("label")
            meta["caption_pdf"] = context.get("caption", "")
            meta["footnote_pdf"] = context.get("footnote", "")
            meta["caption_source"] = context.get("caption_source", "missing")
        ev["meta"] = meta
        letter = (ev.get("text_evidence") or {}).get("panel_marker")
        seg = panel_caption(context.get("caption") or meta.get("caption_pdf") or "", letter) if letter else None
        if seg:
            ev["panel_caption"] = seg
    return infer_legacy_facts(ev) if source_type == "figure" else ev


_PANEL_SPLIT_RE = re.compile(r"(?<![A-Za-z0-9])\(?([a-h])\)\s*:?\s*", re.I)


def panel_caption(caption: str, letter: str) -> Optional[str]:
    """The part of a multi-panel caption that describes panel ``letter``:
    "(a) tert-butyl ester 1a and (b) isopropyl ester 1b" -> "tert-butyl ester 1a"
    for "a".  None when the caption has no such marker."""
    if not caption or not letter:
        return None
    marks = list(_PANEL_SPLIT_RE.finditer(caption))
    for i, m in enumerate(marks):
        if m.group(1).lower() != letter.lower():
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(caption)
        seg = caption[m.end():end].strip()
        seg = re.sub(r"[\s,;]*(?:and|or)?\s*$", "", seg, flags=re.I).strip(" ,;")
        return seg or None
    return None


def infer_legacy_facts(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a FIGURE evidence dict with ``meta.facts`` /
    ``meta.figure_type`` filled in when the packet predates the facts field.

    Only deterministic inferences: a chart whose points mostly carry a
    ``Y_Right/Data_Value`` label was processed in heatmap mode (that column
    is populated only by the point-label OCR path).
    """
    meta = dict(evidence.get("meta", {}) or {})
    raw = evidence.get("raw_data", []) or []
    if meta.get("facts"):
        # Facts exist but may predate the point-label heatmap signal: a chart
        # whose points mostly carry OCR'd labels is a value map regardless of
        # what the legend OCR said.
        f = dict(meta["facts"])
        n = len(raw)
        # The pipeline's own label count wins when it recorded one: on a chart with a
        # right Y axis EVERY point carries a Y_Right/Data_Value (the right-axis reading),
        # so counting that column would turn dual-axis scatter plots into heatmaps
        # (rb2_67 / rb2_38: conversion and selectivity filed as temperature).
        if "n_point_labels" in f:
            n_dv = int(f.get("n_point_labels") or 0)
        else:
            n_dv = sum(1 for r in raw if isinstance(r, dict) and r.get("Y_Right/Data_Value") is not None)
        if f.get("chart_type") != "heatmap" and n and n_dv / n >= 0.5 and is_grid_like(raw)["grid"]:
            f["chart_type"] = "heatmap"; f["heatmap_signal"] = "point_labels+grid(reclassified)"
            meta["facts"] = f; meta["figure_type"] = "heatmap"
            ev = dict(evidence); ev["meta"] = meta
            return ev
        return evidence
    n = len(raw)
    n_dv = sum(1 for r in raw if isinstance(r, dict) and r.get("Y_Right/Data_Value") is not None)
    facts: Dict[str, Any] = {"legacy_inferred": True, "n_points": n, "n_point_labels": n_dv,
                             "has_point_labels": n_dv > 0}
    if n and n_dv / n >= 0.5 and is_grid_like(raw)["grid"]:
        facts["chart_type"] = "heatmap"
        facts["x_scale"] = "log"
    series = {r.get("Series") for r in raw if isinstance(r, dict)}
    facts["series_matched_ratio"] = 0.0 if series <= {None, "", "Default"} else None
    meta["facts"] = facts
    if facts.get("chart_type"):
        meta["figure_type"] = facts["chart_type"]
    ev = dict(evidence)
    ev["meta"] = meta
    return ev


def _context_anchor_keywords(context: Optional[Dict[str, Any]]) -> Tuple[str, ...]:
    cap = (context or {}).get("caption") or ""
    return (cap[:60].lower(),) if len(cap) >= 5 else ()


def _table_anchor_keywords(evidence: Dict[str, Any]) -> Tuple[str, ...]:
    caption = evidence.get("caption_text", "") or ""
    if caption and len(caption) >= 5:
        return (caption[:60].lower(),)
    return ()


_RAW_VALUE_COLS = ("X", "Y_Left", "Y_Right/Data_Value")


def load_figure_evidence(path: str) -> Dict[str, Any]:
    """Load a figure evidence file with non-finite numbers in ``raw_data`` turned
    into None.  Partially labelled charts were written with pandas NaN for the
    unlabelled points; ``is not None`` then counted them as read labels and a
    scatter panel was reclassified as a heatmap (rb1_69 Figure 7a: concentrations
    filed as temperature)."""
    evidence = json.load(open(path))
    raw = evidence.get("raw_data")
    if isinstance(raw, list):
        evidence["raw_data"] = [
            {k: (None if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))) else v) for k, v in p.items()}
            if isinstance(p, dict) else p for p in raw]
    return evidence


def figure_value_problem(evidence: Dict[str, Any]) -> Optional[str]:
    """Why a figure has nothing to assemble, else None.  A figure whose axes
    could not be calibrated keeps only pixel positions (``note: CoordMapping
    Failed``); records built from it would be empty shells that inherit the
    paper's conditions and look like measurements."""
    raw = [p for p in (evidence.get("raw_data") or []) if isinstance(p, dict)]
    if not raw:
        return "no data points extracted"
    def _num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if not any(_num(p.get(c)) for p in raw for c in _RAW_VALUE_COLS):
        return f"coordinate mapping failed: {len(raw)} points carry pixel positions only"
    return None


def discover(intermediate_dir: str, basename: str, paper_text: str) -> List[SourcePacket]:
    """Walk the per-PDF intermediate dir; return one packet per relevant source.

    ``intermediate_dir`` is the parent (e.g. ``data/intermediate``);
    sources live under ``intermediate_dir/{basename}/{macro_cleaned,tables}/``.
    """
    pdf_root = os.path.join(intermediate_dir, basename)
    local_vars_dir = os.path.join(pdf_root, "local_vars")
    packets: List[SourcePacket] = []

    # --- Tables: nested layout ``tables/{table_basename}/{...}_evidence.json``
    for ev_path in sorted(glob.glob(os.path.join(glob.escape(pdf_root), "tables", "**", "*_evidence.json"), recursive=True)):
        try:
            evidence = json.load(open(ev_path))
        except Exception as exc:
            logger.warning("source_discovery table evidence load failed path=%s exc=%s", ev_path, exc)
            continue
        source_id = os.path.basename(os.path.dirname(ev_path))
        if str(evidence.get("parse_status", "ok")).startswith("failed"):
            logger.info("source_discovery skip failed table source=%s status=%s", source_id, evidence.get("parse_status"))
            continue
        if not evidence.get("is_relevant", True):
            logger.info("source_discovery skip irrelevant table source=%s", source_id)
            continue
        local_vars, lv_path = _load_local_vars(local_vars_dir, source_id)
        context = load_context(pdf_root, source_id)
        evidence = apply_context_to_evidence(evidence, context, "table")
        csv_content = _read_csv(evidence.get("csv_path"))
        text_window = extract_text_window(
            paper_text, source_id, "table",
            primary_size=6000, experimental_size=2000,
            extra_anchor_keywords=_context_anchor_keywords(context) or _table_anchor_keywords(evidence),
            label=(context or {}).get("label"),
        )
        packets.append(SourcePacket(
            source_id=source_id, source_type="table",
            human_label=_human_label_for_table(source_id, evidence, context),
            evidence=evidence, local_vars=local_vars,
            csv_content=csv_content, text_window=text_window,
            evidence_path=ev_path, local_vars_path=lv_path, context=context,
        ))

    # --- Figures: flat layout ``macro_cleaned/{figure_id}_evidence.json``
    for ev_path in sorted(glob.glob(os.path.join(glob.escape(pdf_root), "macro_cleaned", "*_evidence.json"))):
        try:
            evidence = load_figure_evidence(ev_path)
        except Exception as exc:
            logger.warning("source_discovery figure evidence load failed path=%s exc=%s", ev_path, exc)
            continue
        source_id = os.path.basename(ev_path).replace("_evidence.json", "")
        if not evidence.get("is_relevant", True):
            if _skip_irrelevant_figures():
                logger.info("source_discovery skip irrelevant figure source=%s", source_id)
                continue
            logger.info("source_discovery keep irrelevant figure (soft flag) source=%s", source_id)
        problem = figure_value_problem(evidence)
        if problem:
            logger.info("source_discovery skip figure without values source=%s reason=%s", source_id, problem)
            write_status(pdf_root, source_id, "coord_map", "failed", problem)
            continue
        local_vars, lv_path = _load_local_vars(local_vars_dir, source_id)
        context = load_context(pdf_root, source_id)
        evidence = apply_context_to_evidence(evidence, context, "figure")
        text_window = extract_text_window(
            paper_text, source_id, "figure",
            primary_size=6000, experimental_size=2000,
            extra_anchor_keywords=_context_anchor_keywords(context) or _figure_anchor_keywords(evidence),
            label=(context or {}).get("label"),
        )
        packets.append(SourcePacket(
            source_id=source_id, source_type="figure",
            human_label=_human_label_for_figure(source_id, evidence, local_vars, context),
            evidence=evidence, local_vars=local_vars,
            csv_content="", text_window=text_window,
            evidence_path=ev_path, local_vars_path=lv_path, context=context,
        ))

    # Deterministic order: tables first (already by glob sort), then figures by page+instance.
    table_packets = [p for p in packets if p.source_type == "table"]
    figure_packets = sorted(
        (p for p in packets if p.source_type == "figure"),
        key=lambda p: (_page_index(p.source_id), _instance_index(p.source_id)),
    )
    return table_packets + figure_packets
