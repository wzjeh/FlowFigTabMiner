"""CaptionLocator — resolve each TF-ID crop to its real caption / label / footnote
from the PDF *text layer*, using the crop's page geometry.

Why this exists
---------------
Downstream stages used to learn a figure's identity from two lossy paths:
a YOLO ``caption`` crop run through PaddleOCR (empty for ~96 % of figures,
truncated for tables: "ble 2: …"), and the crop *index* baked into the
source_id (``page_2_figure_0`` → "Figure 0"), which is not the paper's
figure number at all.  The PDF text layer, however, carries the caption
verbatim ("Figure 2. Effects of temperature and residence time …") on the
same page, and 91 % of crops in the corpus have one.  Given the crop's
bbox (``layout.json`` written by ``ActiveAreaDetector.save_crops`` or by
``scripts/backfill_layout.py``) the caption is found geometrically:

* figure / scheme crops → nearest caption block **below** the bbox with
  horizontal overlap (a caption block that intersects the bbox also counts —
  TF-ID boxes often include the caption pixels);
* table crops → nearest caption block **above** the bbox; table footnotes
  (``[a] …``, ``a) …``, ``Reaction conditions: …``) are the blocks directly
  **below** the bbox.

Output: ``{intermediate_dir}/context/{source_id}_context.json`` with the
true label ("Figure 2"), verbatim caption, footnote and a
``caption_source`` provenance tag.  Pure PyMuPDF, no ML.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from src.adjudication.pdf_parser import _normalize_text

# "Figure 2." / "Fig. 3" / "Table 1:" / "Scheme 4" / "Chart 1" / "Figure S3"
CAPTION_RE = re.compile(
    r"^\s*(?P<kind>Figure|Fig\.?|Table|Scheme|Chart)\s*(?P<num>S?\d+[a-z]?)\s*[.:|\-–—]?\s",
    re.I,
)
FOOTNOTE_RE = re.compile(
    r"^\s*(\[?[a-h]\]?[\s.)]|\(?[a-h]\)\s|[a-h]\s*[A-Z]|Conditions|Reaction conditions|General conditions|Yields?\b)",
)
_KIND_NORMAL = {"fig": "Figure", "fig.": "Figure", "figure": "Figure", "table": "Table",
                "scheme": "Scheme", "chart": "Chart"}

# Geometry tolerances (PDF points).
_MAX_GAP_PT = 160.0        # caption may sit this far from the crop edge
_FOOTNOTE_GAP_PT = 140.0   # table footnotes: directly beneath the body
_CONTINUATION_GAP_PT = 6.0 # a block is a caption continuation if this close


def _norm_kind(kind: str) -> str:
    return _KIND_NORMAL.get(kind.lower().rstrip(".") if kind.lower() != "fig." else "fig.", kind.title())


def _h_overlap(a: List[float], b: List[float]) -> float:
    """Horizontal overlap length between two [x0, y0, x1, y1] boxes."""
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


def _blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    out = []
    for b in page.get_text("blocks") or []:
        text = (b[4] if len(b) > 4 else "") or ""
        if not text.strip():
            continue
        out.append({"bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "text": _normalize_text(" ".join(text.splitlines()))})
    out.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return out


def _extend_caption(blocks: List[Dict[str, Any]], start_idx: int) -> str:
    """Join a caption block with immediate continuation blocks (same column,
    tiny vertical gap) — PyMuPDF sometimes splits one caption paragraph."""
    text = blocks[start_idx]["text"]
    cur = blocks[start_idx]["bbox"]
    for nb in blocks[start_idx + 1:start_idx + 4]:
        nbb = nb["bbox"]
        if CAPTION_RE.match(nb["text"]):
            break
        gap = nbb[1] - cur[3]
        same_col = abs(nbb[0] - cur[0]) < 12 and _h_overlap(cur, nbb) > 0
        if 0 <= gap <= _CONTINUATION_GAP_PT and same_col:
            text += " " + nb["text"]
            cur = nbb
        else:
            break
    return text


def _pick_caption(blocks: List[Dict[str, Any]], bbox: List[float], kind: str) -> Optional[Dict[str, Any]]:
    """Choose the caption block for one crop.  Returns dict(idx, kind, num, gap)."""
    cands = []
    for i, blk in enumerate(blocks):
        m = CAPTION_RE.match(blk["text"])
        if not m:
            continue
        bb = blk["bbox"]
        if _h_overlap(bb, bbox) <= 0:
            continue
        ck = _norm_kind(m.group("kind"))
        # vertical relation: negative gap = intersects the crop box
        below_gap = bb[1] - bbox[3]        # block top vs crop bottom
        above_gap = bbox[1] - bb[3]        # crop top vs block bottom
        intersects = not (bb[3] < bbox[1] or bb[1] > bbox[3])
        if kind == "table":
            prefer = ck == "Table"
            if intersects:
                gap = 0.0
            elif 0 <= above_gap <= _MAX_GAP_PT:
                gap = above_gap
            elif 0 <= below_gap <= _MAX_GAP_PT:
                gap = below_gap + 40.0   # captions below a table are rare; penalise
            else:
                continue
        else:
            prefer = ck != "Table"
            if intersects:
                gap = 0.0
            elif 0 <= below_gap <= _MAX_GAP_PT:
                gap = below_gap
            elif 0 <= above_gap <= _MAX_GAP_PT:
                gap = above_gap + 40.0   # figure caption above the figure: rare
            else:
                continue
        cands.append({"idx": i, "kind": ck, "num": m.group("num"), "gap": gap,
                      "rank": (0 if prefer else 1, gap)})
    if not cands:
        return None
    return min(cands, key=lambda c: c["rank"])


def _pick_footnote(blocks: List[Dict[str, Any]], bbox: List[float], skip_idx: Optional[int]) -> Tuple[str, List[List[float]]]:
    """Table footnotes ("[a] …", "a) …", "Reaction conditions: …").

    TF-ID boxes usually include the footnote pixels, so candidates may start
    inside the lower part of the bbox as well as just below it.  The first
    block must match ``FOOTNOTE_RE``; later blocks are accepted only if they
    also match or sit directly beneath an accepted block (continuation).
    """
    height = max(1.0, bbox[3] - bbox[1])
    parts: List[str] = []
    boxes: List[List[float]] = []
    last_bottom: Optional[float] = None
    for i, blk in enumerate(blocks):
        if i == skip_idx:
            continue
        bb = blk["bbox"]
        gap = bb[1] - bbox[3]
        if gap < -0.45 * height or gap > _FOOTNOTE_GAP_PT or _h_overlap(bb, bbox) <= 0:
            continue
        if CAPTION_RE.match(blk["text"]):
            break                    # reached the next float's caption
        if bb[3] - bb[1] > 0.6 * height:
            continue                 # a whole text column, not a footnote
        is_fn = bool(FOOTNOTE_RE.match(blk["text"]))
        is_cont = parts and last_bottom is not None and 0 <= bb[1] - last_bottom <= 4.0
        if is_fn or is_cont:
            parts.append(blk["text"])
            boxes.append(bb)
            last_bottom = bb[3]
        elif parts:
            break                    # footnote run ended
        if len(parts) >= 6:
            break
    return " ".join(parts).strip(), boxes


_INNER_TEXT_MAX_CHARS = 4000


def _inner_lines(page: fitz.Page, bbox: List[float], exclude_boxes: List[List[float]]) -> List[Dict[str, Any]]:
    """Text-layer lines inside the crop bbox, each with its page-space bbox
    (``{"text", "bbox_pt"}``), reading order.  Lines whose centre falls inside
    an ``exclude_boxes`` entry (caption / footnote blocks) are dropped."""
    x0, y0, x1, y1 = bbox
    out: List[Dict[str, Any]] = []
    try:
        d = page.get_text("dict")
    except Exception:
        return out
    for blk in d.get("blocks", []):
        if blk.get("type", 0) != 0:
            continue
        for ln in blk.get("lines", []):
            lb = ln.get("bbox")
            if not lb:
                continue
            cx, cy = (lb[0] + lb[2]) / 2.0, (lb[1] + lb[3]) / 2.0
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            if any(eb[0] <= cx <= eb[2] and eb[1] <= cy <= eb[3] for eb in exclude_boxes):
                continue
            txt = _normalize_text(" ".join(sp.get("text", "") for sp in ln.get("spans", [])))
            if txt:
                out.append({"text": txt, "bbox_pt": [round(float(v), 2) for v in lb]})
    return out


def _inner_text(page: fitz.Page, bbox: List[float], exclude_boxes: List[List[float]]) -> str:
    """Verbatim PDF text layer inside the crop bbox, grouped into rows.

    For tables this is the row content (entry numbers, yields …) that
    TATR/OCR may have lost; for vector figures it is the tick labels and
    legend strings.  Lines whose centre falls inside an ``exclude_boxes``
    entry (the caption / footnote blocks) are dropped.  Lines are clustered
    into rows by vertical position and joined with " | " left-to-right.
    """
    x0, y0, x1, y1 = bbox
    items: List[Tuple[float, float, str]] = []
    try:
        d = page.get_text("dict")
    except Exception:
        return ""
    for blk in d.get("blocks", []):
        if blk.get("type", 0) != 0:
            continue
        for ln in blk.get("lines", []):
            lb = ln.get("bbox")
            if not lb:
                continue
            cx, cy = (lb[0] + lb[2]) / 2.0, (lb[1] + lb[3]) / 2.0
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            if any(eb[0] <= cx <= eb[2] and eb[1] <= cy <= eb[3] for eb in exclude_boxes):
                continue
            txt = _normalize_text(" ".join(sp.get("text", "") for sp in ln.get("spans", [])))
            if txt:
                items.append((cy, lb[0], txt))
    if not items:
        return ""
    items.sort()
    rows: List[List[Tuple[float, str]]] = []
    row_y = None
    for cy, lx, txt in items:
        if row_y is None or abs(cy - row_y) > 3.0:
            rows.append([]); row_y = cy
        rows[-1].append((lx, txt))
    out = "\n".join(" | ".join(t for _, t in sorted(r)) for r in rows)
    return out[:_INNER_TEXT_MAX_CHARS]


def locate_contexts(pdf_path: str, layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return ``{source_id: context}`` for every entry in ``layout['sources']``."""
    doc = fitz.open(pdf_path)
    per_page: Dict[int, List[Dict[str, Any]]] = {}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        for src in layout.get("sources", []):
            sid, kind, page_no = src["source_id"], src["kind"], int(src["page"])
            bbox = [float(v) for v in src["bbox_pt"]]
            ctx: Dict[str, Any] = {
                "source_id": sid, "kind": kind, "page": page_no, "bbox_pt": bbox,
                "label_kind": None, "label_num": None, "label": None,
                "caption": "", "footnote": "", "caption_source": "missing",
                "inner_text": "", "inner_lines": [],
                "geometry_source": src.get("geometry_source", "unknown"),
            }
            if 1 <= page_no <= len(doc):
                if page_no not in per_page:
                    per_page[page_no] = _blocks(doc[page_no - 1])
                blocks = per_page[page_no]
                pick = _pick_caption(blocks, bbox, kind)
                if pick:
                    ctx.update({
                        "label_kind": pick["kind"], "label_num": pick["num"],
                        "label": f"{pick['kind']} {pick['num']}",
                        "caption": _extend_caption(blocks, pick["idx"]),
                        "caption_source": "pdf_text",
                        "caption_gap_pt": round(pick["gap"], 1),
                    })
                ctx["footnote"], fn_boxes = _pick_footnote(blocks, bbox, pick["idx"] if pick else None)
                excl = list(fn_boxes)
                if pick:
                    excl.append(blocks[pick["idx"]]["bbox"])
                ctx["inner_text"] = _inner_text(doc[page_no - 1], bbox, excl)
                ctx["inner_lines"] = _inner_lines(doc[page_no - 1], bbox, excl)
            out[sid] = ctx
    finally:
        doc.close()
    return out


def write_contexts(pdf_path: str, intermediate_dir: str) -> int:
    """Read ``layout.json`` under ``intermediate_dir``; write ``context/*.json``.
    Returns the number of sources resolved to a PDF-text caption."""
    layout_path = os.path.join(intermediate_dir, "layout.json")
    if not os.path.exists(layout_path):
        print(f"[CaptionLocator] no layout.json in {intermediate_dir} — skipping")
        return 0
    with open(layout_path) as f:
        layout = json.load(f)
    contexts = locate_contexts(pdf_path, layout)
    ctx_dir = os.path.join(intermediate_dir, "context")
    os.makedirs(ctx_dir, exist_ok=True)
    resolved = 0
    for sid, ctx in contexts.items():
        with open(os.path.join(ctx_dir, f"{sid}_context.json"), "w") as f:
            json.dump(ctx, f, indent=2, ensure_ascii=False)
        resolved += ctx["caption_source"] == "pdf_text"
    print(f"[CaptionLocator] {resolved}/{len(contexts)} sources resolved to PDF-text captions -> {ctx_dir}")
    return resolved


def load_context(intermediate_dir: str, source_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the context for a source; figure ids may carry a ``_tN`` macro
    suffix (``page_2_figure_1_t0``) which maps to the TF-ID crop id."""
    base = re.sub(r"_t\d+$", "", source_id)
    path = os.path.join(intermediate_dir, "context", f"{base}_context.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
