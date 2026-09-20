"""Pure grid operations for the VLM-transcribed table.

Two deterministic steps sit between the VLM transcription and the CSV:

* ``align_structures`` — the VLM marks every drawn molecule as the literal
  token ``[STRUCTURE]``; the molecule detector (YOLO + MolNexTR) gives boxes
  with SMILES in body-crop pixels.  Boxes are clustered into rows by their
  vertical gaps and matched to the token positions left-to-right, so SMILES
  come only from MolNexTR and land in the cell the VLM saw the drawing in.
* ``text_agreement`` — numbers in the transcribed data rows are checked
  against the PDF text layer of the same crop (``context.inner_text``); a
  low recall marks the table ``unverified`` downstream.

No model calls, no I/O: everything here is unit-testable with lists.
"""
from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

STRUCTURE_TOKEN = "[STRUCTURE]"
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _box_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def cluster_rows(boxes: Sequence[Sequence[float]], gap_factor: float = 0.5) -> List[List[int]]:
    """Group box indices into rows: sort by vertical centre; a box joins the
    current row while its centre lies within the row's vertical span plus
    ``gap_factor`` × the median box height, else it starts a new row.
    Each row is sorted left-to-right."""
    if not boxes:
        return []
    xyxy = [_box_xyxy(b) for b in boxes]
    heights = [b[3] - b[1] for b in xyxy]
    med_h = max(1.0, statistics.median(heights))
    order = sorted(range(len(xyxy)), key=lambda i: (xyxy[i][1] + xyxy[i][3]) / 2.0)
    rows: List[List[int]] = []
    row_bottom = None
    for i in order:
        cy = (xyxy[i][1] + xyxy[i][3]) / 2.0
        # A box belongs to the current row while its centre lies inside the
        # row's vertical span (plus a small tolerance) — tall and short
        # drawings on one line then stay together.
        if row_bottom is None or cy > row_bottom + gap_factor * med_h:
            rows.append([]); row_bottom = xyxy[i][3]
        rows[-1].append(i)
        row_bottom = max(row_bottom, xyxy[i][3])
    for r in rows:
        r.sort(key=lambda i: xyxy[i][0])
    return rows


def _token_slots(grid: Sequence[Sequence[str]], token: str) -> List[List[Tuple[int, int, int]]]:
    """Per grid row, the (row, col, k) slots of ``token`` occurrences, left to
    right; rows without tokens are omitted."""
    out: List[List[Tuple[int, int, int]]] = []
    for r, row in enumerate(grid):
        slots = []
        for c, cell in enumerate(row):
            n = str(cell or "").count(token)
            slots.extend((r, c, k) for k in range(n))
        if slots:
            out.append(slots)
    return out


# Atom labels / substituent abbreviations printed as part of a drawing that the
# VLM sometimes transcribes as text next to the token ("MeO-[STRUCTURE]-F").
_ATOM_LABEL_RE = re.compile(
    r"^[\s\-\u2013\u2014]*(MeO|OMe|EtO|OEt|F3C|CF3|Me3Si|TMS|SiMe3|Me|Et|Ph|Bn|Ac|Boc|Ts|Tf|CN|NC|NO2|OH|NH2|"
    r"tBu|t-Bu|nBu|n-Bu|Bu|iPr|i-Pr|Pr|Br|Cl|F|I|N|O|S|H)?[\s\-\u2013\u2014]*$")


def _strip_atom_labels(cell: str, anchors: Sequence[str]) -> str:
    """Drop atom-label fragments glued to the given anchors (structure tokens
    or inserted SMILES) in a cell; any other text is kept."""
    anchors = [a for a in anchors if a]
    if not anchors:
        return cell
    pattern = "(" + "|".join(re.escape(a) for a in sorted(set(anchors), key=len, reverse=True)) + ")"
    keep = []
    for part in re.split(pattern, cell):
        if not part:
            continue
        if part in anchors or not _ATOM_LABEL_RE.match(part):
            keep.append(part)
    return re.sub(r"\s+", " ", "".join(keep)).strip()


def _put(grid: List[List[str]], slot: Tuple[int, int, int], smiles: str, token: str) -> None:
    r, c, k = slot
    cell = str(grid[r][c])
    parts = cell.split(token)
    # Replace the k-th token occurrence only (cells may hold several).
    idx = 0
    rebuilt = parts[0]
    for i in range(1, len(parts)):
        rebuilt += (smiles if idx == k else token) + parts[i]
        idx += 1
    grid[r][c] = _strip_atom_labels(rebuilt, [token, smiles])


def _norm_cell(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"[\s\[\]()]+", "", s).casefold()


def text_layer_anchors(grid: Sequence[Sequence[str]], n_header: int, inner_lines: Sequence[Dict[str, Any]],
                       bbox_pt: Sequence[float], scale: float, offset_px: Tuple[float, float],
                       img_w: float) -> Tuple[Optional[List[Optional[float]]], Optional[List[float]]]:
    """Row / column anchors (body-crop pixels) from the PDF text layer.

    * row centres: one per grid row (``None`` for header rows) — the y centre
      of the text-layer line whose text equals the row's first non-empty cell
      (the entry label) and lies in the left fifth of the crop.  Returned only
      when every data row is anchored and the anchors increase monotonically.
    * column centres: x centre of the line matching each header cell of the
      first header row.  Returned only when every column is anchored.
    """
    if not inner_lines or not bbox_pt:
        return None, None
    ox, oy = float(bbox_pt[0]), float(bbox_pt[1])
    lines = []
    for ln in inner_lines:
        b = ln.get("bbox_pt")
        if not b:
            continue
        x0, y0, x1, y1 = [(float(v) - (ox if k % 2 == 0 else oy)) * scale for k, v in enumerate(b)]
        lines.append((_norm_cell(ln.get("text", "")), (x0 + x1) / 2 - offset_px[0], (y0 + y1) / 2 - offset_px[1]))
    # rows
    row_centres: List[Optional[float]] = [None] * len(grid)
    used = set(); ok_rows = True
    for r in range(n_header, len(grid)):
        label = next((c for c in grid[r] if str(c).strip()), "")
        key = _norm_cell(label)
        if not key or len(key) > 6:
            ok_rows = False; break
        cands = [(cy, i) for i, (t, cx, cy) in enumerate(lines) if t == key and cx <= 0.2 * img_w and i not in used]
        if not cands:
            ok_rows = False; break
        cy, i = min(cands)                          # topmost unused match keeps reading order
        row_centres[r] = cy; used.add(i)
    data = [c for c in row_centres if c is not None]
    if not ok_rows or len(data) < 2 or any(b <= a for a, b in zip(data, data[1:])):
        row_centres = None
    # columns
    col_centres: Optional[List[float]] = None
    if n_header >= 1 and grid:
        cols = []
        for c, cell in enumerate(grid[0]):
            key = _norm_cell(cell)
            cands = [cx for (t, cx, cy) in lines if key and (t == key or (len(key) >= 4 and t.startswith(key)))]
            if not cands:
                cols = None; break
            cols.append(min(cands))
        if cols and len(cols) >= 2 and all(b > a for a, b in zip(cols, cols[1:])):
            col_centres = cols
    return row_centres, col_centres


def _x_clusters(boxes: Sequence[Tuple[float, float, float, float]], gap_factor: float = 0.5) -> List[float]:
    """Centres of the x-clusters of box centres (new cluster when the gap to
    the previous centre exceeds ``gap_factor`` × the median box width)."""
    if not boxes:
        return []
    med_w = max(1.0, statistics.median(b[2] - b[0] for b in boxes))
    xs = sorted((b[0] + b[2]) / 2 for b in boxes)
    clusters: List[List[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] > gap_factor * med_w:
            clusters.append([])
        clusters[-1].append(x)
    return [sum(c) / len(c) for c in clusters]


def _nearest(centres: Sequence[Optional[float]], v: float) -> Optional[int]:
    best, bd = None, None
    for i, c in enumerate(centres):
        if c is None:
            continue
        d = abs(c - v)
        if bd is None or d < bd:
            best, bd = i, d
    return best


def align_structures(grid: Sequence[Sequence[str]], mol_meta: Sequence[Dict[str, Any]],
                     token: str = STRUCTURE_TOKEN, row_centres: Optional[Sequence[Optional[float]]] = None,
                     col_centres: Optional[Sequence[float]] = None) -> Tuple[List[List[str]], Dict[str, Any]]:
    """Fill ``token`` cells with MolNexTR SMILES by geometry.

    With text-layer anchors (``row_centres`` per grid row, optionally
    ``col_centres`` per column) every box is dropped into the cell whose
    anchors are nearest — extra or missing boxes affect only their own cell
    (``status`` ``anchored`` / ``anchored_partial``).  Without anchors, boxes
    are clustered into rows (``cluster_rows``) and matched to the token rows:
    ``ok`` (row counts all match), ``partial`` (some rows), ``global`` (row
    structure differs but totals match) or ``failed``; ``none`` = no boxes.
    Boxes whose SMILES is empty / invalid keep the token (``unresolved``).
    """
    out = [list(map(lambda v: "" if v is None else str(v), row)) for row in grid]
    metas = [m for m in mol_meta if m.get("box") is not None]
    boxes = [_box_xyxy(m["box"]) for m in metas]
    slots_by_row = _token_slots(out, token)
    n_tokens = sum(len(s) for s in slots_by_row)
    report: Dict[str, Any] = {"status": "none", "n_boxes": len(boxes), "n_tokens": n_tokens,
                              "n_box_rows": 0, "n_token_rows": len(slots_by_row), "assigned": 0, "unresolved": 0,
                              "unplaced": 0, "anchors": "none"}
    if not boxes:
        return out, report
    if n_tokens == 0:
        report["status"] = "failed"
        return out, report

    pairs: List[Tuple[Tuple[int, int, int], int]] = []
    if row_centres is not None and any(c is not None for c in row_centres):
        report["anchors"] = "rows+cols" if col_centres else "rows"
        token_cols = sorted({s[1] for row in slots_by_row for s in row})
        if not col_centres:
            # Column anchors from the boxes themselves: x-clusters (gap > half
            # the median box width) that map 1:1 onto the token columns.
            xc = _x_clusters(boxes)
            if len(xc) == len(token_cols):
                col_centres = [None] * (max(token_cols) + 1)
                for c, x in zip(token_cols, xc):
                    col_centres[c] = x
                report["anchors"] = "rows+box_cols"
        data_c = [c for c in row_centres if c is not None]
        pitch = statistics.median([b - a for a, b in zip(data_c, data_c[1:])]) if len(data_c) > 1 else None
        by_row: Dict[int, List[int]] = {}
        for bi, (x1, y1, x2, y2) in enumerate(boxes):
            cy = (y1 + y2) / 2
            r = _nearest(row_centres, cy)
            if r is None or (pitch and abs(row_centres[r] - cy) > 0.6 * pitch):
                report["unplaced"] += 1          # header-band / footnote drawing, not a cell
                continue
            by_row.setdefault(r, []).append(bi)
        slots_map = {row[0][0]: row for row in slots_by_row}
        for r, bis in by_row.items():
            bis.sort(key=lambda i: boxes[i][0])
            slots = list(slots_map.get(r, []))
            if col_centres:
                # Per cell: the box whose centre is nearest the row anchor wins;
                # competing boxes (duplicate detections, fragments) stay unplaced.
                claims: Dict[Tuple[int, int, int], List[int]] = {}
                for bi in bis:
                    c = _nearest(col_centres, (boxes[bi][0] + boxes[bi][2]) / 2)
                    slot = next((s for s in slots if s[1] == c and s not in claims), None)
                    if slot is None:
                        slot = next((s for s in slots if s[1] == c), None)
                    if slot is None:
                        report["unplaced"] += 1; continue
                    claims.setdefault(slot, []).append(bi)
                for slot, cands in claims.items():
                    best = min(cands, key=lambda i: abs((boxes[i][1] + boxes[i][3]) / 2 - row_centres[r]))
                    pairs.append((slot, best)); report["unplaced"] += len(cands) - 1
            elif len(bis) == len(slots):
                pairs.extend(zip(slots, bis))
            else:
                report["unplaced"] += len(bis)
        report["status"] = "anchored" if report["unplaced"] == 0 and len(pairs) == n_tokens else "anchored_partial"
    else:
        box_rows = cluster_rows(boxes)
        report["n_box_rows"] = len(box_rows)
        if len(box_rows) == len(slots_by_row):
            matched_rows = [i for i in range(len(box_rows)) if len(box_rows[i]) == len(slots_by_row[i])]
            for i in matched_rows:
                pairs.extend(zip(slots_by_row[i], box_rows[i]))
            report["status"] = "ok" if len(matched_rows) == len(box_rows) else ("partial" if matched_rows else "failed")
            if report["status"] == "failed" and len(boxes) == n_tokens:
                pairs = list(zip([s for row in slots_by_row for s in row], [b for row in box_rows for b in row]))
                report["status"] = "global"
        elif len(boxes) == n_tokens:
            pairs = list(zip([s for row in slots_by_row for s in row], [b for row in box_rows for b in row]))
            report["status"] = "global"
        else:
            report["status"] = "failed"

    for slot, bi in pairs:
        smiles = str(metas[bi].get("smiles") or "").strip()
        if not smiles or smiles.startswith("<"):
            report["unresolved"] += 1
            continue
        _put(out, slot, smiles, token)
        report["assigned"] += 1
    return out, report


def _numbers(text: str) -> Counter:
    out: Counter = Counter()
    for tok in _NUM_RE.findall(text or ""):
        try:
            out["%g" % float(tok.replace(",", "."))] += 1
        except ValueError:
            pass
    return out


def text_agreement(data_rows: Sequence[Sequence[str]], inner_text: Optional[str],
                   min_numbers: int = 3) -> Optional[float]:
    """Multiset recall of the grid's numeric tokens in the PDF text layer.
    Unsigned on purpose: the text layer renders minus signs as control
    characters.  ``None`` when there is no text layer or too few numbers."""
    if not inner_text or not inner_text.strip():
        return None
    grid_nums = _numbers(" ".join(str(c) for row in data_rows for c in row))
    total = sum(grid_nums.values())
    if total < min_numbers:
        return None
    text_nums = _numbers(inner_text)
    hit = sum(min(n, text_nums.get(k, 0)) for k, n in grid_nums.items())
    return round(hit / total, 3)
