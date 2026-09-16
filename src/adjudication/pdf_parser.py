import os
import re
from typing import List, Optional, Tuple
import fitz  # PyMuPDF


def _normalize_text(s: str) -> str:
    """Normalize ligatures, quotes, and hyphen line-breaks."""
    # Ligature normalization
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    # Typographic quotes → ASCII
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    # Hyphen at word boundary followed by whitespace (line-break repair)
    s = re.sub(r"\b-\s+", "", s)
    # Collapse multiple spaces
    s = re.sub(r" {2,}", " ", s)
    return s.strip()


class PDFParser:
    def __init__(self):
        pass

    def extract_text(self, pdf_path):
        """
        Extracts full text from a PDF using block-level extraction.
        - Skips header/footer regions (top/bottom 5% of page height)
        - Filters pure page-number lines
        - Normalizes ligatures, quotes, hyphen line-breaks
        - Truncates at References/Acknowledgements
        Caches the result to a .txt file next to the PDF.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        txt_path = pdf_path.replace(".pdf", "_fulltext.txt")
        if os.path.exists(txt_path):
            print(f"[PDFParser] Loading cached text from {txt_path}")
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            truncated = self._truncate_text(content)
            if len(truncated) < len(content):
                with open(txt_path, "w", encoding="utf-8") as fw:
                    fw.write(truncated)
                return truncated
            return content

        print(f"[PDFParser] Parsing {pdf_path}...")
        out_blocks = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                ph = page.rect.height
                top_cut = ph * 0.05
                bot_cut = ph * 0.95
                blocks = page.get_text("blocks") or []
                blocks.sort(key=lambda b: (b[1], b[0]))
                for b in blocks:
                    x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
                    raw_text = b[4] if len(b) > 4 else ""
                    if not raw_text or not raw_text.strip():
                        continue
                    # Skip header/footer regions unless they contain obvious body keywords
                    if y0 < top_cut or y1 > bot_cut:
                        low = raw_text.lower()
                        body_kw = ["introduction", "abstract", "experiment", "method",
                                   "result", "discussion", "conclusion", "flow", "reactor"]
                        if not any(k in low for k in body_kw):
                            continue
                    # Filter pure page-number lines
                    lines = [ln for ln in raw_text.splitlines()
                             if not re.match(r"^\s*\d+\s*$", ln)]
                    if not lines:
                        continue
                    block_text = _normalize_text(" ".join(lines))
                    if block_text:
                        out_blocks.append(block_text)
            doc.close()
        except Exception as e:
            print(f"[PDFParser] Error parsing PDF: {e}")
            return ""

        combined_text = "\n\n".join(out_blocks)
        combined_text = self._truncate_text(combined_text)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

        return combined_text

    def _truncate_text(self, text):
        """
        Truncate text after References/Bibliography/Conclusion to save tokens.
        """
        import re
        text_lower = text.lower()
        
        # We only consider the last 40% of the document to avoid false positives.
        min_pos = int(len(text) * 0.6)
        
        # We can use regex to find section headers robustly
        patterns = [
            r'\n\s*(?:[0-9]*\.?\s*)?references\s*\n',
            r'\n\s*(?:[0-9]*\.?\s*)?bibliography\s*\n',
            r'\n\s*(?:[0-9]*\.?\s*)?acknowledgements?\s*\n',
            r'\n\s*(?:[0-9]*\.?\s*)?conclusions?\s*\n',
            r'\n\s*(?:[0-9]*\.?\s*)?notes\s*and\s*references\s*\n'
        ]
        
        cutoff_idx = len(text)
        
        for pattern in patterns:
            match = re.search(pattern, text_lower[min_pos:])
            if match:
                idx = min_pos + match.start()
                if idx < cutoff_idx:
                    cutoff_idx = idx
                    
        # Fallback to simple rfind for references if regex misses
        if cutoff_idx == len(text):
            fallback_keywords = ["\nreferences", "\nacknowledgement"]
            for kw in fallback_keywords:
                idx = text_lower.rfind(kw)
                if idx > min_pos and idx < cutoff_idx:
                    cutoff_idx = idx
        
        if cutoff_idx < len(text):
            print(f"[PDFParser] Truncated text at index {cutoff_idx}/{len(text)} (detected ending section).")
            return text[:cutoff_idx]

        return text


# ── Dual-anchor text-window helper (shared by LocalVarsBuilder and PerSourceAssembler)

# Keywords that signal the experimental-section / general procedure block.
# Order matters only for tie-breaking on which anchor wins; we accept any.
_EXPERIMENTAL_ANCHORS: Tuple[str, ...] = (
    "general procedure",
    "typical procedure",
    "experimental section",
    "experimental procedure",
    "reaction setup",
    "materials and methods",
    "experimental",  # last resort — broader
)


def _source_id_keywords(source_id: str, source_type: str) -> list[str]:
    """Derive in-text figure/table number keywords from a source_id.

    ``page_2_figure_1_t0`` → ``["figure 1", "fig. 1", "fig 1", "figure1"]``.
    """
    keywords: list[str] = []
    parts = source_id.lower().split("_")
    if source_type == "figure":
        for i, p in enumerate(parts):
            if p == "figure" and i + 1 < len(parts):
                num = parts[i + 1]
                keywords += [f"figure {num}", f"fig. {num}", f"fig {num}", f"figure{num}"]
    elif source_type == "table":
        for i, p in enumerate(parts):
            if p == "table" and i + 1 < len(parts):
                num = parts[i + 1]
                keywords += [f"table {num}", f"table{num}"]
    return keywords


def _slice_around(text: str, center: int, size: int) -> Tuple[int, int]:
    """Return (start, end) indices for a `size`-char slice centred on ``center``.

    Pulls back ~10% before the anchor so the section heading itself is in
    view, then pads forward to ``size``.
    """
    pre = size // 10
    start = max(0, center - pre)
    end = min(len(text), start + size)
    return start, end


def _label_patterns(label: str) -> List[str]:
    """Regexes matching in-text citations of a resolved label.

    ``"Figure 2"`` → ``\bfig(?:ure|\.)?\s*2(?![0-9])`` (also matches "Fig. 2",
    "Fig 2", "Figure2" but not "Figure 20"); ``"Table 1"`` / ``"Scheme 3"``
    analogously.  Supplementary labels ("Figure S3") keep the S.
    """
    m = re.match(r"\s*(figure|fig\.?|table|scheme|chart)\s*(S?\d+)", label or "", re.I)
    if not m:
        return []
    kind, num = m.group(1).lower().rstrip("."), re.escape(m.group(2))
    if kind in ("figure", "fig"):
        return [rf"\bfig(?:ure|\.)?\s*{num}(?![0-9])"]
    return [rf"\b{kind}\s*{num}(?![0-9])"]


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def extract_text_window(
    paper_text: str,
    source_id: str,
    source_type: str,
    primary_size: int = 6000,
    experimental_size: int = 2000,
    extra_anchor_keywords: Optional[Tuple[str, ...]] = None,
    label: Optional[str] = None,
    mention_radius: int = 800,
) -> str:
    """Extract a dual-anchor window relevant to a single source.

    Two anchors are scanned independently:

    1. **Primary anchor(s)** — when ``label`` (the source's real caption
       label, e.g. ``"Figure 2"``, resolved by ``CaptionLocator``) is given,
       EVERY in-text mention of it ("Fig. 2", "Figure 2a" …) plus the
       caption fragments in ``extra_anchor_keywords`` become anchors; a
       ±``mention_radius`` slice is taken around each, overlapping slices
       are merged, and slices are kept in document order until
       ``primary_size`` is exhausted.  Without ``label`` the legacy
       behaviour applies: the first hit of the source_id-derived
       "figure N" keyword (N = crop index on the page — usually NOT the
       paper's figure number) or ``paper_text[:primary_size]``.

    2. **Experimental anchor** — the first occurrence of a General
       Procedure / Materials-and-Methods heading.  Captures paper-wide
       baselines (catalyst loading, default temperature, solvent) that
       authors typically state once in the Experimental section.  Skipped
       if no heading is found.

    Slices are concatenated in document order with ``--- experimental
    section ---`` / ``--- local context ---`` / ``--- … ---`` separators so
    downstream LLM prompts can tell them apart.  Overlapping slices are
    merged, never duplicated.  Total length ≤ ``primary_size +
    experimental_size``.
    """
    if not paper_text:
        return ""

    # Lower-case and map every whitespace char (NBSP included) to ONE space
    # WITHOUT changing string length, so match offsets index ``paper_text``.
    text_lower = re.sub(r"\s", " ", paper_text.lower())

    primary_slices: List[Tuple[int, int]] = []
    patterns = _label_patterns(label) if label else []
    if extra_anchor_keywords:
        patterns += [re.escape(k.lower()) for k in extra_anchor_keywords if k and len(k) >= 5]

    if patterns:
        hits = sorted({m.start() for pat in patterns for m in re.finditer(pat, text_lower)})
        spans = [(max(0, h - mention_radius), min(len(paper_text), h + mention_radius)) for h in hits]
        budget = primary_size
        for a, b in _merge_spans(spans):
            if budget <= 0:
                break
            b = min(b, a + budget)
            primary_slices.append((a, b))
            budget -= (b - a)

    if not primary_slices:
        # Legacy path: source_id-derived keyword, first hit wins.
        primary_pos = -1
        for kw in _source_id_keywords(source_id, source_type):
            idx = text_lower.find(kw)
            if idx != -1 and (primary_pos == -1 or idx < primary_pos):
                primary_pos = idx
        if primary_pos == -1:
            primary_slices = [(0, min(len(paper_text), primary_size))]
        else:
            primary_slices = [_slice_around(paper_text, primary_pos, primary_size)]

    # 2. Experimental anchor — first occurrence wins.
    exp_slice: Optional[Tuple[int, int]] = None
    if experimental_size > 0:
        for anchor in _EXPERIMENTAL_ANCHORS:
            idx = text_lower.find(anchor)
            if idx != -1:
                exp_slice = _slice_around(paper_text, idx, experimental_size)
                break

    # 3. Merge everything in document order, labelling each disjoint piece.
    tagged = [(a, b, "local") for a, b in primary_slices]
    if exp_slice:
        tagged.append((exp_slice[0], exp_slice[1], "experimental"))
    tagged.sort()
    pieces: List[Tuple[int, int, str]] = []
    for a, b, tag in tagged:
        if pieces and a <= pieces[-1][1]:
            pa, pb, ptag = pieces[-1]
            pieces[-1] = (pa, max(pb, b), ptag if ptag == tag else "local+experimental")
        else:
            pieces.append((a, b, tag))

    if len(pieces) == 1:
        a, b, _ = pieces[0]
        return paper_text[a:b]
    # First piece verbatim; every later piece is introduced by a separator
    # naming what it is (the legacy two-piece shape, generalised to N).
    out = [paper_text[pieces[0][0]:pieces[0][1]]]
    for a, b, tag in pieces[1:]:
        sep = {"local": "--- local context ---", "experimental": "--- experimental section ---"}.get(
            tag, "--- local context + experimental section ---")
        out.append(f"{sep}\n\n{paper_text[a:b]}")
    return "\n\n".join(out)
