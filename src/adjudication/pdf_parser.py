import os
import re
from typing import Optional, Tuple
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


def extract_text_window(
    paper_text: str,
    source_id: str,
    source_type: str,
    primary_size: int = 6000,
    experimental_size: int = 2000,
    extra_anchor_keywords: Optional[Tuple[str, ...]] = None,
) -> str:
    """Extract a dual-anchor window relevant to a single source.

    Two anchors are scanned independently:

    1. **Primary anchor** — the figure/table number ("Figure 3", "Table 1")
       derived from ``source_id``.  This window carries the local context
       (caption mentions, immediate prose around the result).  Falls back
       to ``paper_text[:primary_size]`` if no keyword matches.

    2. **Experimental anchor** — the first occurrence of a General
       Procedure / Materials-and-Methods heading.  Captures paper-wide
       baselines (catalyst loading, default temperature, solvent) that
       authors typically state once in the Experimental section.  Skipped
       if no heading is found.

    The two slices are concatenated with a clear ``--- experimental
    section ---`` separator so downstream LLM prompts can tell them
    apart.  If the two windows overlap (e.g. experimental section is
    next to the figure citation), the larger contiguous range is used
    once — never duplicated.

    Total output length is bounded by ``primary_size +
    experimental_size``; in practice the LLM sees ~8KB for a
    ``(6000, 2000)`` call.
    """
    if not paper_text:
        return ""

    # Normalise NBSP and other Unicode whitespace before lowercase
    # find().  Without this, "Materials and\xa0methods" (paper PDF
    # extraction often leaves NBSPs) never matches our anchor list.
    text_lower = re.sub(r"\s+", " ", paper_text.lower())

    # 1. Primary anchor — earliest match wins.  Search both the
    # source_id-derived "figure N" / "table N" keywords AND any
    # extra keywords the caller passed (typically a caption fragment
    # like "Fig. 3 Yield of …").
    primary_pos = -1
    anchor_keywords = list(_source_id_keywords(source_id, source_type))
    if extra_anchor_keywords:
        anchor_keywords += [k.lower() for k in extra_anchor_keywords if k]
    for kw in anchor_keywords:
        idx = text_lower.find(kw)
        if idx != -1 and (primary_pos == -1 or idx < primary_pos):
            primary_pos = idx

    if primary_pos == -1:
        primary_slice = (0, min(len(paper_text), primary_size))
    else:
        primary_slice = _slice_around(paper_text, primary_pos, primary_size)

    # 2. Experimental anchor — first occurrence wins.
    exp_pos = -1
    for anchor in _EXPERIMENTAL_ANCHORS:
        idx = text_lower.find(anchor)
        if idx != -1:
            exp_pos = idx
            break

    if exp_pos == -1 or experimental_size <= 0:
        return paper_text[primary_slice[0] : primary_slice[1]]

    exp_slice = _slice_around(paper_text, exp_pos, experimental_size)

    # 3. Merge — if the two windows overlap, return the contiguous span
    # once; otherwise concatenate with a labelled separator.
    p_start, p_end = primary_slice
    e_start, e_end = exp_slice
    if not (p_end < e_start or e_end < p_start):
        merged_start = min(p_start, e_start)
        merged_end = max(p_end, e_end)
        return paper_text[merged_start:merged_end]

    # Disjoint — keep both, in document order, with separator.
    if p_start < e_start:
        return (
            paper_text[p_start:p_end]
            + "\n\n--- experimental section ---\n\n"
            + paper_text[e_start:e_end]
        )
    return (
        paper_text[e_start:e_end]
        + "\n\n--- local context ---\n\n"
        + paper_text[p_start:p_end]
    )
