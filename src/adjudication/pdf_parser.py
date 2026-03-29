import os
import re
import fitz  # PyMuPDF


def _safe_print(text):
    try:
        import sys
        sys.stdout.buffer.write((str(text) + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    except Exception:
        pass


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
            _safe_print(f"[PDFParser] Loading cached text from {txt_path}")
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
            truncated = self._truncate_text(content)
            if len(truncated) < len(content):
                with open(txt_path, "w", encoding="utf-8") as fw:
                    fw.write(truncated)
                return truncated
            return content

        _safe_print(f"[PDFParser] Parsing {pdf_path}...")
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
            _safe_print(f"[PDFParser] Error parsing PDF: {e}")
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
            _safe_print(f"[PDFParser] Truncated text at index {cutoff_idx}/{len(text)} (detected ending section).")
            return text[:cutoff_idx]
            
        return text
