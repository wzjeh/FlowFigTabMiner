"""
Pre-filter layer: checks if a PDF is a flow-chemistry PRIMARY research
paper (and not a review) before running expensive model inference.

Keywords are NOT hardcoded here — they live in ``keywords.yaml`` under
``paper_filter.flow_chemistry_include`` (whitelist) and
``paper_filter.review_exclude`` (blacklist), so they can be maintained
without touching code.
"""
import os
import logging

import yaml

logger = logging.getLogger(__name__)

# Cached (flow_chemistry_include, review_exclude) loaded from keywords.yaml.
_KEYWORDS = None

_KEYWORDS_CANDIDATES = [
    "keywords.yaml",
    os.path.join(os.getcwd(), "keywords.yaml"),
    os.path.join(os.path.dirname(__file__), "..", "..", "keywords.yaml"),
]


def _load_keywords() -> tuple:
    """Return (flow_include, review_exclude) from keywords.yaml; ([], []) if
    the config is missing (callers treat empty whitelist as fail-safe)."""
    global _KEYWORDS
    if _KEYWORDS is not None:
        return _KEYWORDS
    for path in _KEYWORDS_CANDIDATES:
        if os.path.exists(path):
            try:
                cfg = (yaml.safe_load(open(path)) or {}).get("paper_filter", {})
                _KEYWORDS = (
                    [k.lower() for k in cfg.get("flow_chemistry_include", [])],
                    [k.lower() for k in cfg.get("review_exclude", [])],
                )
                return _KEYWORDS
            except Exception as exc:
                logger.warning("paper_filter: failed to read %s: %s", path, exc)
    logger.warning("paper_filter: keywords.yaml not found; pre-filter disabled (fail-safe)")
    _KEYWORDS = ([], [])
    return _KEYWORDS


def filter_paper(pdf_path: str, pages_to_check: int = 3) -> dict:
    """
    Extract text from the first N pages and decide relevance:
      1. exclude review/overview articles (review_exclude, title region)
      2. require a flow-chemistry keyword (flow_chemistry_include)

    Returns:
        {"is_relevant": bool, "reason": str, "matched_keyword": str | None}
    """
    flow_include, review_exclude = _load_keywords()

    # Fail-safe: no keyword config → don't filter anything.
    if not flow_include:
        return {"is_relevant": True, "reason": "no keyword config (fail-safe)", "matched_keyword": None}

    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning("pypdfium2 not available; skipping paper filter (treating as relevant)")
        return {"is_relevant": True, "reason": "pypdfium2 not installed", "matched_keyword": None}

    if not os.path.exists(pdf_path):
        return {"is_relevant": False, "reason": f"PDF not found: {pdf_path}", "matched_keyword": None}

    try:
        doc = pdfium.PdfDocument(pdf_path)
        n_pages = min(len(doc), pages_to_check)
        text_parts = []
        for i in range(n_pages):
            page = doc[i]
            textpage = page.get_textpage()
            text_parts.append(textpage.get_text_range())
        doc.close()
        full_text = " ".join(text_parts).lower()
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}; treating as relevant")
        return {"is_relevant": True, "reason": f"Text extraction error: {e}", "matched_keyword": None}

    # 1. Exclude review / overview articles first (no extractable reaction data).
    # Restrict to the title/abstract region (first ~1500 chars) so a primary
    # paper citing a review in its intro isn't falsely excluded.
    head = full_text[:1500]
    for kw in review_exclude:
        if kw in head:
            logger.info(f"Paper filter FAIL — review/overview article (matched: '{kw}')")
            return {"is_relevant": False, "reason": f"review article (matched '{kw}')", "matched_keyword": kw}

    # 2. Flow-chemistry relevance whitelist.
    for kw in flow_include:
        if kw in full_text:
            logger.info(f"Paper filter PASS — matched keyword: '{kw}'")
            return {"is_relevant": True, "reason": "flow chemistry keyword found", "matched_keyword": kw}

    logger.info(f"Paper filter FAIL — no flow chemistry keywords in first {n_pages} pages")
    return {
        "is_relevant": False,
        "reason": "no flow chemistry keywords found in abstract/intro",
        "matched_keyword": None,
    }
