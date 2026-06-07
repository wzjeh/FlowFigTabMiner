"""
Pre-filter layer: checks if a PDF is flow chemistry relevant
before running expensive model inference.
"""
import os
import logging

logger = logging.getLogger(__name__)

FLOW_CHEMISTRY_KEYWORDS = [
    "flow chemistry", "continuous flow", "continuous-flow",
    "microreactor", "micro-reactor", "flow reactor",
    "plug flow", "tubular reactor",
    "micropacked", "micro-packed", "packed bed reactor",
    "coil reactor", "microfluidic reactor",
    "flow synthesis", "flow process",
]

# Review / overview articles to EXCLUDE — we only want primary research with
# extractable reaction data, not literature surveys.  Specific phrases only
# (not bare "review"/"account") to avoid killing primary papers that merely
# say "we review the conditions" or "under review".
REVIEW_KEYWORDS = [
    "recent advances", "recent progress", "recent developments",
    "a review of", "this review", "review article", "in this review",
    "tutorial review", "mini-review", "mini review", "minireview",
    "critical review", "comprehensive review", "an overview of",
    "chem. rev.", "chem soc rev", "chem. soc. rev.",
    "chemical reviews", "chemical society reviews",
    "annu. rev.", "annual review", "modern strategies",
]


def filter_paper(pdf_path: str, pages_to_check: int = 3) -> dict:
    """
    Extract text from the first N pages and check for flow chemistry keywords.

    Returns:
        {"is_relevant": bool, "reason": str, "matched_keyword": str | None}
    """
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
    for kw in REVIEW_KEYWORDS:
        if kw in head:
            logger.info(f"Paper filter FAIL — review/overview article (matched: '{kw}')")
            return {"is_relevant": False, "reason": f"review article (matched '{kw}')", "matched_keyword": kw}

    # 2. Flow-chemistry relevance whitelist.
    for kw in FLOW_CHEMISTRY_KEYWORDS:
        if kw in full_text:
            logger.info(f"Paper filter PASS — matched keyword: '{kw}'")
            return {"is_relevant": True, "reason": "flow chemistry keyword found", "matched_keyword": kw}

    logger.info(f"Paper filter FAIL — no flow chemistry keywords in first {n_pages} pages")
    return {
        "is_relevant": False,
        "reason": "no flow chemistry keywords found in abstract/intro",
        "matched_keyword": None,
    }
