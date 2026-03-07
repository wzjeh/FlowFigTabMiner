import logging
from typing import Dict, Any, List

from src.adjudication.llm_engine import LLMEngine
from src.adjudication.global_context import GlobalInfoExtractor
from src.adjudication.context_resolver import ContextResolver

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Handles processing of pure text blocks using LLMs or NLP rules.
    Responsibilities:
    - Extracting explicit candidates (T, P) from contexts (e.g. Captions/Notes).
    - Resolving unknown terms by scanning the full text.
    """
    def __init__(self, llm_engine=None):
        logger.info("Initializing TextProcessor...")
        self.llm = llm_engine if llm_engine else LLMEngine()
        self.extractor = GlobalInfoExtractor(self.llm)

    def extract_candidates(self, context_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts structural candidates and unknown terms from a caption or list of notes.
        Expects context_obj to contain 'text_evidence' such as caption and axis_notes.
        """
        logger.debug("Extracting local candidates via LLM...")
        return self.extractor.extract(context_obj)
        
    def resolve_unknowns(self, unknowns: List[str], full_text: str, external_context: Dict[str, str] = None) -> Dict[str, str]:
        """
        Searches the full text to find meanings for abbreviated or unknown terms.
        """
        if not unknowns:
            return {}
        logger.debug(f"Resolving {len(unknowns)} unknowns against full text...")
        resolver = ContextResolver(full_text, external_context=external_context or {})
        return resolver.resolve(unknowns)
