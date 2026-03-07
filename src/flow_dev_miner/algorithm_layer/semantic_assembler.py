import logging
from typing import Dict, Any, List

from src.adjudication.llm_engine import LLMEngine
from src.adjudication.data_synthesizer import DataSynthesizer
from src.adjudication.global_context import GlobalInfoExtractor
from src.adjudication.context_resolver import ContextResolver

try:
    from src.extraction.context.llm_selector import LLMSelector
except ImportError:
    try:
        from scripts.step_llm_selector import LLMSelector
    except:
        class LLMSelector:
             def select_relevant_assets(self, *args, **kwargs): return {}

logger = logging.getLogger(__name__)

class SemanticAlgorithmCore:
    """
    The Brain of the system.
    Responsibilities:
    - Agentic Selection (Which figures/tables to process).
    - Building the Global Variable Pool (GVP).
    - Semantic Assembly: Merging Figure/Table data points with the GVP.
    """
    def __init__(self, llm_engine=None):
        logger.info("Initializing SemanticAlgorithmCore...")
        self.llm = llm_engine if llm_engine else LLMEngine()
        self.selector = LLMSelector()
        self.synthesizer = DataSynthesizer(self.llm)
        self.extractor = GlobalInfoExtractor(self.llm)

    def generate_execution_plan(self, text_segments: List[Dict[str, str]], pdf_name: str) -> Dict[str, Any]:
        """
        Uses LLM to evaluate parsed captions and determine which ones contain relevant data.
        Returns a whitelist of targets.
        """
        logger.info("Generating Execution Plan (Agentic Selection)...")
        return self.selector.select_relevant_assets(text_segments, pdf_name)

    def extract_gvp(self, context_obj: Dict[str, Any], full_text: str = "") -> Dict[str, Any]:
        """
        Extracts structural candidates and unknown terms from local context using the LLM.
        Then uses the ContextResolver to map unknown abbreviations against the full text.
        """
        # 1. Ask LLM to extract local variables and identify what it doesn't know
        extracted_info = self.extractor.extract(context_obj)
        candidates = extracted_info.get("global_candidates", {})
        unknowns = extracted_info.get("unknown_terms", [])
        
        # 2. Use Context Resolver on the full document Text
        resolver = ContextResolver(full_text=full_text, external_context={})
        resolved_unknowns = resolver.resolve(unknowns)
        
        return {
            "candidates": candidates,
            "resolved": resolved_unknowns,
            "raw_unknowns": unknowns
        }

    def semantic_assembly(self, item_data: Dict[str, Any], gvp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Synthesizes the raw data points from images with the text-based Global Variable Pool 
        into a finalized JSON array of experiments.
        """
        logger.info(f"Assembling Semantics for data points...")
        candidates = gvp.get("candidates", {})
        resolved = gvp.get("resolved", {})
        
        # Uses LLM or rule-based mapping to construct rows
        return self.synthesizer.synthesize(item_data, candidates, resolved)
