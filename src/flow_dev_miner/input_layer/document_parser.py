import os
import logging
from typing import Dict, Any, List

from src.parsing.active_area_detector import ActiveAreaDetector
from src.adjudication.pdf_parser import PDFParser

logger = logging.getLogger(__name__)

class DocumentInputManager:
    """
    Input Layer: Handles the ingestion of scientific PDFs.
    Responsibilities:
    - PDF to Image conversion.
    - TF-ID scanning (Layout Analysis, Figure/Table region extraction).
    - Full text extraction from PDF.
    """
    def __init__(self):
        logger.info("Initializing DocumentInputManager...")
        self.detector = ActiveAreaDetector()
        self.pdf_parser = PDFParser()

    def process_pdf(self, pdf_path: str, intermediate_dir: str) -> Dict[str, Any]:
        """
        Parses the PDF, extracts layout regions (crops for figures/tables), 
        and extracts the full text.
        """
        logger.info(f"Extracting layout and content from {pdf_path}")
        
        # 1. Image Regions (TF-ID)
        detections = self.detector.process_pdf(pdf_path)
        saved_paths = self.detector.save_crops(pdf_path, detections, intermediate_dir)
        
        # 2. Complete Text Extraction
        full_text = self.pdf_parser.extract_text(pdf_path)
        
        return {
            "pdf_path": pdf_path,
            "crops_saved": len(saved_paths),
            "output_dir": intermediate_dir,
            "full_text": full_text
        }
