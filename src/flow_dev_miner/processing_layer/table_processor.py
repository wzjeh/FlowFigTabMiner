import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd

# Re-use existing components for now. 
from src.utils.config import load_config
from src.extraction.table.pipeline import TablePipeline
try:
    from src.parsing.table_filter import TableFilter
except ImportError:
    from src.extraction.table.table_filter import TableFilter
from src.extraction.table.structure import TableStructureRecognizer
from src.extraction.common.molecule_processor import MoleculeProcessor
from src.extraction.common.content_recognizer import ContentRecognizer

logger = logging.getLogger(__name__)

class TableProcessor:
    """
    Handles processing of scientific tables.
    Responsibilities:
    - Table Segmentation: YOLO filtering to locate table body.
    - Structure Recognition: TATR (Table Transformer) cell detection.
    - Molecule Detection: YOLO detection of molecules within cells.
    - Content Recognition: MolNexTR (for SMILES) & OCR (for text).
    - Layout Assembly: Producing final CSV/JSON representation.
    """
    def __init__(self):
        logger.info("Initializing Table Processor...")
        self.cfg = load_config()
        tables_cfg = self.cfg.get("tables", {})

        # Configuration Paths
        self.seg_model = tables_cfg.get("segmentation", {}).get("model_path")
        self.struct_model = tables_cfg.get("structure", {}).get("model_path")
        # MolScribe removed - MolNexTR is auto-loaded by ContentRecognizer
        
        mol_det_cfg = tables_cfg.get("molecule_detection", {})
        self.mol_model_path = mol_det_cfg.get("model_path")
        self.mol_conf = mol_det_cfg.get("confidence_threshold", 0.25)
        
        # Lazy loading components
        self._table_filter = None
        self._structure_recognizer = None
        self._content_recognizer = None
        self._molecule_processor = None
        self._pipeline = None

    @property
    def table_filter(self):
        if self._table_filter is None:
            logger.info(f"Loading Table Filter: {self.seg_model}")
            self._table_filter = TableFilter(model_path=self.seg_model)
        return self._table_filter

    @property
    def structure_recognizer(self):
        if self._structure_recognizer is None:
            logger.info(f"Loading Structure Recognizer: {self.struct_model}")
            self._structure_recognizer = TableStructureRecognizer(model_name=self.struct_model)
        return self._structure_recognizer

    @property
    def content_recognizer(self):
        if self._content_recognizer is None:
            logger.info("Loading Content Recognizer (MolNexTR)")
            self._content_recognizer = ContentRecognizer()
        return self._content_recognizer

    @property
    def molecule_processor(self):
        if self._molecule_processor is None:
            logger.info(f"Loading Molecule Processor: {self.mol_model_path}")
            self._molecule_processor = MoleculeProcessor(model_path=self.mol_model_path, conf_threshold=self.mol_conf)
        return self._molecule_processor

    @property
    def pipeline(self):
        if self._pipeline is None:
            logger.info("Instantiating TablePipeline with underlying models...")
            self._pipeline = TablePipeline(
                table_filter=self.table_filter,
                structure_recognizer=self.structure_recognizer,
                molecule_processor=self.molecule_processor,
                content_recognizer=self.content_recognizer
            )
        return self._pipeline

    def process_table(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Main entry point for single table processing.
        Produces structured CSV/JSON containing text and SMILES.
        """
        logger.info(f"Processing Table: {image_path}")
        
        try:
            res = self.pipeline.process_table(image_path, output_dir=output_dir)
            
            if res.get('is_valid'):
                return {
                    "status": "success",
                    "table_name": os.path.basename(image_path),
                    "csv_path": res.get('csv_path'),
                    "json_path": res.get('json_path')
                }
            else:
                return {
                    "status": "skipped",
                    "table_name": os.path.basename(image_path),
                    "reason": res.get('reason')
                }
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                "status": "error",
                "table_name": os.path.basename(image_path),
                "error": str(e)
            }

    def process_batch(self, table_images: List[str], input_dir: str, output_base_dir: str = None) -> List[Dict[str, Any]]:
        """
        Process multiple tables efficiently by keeping models in memory.
        """
        results_summary = []
        for img_path in table_images:
            # Determine target output dir
            current_output_dir = os.path.dirname(img_path)
            if output_base_dir:
                rel_path = os.path.relpath(os.path.dirname(img_path), input_dir)
                current_output_dir = os.path.join(output_base_dir, rel_path)
                
            res = self.process_table(img_path, current_output_dir)
            results_summary.append(res)
            
        return results_summary
