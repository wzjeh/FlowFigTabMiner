import os
import json
import logging
import pandas as pd
from typing import Dict, Any, List

# Re-use existing components for now. Later they can also be moved to flow_dev_miner/processing_layer/components
from src.parsing.yolo_detector import YoloDetector
from src.parsing.stage2_detector import Stage2Detector
from src.extraction.figure.legend_matcher import LegendMatcher
from src.extraction.figure.coordinate_mapper import CoordinateMapper
from src.assembly.evidence_assembler import EvidenceAssembler
from src.utils.config import load_config

logger = logging.getLogger(__name__)

class FigureProcessor:
    """
    Handles processing of scientific figures.
    Responsibilities:
    - Marco Segmentation: YOLO segmentation for legend, axes, and data areas.
    - Micro Detection: Extraction of scatter points.
    - Coordinate Mapping: Alignment and reading of axes labels to physical units.
    - Assembly: Filtering and JSON storage.
    """
    def __init__(self):
        logger.info("Initializing Figure Processor...")
        self.cfg = load_config()
        
        # Load Configs
        fig_cfg = self.cfg.get("figures", {})
        macro_cfg = fig_cfg.get("step2_macro", {})
        micro_cfg = fig_cfg.get("step3_micro", {})
        
        # Paths
        self.macro_model_path = macro_cfg.get("model_path", "models/bestYOLOn-2-1.pt")
        self.micro_model_path = micro_cfg.get("model_path", "models/bestYOLOm-2-2.pt")
        
        self.macro_conf = macro_cfg.get("confidence_threshold", 0.5)
        self.micro_conf = micro_cfg.get("confidence_threshold", 0.25)
        
        # Lazy loading of models to save memory
        self._yolo_macro = None
        self._yolo_micro = None
        self._legend_matcher = None
        self._coord_mapper = None
        self._assembler = None

    @property
    def yolo_macro(self):
        if self._yolo_macro is None:
            logger.info(f"Loading YOLO Macro: {self.macro_model_path}")
            self._yolo_macro = YoloDetector(model_path=self.macro_model_path)
        return self._yolo_macro

    @property
    def yolo_micro(self):
        if self._yolo_micro is None:
            logger.info(f"Loading YOLO Micro: {self.micro_model_path}")
            self._yolo_micro = Stage2Detector(model_path=self.micro_model_path)
        return self._yolo_micro

    @property
    def legend_matcher(self):
        if self._legend_matcher is None:
            logger.info("Loading LegendMatcher...")
            self._legend_matcher = LegendMatcher(yolo_model=self.yolo_micro)
        return self._legend_matcher

    @property
    def coord_mapper(self):
        if self._coord_mapper is None:
            logger.info("Loading CoordinateMapper...")
            self._coord_mapper = CoordinateMapper()
        return self._coord_mapper

    @property
    def assembler(self):
        if self._assembler is None:
            logger.info("Loading EvidenceAssembler...")
            self._assembler = EvidenceAssembler()
        return self._assembler

    def process_figure(self, image_path: str, output_base_dir: str) -> Dict[str, Any]:
        """
        Main entry point for single figure processing.
        Returns extracted data points and metadata in a structured format.
        """
        logger.info(f"Processing Figure: {image_path}")
        
        # 1. Macro Cleaning
        macro_results = self.yolo_macro.process_images(
            [image_path], 
            output_base_dir=output_base_dir, 
            output_subdir_name="macro_cleaned",
            imgsz=1024 
        )
        
        if not macro_results:
            logger.warning("Macro cleaning returned no results.")
            return {"status": "error", "message": "Macro cleaning failed."}
            
        item = macro_results[0]
        cleaned_plot_path = item['cleaned_image']
        elements = item['elements']
        
        figure_id = os.path.splitext(os.path.basename(cleaned_plot_path))[0]
        if figure_id.endswith("_cleaned"):
            figure_id = figure_id.replace("_cleaned", "")
            
        macro_cleaned_dir = os.path.dirname(cleaned_plot_path)
        
        # 2. Check Relevance (Optimization)
        is_relevant, text_evidence = self.assembler.check_relevance(figure_id, macro_cleaned_dir)
        if not is_relevant:
            logger.info(f"Figure '{figure_id}' is not relevant based on keywords.")
            return {"status": "skipped", "reason": "not relevant", "figure_id": figure_id}

        # 3. Micro Detection
        micro_detections = self.yolo_micro.detect(cleaned_plot_path, conf=self.micro_conf, imgsz=1024, use_tiling=False)
        points = [d for d in micro_detections if d['label'] in ['data_point', 'marker']]
        
        # 4. Legend Matching
        legend_crops = elements.get('legend', [])
        prototypes = self.legend_matcher.parse_legend_crops(legend_crops)
        matched_points = self.legend_matcher.match_points(points, prototypes, cleaned_plot_path)
        
        # 5. Coordinate Mapping
        other_detections = [d for d in micro_detections if d['label'] not in ['data_point', 'marker']]
        full_detections = other_detections + matched_points
        
        try:
            df, _ = self.coord_mapper.map_coordinates(full_detections, cleaned_plot_path)
        except Exception as e:
            logger.warning(f"Coordinate Mapping Warning: {e}")
            df = pd.DataFrame()
            
        extraction_data = []
        if not df.empty:
            extraction_data = df.to_dict(orient='records')
        else:
            for p in matched_points:
                extraction_data.append({
                    "series": p.get('series', 'Unknown'),
                    "x_pixel": p['center'][0],
                    "y_pixel": p['center'][1],
                    "note": "CoordMapping Failed"
                })
                
        # 6. Assembly & Saving
        json_path = self.assembler.assemble(figure_id, extraction_data, macro_cleaned_dir, text_evidence=text_evidence)
        
        return {
            "status": "success",
            "figure_id": figure_id,
            "evidence_path": json_path,
            "points_extracted": len(extraction_data),
            "mapped": not df.empty
        }

    def process_batch(self, figure_images: List[str], output_base_dir: str) -> List[Dict[str, Any]]:
        """
        Process multiple figures and return a list of results.
        """
        results = []
        for img in figure_images:
            res = self.process_figure(img, output_base_dir)
            results.append(res)
        return results
