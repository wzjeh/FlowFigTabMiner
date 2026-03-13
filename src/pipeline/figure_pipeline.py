import os
import re
import glob
import pandas as pd
import json

# Fix OpenMP Conflict (Preserved from original script)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.parsing.yolo_detector import YoloDetector
from src.parsing.stage2_detector import Stage2Detector
from src.extraction.figure.legend_matcher import LegendMatcher
from src.extraction.figure.coordinate_mapper import CoordinateMapper
from src.assembly.evidence_assembler import EvidenceAssembler
from src.utils.config import load_config

class FigurePipeline:
    def __init__(self):
        print("Initializing Figure Pipeline...")
        self.cfg = load_config()
        
        # Load Configs
        fig_cfg = self.cfg.get("figures", {})
        macro_cfg = fig_cfg.get("step2_macro", {})
        micro_cfg = fig_cfg.get("step3_micro", {})
        
        # Paths
        # Fallback to hardcoded defaults if config missing (matching original behavior partially)
        self.macro_model_path = macro_cfg.get("model_path", "models/bestYOLOn-2-1.pt")
        self.micro_model_path = micro_cfg.get("model_path", "models/bestYOLOm-2-2.pt")
        
        self.macro_conf = macro_cfg.get("confidence_threshold", 0.5)
        self.micro_conf = micro_cfg.get("confidence_threshold", 0.25) # Default 0.25 if not set
        self.micro_crop_padding = micro_cfg.get("crop_padding", 10)
        
        # Initialize Models
        print(f"Loading YOLO Macro: {self.macro_model_path}")
        self.yolo_macro = YoloDetector(model_path=self.macro_model_path)
        
        print(f"Loading YOLO Micro: {self.micro_model_path}")
        # Stage2Detector uses 'model_path' in init? Yes.
        self.yolo_micro = Stage2Detector(model_path=self.micro_model_path)
        
        print("Loading LegendMatcher & CoordinateMapper...")
        self.legend_matcher = LegendMatcher(yolo_model=self.yolo_micro)
        self.coord_mapper = CoordinateMapper()
        
        print("Loading EvidenceAssembler...")
        self.assembler = EvidenceAssembler() # Output dir handled per run usually or default
        
    def process_pdf_figures(self, pdf_path):
        """
        Runs Step 2-4 for a given PDF.
        Assumes Step 1 (TF-ID) has already run and populated data/intermediate/{basename}/figures.
        """
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        intermediate_dir = "data/intermediate" # Could be configurable
        # If the user has a different intermediate structure, we might need to adjust.
        # But standard is data/intermediate/{basename}
        
        pdf_intermediate_dir = os.path.join(intermediate_dir, basename)
        figures_dir = os.path.join(pdf_intermediate_dir, "figures")
        
        if not os.path.exists(figures_dir):
            print(f"Error: Figures directory not found: {figures_dir}. Did Step 1 run?")
            return []

        # --- Hybrid Agentic Filtering ---
        whitelist = None
        selection_path = os.path.join(pdf_intermediate_dir, "selected_assets.json")
        if os.path.exists(selection_path):
            print(f"Found Selection File: {selection_path}")
            try:
                with open(selection_path, 'r') as f:
                    sel_data = json.load(f)
                    whitelist = set(sel_data.get("selected_figures", []))
                    print(f"Applying Whitelist: {len(whitelist)} figures selected.")
            except Exception as e:
                print(f"Error reading selection file: {e}")

        all_figures = glob.glob(os.path.join(figures_dir, "*.png"))
        
        if whitelist is not None:
            figure_images = [f for f in all_figures if os.path.basename(f) in whitelist]
            print(f"Filtered {len(all_figures)} -> {len(figure_images)} figures.")
        else:
            figure_images = all_figures

        print(f"--- Steps 2-4: Processing {len(figure_images)} figures from {figures_dir} ---")
        
        if not figure_images:
            return []
            
        return self.process_images(figure_images, pdf_intermediate_dir)

    def process_images(self, figure_images, output_base_dir):
        """
        Process a list of figure images (crops from Step 1).
        """
        extracted_results = []
        
        # Step 2: Macro Cleaning
        print("\nStep 2: Macro Cleaning...")
        # output_base_dir is typically data/intermediate/{pdf_name}
        # output_subdir_name="macro_cleaned" -> data/intermediate/{pdf_name}/macro_cleaned
        macro_results = self.yolo_macro.process_images(
            figure_images, 
            output_base_dir=output_base_dir, 
            output_subdir_name="macro_cleaned",
            imgsz=1024 # Match User Training
        )
        
        # Step 3 & 4
        print("\nStep 3 & 4: Micro Detection, Extraction & Assembly...")
        
        for item in macro_results:
            original_source = item['original_source']
            cleaned_plot_path = item['cleaned_image']
            elements = item['elements']
            
            # ID generation
            figure_id = os.path.splitext(os.path.basename(cleaned_plot_path))[0]
            if figure_id.endswith("_cleaned"):
                figure_id = figure_id.replace("_cleaned", "")
            
            print(f"   >>> Processing Chart: {figure_id}")
            
            try:
                # --- NEW OPTIMIZATION: Check Relevance BEFORE Step 3 ---
                # We need the macro_cleaned directory for text crops
                macro_cleaned_dir = os.path.dirname(cleaned_plot_path)
                
                print("      [Pipeline] Checking Relevance (OCR Captions)...")
                is_relevant, text_evidence = self.assembler.check_relevance(figure_id, macro_cleaned_dir)
                
                if not is_relevant:
                    print(f"      [Pipeline] SKIPPING Step 3. Chart '{figure_id}' is not relevant (no keywords).")
                    continue
                
                # If we are here, it's relevant! Proceed to Step 3.
                
                # 3A: Micro Detection
                # 3A: Micro Detection
                print(f"      [Step 3a] Micro Detection (conf={self.micro_conf})...")
                # User Training: Fit 1024x1024. 
                # Disable tiling to strictly match training distribution.
                micro_detections = self.yolo_micro.detect(cleaned_plot_path, conf=self.micro_conf, imgsz=1024, use_tiling=False)
                points = [d for d in micro_detections if d['label'] in ['data_point', 'marker']]
                print(f"      [Step 3a] Detected {len(points)} data points.")
                
                # 3B: Legend Matching
                legend_crops = elements.get('legend', [])
                prototypes = self.legend_matcher.parse_legend_crops(legend_crops)
                matched_points = self.legend_matcher.match_points(points, prototypes, cleaned_plot_path)
                
                # 3C: Coordinate Mapping
                other_detections = [d for d in micro_detections if d['label'] not in ['data_point', 'marker']]
                full_detections = other_detections + matched_points

                # 热图自动检测：legend 文本含 yield 范围（如 "60 ~ 80%", "< 20%", "> 80%"）
                is_heatmap = self._detect_heatmap(text_evidence)
                if is_heatmap:
                    print(f"      [Heatmap] Detected heatmap-style legend → enabling extract_point_labels + log_x")

                try:
                    df, _ = self.coord_mapper.map_coordinates(
                        full_detections, cleaned_plot_path,
                        force_log_x=is_heatmap,
                        extract_point_labels=is_heatmap,
                    )
                except Exception as e:
                    print(f"      [Mapper Warning] {e}")
                    df = pd.DataFrame()
                
                extraction_data = []
                if not df.empty:
                    extraction_data = df.to_dict(orient='records')
                else:
                    # Fallback: Just points
                     for p in matched_points:
                         extraction_data.append({
                            "series": p.get('series', 'Unknown'),
                            "x_pixel": p['center'][0],
                            "y_pixel": p['center'][1],
                            "note": "CoordMapping Failed"
                        })
                
                # Step 4: Assembly & Filtering
                # Pass pre-computed text_evidence to avoid re-OCR
                json_path = self.assembler.assemble(figure_id, extraction_data, macro_cleaned_dir, text_evidence=text_evidence)
                
                if json_path:
                    print(f"      -> EVIDENCE SAVED: {json_path}")
                    extracted_results.append(json_path)
                else:
                    print(f"      -> DISCARDED (Irrelevant).")
                    
            except Exception as e:
                print(f"      ! Error processing chart {figure_id}: {e}")
                import traceback
                traceback.print_exc()

        return extracted_results

    def _detect_heatmap(self, text_evidence: dict) -> bool:
        """
        检测图是否为热图（yield range legend）。
        判据：legend 文本含 yield 范围模式，如 "60 ~ 80%", "< 20%", "> 80%"
        """
        if not text_evidence:
            return False
        legend_items = text_evidence.get('legend_text', [])
        for item in legend_items:
            text = item.get('text', '')
            # 匹配 "数字 ~ 数字 %" 或 "< 数字 %" 或 "> 数字 %"
            if re.search(r'(\d+\s*[~\-]\s*\d+\s*%|[<>]\s*\d+\s*%)', text):
                return True
        return False
