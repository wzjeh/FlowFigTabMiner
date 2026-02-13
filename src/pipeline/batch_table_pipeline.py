import os
import glob
import time
from tqdm import tqdm
import cv2
import pandas as pd
import json

from src.utils.config import load_config
from src.extraction.table_pipeline import TablePipeline
from src.parsing.table_filter import TableFilter
from src.extraction.table_structure import TableStructureRecognizer
from src.extraction.molecule_processor import MoleculeProcessor
from src.extraction.content_recognizer import ContentRecognizer

class BatchTablePipeline:
    def __init__(self):
        """
        Initialize all models ONCE for batch processing.
        """
        print("--- initializing Batch Table Pipeline (Loading Models Once) ---")
        self.cfg = load_config()
        tables_cfg = self.cfg.get("tables", {})

        # 1. Load Filter (YOLO Segmentation)
        seg_model = tables_cfg.get("segmentation", {}).get("model_path")
        print(f"Loading Table Filter: {seg_model}")
        self.table_filter = TableFilter(model_path=seg_model)

        # 2. Load Structure (Table Transformer)
        struct_model = tables_cfg.get("structure", {}).get("model_path")
        print(f"Loading Structure Recognizer: {struct_model}")
        self.structure_recognizer = TableStructureRecognizer(model_name=struct_model)
        
        # 3. Load Content Recognizer (PaddleOCR + MolScribe)
        molscribe_path = tables_cfg.get("content", {}).get("molscribe_path")
        print(f"Loading Content Recognizer (OCR + MolScribe: {molscribe_path})")
        self.content_recognizer = ContentRecognizer(molscribe_path=molscribe_path)

        # 4. Load Molecule Processor (YOLO Molecules)
        mol_det_cfg = tables_cfg.get("molecule_detection", {})
        mol_model_path = mol_det_cfg.get("model_path")
        mol_conf = mol_det_cfg.get("confidence_threshold", 0.25)
        print(f"Loading Molecule Processor: {mol_model_path}")
        self.molecule_processor = MoleculeProcessor(model_path=mol_model_path, conf_threshold=mol_conf)
        
        # 5. Instantiate Pipeline with injected models
        self.pipeline = TablePipeline(
            table_filter=self.table_filter,
            structure_recognizer=self.structure_recognizer,
            molecule_processor=self.molecule_processor,
            content_recognizer=self.content_recognizer
        )
        print("--- Models Loaded Successfully ---")

    def process_directory(self, input_dir, output_base_dir):
        """
        Process all valid table images in a directory.
        """
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} not found.")
            return

        # Find images
        # We look for ANY image that looks like a table? 
        # Or specifically ones organized by Step 1?
        # Step 1 puts them in data/intermediate/{pdf_name}/tables/
        # User might point to data/intermediate directly and want recursive?
        # Let's support recursive search for *.png
        
        print(f"Scanning {input_dir}...")
        image_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True))
        
        # --- Hybrid Agentic Filtering ---
        # Look for selected_assets.json in the parent folder of input_dir
        # input_dir = data/intermediate/{pdf}/tables -> parent = data/intermediate/{pdf}
        parent_dir = os.path.dirname(input_dir.rstrip(os.sep))
        selection_path = os.path.join(parent_dir, "selected_assets.json")
        whitelist = None
        
        if os.path.exists(selection_path):
            print(f"Found Selection File: {selection_path}")
            try:
                with open(selection_path, 'r') as f:
                    sel_data = json.load(f)
                    whitelist = set(sel_data.get("selected_tables", []))
                    print(f"Applying Whitelist: {len(whitelist)} tables selected.")
            except Exception as e:
                print(f"Error reading selection file: {e}")

        # Filter: Exclude already processed crops (body, cell, smiles)
        # We want the ROOT table images.
        # Heuristic: exclude files containing "_body", "_cell", "_smiles", "_viz", "_crop"
        to_process = []
        for f in image_files:
            name = os.path.basename(f)
            if any(x in name for x in ["_body", "_cell", "_smiles", "_viz", "_crop", "lab_pub_"]):
                continue
            
            # Apply Whitelist
            if whitelist is not None:
                if name not in whitelist:
                    continue
                    
            to_process.append(f)
            
        print(f"Found {len(to_process)} potential raw table images (Filtered).")
        
        results_summary = []
        
        for img_path in tqdm(to_process, desc="Batch Processing Tables"):
            try:
                # Determine output path
                # If input is data/intermediate/PDF_A/tables/table_1.png
                # We want output to be in data/intermediate/PDF_A/tables/table_1/ (handled by pipeline)
                # Or if output_base_dir is specified, we mirror structure?
                # Pipeline.process_table(image_path, output_dir)
                # If output_dir is passed, it creates a subfolder there.
                
                # Best strategy: Output to the SAME directory as the input image, in a subfolder named after the table?
                # Or strict output_base_dir control.
                
                # Let's calculate a specific output dir for this table.
                # If img_path is .../tables/table_1.png
                # output should be .../tables/ (so pipeline creates table_1 subfolder)
                
                current_output_dir = os.path.dirname(img_path)
                if output_base_dir:
                    # If user specified a global output root, we need to map the relative path
                    rel_path = os.path.relpath(os.path.dirname(img_path), input_dir)
                    current_output_dir = os.path.join(output_base_dir, rel_path)
                
                res = self.pipeline.process_table(img_path, output_dir=current_output_dir)
                
                if res.get('is_valid'):
                    results_summary.append({
                        "table": os.path.basename(img_path),
                        "status": "success",
                        "csv": res.get('csv_path')
                    })
                else:
                     results_summary.append({
                        "table": os.path.basename(img_path),
                        "status": "filtered",
                        "reason": res.get('reason')
                    })
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                
        return results_summary
