import os
import cv2
import pandas as pd
from src.parsing.table_filter import TableFilter
from src.extraction.table.structure import TableStructureRecognizer
from src.extraction.table.header_corrector import HeaderCorrector
from src.extraction.common.content_recognizer import ContentRecognizer
from src.extraction.common.molecule_processor import MoleculeProcessor
import json
import glob
import logging

logger = logging.getLogger(__name__)

class TablePipeline:
    def __init__(self, table_filter=None, structure_recognizer=None, molecule_processor=None, content_recognizer=None, sequential_mode=False):
        """
        Initialize Table Pipeline.
        Args:
            table_filter: Optional pre-loaded TableFilter instance.
            structure_recognizer: Optional pre-loaded TableStructureRecognizer instance.
            molecule_processor: Optional pre-loaded MoleculeProcessor instance.
            content_recognizer: Optional pre-loaded ContentRecognizer instance.
            sequential_mode (bool): If True, models are loaded/unloaded on demand to save memory.
        """
        logger.info(f"Initializing Table Pipeline (Sequential Mode: {sequential_mode})")
        from src.utils.config import load_config
        self.cfg = load_config()
        self.tables_cfg = self.cfg.get("tables", {})
        self.sequential_mode = sequential_mode

        # Initialize placeholders
        self.filter = table_filter
        self.structure = structure_recognizer
        self.recognizer = content_recognizer
        self.molecule_processor = molecule_processor
        # Removed CellClassifier - we use YOLO molecule detection instead

        # VLM header corrector (lightweight, no model loaded at init)
        header_cfg = self.tables_cfg.get("header_correction", {})
        self.header_corrector = HeaderCorrector(header_cfg)

        # If NOT sequential, load everything now (Standard Behavior)
        if not self.sequential_mode:
            self._load_all_models()

    def _load_all_models(self):
        # 1. Table Filter (Segmentation)
        if not self.filter:
            seg_model = self.tables_cfg.get("segmentation", {}).get("model_path")
            self.filter = TableFilter(model_path=seg_model)

        # 2. Structure Recognizer
        if not self.structure:
            struct_model = self.tables_cfg.get("structure", {}).get("model_path")
            self.structure = TableStructureRecognizer(model_name=struct_model)

        # 3. Content Recognizer (OCR + MolNexTR)
        if not self.recognizer:
            self.recognizer = ContentRecognizer()

        # 4. Molecule Processor
        if not self.molecule_processor:
            mol_det_cfg = self.tables_cfg.get("molecule_detection", {})
            mol_model_path = mol_det_cfg.get("model_path")
            mol_conf = mol_det_cfg.get("confidence_threshold", 0.25)
            self.molecule_processor = MoleculeProcessor(model_path=mol_model_path, conf_threshold=mol_conf)

    def _get_model(self, model_type):
        """Helper for sequential mode loading"""
        if model_type == 'filter':
            if self.filter: return self.filter
            seg_model = self.tables_cfg.get("segmentation", {}).get("model_path")
            return TableFilter(model_path=seg_model)
            
        elif model_type == 'molecule':
            if self.molecule_processor: return self.molecule_processor
            mol_det_cfg = self.tables_cfg.get("molecule_detection", {})
            mol_model_path = mol_det_cfg.get("model_path")
            mol_conf = mol_det_cfg.get("confidence_threshold", 0.25)
            return MoleculeProcessor(model_path=mol_model_path, conf_threshold=mol_conf)
            
        elif model_type == 'structure':
            if self.structure: return self.structure
            struct_model = self.tables_cfg.get("structure", {}).get("model_path")
            return TableStructureRecognizer(model_name=struct_model)
            
        elif model_type == 'content':
            if self.recognizer: return self.recognizer
            return ContentRecognizer()
        return None

    def _unload_model(self, model_instance):
        """Helper to unload model and clear CUDA cache if possible"""
        if not self.sequential_mode: return
        # Simple deletion, let GC handle it. 
        # For Torch/CUDA, explicit empty_cache might be needed but we are likely on generic memory pressure.
        del model_instance
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif torch.backends.mps.is_available(): torch.mps.empty_cache() # Mac specific
        except: pass
        import gc
        gc.collect()

    def process_table(self, image_path, output_dir=None):
        """
        Process a single table image.
        Args:
            image_path (str): Path to table image.
            output_dir (str, optional): Directory to save debug crops/results.
        Returns:
            dict: {
                'is_valid': bool,
                'csv_path': str,
                'dataframe': pd.DataFrame,
                'cells': list of dicts (debug info),
                'structure': dict
            }
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Processing Table: {image_path}")
        
        # 1. Filter
        filter_model = self._get_model('filter')
        try:
            filter_res = filter_model.filter_tables([image_path], conf_threshold=0.4)[0]
        finally:
            self._unload_model(filter_model)

        if not filter_res['is_table']:
            logger.info(f"   -> Rejected by Table Filter (Conf: {filter_res.get('conf')})")
            return {'is_valid': False, 'reason': 'Filtered by YOLO'}

        # Prepare output structure: data/intermediate/{pdf_name}/tables/{table_basename}/
        current_image_path = image_path
        table_basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # If output_dir is provided, append basename to create dedicated folder
        table_output_dir = output_dir
        if output_dir:
            table_output_dir = os.path.join(output_dir, table_basename)
            os.makedirs(table_output_dir, exist_ok=True)

        # Save all segmented components
        components = filter_res.get('components', {})
        if table_output_dir and components:
            logger.info(f"   -> Saving {len(components)} component types to {table_output_dir}")
            for label, items in components.items():
                for i, item in enumerate(items):
                    comp_filename = f"{table_basename}_{label}_{i}.png"
                    comp_path = os.path.join(table_output_dir, comp_filename)
                    cv2.imwrite(comp_path, item['crop'])
                    item['saved_path'] = comp_path

        # Use the cropped table body for structure recognition
        if 'table_body' in components:
            body_crop = filter_res['table_body_crop']
            if table_output_dir:
                body_path = os.path.join(table_output_dir, f"{table_basename}_body_main.png")
                cv2.imwrite(body_path, body_crop)
                current_image_path = body_path
                logger.info(f"   -> Using cropped table body: {body_path}")
            else:
                import tempfile
                fd, body_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                cv2.imwrite(body_path, body_crop)
                current_image_path = body_path
        
        # --- Load ContentRecognizer ONCE for the entire table ---
        # (PaddleOCR + MolNexTR ~1 GB — load once, reuse for molecule/OCR/context, unload at end)
        shared_recognizer = self._get_model('content')

        # --- Process Molecules (Detect -> MolNexTR -> Mask) ---
        logger.info("   -> Detecting and masking molecules with YOLO...")
        mol_processor = self._get_model('molecule')
        mol_meta = []
        try:
            base_body = os.path.splitext(os.path.basename(current_image_path))[0]
            debug_mol_path = os.path.join(os.path.dirname(current_image_path), f"{base_body}_debug_yolo.png")
            modified_img, mol_meta = mol_processor.process_image(
                current_image_path,
                shared_recognizer,
                mask_only=True,
                output_path=debug_mol_path
            )
        finally:
            self._unload_model(mol_processor)

        if modified_img is not None and mol_meta:
            logger.info(f"      -> Masked {len(mol_meta)} molecules with white fills")
            modified_body_path = os.path.join(os.path.dirname(current_image_path), f"{base_body}_masked.png")
            cv2.imwrite(modified_body_path, modified_img)
            current_image_path = modified_body_path
        else:
            logger.info("      -> No molecules detected in table")
            
        # 2. Structure (on the body, potentially modified)
        structure_model = self._get_model('structure')
        cells = []
        try:
            struct_res = structure_model.recognize_structure(current_image_path)
            cells = struct_res.get('cells', [])
            if not cells:
                 cells = structure_model.get_cells_from_grid(struct_res)
        finally:
            self._unload_model(structure_model)

        if not cells:
            logger.warning("   -> No cells detected by TATR")
            return {'is_valid': False, 'reason': 'No cells detected'}

        logger.info(f"   -> Detected {len(cells)} cells via TATR")

        # 3. Process Cells (Crop -> Classify -> Recognize)
        original_img = cv2.imread(current_image_path)
        if original_img is None:
             return {'is_valid': False, 'reason': 'Image read error'}
             
        cell_crops = []
        cell_meta = []
        
        for i, cell in enumerate(cells):
            x1, y1, x2, y2 = map(int, cell['box'])
            h, w = original_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = original_img[y1:y2, x1:x2]
            
            if output_dir:
                cells_dir = os.path.join(output_dir, "cells")
                os.makedirs(cells_dir, exist_ok=True)
                crop_filename = f"cell_{i}.png"
                crop_path = os.path.join(cells_dir, crop_filename)
                cv2.imwrite(crop_path, crop)
                cell['crop_path'] = crop_path
            
            if output_dir:
                 cell_crops.append(cell['crop_path'])
            else:
                 import PIL.Image
                 img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                 cell_crops.append(PIL.Image.fromarray(img_rgb))
            
            cell_meta.append(cell)

        # 4. Cell Content Recognition (Optimized: No Classification Needed)
        logger.info("   -> Mapping cells to molecule detections...")

        # Map cells to YOLO-detected molecules using IoU overlap
        def calculate_iou_overlap(boxA, boxB):
            """Calculate IoU overlap between two boxes [x1, y1, x2, y2]"""
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interArea = max(0, xB - xA) * max(0, yB - yA)
            if interArea == 0:
                return 0.0

            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            if boxAArea <= 0:
                return 0.0

            return interArea / boxAArea

        # Mark cells that contain molecules
        cells_with_molecules = []
        cells_for_ocr = []

        for i, cell in enumerate(cell_meta):
            match_mol = None
            best_overlap = 0.0

            if mol_meta:
                for m in mol_meta:
                    overlap = calculate_iou_overlap(m['box'], cell['box'])
                    if overlap > 0.5 and overlap > best_overlap:
                        match_mol = m
                        best_overlap = overlap

            if match_mol:
                cell['class'] = 'Molecule'
                cell['content'] = match_mol.get('smiles', '')
                cell['cell_index'] = i
                cells_with_molecules.append(i)
                logger.debug(f"      Cell {i} -> Molecule: {cell['content'][:20]}...")
            else:
                cell['class'] = 'Text'  # Default to text/number (OCR will handle both)
                cell['cell_index'] = i
                cells_for_ocr.append(i)

        logger.info(f"   -> {len(cells_with_molecules)} cells contain molecules, {len(cells_for_ocr)} cells need OCR")

        # 5. OCR for non-molecule cells
        logger.info("   -> Running OCR on text/number cells...")

        extracted_data = []
        if cell_meta and 'row_index' not in cell_meta[0]:
             self._assign_grid_indices(cell_meta)

        # Run OCR on non-molecule cells (reuse shared_recognizer)
        for i, cell in enumerate(cell_meta):
            if cell.get('class') == 'Molecule':
                content = cell['content']
                logger.debug(f"      Cell {i} (Molecule): {content[:30]}...")
            else:
                content = shared_recognizer.recognize_content(cell_crops[i], 'Text')
                cell['content'] = content
                logger.debug(f"      Cell {i} (Text/Number): {content[:30]}...")

            extracted_data.append({
                'row': cell['row_index'],
                'col': cell['col_index'],
                'content': content,
                'type': cell.get('class', 'Text')
            })

        # 6. Construct DataFrame
        if not extracted_data:
             return {'is_valid': False, 'reason': 'No content extracted'}

        max_row = max(d['row'] for d in extracted_data)
        max_col = max(d['col'] for d in extracted_data)

        grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for item in extracted_data:
            grid[item['row']][item['col']] = item['content']

        # 6b. VLM Header Correction (if enabled and heuristic triggers)
        if self.header_corrector.enabled and self.header_corrector.needs_correction(grid, struct_res):
            logger.info("   -> Header quality low, calling VLM for correction...")
            grid = self.header_corrector.correct(current_image_path, grid, struct_res)

        df = pd.DataFrame(grid)
        
        csv_path = None
        if output_dir:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            csv_path = os.path.join(output_dir, f"{basename}_extracted.csv")
            try:
                df.to_csv(csv_path, index=False, header=False)
                logger.info(f"   -> Saved CSV to {csv_path}")
            except Exception as e:
                logger.error(f"   -> Failed to save CSV: {e}")

        # --- 7. Synthesize Context (Caption/Note) via OCR ---
        context_data = {
            "caption": [],
            "table_note": []
        }
        
        if table_output_dir:
            # We assume table filter put captions/notes into table_output_dir just earlier!
            # The filenames are e.g. {table_basename}_table_caption_*.png
            cap_pattern = os.path.join(table_output_dir, f"{table_basename}_table_caption_*.png")
            note_pattern = os.path.join(table_output_dir, f"{table_basename}_table_note_*.png")
            
            # Reuse shared_recognizer for caption/note OCR
            for c_path in sorted(glob.glob(cap_pattern)):
                c_img = cv2.imread(c_path)
                if c_img is not None:
                    c_img = cv2.resize(c_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    pad = 50
                    c_img = cv2.copyMakeBorder(c_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    c_rgb = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
                    txt = shared_recognizer._recognize_text(c_rgb)
                    if txt.strip(): context_data["caption"].append(txt)

            for n_path in sorted(glob.glob(note_pattern)):
                n_img = cv2.imread(n_path)
                if n_img is not None:
                    n_img = cv2.resize(n_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    pad = 50
                    n_img = cv2.copyMakeBorder(n_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    n_rgb = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
                    txt = shared_recognizer._recognize_text(n_rgb)
                    if txt.strip(): context_data["table_note"].append(txt)

        # 8. Check Relevance
        import yaml
        kw_candidates = [
            "keywords.yaml",
            os.path.join(os.getcwd(), "keywords.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "keywords.yaml"),
        ]
        keywords = ['yield', 'conversion', 'selectivity', 'product', 'composition', 'conditions', 'reaction']
        for kw_path in kw_candidates:
            if os.path.exists(kw_path):
                try:
                    with open(kw_path, 'r') as f:
                        kw_config = yaml.safe_load(f)
                        keywords = kw_config.get('keywords', keywords)
                    break
                except Exception:
                    pass

        check_text = (" ".join(context_data["caption"]) + " " + " ".join(context_data["table_note"])).lower()
        if not df.empty:
            check_text += " " + df.head(3).to_string(index=False, header=False).lower()

        is_relevant = any(kw in check_text for kw in keywords)

        if not is_relevant:
            logger.info(f"   -> Table marked not relevant by keyword filter (soft reject, CSV still written)")

        # 9. Format Evidence JSON output
        result_packet = {
            'is_valid': True,
            'is_relevant': is_relevant,
            'csv_path': csv_path,
            'dataframe': df,
            'cells': cell_meta,
            'structure': struct_res,
            'num_extracted': len(extracted_data),
            'caption_text': " ".join(context_data["caption"]),
            'table_note_text': " ".join(context_data["table_note"])
        }

        if table_output_dir:
             evidence_data = {
                 "csv_path": csv_path,
                 "num_extracted": len(extracted_data),
                 "caption_text": result_packet["caption_text"],
                 "table_note_text": result_packet["table_note_text"],
                 "is_relevant": is_relevant
             }
             json_filename = f"{table_basename}_evidence.json"
             json_path = os.path.join(table_output_dir, json_filename)
             with open(json_path, 'w') as f:
                 json.dump(evidence_data, f, indent=2)
             result_packet['json_path'] = json_path
             logger.info(f"   -> Saved Evidence JSON to: {json_path}")

        # Unload ContentRecognizer once per table (not 3× per table)
        self._unload_model(shared_recognizer)

        return result_packet

    def _assign_grid_indices(self, cells):
        # Very simple heuristic:
        # Sort by Y center to find rows
        # Sort by X center to find cols
        # Better: Clustering
        
        # Centroids
        for c in cells:
            c['cx'] = (c['box'][0] + c['box'][2]) / 2
            c['cy'] = (c['box'][1] + c['box'][3]) / 2
            
        # Cluster Y for Rows
        ys = sorted([c['cy'] for c in cells])
        # Simple threshold based clustering
        rows = []
        if ys:
            current_row = [ys[0]]
            for y in ys[1:]:
                if y - current_row[-1] > 10: # Threshold 10 pixels?
                    rows.append(sum(current_row)/len(current_row))
                    current_row = [y]
                else:
                    current_row.append(y)
            rows.append(sum(current_row)/len(current_row))
            
        # Cluster X for Cols
        xs = sorted([c['cx'] for c in cells])
        cols = []
        if xs:
            current_col = [xs[0]]
            for x in xs[1:]:
                if x - current_col[-1] > 10:
                    cols.append(sum(current_col)/len(current_col))
                    current_col = [x]
                else:
                    current_col.append(x)
            cols.append(sum(current_col)/len(current_col))
            
        # Assign
        for c in cells:
            # Find closest row/col index
            r_idx = min(range(len(rows)), key=lambda i: abs(rows[i] - c['cy']))
            c_idx = min(range(len(cols)), key=lambda i: abs(cols[i] - c['cx']))
            c['row_index'] = r_idx
            c['col_index'] = c_idx

