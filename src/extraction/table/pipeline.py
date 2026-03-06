import os
import cv2
import pandas as pd
from src.parsing.table_filter import TableFilter
from src.extraction.table.structure import TableStructureRecognizer
from src.extraction.table.cell_classifier import CellClassifier
from src.extraction.common.content_recognizer import ContentRecognizer
from src.extraction.common.molecule_processor import MoleculeProcessor

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
        print(f"Initializing Table Pipeline (Sequential Mode: {sequential_mode})...")
        from src.utils.config import load_config
        self.cfg = load_config()
        self.tables_cfg = self.cfg.get("tables", {})
        self.sequential_mode = sequential_mode
        
        # Initialize placeholders
        self.filter = table_filter
        self.structure = structure_recognizer
        self.recognizer = content_recognizer
        self.molecule_processor = molecule_processor
        self.classifier = CellClassifier()

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

        # 3. Content Recognizer (OCR + MolScribe)
        if not self.recognizer:
            molscribe_path = self.tables_cfg.get("content", {}).get("molscribe_path")
            self.recognizer = ContentRecognizer(molscribe_path=molscribe_path)

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
            molscribe_path = self.tables_cfg.get("content", {}).get("molscribe_path")
            return ContentRecognizer(molscribe_path=molscribe_path)
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
            
        print(f"Processing Table: {image_path}")
        
        # 1. Filter
        filter_model = self._get_model('filter')
        try:
            filter_res = filter_model.filter_tables([image_path], conf_threshold=0.4)[0]
        finally:
            self._unload_model(filter_model)

        if not filter_res['is_table']:
            print(f"   -> Rejected by Table Filter (Conf: {filter_res.get('conf')}).")
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
            print(f"   -> Saving components to {table_output_dir}...")
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
                print(f"   -> Using cropped table body: {body_path}")
            else:
                import tempfile
                fd, body_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                cv2.imwrite(body_path, body_crop)
                current_image_path = body_path
        
        # --- Process Molecules (Detect -> MolScribe -> Replace) ---
        print("   -> Processing Molecules...")
        mol_processor = self._get_model('molecule')
        recognizer_for_mol = self._get_model('content') # Needed for MolScribe
        mol_meta = []
        try:
            # Construct debug path for molecule detection visualization
            base_body = os.path.splitext(os.path.basename(current_image_path))[0]
            debug_mol_path = os.path.join(os.path.dirname(current_image_path), f"{base_body}_debug_yolo.png")
            
            # Use mask_only=True to white-out molecules without writing text, preventing TATR interference.
            modified_img, mol_meta = mol_processor.process_image(
                current_image_path, 
                recognizer_for_mol, 
                mask_only=True,
                output_path=debug_mol_path
            )
        finally:
            self._unload_model(mol_processor)
            self._unload_model(recognizer_for_mol)
        
        if modified_img is not None and mol_meta:
            print(f"      Replaced {len(mol_meta)} molecules with White Masks (for Structure Rec).")
            # Rename to _masked to reflect that it is just masked, not text-replaced
            modified_body_path = os.path.join(os.path.dirname(current_image_path), f"{base_body}_masked.png")
            cv2.imwrite(modified_body_path, modified_img)
            current_image_path = modified_body_path
        else:
            print("      No molecules detected or model not loaded.")
            
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
            print("   -> No cells detected.")
            return {'is_valid': False, 'reason': 'No cells detected'}
            
        print(f"   -> Detected {len(cells)} cells.")

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

        # 4. Classify Batch
        print("   -> Classifying cells...")
        
        cells_to_classify_indices = []
        cells_to_classify_crops = []
        
        # Reuse robust overlapping logic from previous version
        def get_overlap(box1, box2):
             # Simplified for brevity in replacement, but ideally should be robust
             # Let's rely on overlap logic implemented below
             pass 

        for i, cell in enumerate(cell_meta):
             match_mol = None
             if mol_meta:
                 for m in mol_meta:
                     # Robust IoU Logic inline
                     boxA = m['box']
                     boxB = cell['box']
                     xA = max(boxA[0], boxB[0])
                     yA = max(boxA[1], boxB[1])
                     xB = min(boxA[2], boxB[2])
                     yB = min(boxA[3], boxB[3])
                     interArea = max(0, xB - xA) * max(0, yB - yA)
                     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                     
                     overlap_mol = 0.0
                     if boxAArea > 0: overlap_mol = interArea / boxAArea
                     
                     if overlap_mol > 0.5: 
                         match_mol = m
                         break
             
             if match_mol:
                 cell['class'] = 'Molecule'
                 cell['content'] = match_mol.get('smiles', '')
             else:
                 cells_to_classify_indices.append(i)
                 cells_to_classify_crops.append(cell_crops[i])
        
        if cells_to_classify_crops:
             cls_results = self.classifier.classify_cells(cells_to_classify_crops)
        else:
             cls_results = []
             
        # 5. Recognize Content (for non-molecules)
        print("   -> Recognizing content...")
        if not cells_to_classify_indices:
             print("      (All cells matched to molecules)")

        extracted_data = []
        if cell_meta and 'row_index' not in cell_meta[0]:
             self._assign_grid_indices(cell_meta)
        
        # Load Content Recognizer AGAIN for OCR
        recognizer_for_ocr = self._get_model('content')
        try:
            cls_idx = 0
            for i, cell in enumerate(cell_meta):
                 if 'content' in cell and cell.get('class') == 'Molecule':
                     content = cell['content']
                     cls = 'Molecule'
                 else:
                     cls = cls_results[cls_idx]
                     content = recognizer_for_ocr.recognize_content(cell_crops[i], cls)
                     cell['class'] = cls
                     cell['content'] = content
                     cls_idx += 1
                
                 extracted_data.append({
                    'row': cell['row_index'],
                    'col': cell['col_index'],
                    'content': content
                 })
        finally:
            self._unload_model(recognizer_for_ocr)

        # 6. Construct DataFrame
        if not extracted_data:
             return {'is_valid': False, 'reason': 'No content extracted'}

        max_row = max(d['row'] for d in extracted_data)
        max_col = max(d['col'] for d in extracted_data)
        
        grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for item in extracted_data:
            grid[item['row']][item['col']] = item['content']
            
        df = pd.DataFrame(grid)
        
        csv_path = None
        if output_dir:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            csv_path = os.path.join(output_dir, f"{basename}.csv")
            try:
                df.to_csv(csv_path, index=False, header=False)
                print(f"   -> Saved CSV to {csv_path}")
            except Exception as e:
                print(f"   -> Failed to save CSV: {e}")

        return {
            'is_valid': True,
            'csv_path': csv_path,
            'dataframe': df,
            'cells': cell_meta,
            'structure': struct_res
        }

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

