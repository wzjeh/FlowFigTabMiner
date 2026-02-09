import os
import cv2
import pandas as pd
from src.parsing.table_filter import TableFilter
from src.extraction.table_structure import TableStructureRecognizer
from src.extraction.cell_classifier import CellClassifier
from src.extraction.content_recognizer import ContentRecognizer
from src.extraction.molecule_processor import MoleculeProcessor

class TablePipeline:
    def __init__(self):
        print("Initializing Table Pipeline...")
        from src.utils.config import load_config
        cfg = load_config()
        tables_cfg = cfg.get("tables", {})
        
        seg_model = tables_cfg.get("segmentation", {}).get("model_path")
        struct_model = tables_cfg.get("structure", {}).get("model_path")
        molscribe_path = tables_cfg.get("content", {}).get("molscribe_path")
        
        # Molecule Detection
        mol_det_cfg = tables_cfg.get("molecule_detection", {})
        mol_model_path = mol_det_cfg.get("model_path")
        mol_conf = mol_det_cfg.get("confidence_threshold", 0.25)

        self.filter = TableFilter(model_path=seg_model)
        self.structure = TableStructureRecognizer(model_name=struct_model)
        self.classifier = CellClassifier()
        self.recognizer = ContentRecognizer(molscribe_path=molscribe_path)
        
        # Initialize Molecule Processor
        # Check if model path exists to avoid error if user hasn't downloaded yet (though config verify checks this)
        self.molecule_processor = MoleculeProcessor(model_path=mol_model_path, conf_threshold=mol_conf)

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
        filter_res = self.filter.filter_tables([image_path], conf_threshold=0.4)[0]
        if not filter_res['is_table']:
            print(f"   -> Rejected by Table Filter (Conf: {filter_res.get('conf')}).")
            return {'is_valid': False, 'reason': 'Filtered by YOLO'}

        # Prepare output structure: data/intermediate/{pdf_name}/tables/{table_basename}/
        # 'output_dir' passed in is usually the base tables dir. We need a subfolder per table.
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
                    # Store path back in item for reference
                    item['saved_path'] = comp_path

        # Use the cropped table body for structure recognition
        if 'table_body' in components:
            # Taking the best body (already sorted or max conf logic in filter)
            # TableFilter returns 'table_body_crop' which is the best one.
            # But let's use the one from components list to be consistent if we saved it.
            # Actually TableFilter provides 'table_body_crop' key for convenience.
            body_crop = filter_res['table_body_crop']
            
            if table_output_dir:
                # It might have been saved in loop above, but let's ensure we know which one is the MAIN body
                # The loop saves ALL bodies. We need the specific one used for extraction.
                # Let's just save it as separate 'body_master.png' or reuse.
                # Simple: Overwrite/Ensure 'table_body.png' exists
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
        
                cv2.imwrite(body_path, body_crop)
                current_image_path = body_path
        
        # --- NEW: Process Molecules (Detect -> MolScribe -> Replace) ---
        print("   -> Processing Molecules...")
        # We need to pass the recognizer so it can use MolScribe
        # Function returns modified image (with SMILES text) and metadata
        modified_img, mol_meta = self.molecule_processor.process_image(current_image_path, self.recognizer)
        
        if modified_img is not None and mol_meta:
            print(f"      Replaced {len(mol_meta)} molecules with SMILES.")
            # Save modified image for structure recognition
            # Use a slightly different name to distinguish
            base_body = os.path.splitext(os.path.basename(current_image_path))[0]
            modified_body_path = os.path.join(os.path.dirname(current_image_path), f"{base_body}_smiles.png")
            cv2.imwrite(modified_body_path, modified_img)
            
            # Switch current_image_path to the modified one for Structure Recognition
            current_image_path = modified_body_path
        else:
            print("      No molecules detected or model not loaded.")
            
        # 2. Structure (on the body, potentially modified)
        struct_res = self.structure.recognize_structure(current_image_path)
        cells = struct_res.get('cells', [])
        
        if not cells:
             # Try fallback: intersection of rows and cols
             cells = self.structure.get_cells_from_grid(struct_res)
             
        if not cells:
            print("   -> No cells detected.")
            return {'is_valid': False, 'reason': 'No cells detected'}
            
        print(f"   -> Detected {len(cells)} cells.")

        # 3. Process Cells (Crop -> Classify -> Recognize)
        # Use the cropped body image for cell cropping
        original_img = cv2.imread(current_image_path)
        if original_img is None:
             return {'is_valid': False, 'reason': 'Image read error'}
             
        cell_crops = []
        cell_meta = []
        
        for i, cell in enumerate(cells):
            x1, y1, x2, y2 = map(int, cell['box'])
            # Pad slightly?
            h, w = original_img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = original_img[y1:y2, x1:x2]
            
            # Save crop for debug/classifier
            crop_filename = f"cell_{i}.png"
            if output_dir:
                crop_path = os.path.join(output_dir, crop_filename)
                cv2.imwrite(crop_path, crop)
                cell['crop_path'] = crop_path
            
            # Use crop directly for classification (convert BGR to RGB PIL) or path
            # Classifier handles PIL or Path. Here we use path if saved, else Convert
            if output_dir:
                 cell_crops.append(cell['crop_path'])
            else:
                 # In-memory PIL
                 import PIL.Image
                 img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                 cell_crops.append(PIL.Image.fromarray(img_rgb))
            
            cell_meta.append(cell)

        # 4. Classify Batch
        print("   -> Classifying cells...")
        classifications = self.classifier.classify_cells(cell_crops)
        
        # 5. Recognize Content
        print("   -> Recognizing content...")
        extracted_data = []
        
        # We need row/col info. 
        # If Table Transformer gave row/col indices (some models do, some don't).
        # Our `recognize_structure` implementation returns raw DETR boxes usually.
        # We need to assign row/col indices based on grid if not present.
        # Let's assume we need to assign them if missing.
        
        # Simple clustering for row/col assignment if missing
        if 'row_index' not in cell_meta[0]:
             self._assign_grid_indices(cell_meta)

        for i, (cell, cls) in enumerate(zip(cell_meta, classifications)):
            content = self.recognizer.recognize_content(cell_crops[i], cls)
            
            cell['class'] = cls
            cell['content'] = content
            
            extracted_data.append({
                'row': cell['row_index'],
                'col': cell['col_index'],
                'content': content
            })

        # 6. Construct DataFrame
        if not extracted_data:
             return {'is_valid': False, 'reason': 'No content extracted'}

        df_data = {}
        # Find max rows/cols
        max_row = max(d['row'] for d in extracted_data)
        max_col = max(d['col'] for d in extracted_data)
        
        # Populate grid
        grid = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        for item in extracted_data:
            grid[item['row']][item['col']] = item['content']
            
        df = pd.DataFrame(grid)
        
        # Save CSV
        csv_path = None
        if output_dir:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            csv_path = os.path.join(output_dir, f"{basename}.csv")
            df.to_csv(csv_path, index=False, header=False)
            print(f"   -> Saved CSV to {csv_path}")

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

