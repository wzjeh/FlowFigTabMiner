from transformers import TableTransformerForObjectDetection, AutoImageProcessor
import torch
from PIL import Image
import numpy as np

class TableStructureRecognizer:
    def __init__(self, model_name="microsoft/table-transformer-structure-recognition-v1.1-all"):
        """
        Initialize Table Transformer for structure recognition.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Table Structure model: {model_name} on {self.device}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = TableTransformerForObjectDetection.from_pretrained(model_name).to(self.device)
        except Exception as e:
            print(f"Error loading Table Structure model: {e}")
            self.model = None

    def recognize_structure(self, image_path):
        """
        Detect cells, rows, and columns in a table image.
        Returns:
            dict: {
                'cells': [{'box': [x1, y1, x2, y2], 'score': float}],
                'rows': [...],
                'columns': [...]
            }
        """
        if self.model is None:
            return {'cells': []}

        try:
            image = Image.open(image_path).convert("RGB")
            # Explicitly pass size to avoid error with deprecated/malformed model config
            # DETR standard: shortest_edge=800, longest_edge=1333
            # But let's check what the model actually expects. 
            # Safest is to provide what standard DETR expects if config is broken.
            # The error "Got dict_keys(['longest_edge'])" suggests it has one but misses the other.
            
            # Using DETR defaults
            inputs = self.processor(images=image, return_tensors="pt", size={"shortest_edge": 800, "longest_edge": 1333})
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process
            # target_sizes should be (height, width)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
            
            cells = []
            rows = []
            columns = []
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                label_str = self.model.config.id2label[label.item()]
                item = {
                    'box': box,
                    'score': round(score.item(), 2),
                    'label': label_str
                }
                
                if label_str == 'table row':
                    rows.append(item)
                elif label_str == 'table column':
                    columns.append(item)
                elif label_str == 'table column header':
                    pass
                elif label_str == 'table': 
                    pass 
                else:
                    # PubTables-1M specific: 'table cell', 'table header', 'table spanning cell'
                    # It natively detects cells much better.
                    cells.append(item)
            
            # Prefer grid intersection if we have rows and columns, as it provides a complete grid
            # The 'cells' list often only contains spanning cells or specific types in this model
            structure = {
                'cells': cells,
                'rows': sorted(rows, key=lambda x: x['box'][1]), # Sort by Y
                'columns': sorted(columns, key=lambda x: x['box'][0]), # Sort by X
                'image_size': image.size
            }
            
            # Generate base grid cells from rows/cols intersection
            grid_cells = self.get_cells_from_grid(structure)
            
            # Identify spanning cells (detected as objects)
            spanning_cells = [c for c in cells if c['label'] == 'table spanning cell']
            
            # Merge Strategy:
            # 1. Start with all grid cells.
            # 2. For each spanning cell, find grid cells that overlap significantly.
            # 3. Replace those grid cells with the spanning cell (or a unified box).
            
            final_cells = []
            
            if not spanning_cells:
                final_cells = grid_cells
            else:
                # Mark grid cells to be removed
                cells_to_remove = set()
                
                for span in spanning_cells:
                    sx1, sy1, sx2, sy2 = span['box']
                    span_area = (sx2 - sx1) * (sy2 - sy1)
                    
                    # Find covered grid cells
                    covered_indices = []
                    for i, gc in enumerate(grid_cells):
                        if i in cells_to_remove: continue
                        
                        gx1, gy1, gx2, gy2 = gc['box']
                        
                        # Intersection
                        ix1 = max(sx1, gx1)
                        iy1 = max(sy1, gy1)
                        ix2 = min(sx2, gx2)
                        iy2 = min(sy2, gy2)
                        
                        if ix2 > ix1 and iy2 > iy1:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            gc_area = (gx2 - gx1) * (gy2 - gy1)
                            
                            # If overlap covers most of the grid cell
                            if inter_area / gc_area > 0.6:
                                covered_indices.append(i)
                    
                    if covered_indices:
                        # Add span cell (using union of covered grid cells for alignment?)
                        # For now, use the detected span box as it's cleaner than union of potentially noisy grid
                        # But we need row/col indices.
                        # Min row/col of covered cells
                        min_r = min([grid_cells[i]['row_index'] for i in covered_indices])
                        max_r = max([grid_cells[i]['row_index'] for i in covered_indices])
                        min_c = min([grid_cells[i]['col_index'] for i in covered_indices])
                        max_c = max([grid_cells[i]['col_index'] for i in covered_indices])
                        
                        span['row_index'] = min_r
                        span['col_index'] = min_c
                        span['row_span'] = max_r - min_r + 1
                        span['col_span'] = max_c - min_c + 1
                        
                        final_cells.append(span)
                        cells_to_remove.update(covered_indices)
                    else:
                        # Span didn't cover any grid cell significantly (weird), keep it?
                        pass 
                        
                # Add remaining grid cells
                for i, gc in enumerate(grid_cells):
                    if i not in cells_to_remove:
                        final_cells.append(gc)
                        
            structure['cells'] = final_cells
            return structure
            
        except Exception as e:
            print(f"Error extracting structure from {image_path}: {e}")
            return {'cells': [], 'error': str(e)}

    def get_cells_from_grid(self, structure_data):
        """
        Compute cell bounding boxes from rows and columns intersections.
        """
        rows = structure_data.get('rows', [])
        cols = structure_data.get('columns', [])
        
        cells = []
        for r_idx, row in enumerate(rows):
            for c_idx, col in enumerate(cols):
                # Intersection
                rx1, ry1, rx2, ry2 = row['box']
                cx1, cy1, cx2, cy2 = col['box']
                
                x1 = max(rx1, cx1)
                y1 = max(ry1, cy1)
                x2 = min(rx2, cx2)
                y2 = min(ry2, cy2)
                
                if x2 > x1 and y2 > y1:
                    cells.append({
                        'box': [x1, y1, x2, y2],
                        'row_index': r_idx,
                        'col_index': c_idx,
                        'row_span': 1,
                        'col_span': 1,
                        'label': 'table cell'
                    })
        return cells
