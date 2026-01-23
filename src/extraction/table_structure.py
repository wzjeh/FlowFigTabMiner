from transformers import TableTransformerForObjectDetection, AutoImageProcessor
import torch
from PIL import Image
import numpy as np

class TableStructureRecognizer:
    def __init__(self, model_name="microsoft/table-transformer-structure-recognition"):
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
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
            
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
                elif label_str == 'table column header': # Treat as cell container or specific row
                    pass
                else:
                    # Generic cell or spanning cell
                    cells.append(item)
            
            # If we only get rows/cols, we might need to compute grid intersections to define cells
            # But Table Transformer usually outputs 'table spanning cell' etc.
            # Many implementations reconstruct cells from rows/cols intersections.
            
            # Simplified logic: If explicit cells are few, assume intersection of rows/cols
            # But 'microsoft/table-transformer-structure-recognition' effectively detects rows and columns.
            # We need to construct the grid.
            
            return {
                'cells': cells,
                'rows': sorted(rows, key=lambda x: x['box'][1]), # Sort by Y
                'columns': sorted(columns, key=lambda x: x['box'][0]), # Sort by X
                'image_size': image.size
            }
            
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
                        'is_header': False # Logic to detect header? Maybe first row?
                    })
        return cells
