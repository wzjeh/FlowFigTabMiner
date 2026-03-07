import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DataExporter:
    """
    Output Layer: Handles saving results accurately.
    Responsibilities:
    - Exporting to JSON / CSV formats.
    - Interfacing with databases (future).
    """
    def __init__(self):
        pass

    def export_to_json(self, assembled_data: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Saves the final unified array of experiments to a JSON file.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(assembled_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully exported exactly {len(assembled_data)} unified records to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False

    def export_debug_info(self, debug_data: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Saves debug metadata separately.
        """
        try:
             os.makedirs(os.path.dirname(output_path), exist_ok=True)
             with open(output_path, 'w', encoding='utf-8') as f:
                 json.dump(debug_data, f, indent=2, ensure_ascii=False)
             return True
        except Exception as e:
             logger.error(f"Failed to export debug info: {e}")
             return False
