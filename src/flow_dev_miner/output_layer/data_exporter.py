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

    def export_to_excel(self, assembled_data: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Saves the final unified array of experiments to a human-readable Excel file.
        Flattens nested chemical structures for tabular viewing.
        """
        try:
            import pandas as pd
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            flattened_data = [self._flatten_record(r) for r in assembled_data]
            df = pd.DataFrame(flattened_data)
            
            # Smart Column Ordering
            priority_cols = [
                'Source_ID', 'Reactant_1', 'Product_1', 'Yield', 'Conversion', 
                'Temperature', 'Pressure', 'Time', 'Flow Rate', 'SMILES_Reactant', 'SMILES_Product'
            ]
            last_cols = ['Verification_Image']
            
            # Get existing cols
            existing_cols = list(df.columns)
            order = [c for c in priority_cols if c in existing_cols]
            middle_cols = [c for c in existing_cols if c not in order and c not in last_cols]
            final_order = order + middle_cols + [c for c in last_cols if c in existing_cols]
            
            df = df[final_order]
            df.to_excel(output_path, index=False)
            logger.info(f"Successfully exported {len(assembled_data)} records to Excel: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to Excel: {e}")
            return False

    def _flatten_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flattens a nested experiment record into a single row dictionary.
        Optimized for human readability with aliases for common fields.
        """
        flat = {}
        
        # 1. Base Info
        flat['Source_ID'] = record.get('_source_key', 'Unknown')
        flat['Temperature'] = record.get('Temperature')
        flat['Pressure'] = record.get('Pressure')
        flat['Time'] = record.get('Time')
        flat['Catalyst'] = record.get('Catalyst')
        flat['Solvent'] = record.get('Solvent')
        
        # 2. Reactants
        reactants = record.get('Reactants', [])
        if reactants:
            flat['Reactant_1'] = reactants[0].get('name')
            flat['SMILES_Reactant'] = reactants[0].get('smiles')
        
        # 3. Products
        products = record.get('Products', [])
        if products:
            first_prod = products[0]
            flat['Product_1'] = first_prod.get('name')
            flat['SMILES_Product'] = first_prod.get('smiles')
            flat['Yield'] = first_prod.get('yield')
            
        # 4. Other Parameters & Aliases
        others = record.get('Other Parameters', {})
        if isinstance(others, dict):
             for k, v in others.items():
                 k_low = k.lower().strip()
                 # Normalize common aliases
                 if k_low in ['flow rate', 'flowrate', 'liquid flow rate']:
                     flat['Flow Rate'] = v
                 elif k_low in ['conversion', 'conv.']:
                     flat['Conversion'] = v
                 elif k_low in ['yield', 'yld']:
                     if not flat.get('Yield'): flat['Yield'] = v
                 else:
                     flat[k] = v
        
        # 5. Image Link
        flat['Verification_Image'] = record.get('_visual_verification', '')
        
        return flat

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
