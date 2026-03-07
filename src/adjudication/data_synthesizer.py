
import json
from src.adjudication.llm_engine import LLMEngine

class DataSynthesizer:
    """
    Step 5c: Data Synthesis
    Combines Global Variable Pool + Row Data to produce final schema-compliant JSON.
    """
    def __init__(self, llm_engine=None):
        self.llm = llm_engine if llm_engine else LLMEngine()

    def synthesize(self, evidence_item, global_vars, resolved_context):
        """
        Synthesize final data for an evidence item.
        Args:
            evidence_item (dict): Original evidence (Table/Figure).
            global_vars (dict): { "Temperature": "20C", ... }
            resolved_context (dict): { "NB": "nitrobenzene", ... }
        Returns:
            list: List of reaction dictionaries.
        """
        is_table = "csv_path" in evidence_item
        all_results = []
        
        if is_table:
            data_content = evidence_item.get('data_content', "")
            lines = data_content.strip().split('\n')
            if not lines:
                return []
            
            header = lines[0]
            rows = lines[1:]
            
            # Chunking for large tables (20 rows per chunk)
            chunk_size = 20
            for i in range(0, len(rows), chunk_size):
                chunk_rows = rows[i:i + chunk_size]
                chunk_data = "\n".join([header] + chunk_rows)
                all_results.extend(self._call_llm_synthesize(evidence_item, global_vars, resolved_context, chunk_data))
        else:
            # Figure data points (also chunk for figures to avoid truncation)
            points = evidence_item.get('raw_data', [])
            chunk_size = 20
            for i in range(0, len(points), chunk_size):
                chunk_points = points[i:i + chunk_size]
                data_preview = json.dumps(chunk_points)
                all_results.extend(self._call_llm_synthesize(evidence_item, global_vars, resolved_context, data_preview))
            
        return all_results

    def _call_llm_synthesize(self, evidence_item, global_vars, resolved_context, data_preview):
        system_prompt = """
        You are an expert chemist. Your task is to synthesize structured reaction data from the provided "Global Variables", "Term Definitions", and "Raw Data Rows".
        
        Input:
        1. Global Variables: Conditions applying to all rows (unless overridden).
        2. Term Definitions: Meanings of abbreviations (e.g., "NB": "nitrobenzene").
        3. Raw Data: Rows from a table or points from a figure. 
           **Note**: For Figures, data points may have "X", "Y_Left", and/or "Y_Right".
        
        Logic:
        - For EACH row/point in Raw Data, output a standardized JSON object.
        - Merge Global Variables into each row. 
        - IF a row specifies a value (like X coordinate), it OVERRIDES the Global Variable.
        - **FOR FIGURES/CHARTS**: 
          - Axis Labels indicate what X, Y_Left, and Y_Right represent.
          - 'Series' often refers to a Product or a Result category.
          - **Dual Axes**: If 'Y_Left' and 'Y_Right' are present, map them to the corresponding fields (e.g. Left might be 'Yield', Right might be 'Selectivity') using Axis Titles.
        - **NON-HALLUCINATION**: 
          - Every value must have a basis in the provided metadata or data points. 
          - If you supplement a global condition (e.g. Temperature), ensure it can be found in the Caption or Global Variables.
          - Provide a "Source" field for EACH record indicating where the primary data came from (e.g., "Figure 6, Page 6").
        - **Substitution**: Substitute ALL abbreviations with their full names from "Term Definitions". 
        - **Product Inference**: If the Product is generic (e.g., "Main product"), infer the chemical name from context.
        
        Output Schema (List of Objects):
        [
          {
            "Reactants": [ {"name": "substrate_name", "smiles": "...", "role": "substrate"} ],
            "Products": [ {"name": "product_name", "smiles": "...", "yield": "95%", "conversion": "100%", "selectivity": "..."} ],
            "Catalyst": "...",
            "Solvent": "...",
            "Temperature": "...",
            "Pressure": "...",
            "Time": "...",
            "Other Parameters": { "Flow Rate": "...", "Conversion": "...", ... },
            "Source": "e.g., Figure 6, Page 6"
          },
          ...
        ]
        
        IMPORTANT: Return ONLY valid JSON list.
        """
        
        user_prompt = f"""
        --- GLOBAL VARIABLES ---
        {json.dumps(global_vars, indent=2)}
        
        --- TERM DEFINITIONS ---
        {json.dumps(resolved_context, indent=2)}
        
        --- RAW DATA ---
        {data_preview}
        """
        
        try:
            response = self.llm.chat(system_prompt, user_prompt)
            
            # Parse JSON
            cleaned_json = response
            if "```" in response:
                cleaned_json = response.split("```json")[-1].split("```")[0].strip()
            
            cleaned_json = cleaned_json.strip('`').strip()
            if cleaned_json.startswith('json'):
                  cleaned_json = cleaned_json[4:].strip()

            print(f"[DataSynthesizer] Raw LLM Response for {evidence_item.get('meta', {}).get('figure_id', 'unknown')}:\n{response}")
            return json.loads(cleaned_json)
        except Exception as e:
            print(f"DataSynthesizer: Failed to parse LLM response for chunk: {e}")
            return []
