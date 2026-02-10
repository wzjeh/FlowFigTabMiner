
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
        # Prepare Context
        is_table = "csv_path" in evidence_item
        
        data_preview = ""
        if is_table:
            # We need the actual data! 
            # Step 5 orchestrator should have loaded CSV content.
            # Assuming evidence_item has 'data_content' key injected by orchestrator.
            data_preview = evidence_item.get('data_content', "[No CSV Data]")
        else:
            # Figure data points
            # Assuming raw_data is list of points
            points = evidence_item.get('raw_data', [])
            # Truncate if too many?
            data_preview = json.dumps(points[:50]) 
            
        system_prompt = """
        You are an expert chemist. Your task is to synthesize structured reaction data from the provided "Global Variables", "Term Definitions", and "Raw Data Rows".
        
        Input:
        1. Global Variables: Conditions applying to all rows (unless overridden).
        2. Term Definitions: Meanings of abbreviations (e.g., "NB": "nitrobenzene").
        3. Raw Data: Rows from a table or points from a figure.
        
        Logic:
        - For EACH row/point in Raw Data, output a standardized JSON object.
        - Merge Global Variables into each row. 
        - IF a row specifies a value, it OVERRIDES the Global Variable.
        - **CRITICAL 1**: Substitute ALL abbreviations with their full names from "Term Definitions". 
          - e.g. If "NB" is resolved to "nitrobenzene", output "nitrobenzene" in the "name" field.
        - **CRITICAL 2**: If the Product is listed generically (e.g., "Main product", "Product 3a", "Yield"), 
          INFER the chemical name based on the Substrate and Reaction Type.
        - **CRITICAL 3**: Do NOT list equipment (e.g., "H-flow", "FlowSyn", "Reactor") as "Catalyst". 
          Catalyst must be a chemical substance (e.g., Pd/C, Raney Ni). If none, use null.
        
        Output Schema (List of Objects):
        [
          {
            "Reactants": [ {"name": "nitrobenzene", "smiles": "...", "role": "substrate"} ],
            "Products": [ {"name": "aniline", "smiles": "...", "yield": "95%"} ],
            "Catalyst": "Pd/C",
            "Solvent": "MeOH",
            "Temperature": "e.g., 50 C",
            "Pressure": "e.g., 10 bar",
            "Time": "Residence time or Reaction time",
            "Other Parameters": "Any other fixed conditions (e.g., H2 Flow Rate, Base, Additive)",
            "Source": "Table 1, Row 3"
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
            
            cleaned_json = cleaned_json.strip('`')
            if cleaned_json.startswith('json'):
                 cleaned_json = cleaned_json[4:]

            return json.loads(cleaned_json)
        except Exception as e:
            print(f"DataSynthesizer: Failed to parse LLM response: {e}")
            return []
