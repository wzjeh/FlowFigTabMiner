
import json
import os
import sys

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.adjudication.llm_engine import LLMEngine

class GlobalInfoExtractor:
    """
    Step 5a: Context Extraction
    Extracts 'Global Candidate Variables' and 'Unknown Terms' from Table/Figure metadata.
    """
    def __init__(self, llm_engine=None):
        self.llm = llm_engine if llm_engine else LLMEngine()

    def extract(self, evidence_item):
        """
        Extract context from a single evidence item (Table or Figure).
        Args:
            evidence_item (dict): Evidence JSON.
        Returns:
            dict: {
                "global_candidates": {"Temperature": "20C", ...},
                "unknown_terms": ["NB", "3a"]
            }
        """
        # Prepare Prompt Context
        is_table = "csv_path" in evidence_item
        
        context_text = ""
        if is_table:
            context_text += f"Caption: {evidence_item.get('caption_text', '')}\n"
            context_text += f"Note: {evidence_item.get('table_note_text', '')}\n"
            
            # Add table headers and first few rows
            data_str = evidence_item.get('data_content', '')
            if data_str:
                lines = data_str.strip().splitlines()
                # Take first 5 lines (header + 4 rows)
                snippet = "\n".join(lines[:5])
                context_text += f"Data Snippet:\n{snippet}\n"

        else:
            # Figure
            context_text += f"Caption: {evidence_item.get('caption', '')}\n"
            context_text += f"Axis Labels: {evidence_item.get('axis_labels', '')}\n" # specific to figure
            
        system_prompt = """
        You are an expert chemist. Your task is to extract "Global Experimental Conditions" and "Unknown Abbreviations/Chemical Codes" from the provided Table/Figure metadata.
        
        1. "global_candidates": Conditions that likely apply to the WHOLE table/figure (e.g., "Reaction at 20°C", "Solvent: MeOH"). 
           Ignore variables that change per row/column.
        2. "unknown_terms": 
           - Any chemical abbreviation or code (e.g., "3,4-DCAN", "NB", "1a", "2b", "P1").
           - Any acronym (e.g., "DMF", "THF" - though common, listing them allows checking specific grades if needed, but prioritize non-standard ones).
           - Look at Column Headers and Cell Values in the snippet. If you see "3,4-DCAN" as a substrate, list it!
           
        Output only valid JSON:
        {
            "global_candidates": { "Temperature": "25C", ... },
            "unknown_terms": [ "3,4-DCAN", "NB", "1a" ]
        }
        """
        
        user_prompt = f"Metadata:\n{context_text}"
        
        try:
            response = self.llm.chat(system_prompt, user_prompt)
            
            # Parse JSON
            cleaned_json = response
            if "```" in response:
                cleaned_json = response.split("```json")[-1].split("```")[0].strip()
            # If the response doesn't contain '```', try parsing directly
            # Remove markdown if any
            cleaned_json = cleaned_json.strip('`')
            if cleaned_json.startswith('json'):
                 cleaned_json = cleaned_json[4:]
                 
            return json.loads(cleaned_json)
        except Exception as e:
            print(f"GlobalInfoExtractor: Failed to parse LLM response: {e}")
            return {"global_candidates": {}, "unknown_terms": []}
