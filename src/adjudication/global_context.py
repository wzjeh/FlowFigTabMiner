
import json
import os
import sys
import re

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
        
        1. "global_candidates": Conditions that likely apply to the WHOLE table/figure.
           - Examples: "Reaction at 20°C", "Solvent: MeOH", "Base: NaH", "H2 Flow rate: 15 sccm" (Note: H2 might appear as "H 2" or with subscripts).
           - **CRITICAL**: Do NOT label reactor systems (e.g., "H-flow", "H-Cube", "FlowSyn", "Micropacked bed") as "Catalyst". They are Equipment.
           - Ignore variables that change per row/column.
        2. "unknown_terms": 
           - Any chemical abbreviation or code (e.g., "3,4-DCAN", "NB", "1a", "2b", "P1").
           - Any acronym (e.g., "DMF", "THF" - though common, listing them allows checking specific grades if needed, but prioritize non-standard ones).
           - Look at Column Headers and Cell Values in the snippet. If you see "3,4-DCAN" as a substrate, list it!
           
        Output only valid JSON:
        {
            "global_candidates": { "Temperature": "25C", "Hydrogen Flow Rate": "15 sccm", ... },
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
            cleaned_json = cleaned_json.strip('`')
            if cleaned_json.startswith('json'):
                cleaned_json = cleaned_json[4:]
            
            result = json.loads(cleaned_json)
            
            # Post-processing: Filter out Equipment from Catalyst
            candidates = result.get("global_candidates", {})
            bad_catalysts = ["h-flow", "h-cube", "flowsyn", "micropacked bed", "flow system", "reactor"]
            
            keys_to_remove = []
            for k, v in candidates.items():
                val_lower = str(v).lower()
                key_lower = str(k).lower()
                
                # If key is specifically "Catalyst" and value is bad, remove it
                if "catalyst" in key_lower:
                    if any(bad in val_lower for bad in bad_catalysts):
                        keys_to_remove.append(k)
                        
                # If the Key itself is "H-flow", remove it (it's not a condition)
                if any(bad in key_lower for bad in bad_catalysts):
                     keys_to_remove.append(k)

            for k in keys_to_remove:
                del candidates[k]

            # Post-processing: Regex fallback for H2 Flow Rate
            # Pattern: H2 (subscript ignored) flow rate [:] value
            # Matches: "H2 flow rate: 15 sccm", "H 2 flow rate 15 mL/min"
            # Tweaked regex to catch "H2 flow rate: 15 sccm"
            h2_pattern = r"(?:H\s*2|Hydrogen)\s+flow\s+rate\D{0,10}?(\d+(?:\.\d+)?\s*[a-zA-Z]+(?:/[a-zA-Z]+)?)"
            match_h2 = re.search(h2_pattern, context_text, re.IGNORECASE)
            
            # Only add if not already present
            has_h2 = any("flow" in k.lower() and ("h2" in k.lower() or "hydrogen" in k.lower()) for k in candidates)
            if match_h2 and not has_h2:
                candidates["Hydrogen Flow Rate"] = match_h2.group(1).strip()

            result["global_candidates"] = candidates
            return result

        except Exception as e:
            print(f"GlobalInfoExtractor: Failed to parse LLM response: {e}")
            return {"global_candidates": {}, "unknown_terms": []}
