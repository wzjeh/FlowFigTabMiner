import json
import re
from src.llm_factory import LLMFactory

def extract_table(table_content, context=""):
    """
    Extracts structured data from a table (Markdown or Image).
    
    Args:
        table_content (str): Path to the table image file.
        context (str): Global context from the paper (optional).
        
    Returns:
        dict: Structured JSON representation of the table.
    """
    print(f"Extracting table from: {table_content}")
    
    prompt = """
    You are an expert research assistant. 
    Analyze the provided image.
    
    1. **Validation**: Determine if this is a valid scientific data table.
       - IGNORE if it is: a generic page header/footer, a DOI string, a plain text paragraph, a list of references, or a figure caption without data.
       - ACCEPT if it is: a grid or list of data values with headers.
    
    2. **Extraction**: If valid, convert the table content into a standard CSV string.
       - Use comma separation.
       - Enclose fields with quotes if they contain commas.
    
    Output the result in the following JSON format ONLY:
    {
        "is_valid": boolean,
        "reason": "Brief reason for acceptance or rejection",
        "csv_content": "The extracted CSV string (or empty if invalid)"
    }
    """
    
    if context:
        prompt += f"\nContext from paper: {context[:500]}..." # Truncate context if too long
        
    try:
        # Assuming table_content is a file path to an image for now
        response_text = LLMFactory.create_completion(prompt, image_path=table_content)
        
        if not response_text:
            print("Error: No response from LLM.")
            return {"is_valid": False, "reason": "No response"}

        # Clean markdown code blocks if present
        cleaned_text = re.sub(r'```json\s*', '', response_text)
        cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        result = json.loads(cleaned_text)
        
        # If valid, maybe we want to parse the CSV effectively or just pass it through
        # For now, return the raw dictionary
        return result
        
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from LLM response. Response: {response_text[:100]}...")
        return {"is_valid": False, "reason": "JSON Parse Error"}
    except Exception as e:
        print(f"Error during table extraction: {e}")
        return {"is_valid": False, "reason": f"Exception: {str(e)}"}
