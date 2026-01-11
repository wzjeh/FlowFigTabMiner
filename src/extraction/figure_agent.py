import json
import re
from src.llm_factory import LLMFactory

def extract_figure(image_path, context=""):
    """
    Extracts data from a figure image using Vision LLM.
    
    Args:
        image_path (str): Path to the figure image file.
        context (str): Global context from the paper.
        
    Returns:
        dict: Extracted data/parameters.
    """
    print(f"Extracting figure info from: {image_path}")
    
    prompt = """
    You are an expert data analyst. 
    Analyze the provided image from a scientific paper.
    
    1. **Classification**: Identify the type of figure.
       - **ACCEPTABLE TYPES**: "line_chart" (折线图), "bar_chart" (柱状图).
       - **UNACCEPTABLE TYPES**: "reaction_scheme", "flow_diagram", "equipment_photo", "table_screenshot", "other".
    
    2. **Extraction**: 
       - IF ACCEPTABLE: Extract the following:
         - "chart_type": "line_chart" or "bar_chart"
         - "x_axis_label": Label of X axis
         - "y_axis_label": Label of Y axis
         - "data_summary": A brief textual summary of the trend or data shown.
       - IF UNACCEPTABLE: Set "is_valid" to false.

    Output the result in the following JSON format ONLY:
    {
        "is_valid": boolean,
        "reason": "Detected type [Type]",
        "chart_data": { 
             "chart_type": "...", 
             "x_axis_label": "...", 
             "y_axis_label": "...",
             "data_summary": "..." 
        } (or null if invalid)
    }
    """
    
    try:
        response_text = LLMFactory.create_completion(prompt, image_path=image_path)
        
        if not response_text:
            print("Error: No response from LLM.")
            return {"is_valid": False, "reason": "No response"}

        # Clean markdown code blocks if present
        cleaned_text = re.sub(r'```json\s*', '', response_text)
        cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        figure_data = json.loads(cleaned_text)
        return figure_data
        
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from LLM response. Response: {response_text[:100]}...")
        return {"is_valid": False, "reason": "JSON Parse Error"}
    except Exception as e:
        print(f"Error during figure extraction: {e}")
        return {"is_valid": False, "reason": f"Exception: {str(e)}"}
