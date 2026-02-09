import os
import sys
import argparse
import json
import glob

# Ensure src is importable
sys.path.insert(0, os.getcwd())

from src.adjudication.llm_engine import LLMEngine
from src.adjudication.pdf_parser import PDFParser

def load_evidence(intermediate_dir, pdf_id):
    """
    Gather all *_evidence.json files from figures and tables.
    """
    evidence = {
        "figures": [],
        "tables": []
    }
    
    # 1. Figures (in data/evidence)
    # We look for evidence files that match the pdf_id prefix
    # Fig evidence naming: {pdf_id}_figure_{i}_evidence.json ??
    # Actually FigurePipeline naming is: {pdf_id without page}_page_{p}_figure_{f}.png
    # And evidence is saved as {figure_id}_evidence.json
    # So we scan data/evidence for files starting with pdf_id
    
    ev_dir = "data/evidence"
    if os.path.exists(ev_dir):
        # We need to be careful with matching. 
        # If pdf_id is "paper1", we match "paper1_page_*.json"
        pattern = os.path.join(ev_dir, f"*{pdf_id}*_evidence.json")
        for f in glob.glob(pattern):
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    if data.get('is_relevant', True): # Default to true if missing, but usually false if checked
                         evidence["figures"].append(data)
            except: pass

    # 2. Tables (in intermediate_dir/tables/{table_subfolder})
    # Table evidence naming: {pdf_id}_table_{i}_evidence.json inside valid subfolders
    tables_dir = os.path.join(intermediate_dir, "tables")
    if os.path.exists(tables_dir):
        # Walk through subdirectories
        for root, dirs, files in os.walk(tables_dir):
            for file in files:
                if file.endswith("_evidence.json"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r') as fp:
                            data = json.load(fp)
                            if data.get('is_relevant', False): # Table pipeline explicitly adds this
                                evidence["tables"].append(data)
                    except: pass
                    
    return evidence

def main():
    parser = argparse.ArgumentParser(description="Step 5: Global Assembly (LLM)")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument("--output_dir", help="Output directory", default="data/final_output")
    parser.add_argument("--prompt_file", help="Path to prompt template", default="prompts/global_summary.txt")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    print(f"--- Step 5: Global Assembly for {basename} ---")
    
    # 1. Extract Text
    print("[1/4] Extracting Text...")
    parser_tool = PDFParser()
    full_text = parser_tool.extract_text(pdf_path)
    print(f"      Extracted {len(full_text)} chars.")
    
    # 2. Gather Evidence
    print("[2/4] Gathering Evidence...")
    # Assume intermediate dir structure
    # Scripts usually output to data/intermediate by default config, but here we can try to find it
    # We assume 'data/intermediate/{basename}'
    intermediate_dir = os.path.join("data/intermediate", basename)
    
    evidence = load_evidence(intermediate_dir, basename)
    print(f"      Found {len(evidence['figures'])} relevant figures.")
    print(f"      Found {len(evidence['tables'])} relevant tables.")
    
    if not evidence['figures'] and not evidence['tables']:
        print("Warning: No relevant evidence found. Proceeding with text only.")

    # 3. Load Prompt
    print("[3/4] Preparing LLM Prompt...")
    if os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as f:
            system_prompt = f.read()
    else:
        print(f"Warning: Prompt file {args.prompt_file} not found. Using default.")
        system_prompt = "You are a scientist. Summarize the experimental results in JSON."
        
    # Construct Context
    # We strip huge extracted_data logs if they exist to save tokens, 
    # but we usually want the CSV content (Tables) and Data Points (Figures).
    
    # Minimize Table JSONs (already done in Step 3 saving, but good to be sure)
    tables_context = []
    for t in evidence['tables']:
        # We want path or content? 
        # Only CSV path is there. We should read the CSV content for the LLM!
        csv_path = t.get('csv_path')
        csv_content = "[CSV Not Found]"
        if csv_path and os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                csv_content = f.read()
        
        tables_context.append({
            "caption": t.get('caption_text'),
            "note": t.get('table_note_text'),
            "data": csv_content
        })

    figures_context = []
    for f in evidence['figures']:
        figures_context.append({
            "id": f.get('meta', {}).get('figure_id'),
            "caption": f.get('text_evidence', {}).get('caption', ''),
            "data_points": f.get('raw_data', [])
        })

    user_prompt = f"""
    Paper Content (Truncated):
    {full_text[:50000]}
    
    --- RELEVANT TABLES ---
    {json.dumps(tables_context, indent=2)}
    
    --- RELEVANT FIGURES ---
    {json.dumps(figures_context, indent=2)}
    """
    
    # 4. Call LLM
    print("[4/4] sending to LLM...")
    llm = LLMEngine()
    response = llm.chat(system_prompt, user_prompt)
    
    # 5. Save
    out_path = os.path.join(output_dir, f"{basename}_final_summary.json")
    
    # Try to clean json
    cleaned_json = response
    try:
        if "```" in response:
            cleaned_json = response.split("```json")[-1].split("```")[0].strip()
        # Parse to ensure valid
        parsed = json.loads(cleaned_json)
        with open(out_path, 'w') as f:
            json.dump(parsed, f, indent=2)
    except:
        # Save raw if parse fails
        with open(out_path, 'w') as f:
            f.write(response)

    print(f"Success! Final Summary saved to: {out_path}")
    
    # Output for UI
    result = {
        "status": "success",
        "output_path": out_path
    }
    print("---JSON_START---")
    print(json.dumps(result))
    print("---JSON_END---")

if __name__ == "__main__":
    main()
