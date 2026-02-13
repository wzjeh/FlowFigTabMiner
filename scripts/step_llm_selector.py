import os
import sys
import json
import glob
import argparse

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.adjudication.llm_engine import LLMEngine
from src.utils.config import load_config

class AssetSelector:
    def __init__(self):
        self.llm = LLMEngine()
        self.config = load_config()

    def select_assets(self, pdf_path):
        """
        Identify relevant figures and tables for extraction.
        """
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 1. Load PDF Text
        text_path = os.path.join("data/input", f"{basename}_fulltext.txt")
        if not os.path.exists(text_path):
            print(f"Warning: Text file {text_path} not found. Attempting extraction...")
            try:
                from src.adjudication.pdf_parser import PDFParser
                parser = PDFParser()
                full_text = parser.extract_text(pdf_path)
                # Save it for future use
                with open(text_path, 'w') as f:
                    f.write(full_text)
                print(f"Extracted and saved text to {text_path}")
            except Exception as e:
                print(f"Error extracting text: {e}")
                return None
        else:
            with open(text_path, 'r') as f:
                full_text = f.read()

        # 2. List Assets
        intermediate_dir = os.path.join("data/intermediate", basename)
        fig_dir = os.path.join(intermediate_dir, "figures")
        tab_dir = os.path.join(intermediate_dir, "tables")
        
        figures = [os.path.basename(f) for f in glob.glob(os.path.join(fig_dir, "*.png"))]
        tables = [os.path.basename(f) for f in glob.glob(os.path.join(tab_dir, "*.png")) 
                  if "_body" not in f and "_crop" not in f and "_viz" not in f] # Filter raw tables
        
        print(f"[DEBUG] Detected Figures in {fig_dir}: {figures}")
        print(f"[DEBUG] Detected Tables in {tab_dir}: {tables}")

        if not figures and not tables:
            print("No assets found.")
            return {"figures": [], "tables": []}

        # 2.5 Prepare Context
        # Truncate text to avoid token limits (intro + results usually < 10k tokens)
        truncated_text = full_text[:15000] + "\n...\n" + full_text[-5000:]
        
        asset_list_str = "Available Assets:\n"
        if figures:
            asset_list_str += "Figures: " + ", ".join(figures) + "\n"
        if tables:
            asset_list_str += "Tables: " + ", ".join(tables) + "\n"

        # 3. Load System Prompt from File
        prompt_path = "prompts/global_summary.txt"
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                system_prompt = f.read()
            print(f"Loaded System Prompt (Selector) from {prompt_path}")
        else:
            print(f"Warning: Prompt file {prompt_path} not found. Using default.")
            system_prompt = """
            You are an expert chemist and data miner. 
            Your goal is to identify which specific Figures and Tables in a paper contain the **Substrate Scope** or **Reaction Yields**.
            
            We have detected several images (Figures/Tables). Based on the paper text, determine which of these files are effectively "Substrate Scope" or "Results Tables".
            
            Rules:
            1. Select tables/figures that list specific chemical structures (substrates/products) and their yields/conversion.
            2. IGNORE "Optimization Tables" (Condition screening) unless they contain the only yield data.
            3. IGNORE "Mechanism Schemes" or "X-ray structures" or "General Reaction Schemes" (unless they list specific scope yields).
            4. IGNORE "Spectra" (NMR, HPLC traces).
            
            Return a JSON object with two lists: "selected_figures" and "selected_tables".
            Use the exact filenames provided.
            """
        
        user_prompt = f"""
        Paper Text Summary:
        {truncated_text}
        
        {asset_list_str}
        
        Return JSON format: {{ "selected_figures": [...], "selected_tables": [...] }}
        """

        # 4. Call LLM
        print("Consulting LLM for Asset Selection...")
        try:
            response = self.llm.chat(system_prompt, user_prompt)
            print(f"--- [DEBUG] Raw LLM Response ---\n{response}\n--------------------------------")
            
            # Parse JSON
            cleaned_json = response
            if "```" in response:
                cleaned_json = response.split("```json")[-1].split("```")[0].strip()
            cleaned_json = cleaned_json.strip('`').strip()
            if cleaned_json.startswith('json'): cleaned_json = cleaned_json[4:]
            
            # The LLM returns { "selected_items": [ ... ] } OR { "selected_figures": [], "selected_tables": [] }
            llm_output = json.loads(cleaned_json)
            selected_items = llm_output.get("selected_items", [])
            
            # If empty, try the other format
            if not selected_items:
                selected_items.extend(llm_output.get("selected_figures", []))
                selected_items.extend(llm_output.get("selected_tables", []))
            
            # Convert to flat lists for downstream compatibility
            # Compatibility keys: "selected_figures", "selected_tables"
            sel_figs = []
            sel_tabs = []
            
            for item in selected_items:
                fname = item.get("filename")
                if not fname: continue
                
                # Robust Matching Strategy
                # 1. Exact Match
                if fname in figures:
                    sel_figs.append(fname)
                    continue
                if fname in tables:
                    sel_tabs.append(fname)
                    continue
                    
                # 2. Fuzzy Match (if LLM omitted extension or used partial name)
                # Check if fname is substring of any real file (or vice versa)
                matched = False
                
                # Check Figures
                for real_fig in figures:
                    # e.g. LLM says "Figure 1", Real is "page_2_Figure_1.png"
                    # OR LLM says "page_2_Figure_1" (no .png)
                    if fname in real_fig or real_fig in fname:
                        # Ensure we don't match "Table 1" to "Table 10" loosely, but here names are distinct usually.
                        # safer: check if filename without extension matches
                        base_real = os.path.splitext(real_fig)[0]
                        base_llm = os.path.splitext(fname)[0]
                        if base_llm in base_real or base_real in base_llm:
                            sel_figs.append(real_fig)
                            # Update item filename to real one for downstream consistency?
                            item['filename'] = real_fig 
                            matched = True
                            print(f"   [Fuzzy Match] '{fname}' -> '{real_fig}'")
                            break
                if matched: continue

                # Check Tables
                for real_tab in tables:
                    if fname in real_tab or real_tab in fname:
                         base_real = os.path.splitext(real_tab)[0]
                         base_llm = os.path.splitext(fname)[0]
                         if base_llm in base_real or base_real in base_llm:
                            sel_tabs.append(real_tab)
                            item['filename'] = real_tab
                            matched = True
                            print(f"   [Fuzzy Match] '{fname}' -> '{real_tab}'")
                            break
                if matched: continue
                
                print(f"   [Warning] Agent selected '{fname}' but it wasn't found in assets.") 

            print(f"Agent Selected: {len(sel_figs)} Figures, {len(sel_tabs)} Tables")
            
            # Return the RICH object, but inject the compatibility keys
            result = {
                "selected_figures": sel_figs,
                "selected_tables": sel_tabs,
                "rich_metadata": selected_items
            }
            return result

        except Exception as e:
            print(f"Selection failed: {e}")
            # Fallback: Select ALL
            print("Fallback: Selecting ALL assets.")
            return {"selected_figures": figures, "selected_tables": tables, "rich_metadata": []}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()
    
    selector = AssetSelector()
    selection = selector.select_assets(args.pdf_path)
    
    if selection:
        basename = os.path.splitext(os.path.basename(args.pdf_path))[0]
        out_path = os.path.join("data/intermediate", basename, "selected_assets.json")
        with open(out_path, 'w') as f:
            json.dump(selection, f, indent=2)
        print(f"Selection saved to {out_path}")
        
        # Print JSON for subprocess capture
        print("---JSON_START---")
        print(json.dumps(selection))
        print("---JSON_END---")

if __name__ == "__main__":
    main()
