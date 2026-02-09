import os
import json
import glob
from src.adjudication.llm_engine import LLMEngine
from src.adjudication.pdf_parser import PDFParser

class GlobalAssembly:
    def __init__(self, output_dir="data/final_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.llm = LLMEngine()
        self.pdf_parser = PDFParser()

    def run(self, pdf_path, intermediate_dir=None):
        """
        Run global assembly for a PDF.
        """
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        if not intermediate_dir:
            intermediate_dir = os.path.join("data/intermediate", basename)
        
        print(f"[GlobalAssembly] Assembling final dataset for {basename}...")
        
        # 1. Get Text
        full_text = self.pdf_parser.extract_text(pdf_path)
        
        # 2. Get Figure Data
        # Evidence JSONs are usually in data/evidence, named {figure_id}_evidence.json
        # Figure IDs usually contain the pdf basename or page info?
        # The assembler saves them to data/evidence.
        # We need to filter for this PDF.
        # If Figure ID scheme is page_X_figure_Y, we might need a mapping.
        # Currently, FigurePipeline processes images in `data/intermediate/{basename}/figures`.
        # The generated JSONs are in `data/evidence`.
        # Problem: 'data/evidence' is a flat folder? 
        # Yes, based on scripts/run_steps2_to_4.py defaulting to EvidenceAssembler default.
        
        # We should look for JSONs that correspond to this PDF.
        # If FigurePipeline used `figure_id` derived from filename like `page_0_figure_0`, 
        # it doesn't strictly have the PDF basename in it unless we added it.
        # TF-ID naming convention: `page_{page_num}_figure_{fig_num}.png` inside `{basename}/figures`.
        # So the IDs are generic "page_0_figure_0". 
        # This implies `data/evidence` might mix evidence from different PDFs if we are not careful?
        # Or maybe we assume we run one PDF at a time and clear evidence?
        # Or we should check `meta['source_intermediate_dir']` inside JSONs?
        
        evidence_dir = "data/evidence"
        figure_data = []
        if os.path.exists(evidence_dir):
            all_jsons = glob.glob(os.path.join(evidence_dir, "*.json"))
            for jpath in all_jsons:
                try:
                    with open(jpath, 'r') as f:
                        data = json.load(f)
                    
                    # check source
                    meta = data.get('meta', {})
                    source_dir = meta.get('source_intermediate_dir', '')
                    if basename in source_dir:
                        figure_data.append(data)
                except: pass
                
        print(f"   -> Found {len(figure_data)} figure evidence packets.")
        
        # 3. Get Table Data
        # TablePipeline saves CSVs and internal JSONs in `data/intermediate/{basename}/tables/`
        tables_dir = os.path.join(intermediate_dir, "tables")
        table_data = []
        if os.path.exists(tables_dir):
            # Recurse? Or just check subfolders
            # Logic: For each subfolder in tables_dir, look for .csv
            for root, dirs, files in os.walk(tables_dir):
                for file in files:
                    if file.endswith(".csv"):
                         csv_path = os.path.join(root, file)
                         # Load CSV content
                         with open(csv_path, 'r') as f:
                             csv_content = f.read()
                         table_data.append({
                             "table_name": file,
                             "content": csv_content
                         })
        
        print(f"   -> Found {len(table_data)} table data packets.")

        # 4. LLM Prompt
        # Construct the prompt
        system_prompt = "You are an expert material scientist. Your goal is to assemble a final dataset from extracted figures and tables, using the full paper text to strictly verify and enrich the data."
        
        user_prompt = f"""
        Paper Text (Truncated):
        {full_text[:50000]} # Hard cap just in case

        --- Extracted Tables ---
        {json.dumps(table_data, indent=2)}

        --- Extracted Figures ---
        {json.dumps(figure_data, indent=2)}

        --- Request ---
        1. contextualize the data points (e.g. adding reaction conditions found in text).
        2. assemble the final dataset where the core is the data points from figures/tables.
        3. Output strict JSON format.
        """
        
        print("   -> Sending to LLM...")
        response = self.llm.chat(system_prompt, user_prompt)
        
        # 5. Save Output
        out_file = os.path.join(self.output_dir, f"{basename}_final.json")
        with open(out_file, 'w') as f:
            # Try to save raw response or parsed JSON if possible
            f.write(response) # LLMEngine returns string usually
            
        print(f"   -> Saved final result to {out_file}")
        return out_file
