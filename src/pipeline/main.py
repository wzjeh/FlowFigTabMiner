import os
import sys
import argparse
import glob
import subprocess

# Ensure src is importable from project root
sys.path.insert(0, os.getcwd())

from src.pipeline.figure_pipeline import FigurePipeline
from src.extraction.table.pipeline import TablePipeline
from src.adjudication.global_assembly import GlobalAssembly

def run_step1_tfid(pdf_path):
    print("\n=== Step 1: TF-ID Parsing ===")
    # Using existing script wrapper for now as Step 1 logic is simple but relies on subprocess or specific imports
    # scripts/step1_tfid.py exists. We can import its logic or subprocess call it.
    # To avoid import issues if it's not in src, let's subprocess it for safety/cleanliness, 
    # or better: refactor step1 logic into src/parsing/tfid.py later.
    # For now, subprocess is robust.
    
    cmd = [sys.executable, "scripts/step1_tfid.py", pdf_path]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("Step 1 Failed.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="FlowFigTabMiner Unified Pipeline")
    parser.add_argument("pdf_path", help="Path to input PDF")
    args = parser.parse_args()
    
    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    intermediate_dir = os.path.join("data/intermediate", basename)

    # 1. TF-ID
    if not run_step1_tfid(pdf_path):
        return

    # 2. Figure Pipeline (Path A)
    print("\n=== Step 2-4: Figure Extraction ===")
    fig_pipeline = FigurePipeline()
    fig_pipeline.process_pdf_figures(pdf_path)

    # 3. Table Pipeline (Path B)
    print("\n=== Step Table: Table Extraction ===")
    tab_pipeline = TablePipeline()
    
    # Needs to find table images extracted by TF-ID
    tables_dir = os.path.join(intermediate_dir, "tables")
    if os.path.exists(tables_dir):
        table_imgs = glob.glob(os.path.join(tables_dir, "*.png"))
        # Filter out debug crops if any leak in
        table_imgs = [f for f in table_imgs if "_body" not in f and "_crop" not in f]
        
        print(f"Processing {len(table_imgs)} tables...")
        for t_img in table_imgs:
            try:
                # Output to a subfolder per table to keep clean
                t_base = os.path.splitext(os.path.basename(t_img))[0]
                t_out = os.path.join(tables_dir, t_base)
                tab_pipeline.process_table(t_img, output_dir=tables_dir) # process_table creates subfolder logic?
                # Looking at table_pipeline.py:
                # if output_dir: table_output_dir = os.path.join(output_dir, table_basename)
                # So yes, we pass tables_dir and it creates the subfolder.
            except Exception as e:
                print(f"Error processing table {t_img}: {e}")
    else:
        print("No tables directory found.")

    # 4. Global Assembly
    print("\n=== Step 5: Global Assembly ===")
    assembler = GlobalAssembly()
    assembler.run(pdf_path, intermediate_dir)

    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
