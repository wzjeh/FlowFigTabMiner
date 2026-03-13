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
    parser.add_argument("--skip-tfid", action="store_true",
                        help="Skip Step 1 if intermediate figures already exist")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    intermediate_dir = os.path.join("data/intermediate", basename)

    # 1. TF-ID
    figures_exist = len(glob.glob(os.path.join(intermediate_dir, "figures", "*.png"))) > 0
    if args.skip_tfid and figures_exist:
        print("\n=== Step 1: TF-ID Parsing (SKIPPED - intermediate figures found) ===")
    elif not run_step1_tfid(pdf_path):
        return

    # 2. Figure Pipeline (Path A)
    print("\n=== Step 2-4: Figure Extraction ===")
    fig_pipeline = FigurePipeline()
    fig_pipeline.process_pdf_figures(pdf_path)
    del fig_pipeline
    import gc; gc.collect()

    # 3. Table Pipeline (Path B)
    print("\n=== Step Table: Table Extraction ===")
    tab_pipeline = TablePipeline(sequential_mode=True)
    
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

    # 3.5. Tab-Scheme-Seg
    print("\n=== Step 3.5: Tab-Scheme-Seg (Scheme Parsing) ===")
    import json as _json
    from src.utils.config import load_config as _load_config

    reactant_pool = {}
    product_pool = {}
    compound_pool = {}
    scheme_conditions_texts = []

    if os.path.exists(tables_dir):
        scheme_imgs = glob.glob(os.path.join(tables_dir, "**", "*_table_scheme_*.png"), recursive=True)
        if scheme_imgs:
            from src.extraction.table.scheme_seg_parser import SchemeSegParser
            _cfg = _load_config()
            _scheme_cfg = _cfg.get("tables", {}).get("scheme_parsing", {})
            scheme_parser = SchemeSegParser(
                model_path=_scheme_cfg.get("model_path", "models/tab-scheme-seg/best.pt"),
                conf_threshold=_scheme_cfg.get("confidence_threshold", 0.3)
            )
            for sp in scheme_imgs:
                print(f"   Processing scheme: {os.path.basename(sp)}")
                result = scheme_parser.parse_scheme(sp)
                reactant_pool.update(result.get("reactant_pool", {}))
                product_pool.update(result.get("product_pool", {}))
                compound_pool.update(result.get("compound_pool", {}))
                ct = result.get("conditions_text", "")
                if ct:
                    scheme_conditions_texts.append(ct)
            del scheme_parser
            import gc; gc.collect()
            print(f"   -> reactant_pool: {len(reactant_pool)} entries, product_pool: {len(product_pool)} entries, compound_pool: {len(compound_pool)} entries")
        else:
            print("   No scheme images found.")
    else:
        print("   No tables directory found, skipping scheme parsing.")

    if reactant_pool or product_pool or compound_pool:
        pool_path = os.path.join(intermediate_dir, "compound_pool.json")
        pool_to_save = {
            "reactant_pool": reactant_pool,
            "product_pool": product_pool,
            "compound_pool": compound_pool,
        }
        with open(pool_path, 'w') as f:
            _json.dump(pool_to_save, f, indent=2)
        total = len(reactant_pool) + len(product_pool) + len(compound_pool)
        print(f"   -> Compound pool ({total} total) -> {pool_path}")
    if scheme_conditions_texts:
        cond_path = os.path.join(intermediate_dir, "scheme_conditions.txt")
        with open(cond_path, 'w') as f:
            f.write("\n".join(scheme_conditions_texts))
        print(f"   -> Scheme conditions saved -> {cond_path}")

    # 4.5. Sub-Variable Libraries
    print("\n=== Step 4.5: Build Sub-Variable Libraries ===")
    import json as _json2
    from src.adjudication.local_vars_builder import LocalVarsBuilder
    from src.adjudication.pdf_parser import PDFParser

    paper_text = PDFParser().extract_text(pdf_path)  # cached by PDFParser

    local_vars_dir = os.path.join(intermediate_dir, "local_vars")
    os.makedirs(local_vars_dir, exist_ok=True)
    builder = LocalVarsBuilder()

    # Figure evidence
    macro_cleaned_dir = os.path.join(intermediate_dir, "macro_cleaned")
    for jpath in glob.glob(os.path.join(macro_cleaned_dir, "*_evidence.json")):
        try:
            with open(jpath) as f:
                ev = _json2.load(f)
            src_id = ev.get("meta", {}).get("figure_id", os.path.basename(jpath).replace("_evidence.json", ""))
            builder.build(src_id, "figure", ev, paper_text, local_vars_dir)
        except Exception as e:
            print(f"   [LocalVars] Figure error {jpath}: {e}")

    # Table evidence
    if os.path.exists(tables_dir):
        for root, _, files in os.walk(tables_dir):
            for fname in files:
                if fname.endswith("_evidence.json"):
                    ev_path = os.path.join(root, fname)
                    try:
                        with open(ev_path) as f:
                            ev = _json2.load(f)
                        src_id = fname.replace("_evidence.json", "")
                        csv_path = ev.get("csv_path", "")
                        csv_head = ""
                        if csv_path and os.path.exists(csv_path):
                            with open(csv_path) as cf:
                                csv_head = "".join(cf.readlines()[:6])
                        builder.build(src_id, "table", ev, paper_text, local_vars_dir, csv_head=csv_head)
                    except Exception as e:
                        print(f"   [LocalVars] Table error {ev_path}: {e}")

    # 4. Global Assembly
    print("\n=== Step 5: Global Assembly ===")
    assembler = GlobalAssembly()
    assembler.run(pdf_path, intermediate_dir)

    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
